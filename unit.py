"""Losses, structure teacher, metrics, and reproducibility utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

from model import CausalMVC, ModelOutput


# Frozen stable-recipe constants. They are not routine command-line knobs.
NUM_CLUSTERS = 5
TEACHER_DIM = 256
RIDGE_ALPHA = 1.0
PROTOTYPE_TEMPERATURE = 0.2
PROTOTYPE_MOMENTUM = 0.9


@dataclass(frozen=True)
class TeacherOutput:
    views: list[np.ndarray]
    pseudo_labels: np.ndarray
    diagnostics: dict


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    size = max(int(y_true.max()), int(y_pred.max())) + 1
    table = np.zeros((size, size), dtype=np.int64)
    for predicted, true in zip(y_pred, y_true):
        table[predicted, true] += 1
    rows, columns = linear_sum_assignment(table.max() - table)
    return float(table[rows, columns].sum() / len(y_true))


def purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = 0
    for cluster in np.unique(y_pred):
        members = y_true[y_pred == cluster]
        total += np.bincount(members).max()
    return float(total / len(y_true))


def clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "ACC": cluster_accuracy(y_true, y_pred),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "PUR": purity(y_true, y_pred),
    }


def kmeans_metrics(embedding: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    prediction = KMeans(NUM_CLUSTERS, n_init=20, random_state=seed).fit_predict(embedding)
    return clustering_metrics(labels, prediction)


def recover_teacher_views(full_views: list[np.ndarray], mask: np.ndarray, seed: int) -> list[np.ndarray]:
    """Build leakage-free PCA spaces and recover missing teacher features with Ridge."""

    paired = mask.astype(bool).all(axis=1)
    if not paired.any():
        raise ValueError("The structure teacher requires at least one complete sample")

    projected: list[np.ndarray] = []
    for view, features in enumerate(full_views):
        observed = mask[:, view].astype(bool)
        dimension = min(TEACHER_DIM, features.shape[1], int(observed.sum()))
        pca = PCA(dimension, random_state=seed).fit(features[observed])
        latent = np.zeros((len(features), dimension), dtype=np.float32)
        latent[observed] = pca.transform(features[observed]).astype(np.float32)
        projected.append(latent)

    recovered = [latent.copy() for latent in projected]
    for target, source in ((0, 1), (1, 0)):
        missing = ~mask[:, target].astype(bool)
        mapper = Ridge(RIDGE_ALPHA).fit(projected[source][paired], projected[target][paired])
        recovered[target][missing] = mapper.predict(projected[source][missing]).astype(np.float32)
    return [Normalizer().fit_transform(latent).astype(np.float32) for latent in recovered]


def graph_from_embedding(embedding: np.ndarray, neighbors: int) -> csr_matrix:
    distances, indices = NearestNeighbors(
        n_neighbors=neighbors + 1, metric="cosine", n_jobs=-1
    ).fit(embedding).kneighbors(embedding)
    distances, indices = distances[:, 1:], indices[:, 1:]
    local_scale = np.maximum(distances[:, -1:], 1e-6)
    weights = np.exp(-np.square(distances / local_scale)).ravel()
    rows = np.repeat(np.arange(len(embedding)), neighbors)
    graph = csr_matrix(
        (weights, (rows, indices.ravel())), shape=(len(embedding), len(embedding))
    )
    return graph.maximum(graph.T)


def select_graph_labels(embedding: np.ndarray) -> tuple[np.ndarray, dict]:
    """Choose graph density using cluster balance and cosine silhouette only."""

    expected_size = len(embedding) / NUM_CLUSTERS
    diagnostics: dict = {}
    pseudo_labels: np.ndarray | None = None
    for neighbors in (20, 25, 30, 35, 40):
        graph = graph_from_embedding(embedding, neighbors)
        pseudo_labels = SpectralClustering(
            NUM_CLUSTERS,
            affinity="precomputed",
            assign_labels="cluster_qr",
            random_state=0,
        ).fit_predict(graph).astype(np.int64)
        sizes = np.bincount(pseudo_labels, minlength=NUM_CLUSTERS)
        silhouette = float(
            silhouette_score(
                embedding,
                pseudo_labels,
                metric="cosine",
                sample_size=min(1000, len(embedding)),
                random_state=0,
            )
        )
        diagnostics = {
            "graph_neighbors": neighbors,
            "silhouette": silhouette,
            "cluster_sizes": sizes.tolist(),
        }
        balanced = sizes.min() >= 0.5 * expected_size and sizes.max() <= 1.5 * expected_size
        if balanced and silhouette >= 0.187:
            break
    assert pseudo_labels is not None
    return pseudo_labels, diagnostics


def _svc_vote(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    predict_features: np.ndarray,
    c_value: float,
) -> tuple[float, np.ndarray]:
    classifier = SVC(C=c_value, gamma="scale", random_state=0)
    class_counts = np.bincount(train_labels, minlength=NUM_CLUSTERS)
    present_counts = class_counts[class_counts > 0]
    folds = min(3, int(present_counts.min()))
    if folds >= 2:
        cv = StratifiedKFold(folds, shuffle=True, random_state=0)
        score = float(cross_val_score(classifier, train_features, train_labels, cv=cv, n_jobs=1).mean())
    else:
        score = 0.2
    classifier.fit(train_features, train_labels)
    decisions = classifier.decision_function(predict_features)
    if decisions.ndim == 1:
        decisions = np.column_stack((-decisions, decisions))
    probabilities = softmax(decisions, axis=1)
    full_probabilities = np.zeros((len(predict_features), NUM_CLUSTERS), dtype=np.float64)
    full_probabilities[:, classifier.classes_.astype(int)] = probabilities
    return score, full_probabilities


def refine_view1_missing_labels(
    full_views: list[np.ndarray],
    mask: np.ndarray,
    pseudo_labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Refine the empirically weak BDGP direction: view 0 observed, view 1 missing."""

    paired = mask.astype(bool).all(axis=1)
    weak_missing = (mask[:, 0] == 1) & (mask[:, 1] == 0)
    if not weak_missing.any():
        return pseudo_labels.copy(), {"refined_samples": 0, "svc_weights": []}

    observed0 = mask[:, 0].astype(bool)
    observed1 = mask[:, 1].astype(bool)
    votes: list[np.ndarray] = []
    weights: list[float] = []

    view0_features: dict[str, np.ndarray] = {"raw": full_views[0]}
    for requested_dim in (128, 512):
        dimension = min(requested_dim, full_views[0].shape[1], int(observed0.sum()))
        pca = PCA(dimension, random_state=0).fit(full_views[0][observed0])
        view0_features[f"pca_{requested_dim}"] = pca.transform(full_views[0]).astype(np.float32)

    for name, c_value in (("pca_128", 1.0), ("pca_512", 10.0), ("raw", 10.0)):
        score, vote = _svc_vote(
            view0_features[name][paired], pseudo_labels[paired], view0_features[name][weak_missing], c_value
        )
        votes.append(vote)
        weights.append(max(score - 0.2, 0.0))

    pca0_dim = min(TEACHER_DIM, full_views[0].shape[1], int(observed0.sum()))
    pca1_dim = min(79, full_views[1].shape[1], int(observed1.sum()))
    pca0 = PCA(pca0_dim, random_state=seed).fit(full_views[0][observed0])
    pca1 = PCA(pca1_dim, random_state=seed).fit(full_views[1][observed1])
    latent0 = pca0.transform(full_views[0])
    latent1 = pca1.transform(full_views[1])
    mapper = Ridge(RIDGE_ALPHA).fit(latent0[paired], latent1[paired])
    simulated = np.concatenate(
        (
            Normalizer().fit_transform(latent0),
            Normalizer().fit_transform(mapper.predict(latent0)),
        ),
        axis=1,
    )
    candidates = [
        (*_svc_vote(simulated[paired], pseudo_labels[paired], simulated[weak_missing], c_value), c_value)
        for c_value in (0.1, 1.0, 10.0)
    ]
    score, vote, _ = max(candidates, key=lambda item: item[0])
    votes.append(vote)
    weights.append(max(score - 0.2, 0.0))

    refined = pseudo_labels.copy()
    weight_sum = sum(weights)
    if weight_sum > 1e-12:
        combined = sum(weight * vote for weight, vote in zip(weights, votes)) / weight_sum
        refined[weak_missing] = combined.argmax(axis=1)
    return refined, {"refined_samples": int(weak_missing.sum()), "svc_weights": weights}


def build_structure_teacher(full_views: list[np.ndarray], mask: np.ndarray, seed: int) -> TeacherOutput:
    """Return continuous teacher views and label-free cluster assignments."""

    recovered = recover_teacher_views(full_views, mask, seed)
    embedding = np.concatenate(recovered, axis=1)
    pseudo_labels, graph_diagnostics = select_graph_labels(embedding)
    pseudo_labels, refinement_diagnostics = refine_view1_missing_labels(
        full_views, mask, pseudo_labels, seed
    )
    diagnostics = {**graph_diagnostics, **refinement_diagnostics}
    return TeacherOutput(recovered, pseudo_labels, diagnostics)


def masked_reconstruction_loss(
    output: ModelOutput, views: list[torch.Tensor], mask: torch.Tensor
) -> torch.Tensor:
    losses = []
    for view in range(2):
        valid = mask[:, view]
        if valid.any():
            losses.append(F.mse_loss(output.reconstructions[view][valid], views[view][valid]))
    if not losses:
        raise ValueError("A batch cannot have every view missing")
    return torch.stack(losses).mean()


def prototype_loss(common: torch.Tensor, targets: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    similarity = F.normalize(common, dim=1) @ F.normalize(prototypes, dim=1).t()
    return F.cross_entropy(similarity / PROTOTYPE_TEMPERATURE, targets)


@torch.no_grad()
def update_prototypes(
    model: CausalMVC,
    commons: list[torch.Tensor],
    targets: torch.Tensor,
    valid_masks: list[torch.Tensor],
) -> None:
    for cluster in range(model.num_clusters):
        members = []
        for view in range(2):
            selected = valid_masks[view] & (targets == cluster)
            if selected.any():
                members.append(commons[view][selected])
        if members:
            center = F.normalize(torch.cat(members).mean(dim=0), dim=0)
            model.prototypes[cluster].mul_(PROTOTYPE_MOMENTUM).add_(
                center * (1.0 - PROTOTYPE_MOMENTUM)
            )
    model.prototypes.copy_(F.normalize(model.prototypes, dim=1))


def linear_structure_alignment(common: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Linear centered-kernel alignment of relations; higher is better."""

    common = F.normalize(common - common.mean(dim=0, keepdim=True), dim=1)
    teacher = F.normalize(teacher - teacher.mean(dim=0, keepdim=True), dim=1)
    cross = common.t().matmul(teacher).pow(2).sum()
    common_norm = common.t().matmul(common).pow(2).sum().sqrt()
    teacher_norm = teacher.t().matmul(teacher).pow(2).sum().sqrt()
    return cross / (common_norm * teacher_norm).clamp_min(1e-8)


def cross_view_loss(
    model: CausalMVC,
    output: ModelOutput,
    views: list[torch.Tensor],
    paired: torch.Tensor,
) -> torch.Tensor:
    if not paired.any():
        return output.latents[0].new_zeros(())

    predicted1 = model.cross_predictors[0](output.latents[0][paired])
    predicted0 = model.cross_predictors[1](output.latents[1][paired])
    latent_loss = (
        1.0 - F.cosine_similarity(predicted1, output.latents[1][paired].detach()).mean()
        + 1.0 - F.cosine_similarity(predicted0, output.latents[0][paired].detach()).mean()
    ) / 2.0
    feature_loss = (
        F.mse_loss(model.decode_latent(predicted1, 1), views[1][paired])
        + F.mse_loss(model.decode_latent(predicted0, 0), views[0][paired])
    ) / 2.0
    return latent_loss + feature_loss
