"""Losses, structure teacher, metrics, and reproducibility utilities."""

from __future__ import annotations

import math
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


# Stable internal defaults. Dataset-dependent quantities are derived at runtime.
TEACHER_DIM = 256
RIDGE_ALPHA = 1.0
PROTOTYPE_TEMPERATURE = 0.2
PROTOTYPE_MOMENTUM = 0.9


@dataclass(frozen=True)
class TeacherOutput:
    views: list[np.ndarray]
    pseudo_labels: np.ndarray
    probabilities: np.ndarray
    confidence: np.ndarray
    sample_weights: np.ndarray
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


def kmeans_metrics(
    embedding: np.ndarray, labels: np.ndarray, seed: int, num_clusters: int
) -> dict[str, float]:
    prediction = KMeans(num_clusters, n_init=20, random_state=seed).fit_predict(embedding)
    return clustering_metrics(labels, prediction)


def recover_teacher_views(full_views: list[np.ndarray], mask: np.ndarray, seed: int) -> list[np.ndarray]:
    """Build leakage-free PCA spaces and recover missing teacher features with Ridge."""

    paired = mask.astype(bool).all(axis=1)
    if not paired.any():
        raise ValueError("The structure teacher requires at least one complete sample")

    projected: list[np.ndarray] = []
    for view, features in enumerate(full_views):
        observed = mask[:, view].astype(bool)
        dimension = min(TEACHER_DIM, features.shape[1], max(1, int(observed.sum()) - 1))
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
    neighbors = min(max(1, neighbors), len(embedding) - 1)
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


def _neighbor_candidates(num_samples: int, num_clusters: int) -> list[int]:
    """Scale graph density with the expected local cluster population."""

    upper = max(1, min(50, num_samples - 1))
    scale = math.sqrt(max(num_samples / max(num_clusters, 1), 1.0))
    values = {
        min(upper, max(2, int(round(scale * multiplier))))
        for multiplier in (0.5, 1.0, 2.0)
    }
    return sorted(values)


def _align_assignments(
    reference: np.ndarray, candidate: np.ndarray, num_clusters: int
) -> np.ndarray:
    """Align arbitrary spectral-cluster IDs without consulting ground truth."""

    table = np.zeros((num_clusters, num_clusters), dtype=np.int64)
    for source, target in zip(candidate, reference):
        table[int(source), int(target)] += 1
    rows, columns = linear_sum_assignment(table.max() - table)
    mapping = np.arange(num_clusters, dtype=np.int64)
    mapping[rows] = columns
    return mapping[candidate]


def select_graph_labels(
    embedding: np.ndarray, num_clusters: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build a soft graph ensemble without assuming balanced true classes."""

    candidates: list[dict] = []
    for neighbors in _neighbor_candidates(len(embedding), num_clusters):
        graph = graph_from_embedding(embedding, neighbors)
        labels = SpectralClustering(
            num_clusters,
            affinity="precomputed",
            assign_labels="cluster_qr",
            random_state=0,
        ).fit_predict(graph).astype(np.int64)
        sizes = np.bincount(labels, minlength=num_clusters)
        proportions = sizes / max(len(labels), 1)
        entropy = float(
            -(proportions[proportions > 0] * np.log(proportions[proportions > 0])).sum()
            / max(math.log(num_clusters), 1e-12)
        )
        silhouette = float(
            silhouette_score(
                embedding,
                labels,
                metric="cosine",
                sample_size=min(1000, len(embedding)),
                random_state=0,
            )
        )
        candidates.append(
            {
                "neighbors": neighbors,
                "labels": labels,
                "sizes": sizes,
                "silhouette": silhouette,
                "entropy": entropy,
                "collapsed": bool(proportions.max() > 0.8 or np.count_nonzero(sizes) < num_clusters),
            }
        )

    admissible = [item for item in candidates if not item["collapsed"]] or candidates
    selected = max(admissible, key=lambda item: (item["silhouette"], item["entropy"]))
    aligned = [
        _align_assignments(selected["labels"], item["labels"], num_clusters)
        for item in candidates
    ]
    probabilities = np.zeros((len(embedding), num_clusters), dtype=np.float64)
    for labels in aligned:
        probabilities[np.arange(len(labels)), labels] += 1.0
    probabilities /= len(aligned)
    pseudo_labels = probabilities.argmax(axis=1).astype(np.int64)
    confidence = probabilities.max(axis=1).astype(np.float32)
    stability = float(
        np.mean(
            [adjusted_rand_score(selected["labels"], item["labels"]) for item in candidates]
        )
    )
    diagnostics = {
        "graph_neighbors": int(selected["neighbors"]),
        "silhouette": float(selected["silhouette"]),
        "cluster_sizes": np.bincount(pseudo_labels, minlength=num_clusters).tolist(),
        "cluster_entropy": float(selected["entropy"]),
        "graph_stability": stability,
        "graph_candidates": [
            {
                "neighbors": int(item["neighbors"]),
                "silhouette": float(item["silhouette"]),
                "cluster_entropy": float(item["entropy"]),
                "collapsed": bool(item["collapsed"]),
            }
            for item in candidates
        ],
    }
    return pseudo_labels, probabilities.astype(np.float32), confidence, diagnostics


def _svc_vote(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    predict_features: np.ndarray,
    c_value: float,
    num_clusters: int,
) -> tuple[float, np.ndarray]:
    if np.unique(train_labels).size < 2:
        return 0.0, np.full(
            (len(predict_features), num_clusters), 1.0 / num_clusters, dtype=np.float64
        )
    classifier = SVC(C=c_value, gamma="scale", random_state=0)
    class_counts = np.bincount(train_labels, minlength=num_clusters)
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
    full_probabilities = np.zeros((len(predict_features), num_clusters), dtype=np.float64)
    full_probabilities[:, classifier.classes_.astype(int)] = probabilities
    return score, full_probabilities


def _pca_features(
    features: np.ndarray, observed: np.ndarray, seed: int
) -> dict[str, np.ndarray]:
    spaces: dict[str, np.ndarray] = {"raw": features}
    maximum = max(1, int(observed.sum()) - 1)
    for requested in (128, TEACHER_DIM):
        dimension = min(requested, features.shape[1], maximum)
        name = f"pca_{dimension}"
        if name not in spaces:
            pca = PCA(dimension, random_state=seed).fit(features[observed])
            spaces[name] = pca.transform(features).astype(np.float32)
    return spaces


def refine_missing_probabilities(
    full_views: list[np.ndarray],
    mask: np.ndarray,
    probabilities: np.ndarray,
    seed: int,
    num_clusters: int,
) -> tuple[np.ndarray, dict]:
    """Refine both missing directions using only the genuinely observed source."""

    paired = mask.astype(bool).all(axis=1)
    labels = probabilities.argmax(axis=1)
    refined = probabilities.astype(np.float64).copy()
    chance = 1.0 / num_clusters
    direction_diagnostics: list[dict] = []

    for target, source in ((0, 1), (1, 0)):
        missing = (mask[:, target] == 0) & (mask[:, source] == 1)
        if not missing.any():
            direction_diagnostics.append(
                {"source": source, "target": target, "samples": 0, "weights": []}
            )
            continue

        source_observed = mask[:, source].astype(bool)
        target_observed = mask[:, target].astype(bool)
        feature_spaces = _pca_features(full_views[source], source_observed, seed)
        votes: list[np.ndarray] = []
        weights: list[float] = []
        for name, features in feature_spaces.items():
            c_value = 10.0 if name == "raw" else 1.0
            score, vote = _svc_vote(
                features[paired], labels[paired], features[missing], c_value, num_clusters
            )
            votes.append(vote)
            weights.append(max(score - chance, 0.0))

        source_dim = min(
            TEACHER_DIM, full_views[source].shape[1], max(1, int(source_observed.sum()) - 1)
        )
        target_dim = min(
            TEACHER_DIM, full_views[target].shape[1], max(1, int(target_observed.sum()) - 1)
        )
        pca_source = PCA(source_dim, random_state=seed).fit(full_views[source][source_observed])
        pca_target = PCA(target_dim, random_state=seed).fit(full_views[target][target_observed])
        source_latent = pca_source.transform(full_views[source])
        target_latent = pca_target.transform(full_views[target])
        mapper = Ridge(RIDGE_ALPHA).fit(source_latent[paired], target_latent[paired])
        simulated = np.concatenate(
            (
                Normalizer().fit_transform(source_latent),
                Normalizer().fit_transform(mapper.predict(source_latent)),
            ),
            axis=1,
        )
        simulated_candidates = [
            (*_svc_vote(simulated[paired], labels[paired], simulated[missing], c, num_clusters), c)
            for c in (0.1, 1.0, 10.0)
        ]
        score, vote, selected_c = max(simulated_candidates, key=lambda item: item[0])
        votes.append(vote)
        weights.append(max(score - chance, 0.0))

        weight_sum = sum(weights)
        if weight_sum > 1e-12:
            combined = refined[missing] + sum(
                weight * vote for weight, vote in zip(weights, votes)
            )
            refined[missing] = combined / (1.0 + weight_sum)
        direction_diagnostics.append(
            {
                "source": source,
                "target": target,
                "samples": int(missing.sum()),
                "weights": [float(weight) for weight in weights],
                "simulated_c": float(selected_c),
            }
        )

    refined /= refined.sum(axis=1, keepdims=True).clip(min=1e-12)
    return refined.astype(np.float32), {
        "refined_samples": int((~mask.astype(bool)).sum()),
        "refinement_directions": direction_diagnostics,
    }


def build_structure_teacher(
    full_views: list[np.ndarray], mask: np.ndarray, seed: int, num_clusters: int
) -> TeacherOutput:
    """Return continuous teacher views and label-free cluster assignments."""

    recovered = recover_teacher_views(full_views, mask, seed)
    embedding = np.concatenate(recovered, axis=1)
    _, probabilities, _, graph_diagnostics = select_graph_labels(embedding, num_clusters)
    probabilities, refinement_diagnostics = refine_missing_probabilities(
        full_views, mask, probabilities, seed, num_clusters
    )
    pseudo_labels = probabilities.argmax(axis=1).astype(np.int64)
    confidence = probabilities.max(axis=1).astype(np.float32)
    confidence_floor = float(np.quantile(confidence, 0.25))
    if confidence_floor >= 1.0 - 1e-6:
        sample_weights = np.ones_like(confidence)
    else:
        sample_weights = np.clip(
            (confidence - confidence_floor) / max(1.0 - confidence_floor, 1e-6), 0.0, 1.0
        ).astype(np.float32)
    for cluster in range(num_clusters):
        members = np.flatnonzero(pseudo_labels == cluster)
        if members.size and not np.any(sample_weights[members] > 0):
            sample_weights[members[np.argmax(confidence[members])]] = 1.0
    diagnostics = {
        **graph_diagnostics,
        **refinement_diagnostics,
        "confidence_floor": confidence_floor,
        "confidence_mean": float(confidence.mean()),
        "confidence_min": float(confidence.min()),
        "high_confidence_samples": int((sample_weights > 0).sum()),
    }
    return TeacherOutput(
        recovered,
        pseudo_labels,
        probabilities,
        confidence,
        sample_weights,
        diagnostics,
    )


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


def prototype_loss(
    common: torch.Tensor,
    soft_targets: torch.Tensor,
    sample_weights: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    """Confidence-weighted soft distillation normalized for the class count."""

    similarity = F.normalize(common, dim=1) @ F.normalize(prototypes, dim=1).t()
    log_probabilities = F.log_softmax(similarity / PROTOTYPE_TEMPERATURE, dim=1)
    per_sample = -(soft_targets * log_probabilities).sum(dim=1)
    weighted = (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
    return weighted / max(math.log(prototypes.shape[0]), 1e-8)


@torch.no_grad()
def update_global_prototypes(
    model: CausalMVC,
    views: list[torch.Tensor],
    mask: torch.Tensor,
    soft_targets: torch.Tensor,
    sample_weights: torch.Tensor,
    batch_size: int,
    *,
    momentum: float = PROTOTYPE_MOMENTUM,
) -> list[float]:
    """Update shared prototypes once from the complete dataset, not per batch."""

    sums = model.prototypes.new_zeros(model.num_clusters, model.prototypes.shape[1])
    masses = model.prototypes.new_zeros(model.num_clusters)
    was_training = model.training
    model.eval()
    for start in range(0, len(mask), batch_size):
        end = min(start + batch_size, len(mask))
        output = model([view[start:end] for view in views])
        probabilities = soft_targets[start:end]
        confidence = sample_weights[start:end]
        for view in range(2):
            valid = mask[start:end, view]
            if not valid.any():
                continue
            common = F.normalize(output.commons[view][valid], dim=1)
            assignments = probabilities[valid] * confidence[valid].unsqueeze(1)
            sums.add_(assignments.t().matmul(common))
            masses.add_(assignments.sum(dim=0))
    for cluster in range(model.num_clusters):
        if masses[cluster] > 1e-8:
            center = F.normalize(sums[cluster] / masses[cluster], dim=0)
            model.prototypes[cluster].mul_(momentum).add_(center * (1.0 - momentum))
    model.prototypes.copy_(F.normalize(model.prototypes, dim=1))
    model.train(was_training)
    return masses.detach().cpu().tolist()


def common_variance_loss(
    commons: list[torch.Tensor], valid_masks: list[torch.Tensor]
) -> torch.Tensor:
    """Prevent the normalized common representation from becoming constant."""

    available = [common[valid] for common, valid in zip(commons, valid_masks) if valid.any()]
    if not available:
        return commons[0].new_zeros(())
    representation = F.normalize(torch.cat(available, dim=0), dim=1)
    standard_deviation = torch.sqrt(representation.var(dim=0, unbiased=False) + 1e-4)
    target = 1.0 / math.sqrt(representation.shape[1])
    return F.relu(target - standard_deviation).pow(2).mean() / (target * target)


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
    paired: torch.Tensor,
) -> torch.Tensor:
    if not paired.any():
        return output.latents[0].new_zeros(())

    predicted1 = model.cross_predictors[0](output.commons[0][paired])
    predicted0 = model.cross_predictors[1](output.commons[1][paired])
    return (
        1.0 - F.cosine_similarity(predicted1, output.commons[1][paired].detach()).mean()
        + 1.0 - F.cosine_similarity(predicted0, output.commons[0][paired].detach()).mean()
    ) / 2.0
