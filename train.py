"""Single authoritative training entry for stable multi-dataset ST-MCC."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import Config, parse_config
from data_load import load_dataset
from model import CausalMVC, GradientReverse
from unit import (
    build_structure_teacher,
    clustering_metrics,
    common_variance_loss,
    cross_view_loss,
    kmeans_metrics,
    linear_structure_alignment,
    masked_reconstruction_loss,
    prototype_loss,
    seed_everything,
    update_global_prototypes,
)


# Compact defaults shared by every dataset; class-count effects are normalized.
LOSS_WEIGHTS = {
    "grl": 0.1,
    "pair": 0.2,
    "pam": 0.5,
    "structure": 0.2,
    "cross": 0.2,
    "variance": 0.1,
}
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_NORM = 5.0
MIN_SEMANTIC_STEPS = 200
MIN_IMPUTATION_STEPS = 100
MIN_ADVERSARIAL_STEPS = 100


def evaluate_final(
    model: CausalMVC,
    views: list[torch.Tensor],
    mask: torch.Tensor,
    labels: np.ndarray,
    seed: int,
    num_clusters: int,
) -> dict[str, float]:
    """Stable final rule: completed common mean followed by KMeans."""

    model.eval()
    embedding = model.completed_common(views, mask).cpu().numpy()
    return kmeans_metrics(embedding, labels, seed, num_clusters)


def _epochs_for_steps(minimum_epochs: int, minimum_steps: int, batches: int) -> int:
    return max(minimum_epochs, math.ceil(minimum_steps / max(batches, 1)))


def _make_optimizer(parameters, learning_rate: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=WEIGHT_DECAY)


def train(config: Config) -> dict:

    config.validate()
    seed_everything(config.seed)

    data = load_dataset(config.data, config.missing_rate, config.effective_mask_seed)
    batch_size = data.batch_size
    teacher = build_structure_teacher(
        data.full_views, data.mask, config.seed, data.num_clusters
    )
    teacher_metrics = clustering_metrics(data.labels, teacher.pseudo_labels)
    zero_baseline = kmeans_metrics(
        np.concatenate(data.observed_views, axis=1),
        data.labels,
        config.seed,
        data.num_clusters,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    views = [torch.from_numpy(view).to(device) for view in data.observed_views]
    mask = torch.from_numpy(data.mask.astype(bool)).to(device)
    soft_targets = torch.from_numpy(teacher.probabilities).to(device)
    sample_weights = torch.from_numpy(teacher.sample_weights).to(device)
    teacher_views = [torch.from_numpy(view).to(device) for view in teacher.views]

    model = CausalMVC(data.input_dims, data.num_clusters).to(device)
    main_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("view_discriminator.")
    ]
    rng = np.random.RandomState(config.seed)
    history: list[dict] = []
    loss_names = (
        "total",
        "reconstruction",
        "grl",
        "pair",
        "pam",
        "structure",
        "cross",
        "variance",
        "discriminator",
        "discriminator_accuracy",
    )
    batches_per_epoch = math.ceil(len(data.labels) / batch_size)
    semantic_epochs = _epochs_for_steps(
        config.joint_epochs, MIN_SEMANTIC_STEPS, batches_per_epoch
    )
    half_joint = max(1, config.joint_epochs // 2)
    imputation_epochs = _epochs_for_steps(
        half_joint, MIN_IMPUTATION_STEPS, batches_per_epoch
    )
    adversarial_epochs = _epochs_for_steps(
        half_joint, MIN_ADVERSARIAL_STEPS, batches_per_epoch
    )
    schedule = [
        ("reconstruction", config.pretrain_epochs),
        ("semantic", semantic_epochs),
        ("imputation", imputation_epochs),
        ("adversarial", adversarial_epochs),
    ]
    global_epoch = 0
    prototype_masses: list[float] = []

    for stage, stage_epochs in schedule:
        if stage_epochs <= 0:
            continue
        if stage == "semantic":
            prototype_masses = update_global_prototypes(
                model,
                views,
                mask,
                soft_targets,
                sample_weights,
                batch_size,
                momentum=0.0,
            )

        optimizer = _make_optimizer(main_parameters, config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, stage_epochs, eta_min=config.learning_rate * 0.05
        )
        discriminator_optimizer = None
        if stage == "adversarial":
            discriminator_optimizer = _make_optimizer(
                model.view_discriminator.parameters(), config.learning_rate
            )

        for stage_epoch in range(stage_epochs):
            global_epoch += 1
            model.train()
            order = rng.permutation(len(data.labels))
            epoch_sums = {name: 0.0 for name in loss_names}
            batches = 0

            for start in range(0, len(order), batch_size):
                ids = torch.from_numpy(order[start : start + batch_size]).to(device)
                batch_views = [view[ids] for view in views]
                batch_mask = mask[ids]
                batch_soft = soft_targets[ids]
                batch_weights = sample_weights[ids]
                batch_teacher = [view[ids] for view in teacher_views]
                paired = batch_mask.all(dim=1)

                discriminator_loss = views[0].new_zeros(())
                discriminator_accuracy = views[0].new_zeros(())
                if stage == "adversarial" and paired.any():
                    assert discriminator_optimizer is not None
                    with torch.no_grad():
                        detached = model(batch_views).commons
                    discriminator_features = torch.cat(
                        (detached[0][paired], detached[1][paired]), dim=0
                    )
                    discriminator_targets = torch.cat(
                        (
                            torch.zeros(int(paired.sum()), dtype=torch.long, device=device),
                            torch.ones(int(paired.sum()), dtype=torch.long, device=device),
                        )
                    )
                    discriminator_logits = model.view_discriminator(discriminator_features)
                    discriminator_loss = F.cross_entropy(
                        discriminator_logits, discriminator_targets
                    )
                    discriminator_optimizer.zero_grad(set_to_none=True)
                    discriminator_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.view_discriminator.parameters(), GRADIENT_CLIP_NORM
                    )
                    discriminator_optimizer.step()
                    discriminator_accuracy = (
                        discriminator_logits.argmax(dim=1) == discriminator_targets
                    ).float().mean()

                output = model(batch_views)
                reconstruction = masked_reconstruction_loss(output, batch_views, batch_mask)
                grl, pair, pam, structure, cross, variance = [
                    reconstruction.new_zeros(()) for _ in range(6)
                ]

                if stage != "reconstruction":
                    variance = common_variance_loss(
                        output.commons, [batch_mask[:, 0], batch_mask[:, 1]]
                    )
                    for view in range(2):
                        valid = batch_mask[:, view]
                        if not valid.any():
                            continue
                        pam = pam + prototype_loss(
                            output.commons[view][valid],
                            batch_soft[valid],
                            batch_weights[valid],
                            model.prototypes,
                        ) / 2.0
                        trusted = valid & (batch_weights > 0)
                        if int(trusted.sum()) >= 2:
                            structure = structure + (
                                1.0
                                - linear_structure_alignment(
                                    output.commons[view][trusted],
                                    batch_teacher[view][trusted],
                                )
                            ) / 2.0

                if stage in ("imputation", "adversarial") and paired.any():
                    cross = cross_view_loss(model, output, paired)

                if stage == "adversarial" and paired.any():
                    pair = 1.0 - F.cosine_similarity(
                        output.commons[0][paired], output.commons[1][paired]
                    ).mean()
                    grl_coefficient = (stage_epoch + 1) / stage_epochs
                    for parameter in model.view_discriminator.parameters():
                        parameter.requires_grad_(False)
                    reversed_features = torch.cat(
                        (
                            GradientReverse.apply(
                                output.commons[0][paired], grl_coefficient
                            ),
                            GradientReverse.apply(
                                output.commons[1][paired], grl_coefficient
                            ),
                        ),
                        dim=0,
                    )
                    view_targets = torch.cat(
                        (
                            torch.zeros(int(paired.sum()), dtype=torch.long, device=device),
                            torch.ones(int(paired.sum()), dtype=torch.long, device=device),
                        )
                    )
                    grl = F.cross_entropy(
                        model.view_discriminator(reversed_features), view_targets
                    )
                    for parameter in model.view_discriminator.parameters():
                        parameter.requires_grad_(True)

                total = reconstruction
                if stage != "reconstruction":
                    total = (
                        total
                        + LOSS_WEIGHTS["pam"] * pam
                        + LOSS_WEIGHTS["structure"] * structure
                        + LOSS_WEIGHTS["variance"] * variance
                    )
                if stage in ("imputation", "adversarial"):
                    total = total + LOSS_WEIGHTS["cross"] * cross
                if stage == "adversarial":
                    total = (
                        total
                        + LOSS_WEIGHTS["grl"] * grl
                        + LOSS_WEIGHTS["pair"] * pair
                    )

                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(main_parameters, GRADIENT_CLIP_NORM)
                optimizer.step()

                batch_losses = (
                    total,
                    reconstruction,
                    grl,
                    pair,
                    pam,
                    structure,
                    cross,
                    variance,
                    discriminator_loss,
                    discriminator_accuracy,
                )
                for name, value in zip(loss_names, batch_losses):
                    epoch_sums[name] += float(value.detach())
                batches += 1

            scheduler.step()
            if stage != "reconstruction":
                prototype_masses = update_global_prototypes(
                    model,
                    views,
                    mask,
                    soft_targets,
                    sample_weights,
                    batch_size,
                )
            if stage in ("imputation", "adversarial"):
                model.eval()
                model.calibrate_imputation_confidence(views, mask)

            record = {
                "epoch": global_epoch,
                "stage": stage,
                "stage_epoch": stage_epoch + 1,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "imputation_confidence": model.imputation_confidence.detach().cpu().tolist(),
                **{name: value / batches for name, value in epoch_sums.items()},
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)

    model.eval()
    model.calibrate_imputation_confidence(views, mask)

    final_metrics = evaluate_final(
        model, views, mask, data.labels, config.seed, data.num_clusters
    )
    config_payload = config.to_dict()
    config_payload["batch_size"] = batch_size
    config_payload["epochs"] = sum(epochs for _, epochs in schedule)
    return {
        "method_version": "st_mcc_stable_v2",
        "config": config_payload,
        "device": str(device),
        "protocol": {
            "mask": "label-free global random; incomplete samples retain one view",
            "views": "first two source views; fixed for every dataset",
            "batch_size": "64 if n<500; 256 if n<5000; otherwise 1024",
            "selection": "fixed final epoch; no label-based model selection",
            "training": "reconstruction -> semantic distillation -> common imputation -> gradual GRL",
            "inference": "confidence-weighted observed/predicted common representation + KMeans",
            "labels_used_for": "reporting metrics only",
        },
        "loss_weights": LOSS_WEIGHTS,
        "dataset": {
            "name": data.dataset_name,
            "source_path": data.source_path,
            "samples": int(len(data.labels)),
            "batch_size": batch_size,
            "num_clusters": data.num_clusters,
            "source_view_count": data.source_view_count,
            "selected_view_indices": list(data.selected_view_indices),
            "input_dims": data.input_dims,
        },
        "teacher_diagnostics": teacher.diagnostics,
        "training_schedule": {stage: epochs for stage, epochs in schedule},
        "prototype_masses": prototype_masses,
        "imputation_confidence": model.imputation_confidence.detach().cpu().tolist(),
        "zero_baseline": zero_baseline,
        "teacher_baseline": teacher_metrics,
        "final": final_metrics,
        "history": history,
        "state_dict": model.state_dict(),
        "input_dims": data.input_dims,
    }


def save_result(result: dict, output_directory: str, seed: int) -> tuple[Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / f"seed_{seed}.json"
    checkpoint_path = output / f"seed_{seed}.pt"

    payload = {key: value for key, value in result.items() if key != "state_dict"}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save(
        {
            "model_state": result["state_dict"],
            "config": result["config"],
            "input_dims": result["input_dims"],
        },
        checkpoint_path,
    )
    return json_path, checkpoint_path


def main() -> None:
    config = parse_config()
    print("=========DATASET==========")
    print("dataset name: ", config.data)
    started = time.time()
    result = train(config)
    result["runtime_seconds"] = time.time() - started
    json_path, checkpoint_path = save_result(result, config.output, config.seed)
    print(
        json.dumps(
            {
                "final": result["final"],
                "json": str(json_path),
                "checkpoint": str(checkpoint_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
