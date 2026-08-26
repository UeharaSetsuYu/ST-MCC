"""Single authoritative training entry for the clean CausalMVC BDGP project."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import Config, parse_config
from data_load import load_bdgp
from model import CausalMVC, GradientReverse
from unit import (
    NUM_CLUSTERS,
    build_structure_teacher,
    clustering_metrics,
    cross_view_loss,
    kmeans_metrics,
    linear_structure_alignment,
    masked_reconstruction_loss,
    prototype_loss,
    seed_everything,
    update_prototypes,
)


# Frozen weights of the verified stable recipe.
LOSS_WEIGHTS = {
    "grl": 0.1,
    "pair": 0.2,
    "pam": 0.5,
    "structure": 0.2,
    "cross": 0.2,
}
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_NORM = 5.0


def evaluate_final(
    model: CausalMVC,
    views: list[torch.Tensor],
    mask: torch.Tensor,
    labels: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """Stable final rule: completed common mean followed by KMeans."""

    model.eval()
    embedding = model.completed_common(views, mask).cpu().numpy()
    return kmeans_metrics(embedding, labels, seed)


def train(config: Config) -> dict:
    config.validate()
    seed_everything(config.seed)

    data = load_bdgp(config.data, config.missing_rate, config.effective_mask_seed)
    teacher = build_structure_teacher(data.full_views, data.mask, config.seed)
    teacher_metrics = clustering_metrics(data.labels, teacher.pseudo_labels)
    zero_baseline = kmeans_metrics(
        np.concatenate(data.observed_views, axis=1), data.labels, config.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    views = [torch.from_numpy(view).to(device) for view in data.observed_views]
    mask = torch.from_numpy(data.mask.astype(bool)).to(device)
    pseudo_labels = torch.from_numpy(teacher.pseudo_labels).to(device)
    teacher_views = [torch.from_numpy(view).to(device) for view in teacher.views]

    model = CausalMVC(data.input_dims, NUM_CLUSTERS).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.epochs, eta_min=config.learning_rate * 0.05
    )
    rng = np.random.RandomState(config.seed)
    history: list[dict] = []
    loss_names = ("total", "reconstruction", "grl", "pair", "pam", "structure", "cross")

    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(len(data.labels))
        epoch_sums = {name: 0.0 for name in loss_names}
        batches = 0

        for start in range(0, len(order), config.batch_size):
            ids = torch.from_numpy(order[start : start + config.batch_size]).to(device)
            batch_views = [view[ids] for view in views]
            batch_mask = mask[ids]
            batch_pseudo = pseudo_labels[ids]
            batch_teacher = [view[ids] for view in teacher_views]

            output = model(batch_views)
            reconstruction = masked_reconstruction_loss(output, batch_views, batch_mask)
            grl, pair, pam, structure, cross = [
                reconstruction.new_zeros(()) for _ in range(5)
            ]
            total = reconstruction

            if epoch >= config.pretrain_epochs:
                joint_epoch = epoch - config.pretrain_epochs + 1
                grl_coefficient = min(1.0, joint_epoch / config.joint_epochs)

                for view in range(2):
                    valid = batch_mask[:, view]
                    reversed_common = GradientReverse.apply(
                        output.commons[view][valid], grl_coefficient
                    )
                    view_targets = torch.full(
                        (int(valid.sum()),), view, dtype=torch.long, device=device
                    )
                    grl = grl + F.cross_entropy(
                        model.view_discriminator(reversed_common), view_targets
                    ) / 2.0
                    pam = pam + prototype_loss(
                        output.commons[view][valid], batch_pseudo[valid], model.prototypes
                    ) / 2.0
                    structure = structure + (
                        1.0
                        - linear_structure_alignment(
                            output.commons[view][valid], batch_teacher[view][valid]
                        )
                    ) / 2.0

                paired = batch_mask.all(dim=1)
                if paired.any():
                    pair = 1.0 - F.cosine_similarity(
                        output.commons[0][paired], output.commons[1][paired]
                    ).mean()
                    cross = cross_view_loss(model, output, batch_views, paired)

                total = (
                    reconstruction
                    + LOSS_WEIGHTS["grl"] * grl
                    + LOSS_WEIGHTS["pair"] * pair
                    + LOSS_WEIGHTS["pam"] * pam
                    + LOSS_WEIGHTS["structure"] * structure
                    + LOSS_WEIGHTS["cross"] * cross
                )

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()

            if epoch >= config.pretrain_epochs:
                update_prototypes(
                    model,
                    [common.detach() for common in output.commons],
                    batch_pseudo,
                    [batch_mask[:, 0], batch_mask[:, 1]],
                )

            batch_losses = (total, reconstruction, grl, pair, pam, structure, cross)
            for name, value in zip(loss_names, batch_losses):
                epoch_sums[name] += float(value.detach())
            batches += 1

        scheduler.step()
        stage = "reconstruction" if epoch < config.pretrain_epochs else "joint"
        record = {
            "epoch": epoch + 1,
            "stage": stage,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            **{name: value / batches for name, value in epoch_sums.items()},
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    final_metrics = evaluate_final(model, views, mask, data.labels, config.seed)
    return {
        "config": config.to_dict(),
        "device": str(device),
        "protocol": {
            "mask": "label-free global random; incomplete samples retain one view",
            "selection": "fixed final epoch; no label-based model selection",
            "inference": "normalized mean of completed common representations + KMeans",
            "labels_used_for": "reporting metrics only",
        },
        "loss_weights": LOSS_WEIGHTS,
        "teacher_diagnostics": teacher.diagnostics,
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
