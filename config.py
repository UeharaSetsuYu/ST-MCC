"""Small, explicit configuration for the stable BDGP experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    """Only expose settings that are useful in normal experiments.

    Architecture sizes, loss weights, and teacher details implement the frozen
    stable recipe and stay next to the code that uses them.
    """

    data: str = r"D:\Data_Mining\Code\Datasets\BDGP\BDGP.mat"
    output: str = "outputs/bdgp_clean"
    seed: int = 61
    mask_seed: int | None = None
    missing_rate: float = 0.5
    batch_size: int = 256
    learning_rate: float = 1e-3
    pretrain_epochs: int = 20
    joint_epochs: int = 10

    @property
    def epochs(self) -> int:
        return self.pretrain_epochs + self.joint_epochs

    @property
    def effective_mask_seed(self) -> int:
        return self.seed if self.mask_seed is None else self.mask_seed

    def validate(self) -> None:
        if not 0.0 <= self.missing_rate <= 1.0:
            raise ValueError("missing_rate must be in [0, 1]")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.pretrain_epochs < 0 or self.joint_epochs <= 0:
            raise ValueError("pretrain_epochs must be >= 0 and joint_epochs must be > 0")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["epochs"] = self.epochs
        payload["effective_mask_seed"] = self.effective_mask_seed
        return payload


def parse_config() -> Config:
    """Keep the command line small; edit Config only for development settings."""

    parser = argparse.ArgumentParser(description="Train the clean CausalMVC BDGP model")
    parser.add_argument("--data", default=Config.data)
    parser.add_argument("--output", default=Config.output)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--mask-seed", type=int, default=None)
    parser.add_argument("--missing-rate", type=float, default=Config.missing_rate)
    args = parser.parse_args()
    config = Config(**vars(args))
    config.validate()
    return config
