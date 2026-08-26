"""BDGP loading, normalization, and label-free missing-mask generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.preprocessing import Normalizer


@dataclass(frozen=True)
class MultiViewData:
    full_views: list[np.ndarray]
    observed_views: list[np.ndarray]
    mask: np.ndarray
    labels: np.ndarray

    @property
    def input_dims(self) -> list[int]:
        return [view.shape[1] for view in self.full_views]


def make_mask(num_samples: int, missing_rate: float, seed: int) -> np.ndarray:
    """Make the stable two-view mask without consulting labels.

    A global fraction of samples is selected as incomplete. Each incomplete
    sample keeps exactly one randomly chosen view; all other samples keep both.
    """

    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError("missing_rate must be in [0, 1]")
    rng = np.random.RandomState(seed)
    incomplete_count = math.floor(num_samples * missing_rate)
    incomplete = rng.permutation(num_samples)[:incomplete_count]
    retained_views = rng.randint(0, 2, size=incomplete_count)

    mask = np.ones((num_samples, 2), dtype=np.float32)
    mask[incomplete] = np.eye(2, dtype=np.float32)[retained_views]
    return mask


def load_bdgp(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"BDGP dataset not found: {data_path}")

    mat = sio.loadmat(data_path)
    required = {"X1", "X2", "Y"}
    missing_keys = required.difference(mat)
    if missing_keys:
        raise KeyError(f"BDGP file is missing arrays: {sorted(missing_keys)}")

    full_views = [mat["X1"].astype(np.float32), mat["X2"].astype(np.float32)]
    labels = mat["Y"].reshape(-1).astype(np.int64)
    if any(len(view) != len(labels) for view in full_views):
        raise ValueError("X1, X2, and Y must contain the same number of samples")

    full_views = [Normalizer().fit_transform(view).astype(np.float32) for view in full_views]
    labels = labels - labels.min()
    mask = make_mask(len(labels), missing_rate, mask_seed)
    observed_views = [view * mask[:, index : index + 1] for index, view in enumerate(full_views)]
    return MultiViewData(full_views, observed_views, mask, labels)
