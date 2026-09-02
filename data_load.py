"""Named dataset registry, import functions, normalization, and missing masks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.io as sio
from scipy import sparse
from sklearn.preprocessing import Normalizer


@dataclass(frozen=True)
class MultiViewData:
    dataset_name: str
    source_path: str
    full_views: list[np.ndarray]
    observed_views: list[np.ndarray]
    mask: np.ndarray
    labels: np.ndarray
    source_view_count: int
    selected_view_indices: tuple[int, int]

    @property
    def input_dims(self) -> list[int]:
        return [view.shape[1] for view in self.full_views]

    @property
    def num_clusters(self) -> int:
        return int(np.unique(self.labels).size)

    @property
    def batch_size(self) -> int:
        return batch_size_for_samples(len(self.labels))


def batch_size_for_samples(num_samples: int) -> int:
    """Apply the user-specified, dataset-size-only batch-size rule."""

    if num_samples < 500:
        return 64
    if num_samples < 5000:
        return 256
    return 1024


def make_mask(num_samples: int, missing_rate: float, seed: int) -> np.ndarray:
    """Select incomplete samples without consulting labels; retain one of two views."""

    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError("missing_rate must be in [0, 1]")
    rng = np.random.RandomState(seed)
    incomplete_count = math.floor(num_samples * missing_rate)
    incomplete = rng.permutation(num_samples)[:incomplete_count]
    retained_views = rng.randint(0, 2, size=incomplete_count)
    mask = np.ones((num_samples, 2), dtype=np.float32)
    mask[incomplete] = np.eye(2, dtype=np.float32)[retained_views]
    return mask


def _read(path: str | Path) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    return sio.loadmat(source)


def _labels(mat: dict, key: str) -> np.ndarray:
    values = np.asarray(mat[key]).reshape(-1)
    _, values = np.unique(values, return_inverse=True)
    return values.astype(np.int64)


def _cell_views(mat: dict, key: str) -> list[np.ndarray]:
    return [item for item in np.asarray(mat[key], dtype=object).ravel()]


def _orient_dense(view: np.ndarray, num_samples: int) -> np.ndarray:
    if sparse.issparse(view):
        view = view.toarray()
    array = np.asarray(view)
    if array.ndim > 2 and array.shape[0] == num_samples:
        array = array.reshape(num_samples, -1)
    if array.ndim != 2:
        raise ValueError(f"Each view must be sample-aligned, got shape {array.shape}")
    if array.shape[0] == num_samples:
        oriented = array
    elif array.shape[1] == num_samples:
        oriented = array.T
    else:
        raise ValueError(
            f"Cannot align view shape {array.shape} with {num_samples} labels"
        )
    oriented = oriented.astype(np.float32)
    if not np.isfinite(oriented).all():
        raise ValueError("Dataset contains NaN or infinite feature values")
    return oriented


def _select_views_by_feature_dims(
    raw_views: list[np.ndarray],
    labels: np.ndarray,
    feature_dims: tuple[int, int],
    dataset_name: str,
) -> list[np.ndarray]:
    """Select exactly two genuine views by feature dimension and preserve their order."""

    oriented_views = [_orient_dense(view, len(labels)) for view in raw_views]
    selected: list[np.ndarray] = []
    for feature_dim in feature_dims:
        matches = [view for view in oriented_views if view.shape[1] == feature_dim]
        if len(matches) != 1:
            available_dims = [view.shape[1] for view in oriented_views]
            raise ValueError(
                f"{dataset_name} expected exactly one {feature_dim}-dimensional view, "
                f"found {len(matches)}; available dimensions: {available_dims}"
            )
        selected.append(matches[0])
    return selected


def _distance_to_similarity(distance: np.ndarray) -> np.ndarray:
    positive = distance[distance > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    return np.exp(-distance / max(scale, 1e-12)).astype(np.float32)


def _finalize(
    dataset_name: str,
    source_path: str,
    raw_views: list[np.ndarray],
    labels: np.ndarray,
    missing_rate: float,
    mask_seed: int,
    *,
    distance_views: bool = False,
) -> MultiViewData:
    if len(raw_views) < 2:
        raise ValueError(
            f"{dataset_name} contains {len(raw_views)} view, but the current model "
            "requires exactly two genuine views; no synthetic view was created"
        )
    selected_indices = (0, 1)
    views = [_orient_dense(raw_views[index], len(labels)) for index in selected_indices]
    if distance_views:
        views = [_distance_to_similarity(view) for view in views]
    views = [Normalizer().fit_transform(view).astype(np.float32) for view in views]
    mask = make_mask(len(labels), missing_rate, mask_seed)
    observed = [view * mask[:, index : index + 1] for index, view in enumerate(views)]
    return MultiViewData(
        dataset_name=dataset_name,
        source_path=str(Path(source_path)),
        full_views=views,
        observed_views=observed,
        mask=mask,
        labels=labels,
        source_view_count=len(raw_views),
        selected_view_indices=selected_indices,
    )


def load_coil20(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    return _finalize(
        "COIL20", path, [mat["fea"]], _labels(mat, "gnd"), missing_rate, mask_seed
    )


def load_caltech101_20(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    labels = _labels(mat, "Y")
    views = _select_views_by_feature_dims(
        _cell_views(mat, "X"), labels, (1984, 512), "Caltech101-20"
    )
    return _finalize(
        "Caltech101-20", path, views, labels, missing_rate, mask_seed
    )


def load_citeseer(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    return _finalize(
        "CiteSeer", path, _cell_views(mat, "fea"), _labels(mat, "gt"), missing_rate, mask_seed
    )


def load_cub(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    return _finalize(
        "CUB", path, _cell_views(mat, "X"), _labels(mat, "gt"), missing_rate, mask_seed
    )


def load_fashion(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    return _finalize(
        "Fashion",
        path,
        [mat["X1"], mat["X2"], mat["X3"]],
        _labels(mat, "Y"),
        missing_rate,
        mask_seed,
    )


def load_reuters_dim10(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    train = np.asarray(mat["x_train"])
    test = np.asarray(mat["x_test"])
    if train.ndim != 3 or test.ndim != 3 or train.shape[0] != test.shape[0]:
        raise ValueError("Reuters_dim10 must store train/test arrays as [views, samples, dims]")
    views = [np.concatenate((train[v], test[v]), axis=0) for v in range(train.shape[0])]
    labels = np.concatenate(
        (np.asarray(mat["y_train"]).reshape(-1), np.asarray(mat["y_test"]).reshape(-1))
    )
    _, labels = np.unique(labels, return_inverse=True)
    return _finalize(
        "Reuters_dim10", path, views, labels.astype(np.int64), missing_rate, mask_seed
    )


def load_landuse_21(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    labels = _labels(mat, "Y")
    views = _select_views_by_feature_dims(
        _cell_views(mat, "X"), labels, (59, 40), "LandUse_21"
    )
    return _finalize(
        "LandUse_21", path, views, labels, missing_rate, mask_seed
    )


def load_nuswide(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    return _finalize(
        "NUSWIDE", path, [mat["Img"], mat["Txt"]], _labels(mat, "label"), missing_rate, mask_seed
    )


def load_lgg(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    return _finalize(
        "LGG", path, _cell_views(mat, "X"), _labels(mat, "Y"), missing_rate, mask_seed
    )


def load_mnist_usps(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    return _finalize(
        "MNIST_USPS", path, [mat["X1"], mat["X2"]], _labels(mat, "Y"), missing_rate, mask_seed
    )


def load_ngs(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    return _finalize(
        "NGs", path, _cell_views(mat, "X"), _labels(mat, "Y"), missing_rate, mask_seed
    )


def load_rgb_d(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    return _finalize(
        "RGB_D", path, _cell_views(mat, "X"), _labels(mat, "Y"), missing_rate, mask_seed
    )


def load_100leaves(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    return _finalize(
        "100Leaves", path, _cell_views(mat, "data"), _labels(mat, "truth"), missing_rate, mask_seed
    )


def load_aloi_100(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    return _finalize(
        "ALOI_100", path, _cell_views(mat, "fea"), _labels(mat, "gt"), missing_rate, mask_seed
    )


def load_bbcsport(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    return _finalize(
        "BBCSport", path, _cell_views(mat, "X"), _labels(mat, "Y"), missing_rate, mask_seed
    )


def load_dha(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    return _finalize(
        "DHA", path, [mat["X1"], mat["X2"]], _labels(mat, "Y"), missing_rate, mask_seed
    )


def load_flower17(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    return _finalize(
        "flower17",
        path,
        _cell_views(mat, "distance_matrices"),
        _labels(mat, "gt"),
        missing_rate,
        mask_seed,
        distance_views=True,
    )


def load_hw(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    return _finalize(
        "HW", path, [mat[f"X{i}"] for i in range(1, 7)], _labels(mat, "Y"), missing_rate, mask_seed
    )


def load_scene_15(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    labels = _labels(mat, "Y")
    views = _select_views_by_feature_dims(
        _cell_views(mat, "X"), labels, (59, 20), "Scene_15"
    )
    return _finalize(
        "Scene_15", path, views, labels, missing_rate, mask_seed
    )


def load_bdgp(path: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    mat = _read(path)
    return _finalize(
        "BDGP", path, [mat["X1"], mat["X2"]], _labels(mat, "Y"), missing_rate, mask_seed
    )


DATASET_PATHS: dict[str, str] = {
    "coil20": r"D:\Data_Mining\Code\Datasets\COIL20\COIL20.mat",
    "caltech101-20": r"D:\Data_Mining\Code\Datasets\Caltech101-20(.mat)\Caltech101-20.mat",
    "citeseer": r"D:\Data_Mining\Code\Datasets\CiteSeer\CiteSeer.mat",
    "cub": r"D:\Data_Mining\Code\Datasets\CUB\cub_googlenet_doc2vec_c10.mat",
    "fashion": r"D:\Data_Mining\Code\Datasets\Fashion\Fashion.mat",
    "reuters_dim10": r"D:\Data_Mining\Code\Datasets\dataset-20250704T023945Z-1-001\dataset\Reuters_dim10.mat",
    "landuse_21": r"D:\Data_Mining\Code\Datasets\dataset-20250704T023945Z-1-001\dataset\LandUse_21.mat",
    "nuswide": r"D:\Data_Mining\Code\Datasets\dataset-20250704T023945Z-1-001\dataset\nuswide_deep_2_view.mat",
    "lgg": r"D:\Data_Mining\Code\Datasets\LGG\LGG.mat",
    "mnist_usps": r"D:\Data_Mining\Code\Datasets\MNIST_USPS\MNIST_USPS.mat",
    "ngs": r"D:\Data_Mining\Code\Datasets\NGs\NGs.mat",
    "rgb_d": r"D:\Data_Mining\Code\Datasets\RGB_D\RGB_D.mat",
    "100leaves": r"D:\Data_Mining\Code\Experiment\Prototype_Argument\dataset\100Leaves.mat",
    "aloi_100": r"D:\Data_Mining\Code\Experiment\Prototype_Argument\dataset\ALOI_100.mat",
    "bbcsport": r"D:\Data_Mining\Code\Experiment\Prototype_Argument\dataset\BBCSport.mat",
    "dha": r"D:\Data_Mining\Code\Experiment\Prototype_Argument\dataset\DHA.mat",
    "flower17": r"D:\Data_Mining\Code\Experiment\Prototype_Argument\dataset\flower17.mat",
    "hw": r"D:\Data_Mining\Code\Experiment\Prototype_Argument\dataset\HW.mat",
    "scene_15": r"D:\Data_Mining\Code\Experiment\Prototype_Argument\dataset\Scene_15.mat",
    "bdgp": r"D:\Data_Mining\Code\Datasets\BDGP\BDGP.mat",
}


DATASET_LOADERS: dict[str, Callable[[str, float, int], MultiViewData]] = {
    "coil20": load_coil20,
    "caltech101-20": load_caltech101_20,
    "citeseer": load_citeseer,
    "cub": load_cub,
    "fashion": load_fashion,
    "reuters_dim10": load_reuters_dim10,
    "landuse_21": load_landuse_21,
    "nuswide": load_nuswide,
    "lgg": load_lgg,
    "mnist_usps": load_mnist_usps,
    "ngs": load_ngs,
    "rgb_d": load_rgb_d,
    "100leaves": load_100leaves,
    "aloi_100": load_aloi_100,
    "bbcsport": load_bbcsport,
    "dha": load_dha,
    "flower17": load_flower17,
    "hw": load_hw,
    "scene_15": load_scene_15,
    "bdgp": load_bdgp,
}


DATASET_ALIASES = {
    "caltech101_20": "caltech101-20",
    "cub_googlenet_doc2vec_c10": "cub",
    "nuswide_deep_2_view": "nuswide",
    "reuters": "reuters_dim10",
    "landuse": "landuse_21",
    "rgb-d": "rgb_d",
}


def resolve_dataset(dataname: str) -> tuple[str, str]:
    """Resolve a human dataset name to a registered loader key and source path."""

    candidate = Path(dataname)
    if candidate.is_file():
        key = candidate.stem.lower()
        key = DATASET_ALIASES.get(key, key)
        if key not in DATASET_LOADERS:
            raise ValueError(f"No loader registered for file '{candidate.name}'")
        return key, str(candidate)
    key = dataname.strip().lower()
    key = DATASET_ALIASES.get(key, key)
    if key not in DATASET_LOADERS:
        supported = ", ".join(sorted(DATASET_LOADERS))
        raise ValueError(f"Unsupported dataname '{dataname}'. Supported: {supported}")
    return key, DATASET_PATHS[key]


def load_dataset(dataname: str, missing_rate: float, mask_seed: int) -> MultiViewData:
    """Load a dataset by dataname; paths are maintained in DATASET_PATHS."""

    key, path = resolve_dataset(dataname)
    return DATASET_LOADERS[key](path, missing_rate, mask_seed)
