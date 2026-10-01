"""Binary graph/grid dataset and variable-size graph batching."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch


GASES = ("C2H4", "C2H6")


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def target_stats_from_manifest(
    rows: list[dict[str, str]], split_column: str = "random_split"
) -> dict[int, tuple[float, float]]:
    training = [row for row in rows if row[split_column] == "train"]
    if not training:
        raise ValueError("manifest has no training structures")
    stats = {}
    for gas_index, gas in enumerate(GASES):
        values = np.asarray([float(row[f"logd_{gas.lower()}"]) for row in training])
        mean, std = float(values.mean()), float(values.std())
        if not np.isfinite(values).all() or std <= 0:
            raise ValueError(f"nonfinite or constant training target: {gas}")
        stats[gas_index] = (mean, std)
    return stats


class GraphVextDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_dir: Path, split: str, split_column: str = "random_split",
        descriptor_targets: dict[int, dict[str, np.ndarray]] | None = None,
        edge_dir: Path | None = None,
        geometry_dir: Path | None = None,
        pore_features: dict[str, np.ndarray] | None = None,
        neighbor_dir: Path | None = None,
        environment_dir: Path | None = None,
    ):
        if edge_dir is not None and neighbor_dir is not None:
            raise ValueError("edge_dir and neighbor_dir are mutually exclusive")
        self.data_dir = Path(data_dir)
        self.descriptor_targets = descriptor_targets
        self.edge_dir = edge_dir
        self.neighbor_dir = neighbor_dir
        self.environment_dir = environment_dir
        self.geometry_dir = geometry_dir
        self.pore_features = pore_features
        rows = load_manifest(self.data_dir / "manifest.csv")
        self.examples = [
            (row["name"], gas_index, float(row[f"logd_{gas.lower()}"]))
            for row in rows if row[split_column] == split
            for gas_index, gas in enumerate(GASES)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        name, gas_index, logd = self.examples[index]
        path = self.data_dir / "data" / f"{name}.npz"
        with np.load(path, allow_pickle=False) as record:
            result: dict[str, object] = {
                "name": name,
                "gas": gas_index,
                "logd": logd,
                "grid": record[f"grid_{GASES[gas_index]}"].astype(np.float32),
            }
            if self.descriptor_targets is not None:
                result["feature_target"] = self.descriptor_targets[gas_index][name]
            for field in (
                "atom_z", "atom_xyz", "atom_ff", "cell",
                "inner_idx", "inner_dist", "inner_mask",
                "outer_idx", "outer_dist", "outer_mask",
            ):
                result[field] = record[field].copy()
        if self.neighbor_dir is not None:
            with np.load(self.neighbor_dir / f"{name}.npz", allow_pickle=False) as record:
                for shell, width in (("inner", 32), ("outer", 48)):
                    for field in ("idx", "dist", "mask", "vector"):
                        result[f"{shell}_{field}"] = record[f"{shell}_{field}"].copy()
                    index = result[f"{shell}_idx"]
                    mask = result[f"{shell}_mask"].astype(bool)
                    distance = result[f"{shell}_dist"]
                    vector = result[f"{shell}_vector"]
                    if (index.shape != (len(result["atom_z"]), width)
                            or distance.shape != index.shape or mask.shape != index.shape
                            or vector.shape != (*index.shape, 3)
                            or not np.isfinite(distance).all() or not np.isfinite(vector).all()
                            or (index[mask] >= len(result["atom_z"])).any()
                            or (distance[mask] <= 0).any()
                            or not np.allclose(np.linalg.norm(vector[mask].astype(np.float32), axis=-1),
                                               distance[mask], atol=.02, rtol=0)):
                        raise ValueError(f"invalid expanded neighbors: {name} {shell}")
        elif self.edge_dir is not None:
            with np.load(self.edge_dir / f"{name}.npz", allow_pickle=False) as record:
                for shell in ("inner", "outer"):
                    vector = record[f"{shell}_vector"].astype(np.float32)
                    if vector.shape != (*result[f"{shell}_idx"].shape, 3) or not np.isfinite(vector).all():
                        raise ValueError(f"bad {shell} edge vectors for {name}")
                    result[f"{shell}_vector"] = vector
        if self.geometry_dir is not None:
            with np.load(self.geometry_dir / f"{name}.npz", allow_pickle=False) as record:
                clearance = record[f"clearance_{GASES[gas_index]}"].astype(np.float32)
            if clearance.shape != (24, 24, 24) or not np.isfinite(clearance).all():
                raise ValueError(f"bad geometry clearance for {name} {GASES[gas_index]}")
            result["clearance"] = clearance
        if self.pore_features is not None:
            result["pore_features"] = self.pore_features[name]
        if self.environment_dir is not None:
            with np.load(self.environment_dir / f"{name}.npz", allow_pickle=False) as record:
                environment = record[f"environment_{GASES[gas_index]}"].astype(np.float32)
            if environment.shape != (len(result["atom_z"]), 28) or not np.isfinite(environment).all():
                raise ValueError(f"invalid all-neighbor environment: {name}")
            result["atom_environment"] = environment
        return result


def collate_samples(
    samples: list[dict[str, object]], stats: dict[int, tuple[float, float]]
) -> tuple[dict[str, torch.Tensor], list[str]]:
    batch_size = len(samples)
    max_atoms = max(len(sample["atom_z"]) for sample in samples)
    inner_width = samples[0]["inner_idx"].shape[1]
    outer_width = samples[0]["outer_idx"].shape[1]
    data = {
        "atom_z": np.zeros((batch_size, max_atoms), dtype=np.int64),
        "atom_xyz": np.zeros((batch_size, max_atoms, 3), dtype=np.float32),
        "atom_ff": np.zeros((batch_size, max_atoms, 3), dtype=np.float32),
        "atom_mask": np.zeros((batch_size, max_atoms), dtype=np.float32),
        "inner_idx": np.zeros((batch_size, max_atoms, inner_width), dtype=np.int64),
        "inner_dist": np.zeros((batch_size, max_atoms, inner_width), dtype=np.float32),
        "inner_mask": np.zeros((batch_size, max_atoms, inner_width), dtype=np.float32),
        "outer_idx": np.zeros((batch_size, max_atoms, outer_width), dtype=np.int64),
        "outer_dist": np.zeros((batch_size, max_atoms, outer_width), dtype=np.float32),
        "outer_mask": np.zeros((batch_size, max_atoms, outer_width), dtype=np.float32),
        "cell": np.zeros((batch_size, 6), dtype=np.float32),
        "gas": np.zeros(batch_size, dtype=np.int64),
        "target_logd": np.zeros(batch_size, dtype=np.float32),
    }
    vector_fields = ("inner_vector", "outer_vector") if "inner_vector" in samples[0] else ()
    for shell in ("inner", "outer"):
        if f"{shell}_vector" in vector_fields:
            width = samples[0][f"{shell}_vector"].shape[1]
            data[f"{shell}_vector"] = np.zeros(
                (batch_size, max_atoms, width, 3), dtype=np.float32
            )
    grids = []
    names = []
    for row, sample in enumerate(samples):
        n_atoms = len(sample["atom_z"])
        for field in (
            "atom_z", "atom_xyz", "atom_ff", "inner_idx", "inner_dist",
            "inner_mask", "outer_idx", "outer_dist", "outer_mask", *vector_fields,
        ):
            data[field][row, :n_atoms] = sample[field]
        data["atom_mask"][row, :n_atoms] = 1.0
        data["cell"][row] = sample["cell"]
        gas_index = int(sample["gas"])
        data["gas"][row] = gas_index
        mean, std = stats[gas_index]
        data["target_logd"][row] = (float(sample["logd"]) - mean) / std
        grids.append(np.asarray(sample["grid"], dtype=np.float32))
        names.append(str(sample["name"]))
    tensor_batch = {key: torch.from_numpy(value) for key, value in data.items()}
    tensor_batch["target_grid"] = torch.from_numpy(np.stack(grids)[:, None])
    if "clearance" in samples[0]:
        tensor_batch["clearance"] = torch.from_numpy(
            np.stack([np.asarray(sample["clearance"], dtype=np.float32) for sample in samples])[:, None]
        )
    if "pore_features" in samples[0]:
        tensor_batch["pore_features"] = torch.from_numpy(np.stack([
            np.asarray(sample["pore_features"], dtype=np.float32) for sample in samples
        ]))
    if "feature_target" in samples[0]:
        tensor_batch["target_features"] = torch.from_numpy(
            np.stack([np.asarray(sample["feature_target"], dtype=np.float32) for sample in samples])
        )
    if "atom_environment" in samples[0]:
        environment = np.zeros((batch_size, max_atoms, 28), dtype=np.float32)
        for row, sample in enumerate(samples):
            environment[row, :len(sample["atom_z"])] = sample["atom_environment"]
        tensor_batch["atom_environment"] = torch.from_numpy(environment)
    return tensor_batch, names
