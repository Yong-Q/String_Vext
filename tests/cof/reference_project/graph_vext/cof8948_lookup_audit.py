"""Compare saved 64-orientation DIRECT fields with 60-grid site lookup."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from graph_vext.cof8948_oldff_prepare import GASES, ROOT
from graph_vext.string_ff_vext import marginal_energy, periodic_interpolate


DEFAULT_SOURCE = ROOT / "inputs/cof8948_oldff_vext_full_v1"


def lookup_poses(site: np.ndarray, cell: np.ndarray, centers: np.ndarray,
                 axes: np.ndarray, bond: float) -> np.ndarray:
    site, cell = np.asarray(site), np.asarray(cell, dtype=float)
    centers, axes = np.asarray(centers, dtype=float), np.asarray(axes, dtype=float)
    if (site.ndim != 4 or site.shape[0] != 2 or len(set(site.shape[1:])) != 1
            or cell.shape != (3, 3) or np.linalg.det(cell) <= 0
            or centers.ndim != 2 or centers.shape[1] != 3
            or axes.ndim != 2 or axes.shape[1] != 3 or bond <= 0):
        raise ValueError("invalid lookup geometry")
    inverse = np.linalg.inv(cell)
    shifts = (bond / 2) * axes @ inverse
    result = np.zeros((len(centers), len(axes)), dtype=float)
    for index, sign in enumerate((-1, 1)):
        fractional = centers[:, None, :] + sign * shifts[None, :, :]
        result += periodic_interpolate(site[index], fractional.reshape(-1, 3)).reshape(result.shape)
    return result


def summarize_error(reference: np.ndarray, candidate: np.ndarray,
                    selected: np.ndarray) -> dict:
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    selected = np.asarray(selected, dtype=bool)
    if (reference.shape != candidate.shape or selected.shape != reference.shape
            or not np.isfinite(reference).all() or not np.isfinite(candidate).all()):
        raise ValueError("matching finite values and mask required")
    errors = np.abs(candidate[selected] - reference[selected])
    if not len(errors):
        return {"count": 0, "mae_K": None, "median_K": None,
                "p90_K": None, "max_K": None, "within_10K": 0,
                "fraction_within_10K": None}
    return {"count": int(len(errors)), "mae_K": float(errors.mean()),
            "median_K": float(np.median(errors)),
            "p90_K": float(np.percentile(errors, 90)),
            "max_K": float(errors.max()),
            "within_10K": int((errors <= 10).sum()),
            "fraction_within_10K": float((errors <= 10).mean())}


def completed(source: Path) -> list[tuple[str, Path, Path, dict]]:
    rows = []
    for shard in ("shard0", "shard1"):
        records = source / shard / "records"
        if not records.is_dir():
            continue
        for record_path in records.glob("*.json"):
            field = source / shard / "data" / f"{record_path.stem}.npz"
            if not field.is_file():
                continue
            row = json.loads(record_path.read_text())
            if (row.get("name") != record_path.stem or row.get("size") != 60
                    or row.get("orientations") != 64 or row.get("energy_mode") != "direct"
                    or {gas: (row["gases"][gas]["guest_sigma_A"],
                               row["gases"][gas]["guest_epsilon_K"]) for gas in GASES}
                       != {gas: (GASES[gas]["sigma"], GASES[gas]["epsilon"])
                           for gas in GASES}):
                raise ValueError(f"wrong source recipe: {record_path}")
            rows.append((record_path.stem, field, record_path, row))
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("duplicate material across Vext shards")
    return rows


def select_rows(source: Path, count: int, wait_seconds: int) -> list[tuple]:
    deadline = time.monotonic() + wait_seconds
    while True:
        rows = completed(source)
        if len(rows) >= count:
            rows.sort(key=lambda row: (int(row[3]["framework_atoms"]), row[0]))
            indices = np.linspace(0, len(rows) - 1, count, dtype=int)
            return [rows[index] for index in indices]
        if time.monotonic() >= deadline:
            raise TimeoutError(f"only {len(rows)}/{count} completed Vext materials")
        print(json.dumps({"available": len(rows), "needed": count}), flush=True)
        time.sleep(min(60, max(1, deadline - time.monotonic())))


def audit_one(name: str, field: Path, record: dict, gas: str,
              sample_centers: int = 256) -> dict:
    if hashlib.sha256(field.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"field content hash mismatch: {field}")
    with np.load(field, allow_pickle=False) as data:
        direct = data[f"orientation_K_{gas}"]
        marginal = data[f"marginal_K_{gas}"]
        site = data[f"site_K_{gas}"]
        axes = data[f"axes_{gas}"]
        cell = data["cell_matrix"]
        if (direct.shape != (64, 60, 60, 60) or marginal.shape != (60, 60, 60)
                or site.shape != (2, 60, 60, 60) or axes.shape != (64, 3)):
            raise ValueError(f"wrong saved Vext shape: {name} {gas}")
        seed = int(hashlib.sha256(f"{name}:{gas}".encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        all_centers = np.arange(60**3)
        accessible_centers = np.flatnonzero((marginal.ravel() < 2000)
                                         & (np.abs(marginal.ravel()) < 10000))
        uniform = rng.choice(all_centers, min(sample_centers // 2, len(all_centers)), replace=False)
        low = rng.choice(accessible_centers,
                         min(sample_centers - len(uniform), len(accessible_centers)),
                         replace=False)
        selected = np.unique(np.concatenate((uniform, low)))
        xyz = np.stack(np.unravel_index(selected, marginal.shape), axis=1)
        centers = (xyz + .5) / 60
        reference = direct[:, xyz[:, 0], xyz[:, 1], xyz[:, 2]].T.astype(float)
        estimate = lookup_poses(site, cell, centers, axes, GASES[gas]["bond"])
        center_reference = marginal[xyz[:, 0], xyz[:, 1], xyz[:, 2]].astype(float)
        center_estimate = marginal_energy(estimate, 298.)
        accessible_poses = (reference < 2000) & (np.abs(reference) < 10000)
        accessible_grid = (center_reference < 2000) & (np.abs(center_reference) < 10000)
        return {"name": name, "gas": gas, "framework_atoms": record["framework_atoms"],
                "field": str(field), "field_sha256": record["sha256"],
                "sampled_centers": int(len(selected)),
                "sampled_poses": int(reference.size),
                "accessible_grid_centers_in_field": int(len(accessible_centers)),
                "all_pose": summarize_error(reference, estimate, np.ones(reference.shape, bool)),
                "accessible_pose": summarize_error(reference, estimate, accessible_poses),
                "accessible_marginal": summarize_error(center_reference, center_estimate,
                                                          accessible_grid)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materials", type=int, default=50)
    parser.add_argument("--sample-centers", type=int, default=256)
    parser.add_argument("--wait-seconds", type=int, default=21600)
    args = parser.parse_args()
    if args.materials < 1 or args.sample_centers < 2 or args.output.exists():
        raise ValueError("positive sample and new output directory required")
    selected = select_rows(args.source, args.materials, args.wait_seconds)
    args.output.mkdir(parents=True)
    cases = []
    for index, (name, field, record_path, record) in enumerate(selected, 1):
        if json.loads(record_path.read_text()) != record:
            raise ValueError(f"record changed during audit: {record_path}")
        for gas in GASES:
            cases.append(audit_one(name, field, record, gas, args.sample_centers))
        if index % 10 == 0:
            print(json.dumps({"completed_materials": index, "cases": len(cases)}), flush=True)
    report = {"passed": len(cases) == 2 * args.materials,
              "materials": args.materials, "cases": len(cases),
              "sample_centers_per_case": args.sample_centers,
              "reference": "saved DIRECT 60^3 x 64 orientation_K",
              "candidate": "trilinear interpolation of saved 60^3 site_K",
              "sample_strategy": "half uniform centers, half DIRECT-marginal-accessible centers",
              "accessible": "DIRECT energy < 2000 K and abs(energy) < 10000 K",
              "selection_caveat": "first completed materials are large-atom structures, not random COFs",
              "by_gas": {}}
    for gas in GASES:
        rows = [row for row in cases if row["gas"] == gas]
        report["by_gas"][gas] = {}
        for metric in ("all_pose", "accessible_pose", "accessible_marginal"):
            count = sum(row[metric]["count"] for row in rows)
            within = sum(row[metric]["within_10K"] for row in rows)
            report["by_gas"][gas][metric] = {
                "count": count, "within_10K": within,
                "fraction_within_10K": within / count if count else None,
                "median_material_MAE_K": float(np.median(
                    [row[metric]["mae_K"] for row in rows if row[metric]["mae_K"] is not None]))}
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
