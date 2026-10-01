"""Fifty-material full-grid DIRECT/native accuracy audit before continuation."""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from graph_vext.cof8948_oldff_prepare import GASES, ROOT
from graph_vext.cof8948_oldff_vext import PREPARED, digest, load_plan, verify_task
from graph_vext.cof8948_oldff_vext_exact import OLD, code_hashes, compute_one_field_exact


def audit_one(item: tuple) -> dict:
    name, shard, record_path, field_path, selected = item
    record = json.loads(record_path.read_text())
    if (record.get("name") != name or record.get("size") != 60
            or record.get("orientations") != 64 or record.get("energy_mode") != "direct"
            or record.get("sha256") != digest(field_path)):
        raise ValueError(f"invalid saved DIRECT source: {name}")
    metrics = {}
    with np.load(field_path, allow_pickle=False) as saved:
        for gas in GASES:
            case = verify_task(PREPARED, selected[gas])
            fresh = compute_one_field_exact(case)
            direct = saved[f"orientation_K_{gas}"].astype(float)
            candidate = fresh["orientation_K"].astype(float)
            if direct.shape != candidate.shape:
                raise ValueError(f"orientation field shape mismatch: {name} {gas}")
            accessible = (direct < 2000) & (np.abs(direct) < 10000)
            difference = np.abs(candidate[accessible] - direct[accessible])
            all_relative = np.abs(candidate - direct) / np.maximum(np.abs(direct), 1.)
            old_marginal = saved[f"marginal_K_{gas}"].astype(float)
            new_marginal = fresh["marginal_K"].astype(float)
            low = (old_marginal < 2000) & (np.abs(old_marginal) < 10000)
            marginal_error = np.abs(new_marginal[low] - old_marginal[low])
            site_error = np.abs(fresh["site_K"].astype(float)
                                - saved[f"site_K_{gas}"].astype(float))
            grid_error = np.abs(fresh["grid"].astype(float)
                                - saved[f"grid_{gas}"].astype(float))
            axes_equal = np.array_equal(fresh["axes"], saved[f"axes_{gas}"])
            if not accessible.any() or not low.any() or not axes_equal:
                raise ValueError(f"accessible/axes gate failed: {name} {gas}")
            metrics[gas] = {"accessible_poses": int(accessible.sum()),
                            "accessible_centers": int(low.sum()),
                            "orientation_max_abs_error_K": float(difference.max()),
                            "orientation_p99_abs_error_K": float(np.percentile(difference, 99)),
                            "orientation_all_max_relative_floor1": float(all_relative.max()),
                            "marginal_max_abs_error_K": float(marginal_error.max()),
                            "site_max_abs_error_K": float(site_error.max()),
                            "grid_max_abs_error": float(grid_error.max()),
                            "axes_bitwise_equal": axes_equal}
            if (metrics[gas]["orientation_max_abs_error_K"] > 1e-3
                    or metrics[gas]["orientation_all_max_relative_floor1"] > 1e-5
                    or metrics[gas]["marginal_max_abs_error_K"] > 1e-3
                    or metrics[gas]["site_max_abs_error_K"] > 1e-3
                    or metrics[gas]["grid_max_abs_error"] > 1e-6):
                raise ValueError(f"full-grid native/DIRECT mismatch: {name} {gas}")
    return {"name": name, "shard": shard,
            "framework_atoms": int(record["framework_atoms"]),
            "source_record": str(record_path), "source_sha256": record["sha256"],
            "gases": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.workers <= 32:
        raise ValueError("new audit output and 1-32 workers required")
    tasks, binding = load_plan(PREPARED, expected_materials=8948)
    mapping = {}
    for task in tasks:
        if task["direction"] == "1":
            mapping.setdefault(task["name"], {})[task["gas"]] = task
    available = []
    for shard in (0, 1):
        for record_path in (OLD / f"shard{shard}" / "records").glob("*.json"):
            field_path = OLD / f"shard{shard}" / "data" / f"{record_path.stem}.npz"
            if field_path.is_file() and record_path.stem in mapping:
                record = json.loads(record_path.read_text())
                available.append((int(record["framework_atoms"]), record_path.stem,
                                  shard, record_path, field_path))
    if len(available) < 50:
        raise ValueError(f"only {len(available)} completed DIRECT materials")
    available.sort()
    indices = np.linspace(0, len(available) - 1, 50, dtype=int)
    chosen = [available[index] for index in indices]
    if len({entry[1] for entry in chosen}) != 50:
        raise ValueError("duplicate in fifty-material audit")
    args.output.mkdir(parents=True)
    snapshot = {"source_manifest_sha256": binding, "code_hashes": code_hashes(),
                "available_at_selection": len(available),
                "selection": [{"name": entry[1], "framework_atoms": entry[0],
                               "shard": entry[2], "source_record": str(entry[3])}
                              for entry in chosen]}
    (args.output / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    items = [(name, shard, record_path, field_path, mapping[name])
             for _, name, shard, record_path, field_path in chosen]
    with mp.Pool(processes=args.workers, maxtasksperchild=1) as pool:
        cases = []
        for case in pool.imap_unordered(audit_one, items):
            cases.append(case)
            if len(cases) % 10 == 0:
                print(json.dumps({"completed_materials": len(cases)}), flush=True)
    if code_hashes() != snapshot["code_hashes"]:
        raise RuntimeError("exact Vext source changed during fifty-material audit")
    cases.sort(key=lambda row: row["name"])
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n")
    report = {"passed": len(cases) == 50, "materials": len(cases), "gas_fields": 2 * len(cases),
              "framework_atom_range": [min(row["framework_atoms"] for row in cases),
                                       max(row["framework_atoms"] for row in cases)],
              "source_manifest_sha256": binding, "code_hashes": snapshot["code_hashes"],
              "selection_caveat": "snapshot of completed DIRECT fields, not random across all 8,948",
              "by_gas": {}}
    for gas in GASES:
        values = [row["gases"][gas] for row in cases]
        report["by_gas"][gas] = {
            "accessible_poses": sum(row["accessible_poses"] for row in values),
            "orientation_max_abs_error_K": max(row["orientation_max_abs_error_K"] for row in values),
            "orientation_max_p99_abs_error_K": max(row["orientation_p99_abs_error_K"] for row in values),
            "marginal_max_abs_error_K": max(row["marginal_max_abs_error_K"] for row in values),
            "site_max_abs_error_K": max(row["site_max_abs_error_K"] for row in values),
            "grid_max_abs_error": max(row["grid_max_abs_error"] for row in values)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
