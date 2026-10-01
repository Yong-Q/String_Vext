"""Full-grid accuracy/timing check against an existing DIRECT Vext NPZ."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from graph_vext.cof8948_oldff_prepare import GASES, ROOT
from graph_vext.cof8948_oldff_vext import verify_task
from graph_vext.exact_neighbor_reuse_probe import ReusePotential
from graph_vext.exact_neighbor_reuse_native import NativeReusePotential
from graph_vext.orientation_oracle_fields import regular_pose_grid
from graph_vext.string_ff_vext import marginal_energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="bne+C56+C32+L14_COOH")
    parser.add_argument("--shard", choices=("shard0", "shard1"), default="shard0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-library", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prepared = ROOT / "runs/cof8948_string_oldff_v1/prepared_v1"
    base = ROOT / "inputs/cof8948_oldff_vext_full_v1" / args.shard
    field = base / "data" / f"{args.name}.npz"
    record = json.loads((base / "records" / f"{args.name}.json").read_text())
    if hashlib.sha256(field.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("saved DIRECT field hash mismatch")
    with (prepared / "tasks.csv").open(newline="") as handle:
        selected = [row for row in csv.DictReader(handle)
                    if row["name"] == args.name and row["direction"] == "1"]
    if {row["gas"] for row in selected} != set(GASES):
        raise ValueError("both corrected-gas inputs required")
    centers, axes = regular_pose_grid(60, 64)
    report = {"material": args.name, "shard": args.shard, "positions": len(centers),
              "orientations": len(axes), "cases": {}, "probe_only": True}
    with np.load(field, allow_pickle=False) as saved:
        for row in selected:
            gas = row["gas"]
            case = verify_task(prepared, row)
            stamp = time.perf_counter()
            potential = (NativeReusePotential(case, args.native_library)
                         if args.native_library is not None else ReusePotential(case))
            setup_seconds = time.perf_counter() - stamp
            stamp = time.perf_counter()
            reused = potential.evaluate(centers, axes)
            evaluation_seconds = time.perf_counter() - stamp
            direct = saved[f"orientation_K_{gas}"].reshape(64, -1).T.astype(float)
            if reused.shape != direct.shape or not np.isfinite(reused).all():
                raise ValueError("nonfinite or misaligned full orientation field")
            delta = np.abs(reused - direct)
            accessible = (direct < 2000) & (np.abs(direct) < 10000)
            generated_marginal = marginal_energy(reused, case["temperature"])
            saved_marginal = saved[f"marginal_K_{gas}"].ravel().astype(float)
            low = (saved_marginal < 2000) & (np.abs(saved_marginal) < 10000)
            marginal_error = np.abs(generated_marginal[low] - saved_marginal[low])
            metrics = {"framework_atoms": len(case["frame_frac"]),
                       "setup_seconds": setup_seconds,
                       "evaluation_seconds": evaluation_seconds,
                       "parent_neighbor_queries": potential.neighbor_queries,
                       "max_parent_neighbors": potential.max_neighbors,
                       "neighbor_query_seconds": getattr(potential, "query_seconds", None),
                       "native_compute_seconds": getattr(potential, "compute_seconds", None),
                       "accessible_poses": int(accessible.sum()),
                       "accessible_max_abs_error_K": float(delta[accessible].max()),
                       "accessible_p99_abs_error_K": float(np.percentile(delta[accessible], 99)),
                       "accessible_mae_K": float(delta[accessible].mean()),
                       "accessible_marginal_centers": int(low.sum()),
                       "marginal_max_abs_error_K": float(marginal_error.max()),
                       "marginal_p99_abs_error_K": float(np.percentile(marginal_error, 99)),
                       "passed": bool(accessible.any() and low.any()
                                      and delta[accessible].max() <= 1e-3
                                      and marginal_error.max() <= 1e-3)}
            report["cases"][gas] = metrics
            print(json.dumps({"gas": gas, **metrics}), flush=True)
    report["passed"] = all(value["passed"] for value in report["cases"].values())
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["passed"]:
        raise RuntimeError("full-grid reused-neighbor/direct comparison failed")


if __name__ == "__main__":
    main()
