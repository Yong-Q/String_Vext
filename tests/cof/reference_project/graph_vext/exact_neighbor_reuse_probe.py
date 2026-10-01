"""Experimental exact COM-neighbor reuse for two identical String LJ sites."""
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
from graph_vext.orientation_oracle_fields import regular_pose_grid
from graph_vext.string_ff_vext import TriclinicSitePotential, body_offsets, direct_pose_energy


class ReusePotential:
    def __init__(self, case: dict):
        self.case = case
        if (not np.array_equal(case["guest_sigma"], np.repeat(case["guest_sigma"][0], 2))
                or not np.array_equal(case["guest_epsilon"], np.repeat(case["guest_epsilon"][0], 2))):
            raise ValueError("neighbor reuse probe requires identical guest LJ sites")
        self.offsets = body_offsets(case)
        self.cutoff = float(case["cutoff"])
        self.radius = self.cutoff + float(np.abs(self.offsets).max())
        self.potential = TriclinicSitePotential({**case, "cutoff": self.radius})
        sigma = (self.potential.sigma + float(case["guest_sigma"][0])) / 2
        epsilon = np.sqrt(self.potential.epsilon * float(case["guest_epsilon"][0]))
        sigma6 = sigma**6
        self.coeff12 = 4 * epsilon * sigma6**2
        self.coeff6 = 4 * epsilon * sigma6
        cut6 = (sigma / self.cutoff)**6
        self.shift = -4 * epsilon * (cut6**2 - cut6)
        self.floor2 = (.1 * sigma)**2
        self.neighbor_queries = 0
        self.max_neighbors = 0

    def evaluate(self, centers: np.ndarray, axes: np.ndarray) -> np.ndarray:
        centers, axes = np.asarray(centers, dtype=float), np.asarray(axes, dtype=float)
        if (centers.ndim != 2 or centers.shape[1] != 3
                or axes.ndim != 2 or axes.shape[1] != 3
                or not np.isfinite(centers).all() or not np.isfinite(axes).all()
                or (np.linalg.norm(axes, axis=1) <= 0).any()):
            raise ValueError("finite fractional centers and nonzero directors required")
        axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
        cart = (np.mod(centers, 1)) @ self.potential.cell
        output = np.zeros((len(centers), len(axes)), dtype=float)
        for start in range(0, len(cart), 32):
            positions = cart[start:start + 32]
            candidates = self.potential.tree.query_ball_point(positions, self.radius, workers=1)
            self.neighbor_queries += len(positions)
            for local, indices in enumerate(candidates):
                if not indices:
                    continue
                ids = np.asarray(indices, dtype=np.int64)
                self.max_neighbors = max(self.max_neighbors, len(ids))
                sites = positions[local][None, None, :] + self.offsets[:, None, None] * axes[None]
                delta = sites[:, :, None, :] - self.potential.images[ids][None, None]
                distance2 = np.einsum("...i,...i->...", delta, delta)
                active = distance2 < self.cutoff**2
                safe = np.maximum(distance2, self.floor2[ids])
                inverse6 = (1. / safe)**3
                energy = (self.coeff12[ids] * inverse6**2
                          - self.coeff6[ids] * inverse6 + self.shift[ids])
                output[start + local] = np.where(active, energy, 0.).sum(axis=(0, 2))
        if not np.isfinite(output).all():
            raise ValueError("nonfinite reused-neighbor energy")
        return output


def error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    diff = np.abs(reference - candidate)
    accessible = (reference < 2000.) & (np.abs(reference) < 10000.)
    relative = diff / np.maximum(np.abs(reference), 1.)
    return {"poses": int(diff.size), "accessible_poses": int(accessible.sum()),
            "accessible_max_abs_error_K": float(diff[accessible].max()) if accessible.any() else None,
            "accessible_mae_K": float(diff[accessible].mean()) if accessible.any() else None,
            "all_max_relative_error_floor1K": float(relative.max()),
            "all_max_abs_error_K": float(diff.max()),
            "passed": bool(accessible.any() and (diff[accessible] < 1e-4).all()
                           and (relative < 1e-6).all())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "runs/cof8948_string_oldff_v1/prepared_smoke100")
    parser.add_argument("--pilot-report", type=Path, default=ROOT / "inputs/cof8948_oldff_vext_pilot_v1/report.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--centers", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() or args.centers < 8:
        raise ValueError("new output and at least eight centers required")
    report = json.loads(args.pilot_report.read_text())
    if not report.get("passed") or len(report.get("names", [])) != 6:
        raise ValueError("six-material corrected-force-field pilot required")
    with (args.prepared / "tasks.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    lookup = {(row["name"], row["gas"], int(row["direction"])): row for row in rows}
    axes = regular_pose_grid(1, 64)[1]
    rng = np.random.default_rng(20261001)
    centers = rng.random((args.centers - 8, 3))
    edge = np.array([[.001, .999, .5], [.999, .001, .5], [.5, .001, .999],
                     [.001, .5, .999], [.999, .5, .001], [.5, .999, .001],
                     [.001, .001, .001], [.999, .999, .999]])
    centers = np.concatenate((centers, edge))
    cases = []
    for name in report["names"]:
        for gas in GASES:
            case = verify_task(args.prepared, lookup[name, gas, 1])
            skew = float(np.abs((case["cell"] / np.linalg.norm(case["cell"], axis=1)[:, None])
                                @ (case["cell"] / np.linalg.norm(case["cell"], axis=1)[:, None]).T
                                - np.eye(3)).max())
            stamp = time.perf_counter()
            reference = direct_pose_energy(case, centers, axes)
            direct_seconds = time.perf_counter() - stamp
            stamp = time.perf_counter()
            candidate = ReusePotential(case)
            setup_seconds = time.perf_counter() - stamp
            stamp = time.perf_counter()
            values = candidate.evaluate(centers, axes)
            reuse_seconds = time.perf_counter() - stamp
            metrics = error_metrics(reference, values)
            cases.append({"name": name, "gas": gas,
                          "input_sha256": lookup[name, gas, 1]["new_input_sha256"],
                          "framework_atoms": len(case["frame_frac"]), "cell_skew": skew,
                          "direct_seconds": direct_seconds, "reuse_setup_seconds": setup_seconds,
                          "reuse_seconds": reuse_seconds,
                          "kernel_speedup": direct_seconds / reuse_seconds,
                          "parent_neighbor_queries": candidate.neighbor_queries,
                          "maximum_parent_neighbors": candidate.max_neighbors,
                          **metrics})
            print(json.dumps({"name": name, "gas": gas, "passed": metrics["passed"],
                              "speedup": direct_seconds / reuse_seconds}), flush=True)
    result = {"passed": all(row["passed"] for row in cases),
              "materials": 6, "gases": len(GASES), "centers_per_case": args.centers,
              "orientations": 64, "cases": cases,
              "probe_only": True,
              "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "note": "kernel comparison on selected centers, not full-grid end-to-end throughput"}
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["passed"]:
        raise RuntimeError("COM-neighbor reuse failed strict DIRECT comparison")
    print(json.dumps({"passed": True, "cases": len(cases)}), flush=True)


if __name__ == "__main__":
    main()
