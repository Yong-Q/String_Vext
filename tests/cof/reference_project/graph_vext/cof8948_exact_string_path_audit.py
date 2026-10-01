"""Same-pose String barrier replay: DIRECT versus accelerated corrected-FF LJ."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from graph_vext.audit_native_path_cap import stable_logd
from graph_vext.cof8948_oldff_prepare import GASES, ROOT
from graph_vext.cof8948_oldff_vext import PREPARED, load_plan, verify_task
from graph_vext.cof8948_oldff_vext_exact import code_hashes
from graph_vext.cof8948_string_runner import cpu_bodyx_pose_energies
from graph_vext.exact_neighbor_reuse_native import NativeReusePotential
from graph_vext.string_atom_mapping import rotate_sites


SELECTION = ROOT / "runs/cof8948_vext_exact_audit50_v1/snapshot.json"
LIBRARY = ROOT / "native/vext_exact_reuse_v1/kernel.so"


def replay_one(item: tuple) -> list[dict]:
    name, rows = item
    results = []
    for row in rows:
        case = verify_task(PREPARED, row)
        archive = Path(row["saved_path"])
        if hashlib.sha256(archive.read_bytes()).hexdigest() != row["saved_path_sha256"]:
            raise ValueError(f"saved path hash mismatch: {archive}")
        path = np.loadtxt(archive)
        if path.shape != (401, 7) or not np.isfinite(path).all():
            raise ValueError(f"invalid archived String path: {archive}")
        reference = cpu_bodyx_pose_energies(case, path)
        center = (case["guest_xyz"] * case["guest_mass"][:, None]).sum(0) / case["total_mass"]
        sites = rotate_sites(path[:, 3:6], case["guest_xyz"] - center)
        axes = sites[:, 1] - sites[:, 0]
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        kernel = NativeReusePotential(case, LIBRARY)
        candidate = np.array([kernel.evaluate(path[i:i + 1, :3], axes[i:i + 1])[0, 0]
                              for i in range(len(path))])
        if not np.isfinite(candidate).all():
            raise ValueError(f"nonfinite accelerated path energies: {archive}")
        difference = np.abs(reference - candidate)
        accessible = (reference < 2000) & (np.abs(reference) < 10000)
        barrier_ref = float(np.ptp(reference))
        barrier_new = float(np.ptp(candidate))
        path_ref = path.copy()
        path_new = path.copy()
        path_ref[:, 6] = reference
        path_new[:, 6] = candidate
        direction = int(row["direction"]) - 1
        capped_ref = stable_logd(case, path_ref, direction, cap=True)
        capped_new = stable_logd(case, path_new, direction, cap=True)
        result = {"name": name, "gas": row["gas"], "direction": direction + 1,
                  "archived_path": str(archive), "archived_path_sha256": row["saved_path_sha256"],
                  "input_sha256": row["new_input_sha256"],
                  "accessible_points": int(accessible.sum()),
                  "accessible_max_abs_error_K": float(difference[accessible].max())
                  if accessible.any() else None,
                  "all_max_relative_floor1": float((difference /
                       np.maximum(np.abs(reference), 1.)).max()),
                  "direct_barrier_K": barrier_ref,
                  "accelerated_barrier_K": barrier_new,
                  "raw_barrier_abs_error_K": abs(barrier_ref - barrier_new),
                  "raw_barrier_relative_error_floor1": abs(barrier_ref - barrier_new) /
                                                        max(abs(barrier_ref), 1.),
                  "direct_capped_logD": capped_ref,
                  "accelerated_capped_logD": capped_new,
                  "capped_logD_abs_error_dex": abs(capped_ref - capped_new),
                  "path_peak_ge_2000K": bool(reference.max() >= 2000)}
        if (result["all_max_relative_floor1"] > 1e-6
                or result["raw_barrier_abs_error_K"] > 1e-3 + 1e-8 * max(abs(barrier_ref), 1.)
                or result["capped_logD_abs_error_dex"] > 1e-6
                or (accessible.any() and result["accessible_max_abs_error_K"] > 1e-4)):
            raise ValueError(f"DIRECT/native String-path disagreement: {name} {row['gas']} dir{direction + 1}")
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.workers <= 32:
        raise ValueError("new report directory and 1-32 CPU workers required")
    selection = json.loads(SELECTION.read_text())
    names = [row["name"] for row in selection["selection"]]
    if len(names) != 50 or len(set(names)) != 50:
        raise ValueError("fifty unique COFs required")
    tasks, binding = load_plan(PREPARED, expected_materials=8948)
    if selection["source_manifest_sha256"] != binding:
        raise ValueError("Vext/path source manifest changed")
    grouped = {}
    for task in tasks:
        if task["name"] in names:
            grouped.setdefault(task["name"], []).append(task)
    if len(grouped) != 50 or any(len(rows) != 6 for rows in grouped.values()):
        raise ValueError("fifty complete six-direction String source groups required")
    args.output.mkdir(parents=True)
    (args.output / "selection.json").write_text(json.dumps(
        {"names": names, "source_manifest_sha256": binding,
         "code_hashes": code_hashes()}, indent=2) + "\n")
    cases = []
    with mp.Pool(processes=args.workers, maxtasksperchild=1) as pool:
        for material in pool.imap_unordered(replay_one,
                                           [(name, grouped[name]) for name in names]):
            cases.extend(material)
            if len(cases) % 60 == 0:
                print(json.dumps({"paths_completed": len(cases)}), flush=True)
    if code_hashes() != selection["code_hashes"]:
        raise RuntimeError("native Vext code changed during path audit")
    cases.sort(key=lambda row: (row["name"], row["gas"], row["direction"]))
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n")
    report = {"passed": len(cases) == 300, "materials": len(names), "paths": len(cases),
              "source_manifest_sha256": binding,
              "caveat": "archived BODY-X path geometries under corrected 92.8/108-K FF; not newly optimized paths",
              "by_gas": {}}
    for gas in GASES:
        rows = [row for row in cases if row["gas"] == gas]
        report["by_gas"][gas] = {
            "paths": len(rows), "paths_peak_ge_2000K": sum(row["path_peak_ge_2000K"] for row in rows),
            "max_accessible_pose_abs_error_K": max(
                (row["accessible_max_abs_error_K"] or 0) for row in rows),
            "max_raw_barrier_abs_error_K": max(row["raw_barrier_abs_error_K"] for row in rows),
            "max_raw_barrier_relative_error_floor1": max(
                row["raw_barrier_relative_error_floor1"] for row in rows),
            "max_capped_logD_abs_error_dex": max(
                row["capped_logD_abs_error_dex"] for row in rows)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["passed"]:
        raise RuntimeError("fifty-material String path audit incomplete")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
