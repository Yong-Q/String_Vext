"""Versioned exact-neighbor-reuse Vext continuation for the 8,948 COFs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np

from graph_vext.cof8948_oldff_prepare import GASES, ROOT
from graph_vext.cof8948_oldff_vext import (
    PREPARED, _group, _verify_cached, code_hashes as old_code_hashes,
    compute_one_field, digest, load_plan, verify_task,
)
from graph_vext.exact_neighbor_reuse_native import NativeReusePotential
from graph_vext.orientation_oracle_fields import regular_pose_grid
from graph_vext.string_ff_vext import TriclinicSitePotential, direct_pose_energy, marginal_energy


OLD = ROOT / "inputs/cof8948_oldff_vext_full_v1"
NEW = ROOT / "inputs/cof8948_oldff_vext_exact_v2"
PILOT = ROOT / "runs/cof8948_oldff_vext_exact_pilot_v2"
LIBRARY = ROOT / "native/vext_exact_reuse_v1/kernel.so"
SOURCES = ("cof8948_oldff_vext_exact.py", "exact_neighbor_reuse_probe.py",
           "exact_neighbor_reuse_native.py",
           "cof8948_oldff_vext.py", "cof8948_oldff_prepare.py",
           "string_ff_vext.py", "orientation_oracle_fields.py",
           "rebuild_legacy_string_ff_vext.py")


def code_hashes() -> dict[str, str]:
    result = {name: digest(ROOT / "graph_vext" / name) for name in SOURCES}
    result["native/vext_exact_reuse_v1/kernel.cpp"] = digest(
        ROOT / "native/vext_exact_reuse_v1/kernel.cpp")
    result["native/vext_exact_reuse_v1/kernel.so"] = digest(LIBRARY)
    return result


def compute_one_field_exact(case: dict, size: int = 60,
                            orientations: int = 64) -> dict:
    centers, axes = regular_pose_grid(size, orientations)
    reuse = NativeReusePotential(case, LIBRARY)
    energy = reuse.evaluate(centers, axes)
    marginal = marginal_energy(energy, case["temperature"])
    site_potential = TriclinicSitePotential(case)
    if not (np.array_equal(case["guest_sigma"], [case["guest_sigma"][0]] * 2)
            and np.array_equal(case["guest_epsilon"], [case["guest_epsilon"][0]] * 2)):
        raise ValueError("equal guest sites required for shared site grid")
    site = site_potential.evaluate(centers @ case["cell"],
                                   case["guest_sigma"][0], case["guest_epsilon"][0])
    field = {"orientation_K": energy.T.reshape(orientations, size, size, size).astype(np.float32),
             "marginal_K": marginal.reshape(size, size, size).astype(np.float32),
             "site_K": np.stack((site, site)).reshape(2, size, size, size).astype(np.float32),
             "axes": axes.astype(np.float32)}
    field["grid"] = np.arcsinh(field["marginal_K"] / 1000).astype(np.float32)
    if any(not np.isfinite(value).all() for value in field.values()):
        raise ValueError("nonfinite exact-neighbor Vext field")
    return field


def _write_json_exclusive(path: Path, record: dict):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_new_cached(data_path: Path, record_path: Path, name: str,
                       codes: dict, binding: str) -> dict:
    if not data_path.is_file() or not record_path.is_file():
        raise ValueError(f"orphan exact Vext output retained: {name}")
    record = json.loads(record_path.read_text())
    if (record.get("name") != name or record.get("source_manifest_sha256") != binding
            or record.get("code_hashes") != codes
            or record.get("energy_mode") != "exact_neighbor_reuse"
            or record.get("size") != 60 or record.get("orientations") != 64
            or record.get("sha256") != digest(data_path)):
        raise ValueError(f"exact Vext record/content mismatch: {name}")
    return record


def build_one_exact(name: str, tasks: list[dict], prepared: Path,
                    output: Path, size: int, orientations: int,
                    codes: dict, binding: str, resume: bool = False) -> dict:
    if code_hashes() != codes or size < 2 or orientations < 4:
        raise ValueError("exact Vext code/grid changed")
    if (len(tasks) != 6 or {task["name"] for task in tasks} != {name}
            or {(task["gas"], int(task["direction"])) for task in tasks}
               != {(gas, direction) for gas in GASES for direction in (1, 2, 3)}):
        raise ValueError(f"incomplete exact Vext gas/direction definition: {name}")
    output = Path(output)
    data_path = output / "data" / f"{name}.npz"
    record_path = output / "records" / f"{name}.json"
    if data_path.exists() or record_path.exists():
        if not resume or size != 60 or orientations != 64:
            raise FileExistsError(data_path)
        return _verify_new_cached(data_path, record_path, name, codes, binding)
    cases = {}
    frame = None
    for gas in GASES:
        definition = None
        gas_tasks = sorted((task for task in tasks if task["gas"] == gas),
                           key=lambda task: int(task["direction"]))
        for task in gas_tasks:
            case = verify_task(prepared, task)
            current = tuple(case[key].tobytes() for key in
                            ("cell", "frame_frac", "frame_sigma", "frame_epsilon"))
            if frame is None:
                frame = current
            elif current != frame:
                raise ValueError(f"gas/direction framework mismatch: {name}")
            gas_definition = tuple(case[key].tobytes() for key in
                                   ("guest_sigma", "guest_epsilon", "guest_mass", "guest_xyz"))
            if definition is None:
                definition = gas_definition
                cases[gas] = case
            elif definition != gas_definition:
                raise ValueError(f"gas definition changes by direction: {name} {gas}")
    arrays, gases = {}, {}
    for gas, case in cases.items():
        field = compute_one_field_exact(case, size, orientations)
        arrays.update({f"{key}_{gas}": value for key, value in field.items()})
        points = np.array([[0, 0, 0], [size // 2] * 3, [size - 1] * 3])
        centers = (points + .5) / size
        expected = direct_pose_energy(case, centers, field["axes"].astype(float))
        cached = field["orientation_K"][:, points[:, 0], points[:, 1], points[:, 2]].T
        if not np.allclose(cached, expected, rtol=2e-4, atol=1e-3):
            raise ValueError(f"exact Vext independent DIRECT check failed: {name} {gas}")
        gases[gas] = {"guest_sigma_A": GASES[gas]["sigma"],
                      "guest_epsilon_K": GASES[gas]["epsilon"],
                      "input_sha256": [task["new_input_sha256"] for task in tasks
                                       if task["gas"] == gas]}
    common = cases["C2H4"]
    arrays.update(cell_matrix=common["cell"], atom_fractional=common["frame_frac"],
                  framework_sigma_A=common["frame_sigma"],
                  framework_epsilon_K=common["frame_epsilon"])
    data_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = data_path.with_suffix(".tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        os.link(temporary, data_path)
    finally:
        temporary.unlink(missing_ok=True)
    record = {"name": name, "prepared": str(Path(prepared).resolve()),
              "source_manifest_sha256": binding, "code_hashes": codes,
              "size": size, "orientations": orientations,
              "energy_mode": "exact_neighbor_reuse", "sha256": digest(data_path),
              "gases": gases, "framework_atoms": len(common["frame_frac"]),
              "worker_peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    _write_json_exclusive(record_path, record)
    return record


def pilot():
    if PILOT.exists():
        raise FileExistsError(PILOT)
    tasks, binding = load_plan(PREPARED, expected_materials=8948)
    grouped = _group(tasks)
    previous = json.loads((ROOT / "inputs/cof8948_oldff_vext_pilot_v1/report.json").read_text())
    if not previous.get("passed") or len(previous.get("names", [])) != 6:
        raise ValueError("six-material DIRECT pilot required")
    names = previous["names"]
    codes = code_hashes()
    comparisons = []
    for name in names:
        for gas in GASES:
            case = verify_task(PREPARED, next(row for row in grouped[name]
                                                  if row["gas"] == gas and row["direction"] == "1"))
            reference = compute_one_field(case, size=16, orientations=64)
            candidate = compute_one_field_exact(case, size=16, orientations=64)
            checks = {}
            for key in reference:
                error = np.abs(candidate[key].astype(float) - reference[key].astype(float))
                relative = error / np.maximum(np.abs(reference[key]), 1.)
                checks[key] = {"max_absolute": float(error.max()),
                               "max_relative_floor1": float(relative.max())}
                if not np.allclose(candidate[key], reference[key], rtol=1e-6, atol=1e-3):
                    raise ValueError(f"exact Vext pilot mismatch: {name} {gas} {key}")
            comparisons.append({"name": name, "gas": gas,
                                "framework_atoms": len(case["frame_frac"]),
                                "checks": checks})
            print(json.dumps({"name": name, "gas": gas, "passed": True}), flush=True)
    if code_hashes() != codes:
        raise RuntimeError("exact Vext source changed during pilot")
    PILOT.mkdir(parents=True)
    _write_json_exclusive(PILOT / "report.json", {
        "passed": True, "mode": "pilot", "materials": len(names),
        "gas_cases": len(comparisons), "source_manifest_sha256": binding,
        "old_code_hashes": old_code_hashes(), "code_hashes": codes,
        "grid_size": 16, "orientations": 64,
        "comparisons": comparisons,
        "fullgrid_evidence": str(ROOT / "runs/cof8948_oldff_vext_native_probe_v4/result/report.json")})


def _old_completion(name: str, shard: int, rows: list[dict], binding: str):
    old = OLD / f"shard{shard}"
    data_path = old / "data" / f"{name}.npz"
    record_path = old / "records" / f"{name}.json"
    if not data_path.exists() and not record_path.exists():
        return None
    if not record_path.exists():
        # Preserve incomplete old data; exact v2 will publish separately.
        return None
    record = _verify_cached(data_path, record_path, name, rows, 60, 64,
                            old_code_hashes(), binding)
    return {"name": name, "data_path": str(data_path),
            "record_path": str(record_path), "data_sha256": record["sha256"],
            "record_sha256": digest(record_path)}


def run(shard: int, workers: int = 96):
    if shard not in (0, 1) or not 1 <= workers <= 96:
        raise ValueError("two exact Vext shards and at most 96 workers required")
    active_old = subprocess.run(["squeue", "-h", "-j", "81733,81734", "-o", "%i"],
                                capture_output=True, text=True, check=True)
    if active_old.stdout.strip():
        raise RuntimeError("old Vext jobs must stop before taking continuation snapshot")
    tasks, binding = load_plan(PREPARED, expected_materials=8948)
    codes = code_hashes()
    gate = json.loads((PILOT / "report.json").read_text())
    if (not gate.get("passed") or gate.get("code_hashes") != codes
            or gate.get("old_code_hashes") != old_code_hashes()
            or gate.get("source_manifest_sha256") != binding
            or gate.get("gas_cases") != 12):
        raise ValueError("matching exact Vext pilot and input binding required")
    grouped = _group(tasks)
    names = list(grouped)[shard::2]
    if len(names) != 4474:
        raise ValueError("unexpected 8,948-cohort shard size")
    output = NEW / f"shard{shard}"
    output.mkdir(parents=True, exist_ok=True)
    old_records = []
    new_records = []
    pending = []
    for name in names:
        previous = _old_completion(name, shard, grouped[name], binding)
        if previous is not None:
            old_records.append(previous)
            continue
        data_path = output / "data" / f"{name}.npz"
        record_path = output / "records" / f"{name}.json"
        if data_path.exists() or record_path.exists():
            new_records.append(_verify_new_cached(data_path, record_path,
                                                   name, codes, binding))
            continue
        pending.append(name)
    snapshot = {"shard": shard, "source_manifest_sha256": binding,
                "old_completed": old_records,
                "new_preexisting": [row["name"] for row in new_records],
                "pending": pending, "code_hashes": codes}
    _write_json_exclusive(output / f"snapshot_{os.environ.get('SLURM_JOB_ID', 'manual')}.json",
                          snapshot)
    print(json.dumps({"shard": shard, "old_completed": len(old_records),
                      "new_preexisting": len(new_records), "pending": len(pending)}), flush=True)
    pending.sort(key=lambda name: int(grouped[name][0]["framework_atoms"]), reverse=True)
    allocated = int(os.environ.get("SLURM_MEM_PER_NODE", "512000"))
    reserve = max(65536, allocated // 8)
    budget = allocated - reserve
    estimate = 4096
    concurrent = min(workers, budget // estimate)
    if concurrent < 1:
        raise ValueError("node memory budget too small")
    completed, failures = [], []
    started = time.monotonic()
    with mp.Pool(processes=concurrent, maxtasksperchild=1) as pool:
        queue = iter(pending)
        current = next(queue, None)
        active = []
        while current is not None or active:
            while current is not None and len(active) < concurrent:
                future = pool.apply_async(build_one_exact,
                    (current, grouped[current], PREPARED, output, 60, 64,
                     codes, binding, True))
                active.append((current, future))
                current = next(queue, None)
            ready = [(name, future) for name, future in active if future.ready()]
            if not ready:
                time.sleep(2)
                continue
            for name, future in ready:
                try:
                    record = future.get()
                    completed.append(name)
                    if record["worker_peak_rss_MiB"] > estimate:
                        failures.append({"name": name, "error": "worker exceeded 4-GiB allowance"})
                except Exception as exc:
                    failures.append({"name": name, "error": repr(exc)})
                active.remove((name, future))
                if len(completed) and len(completed) % 100 == 0:
                    print(json.dumps({"shard": shard, "new_completed": len(completed),
                                      "failed": len(failures),
                                      "seconds": round(time.monotonic() - started, 1)}), flush=True)
    if code_hashes() != codes:
        raise RuntimeError("exact Vext code changed during calculation")
    report = {"passed": not failures and len(old_records) + len(new_records) + len(completed) == len(names),
              "shard": shard, "expected": len(names), "old_completed": len(old_records),
              "new_preexisting": len(new_records), "new_completed": len(completed),
              "failed": failures, "allocated_memory_MiB": allocated,
              "reserved_memory_MiB": reserve, "worker_allowance_MiB": estimate,
              "concurrent_workers": concurrent,
              "source_manifest_sha256": binding, "code_hashes": codes}
    report_path = output / ("report.json" if report["passed"] else
                            f"report_failed_{os.environ.get('SLURM_JOB_ID', 'manual')}.json")
    _write_json_exclusive(report_path, report)
    if not report["passed"]:
        raise RuntimeError("exact Vext continuation incomplete; outputs retained")
    print(json.dumps(report, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "run"), required=True)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--workers", type=int, default=96)
    args = parser.parse_args()
    if args.mode == "pilot":
        pilot()
    else:
        if args.shard is None:
            parser.error("run mode needs --shard")
        run(args.shard, args.workers)


if __name__ == "__main__":
    main()
