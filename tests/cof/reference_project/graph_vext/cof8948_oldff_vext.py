"""FF-gated DIRECT Vext for corrected-guest 8,948 Pormake COFs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import resource
import time
from pathlib import Path

import numpy as np

from graph_vext.cof8948_oldff_prepare import GASES, ROOT
from graph_vext.rebuild_legacy_string_ff_vext import parse_string_definition
from graph_vext.string_ff_vext import direct_pose_energy, marginal_energy, uniform_fields


PREPARED = ROOT / "runs/cof8948_string_oldff_v1/prepared_v1"
SOURCES = ("cof8948_oldff_vext.py", "cof8948_oldff_prepare.py",
           "string_ff_vext.py", "rebuild_legacy_string_ff_vext.py",
           "orientation_oracle_fields.py")


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes() -> dict[str, str]:
    return {name: digest(ROOT / "graph_vext" / name) for name in SOURCES}


def _input_path(prepared: Path, row: dict) -> Path:
    name = row["name"]
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("unsafe material name")
    return (Path(prepared) / "inputs" / row["gas"] / name /
            f"dir{int(row['direction'])}" / "input.dat")


def verify_task(prepared: Path, row: dict, override: Path | None = None) -> dict:
    """Never allow 85/98-K original input to enter the corrected Vext worker."""
    gas = row["gas"]
    if gas not in GASES or int(row["direction"]) not in (1, 2, 3):
        raise ValueError("unknown gas/direction")
    path = Path(override) if override is not None else _input_path(prepared, row)
    if digest(path) != row["new_input_sha256"]:
        raise ValueError(f"new input hash mismatch: {path}")
    case = parse_string_definition(path.read_bytes())
    source = Path(row["source_input_path"])
    if digest(source) != row["source_input_sha256"]:
        raise ValueError(f"original source input hash mismatch: {source}")
    baseline = parse_string_definition(source.read_bytes())
    for key in ("cell", "frame_frac", "frame_sigma", "frame_epsilon", "frame_mass"):
        if not np.array_equal(case[key], baseline[key]):
            raise ValueError(f"source COF framework changed: {row['name']} {key}")
    expected = GASES[gas]
    body = np.array([[-expected["bond"] / 2, 0, 0],
                     [expected["bond"] / 2, 0, 0]])
    if (not np.allclose(case["guest_sigma"], expected["sigma"], rtol=0, atol=1e-6)
            or not np.allclose(case["guest_epsilon"], expected["epsilon"], rtol=0, atol=1e-6)
            or not np.allclose(case["guest_mass"], expected["site_mass"], rtol=0, atol=1e-6)
            or not np.allclose(case["original_body"], body, rtol=0, atol=1e-6)
            or not np.isclose(case["total_mass"], expected["mass"], rtol=0, atol=1e-6)
            or not np.allclose(case["numeric_conditions"],
                               [12.9, 1, expected["mass"], 298, 10000],
                               rtol=0, atol=1e-6)):
        raise ValueError(f"guest force-field mismatch: {row['name']} {gas}")
    return case


def compute_one_field(case: dict, size: int = 60, orientations: int = 64) -> dict:
    fields = uniform_fields(case, size=size, orientations=orientations,
                            energy_mode="direct")
    fields["grid"] = np.arcsinh(fields["marginal_K"] / 1000).astype(np.float32)
    return fields


def load_plan(prepared: Path = PREPARED, expected_materials: int = 8948):
    prepared = Path(prepared).resolve()
    prepared.relative_to(ROOT)
    report = json.loads((prepared / "report.json").read_text())
    inputs = json.loads((prepared / "inputs_report.json").read_text())
    manifest = prepared / "tasks.csv"
    manifest_sha = digest(manifest)
    if (not report.get("passed") or not inputs.get("passed")
            or report.get("materials") != expected_materials
            or report.get("tasks") != 6 * expected_materials
            or inputs.get("inputs") != 6 * expected_materials
            or report.get("task_csv_sha256") != manifest_sha
            or inputs.get("task_csv_sha256") != manifest_sha):
        raise ValueError("complete matching corrected-guest input gate required")
    for gas, expected in GASES.items():
        actual = inputs["guest_recipe"][gas]
        if any(not np.isclose(actual[key], expected[key], rtol=0, atol=1e-6)
               for key in ("sigma", "epsilon", "bond", "mass", "site_mass")):
            raise ValueError(f"wrong Vext gas recipe: {gas}")
    with manifest.open(newline="") as handle:
        tasks = list(csv.DictReader(handle))
    if len(tasks) != 6 * expected_materials or len({r["task_id"] for r in tasks}) != len(tasks):
        raise ValueError("duplicate/missing Vext source task")
    return tasks, manifest_sha


def _verify_cached(destination: Path, record_path: Path, name: str,
                   tasks: list[dict], size: int, orientations: int,
                   codes: dict, binding: str) -> dict:
    if not destination.is_file() or not record_path.is_file():
        raise ValueError(f"orphan Vext output retained: {name}")
    record = json.loads(record_path.read_text())
    if (record.get("name") != name or record.get("source_manifest_sha256") != binding
            or record.get("code_hashes") != codes or record.get("size") != size
            or record.get("orientations") != orientations
            or record.get("energy_mode") != "direct"
            or record.get("sha256") != digest(destination)):
        raise ValueError(f"resumed Vext source/code/content mismatch: {name}")
    for task in tasks:
        verify_task(PREPARED if "prepared" not in record else Path(record["prepared"]), task)
    with np.load(destination, allow_pickle=False) as data:
        for gas in GASES:
            for key, shape in (("orientation_K", (orientations, size, size, size)),
                               ("marginal_K", (size, size, size)),
                               ("site_K", (2, size, size, size)),
                               ("axes", (orientations, 3)),
                               ("grid", (size, size, size))):
                value = data[f"{key}_{gas}"]
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"resumed Vext shape/finiteness mismatch: {name} {gas} {key}")
    return record


def build_one(name: str, tasks: list[dict], prepared: Path, output: Path,
              size: int, orientations: int, codes: dict, binding: str,
              resume: bool = False) -> dict:
    prepared, output = Path(prepared), Path(output)
    if code_hashes() != codes or size < 2 or orientations < 4:
        raise ValueError("Vext code or grid definition changed")
    if len(tasks) != 6 or {task["name"] for task in tasks} != {name} or {
        (task["gas"], int(task["direction"])) for task in tasks
    } != {(gas, direction) for gas in GASES for direction in (1, 2, 3)}:
        raise ValueError(f"incomplete gas/direction definition: {name}")
    destination = output / "data" / f"{name}.npz"
    record_path = output / "records" / f"{name}.json"
    if destination.exists() or record_path.exists():
        if not resume:
            raise FileExistsError(destination)
        return _verify_cached(destination, record_path, name, tasks,
                              size, orientations, codes, binding)
    cases = {}
    frame = None
    for gas in GASES:
        numeric = None
        gas_tasks = sorted((task for task in tasks if task["gas"] == gas),
                           key=lambda task: int(task["direction"]))
        for task in gas_tasks:
            case = verify_task(prepared, task)
            if frame is None:
                frame = tuple(case[key].tobytes() for key in
                              ("cell", "frame_frac", "frame_sigma", "frame_epsilon"))
            elif frame != tuple(case[key].tobytes() for key in
                                ("cell", "frame_frac", "frame_sigma", "frame_epsilon")):
                raise ValueError(f"gas/direction framework mismatch: {name}")
            definition = tuple(case[key].tobytes() for key in
                               ("guest_sigma", "guest_epsilon", "guest_mass", "guest_xyz"))
            if numeric is None:
                numeric = definition
                cases[gas] = case
            elif definition != numeric:
                raise ValueError(f"three-direction guest mismatch: {name} {gas}")
    arrays, gases = {}, {}
    for gas, case in cases.items():
        field = compute_one_field(case, size, orientations)
        for key, value in field.items():
            arrays[f"{key}_{gas}"] = value
        points = np.array([[0, 0, 0], [size // 2] * 3, [size - 1] * 3])
        centers = (points + .5) / size
        expected = direct_pose_energy(case, centers, field["axes"].astype(float))
        cached = field["orientation_K"][:, points[:, 0], points[:, 1], points[:, 2]].T
        if not np.allclose(cached, expected, rtol=2e-4, atol=1e-3):
            raise ValueError(f"DIRECT source/grid energy mismatch: {name} {gas}")
        if not np.allclose(field["marginal_K"][points[:, 0], points[:, 1], points[:, 2]],
                           marginal_energy(expected, case["temperature"]),
                           rtol=2e-4, atol=1e-3):
            raise ValueError(f"angular marginal mismatch: {name} {gas}")
        gases[gas] = {"guest_sigma_A": GASES[gas]["sigma"],
                      "guest_epsilon_K": GASES[gas]["epsilon"],
                      "input_sha256": [task["new_input_sha256"] for task in tasks
                                       if task["gas"] == gas]}
    common = cases["C2H4"]
    arrays.update(cell_matrix=common["cell"],
                  atom_fractional=common["frame_frac"],
                  framework_sigma_A=common["frame_sigma"],
                  framework_epsilon_K=common["frame_epsilon"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    record = {"name": name, "prepared": str(prepared.resolve()),
              "source_manifest_sha256": binding,
              "code_hashes": codes, "size": size, "orientations": orientations,
              "energy_mode": "direct", "sha256": digest(destination),
              "gases": gases, "framework_atoms": len(common["frame_frac"]),
              "worker_peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    temp_record = record_path.with_suffix(".tmp")
    try:
        temp_record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        os.link(temp_record, record_path)
    finally:
        temp_record.unlink(missing_ok=True)
    return record


def estimated_worker_mib(atoms: int) -> int:
    """Conservative transient allowance; never infer zero cost from idle GPUs."""
    if atoms <= 0:
        raise ValueError("positive framework atom count required")
    return max(4096, 3072 + 4 * atoms)


def _group(tasks: list[dict]) -> dict[str, list[dict]]:
    grouped = {}
    for task in tasks:
        grouped.setdefault(task["name"], []).append(task)
    if any(len(rows) != 6 for rows in grouped.values()):
        raise ValueError("incomplete material gas/direction group")
    return grouped


def _write_report(path: Path, report: dict):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def pilot(prepared: Path, output: Path):
    tasks, binding = load_plan(prepared, expected_materials=100)
    grouped = _group(tasks)
    ordered = sorted(grouped, key=lambda name: int(grouped[name][0]["framework_atoms"]))
    indices = [0, len(ordered) // 5, 2 * len(ordered) // 5,
               3 * len(ordered) // 5, 4 * len(ordered) // 5, len(ordered) - 1]
    names = [ordered[index] for index in indices]
    if len(set(names)) != 6 or output.exists():
        raise ValueError("six unique pilot names and new output required")
    output.mkdir(parents=True)
    codes = code_hashes()
    with mp.Pool(processes=6, maxtasksperchild=1) as pool:
        pending = [pool.apply_async(build_one, (name, grouped[name], prepared,
                                                  output, 16, 64, codes, binding,
                                                  False)) for name in names]
        records = [result.get() for result in pending]
    if code_hashes() != codes:
        raise RuntimeError("Vext source changed during pilot")
    report = {"passed": True, "mode": "pilot", "materials": 6,
              "size": 16, "orientations": 64, "energy_mode": "direct",
              "source_manifest_sha256": binding, "code_hashes": codes,
              "guest_recipe": {gas: {key: GASES[gas][key] for key in
                               ("sigma", "epsilon", "bond", "mass", "site_mass")}
                               for gas in GASES},
              "worker_peak_rss_MiB": [row["worker_peak_rss_MiB"] for row in records],
              "names": names}
    _write_report(output / "report.json", report)
    return report


def run(prepared: Path, output: Path, pilot_report: Path, rank: int,
        workers: int = 96, resume: bool = False):
    tasks, binding = load_plan(prepared, expected_materials=8948)
    codes = code_hashes()
    pilot_gate = json.loads(Path(pilot_report).read_text())
    if (not pilot_gate.get("passed") or pilot_gate.get("mode") != "pilot"
            or pilot_gate.get("size") != 16 or pilot_gate.get("orientations") != 64
            or pilot_gate.get("energy_mode") != "direct"
            or pilot_gate.get("code_hashes") != codes
            or pilot_gate.get("guest_recipe") != {gas: {key: GASES[gas][key]
                 for key in ("sigma", "epsilon", "bond", "mass", "site_mass")}
                 for gas in GASES}):
        raise ValueError("matching six-material old-FF DIRECT pilot required")
    if rank not in (0, 1) or not 1 <= workers <= 256:
        raise ValueError("invalid Vext shard or workers")
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=resume)
    grouped = _group(tasks)
    names = list(grouped)[rank::2]
    names.sort(key=lambda name: int(grouped[name][0]["framework_atoms"]), reverse=True)
    allocated = int(os.environ.get("SLURM_MEM_PER_NODE", "512000"))
    reserve = max(65536, allocated // 8)
    budget = allocated - reserve
    if budget <= 0:
        raise ValueError("node memory allocation too small")
    active = []
    completed = []
    failures = []
    queue = iter(names)
    next_name = next(queue, None)
    reserved = 0
    start = time.monotonic()
    with mp.Pool(processes=workers, maxtasksperchild=1) as pool:
        while next_name is not None or active:
            while next_name is not None and len(active) < workers:
                estimate = estimated_worker_mib(int(grouped[next_name][0]["framework_atoms"]))
                if reserved + estimate > budget:
                    if not active:
                        raise ValueError(f"single Vext material exceeds memory budget: {next_name}")
                    break
                future = pool.apply_async(build_one, (next_name, grouped[next_name],
                                            prepared, output, 60, 64, codes,
                                            binding, resume))
                active.append((next_name, estimate, future))
                reserved += estimate
                next_name = next(queue, None)
            ready = [item for item in active if item[2].ready()]
            if not ready:
                time.sleep(2)
                continue
            for name, estimate, future in ready:
                try:
                    record = future.get()
                    completed.append(name)
                    if record["worker_peak_rss_MiB"] > estimate:
                        failures.append({"name": name, "error": "memory estimate exceeded measured peak"})
                except Exception as exc:
                    failures.append({"name": name, "error": repr(exc)})
                active.remove((name, estimate, future))
                reserved -= estimate
                if len(completed) % 100 == 0 and completed:
                    print(json.dumps({"shard": rank, "completed": len(completed),
                                      "failed": len(failures),
                                      "seconds": round(time.monotonic() - start, 1)}), flush=True)
    if code_hashes() != codes:
        raise RuntimeError("Vext source changed during full calculation")
    report = {"passed": not failures and len(completed) == len(names),
              "mode": "full", "rank": rank, "world_size": 2,
              "materials": len(completed), "expected": len(names),
              "size": 60, "orientations": 64, "energy_mode": "direct",
              "source_manifest_sha256": binding, "code_hashes": codes,
              "allocated_memory_MiB": allocated, "reserved_memory_MiB": reserve,
              "max_workers": workers, "failures": failures}
    report_path = (output / "report.json" if report["passed"] else
                   output / f"report_failed_{os.environ.get('SLURM_JOB_ID', 'manual')}.json")
    _write_report(report_path, report)
    if not report["passed"]:
        raise RuntimeError("corrected-guest Vext shard incomplete; evidence retained")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "pilot":
        result = pilot(args.prepared, args.output)
    else:
        if args.pilot_report is None or args.rank is None:
            parser.error("full mode needs --pilot-report and --rank")
        result = run(args.prepared, args.output, args.pilot_report,
                     args.rank, args.workers, args.resume)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
