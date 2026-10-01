"""Independent BODY-X String rerun for the corrected-guest 8,948 COFs."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time

import numpy as np

from graph_vext import legacy8238_runner_v7 as control
from graph_vext.cof8948_oldff_prepare import GASES, ROOT
from graph_vext.cof8948_oldff_vext import load_plan as load_input_plan, verify_task
from graph_vext.rebuild_uncapped_labels import linear_value, path_logd, path_qc
from graph_vext.audit_native_path_cap import stable_logd
from graph_vext.string_atom_mapping import rotate_sites
from graph_vext.string_ff_vext import TriclinicSitePotential


BASE = ROOT / "runs/cof8948_string_oldff_v1"
PREPARED = BASE / "prepared_v1"
EXE = ROOT / "native/string_legacy8238_v4/GPU_string_legacy8238_v4"
RESERVE_MIB = 1024


def load_plan(prepared: Path = PREPARED):
    tasks, binding = load_input_plan(prepared, expected_materials=8948)
    prepared = Path(prepared)
    for row in tasks:
        row["input_path"] = str(prepared / "inputs" / row["gas"] / row["name"] /
                                f"dir{row['direction']}" / "input.dat")
    if len(tasks) != 53688 or {int(row["shard"]) for row in tasks} != set(range(6)):
        raise ValueError("corrected 8,948-COF six-shard plan required")
    return tasks, binding


def cpu_bodyx_pose_energies(case: dict, pose: np.ndarray) -> np.ndarray:
    p = np.asarray(pose, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 7 or not np.isfinite(p).all()
            or not np.allclose(case["guest_xyz"][:, 1:], 0, atol=1e-10)):
        raise ValueError("finite BODY-X path required")
    center = (case["guest_xyz"] * case["guest_mass"][:, None]).sum(0) / case["total_mass"]
    body = case["guest_xyz"] - center
    sites = rotate_sites(p[:, 3:6], body)
    potential = TriclinicSitePotential(case)
    cart = p[:, :3] @ case["cell"]
    energy = np.zeros(len(p))
    for site in range(2):
        energy += potential.evaluate(cart + sites[:, site],
                                     case["guest_sigma"][site], case["guest_epsilon"][site])
    return energy


def execute(task, attempt, exe, uuid, tracked, lock, replay_all=False, timeout=7200):
    work = attempt / task["task_id"]
    work.mkdir(exist_ok=False)
    gas, direction = task["gas"], int(task["direction"])
    recipe = GASES[gas]
    result = dict(task_id=task["task_id"], name=task["name"], gas=gas,
                  direction=direction, guest_sigma_A=recipe["sigma"],
                  guest_epsilon_K=recipe["epsilon"], guest_bond_A=recipe["bond"],
                  guest_mass_g_mol=recipe["mass"], status="failed", warnings="",
                  attempt_path=str(work), job_id=os.environ.get("SLURM_JOB_ID"), gpu_uuid=uuid)
    proc = None
    try:
        case = verify_task(PREPARED, task)
        source = Path(task["input_path"])
        with (work / "input.dat").open("xb") as handle:
            handle.write(source.read_bytes())
        command = [str(exe), "input.dat"]
        saved = Path(task["saved_path"])
        if control.sha(saved) != task["saved_path_sha256"]:
            raise ValueError("original saved path hash mismatch")
        try:
            candidate = np.loadtxt(saved)
            if candidate.shape == (401, 7) and np.isfinite(candidate).all():
                qc = path_qc(case, candidate, direction - 1)
                if qc["endpoint_winding_error"] <= 5e-4 and qc["max_segment_to_mean"] <= 20:
                    with (work / "initial.dat").open("xb") as handle:
                        handle.write(saved.read_bytes())
                    command.append("initial.dat")
                    result["seed_sha256"] = task["saved_path_sha256"]
        except (ValueError, OSError) as exc:
            result["seed_rejected_reason"] = str(exc)
        command.append("string_path.dat")
        result["warm_started"] = "initial.dat" in command
        with (work / "native.stdout").open("x") as stdout, (work / "native.stderr").open("x") as stderr:
            proc = subprocess.Popen(command, cwd=work, stdout=stdout, stderr=stderr)
            with lock:
                tracked[task["task_id"]]["proc"] = proc
            started = time.monotonic()
            while proc.poll() is None:
                with lock:
                    abort = tracked[task["task_id"]].get("abort")
                if abort or time.monotonic() - started > timeout:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait()
                    raise RuntimeError(abort or "case_timeout")
                time.sleep(1)
        if proc.returncode != 0:
            raise RuntimeError(f"native_exit_{proc.returncode}")
        stdout = (work / "native.stdout").read_text()
        stderr = (work / "native.stderr").read_text()
        if any(word in stderr.lower() for word in ("out of memory", "illegal memory", "fatal error")):
            raise RuntimeError("native CUDA failure")
        iterations = control.optimizer_iterations(stdout)
        import re
        stage_match = re.search(r"legacy_selection:.*chosen_stage=(initial|final)", stdout)
        convergence = re.search(r"final_convergence:.*simultaneous=([01])", stdout)
        if not stage_match or not convergence:
            raise ValueError("missing native stage/convergence contract")
        stage, final_ok = stage_match.group(1), convergence.group(1) == "1"
        path = np.loadtxt(work / "string_path.dat")
        if path.shape != (401, 7) or not np.isfinite(path).all():
            raise ValueError("invalid selected 401x7 path")
        selected = np.loadtxt(work / f"string_path.dat.{stage}")
        if not np.array_equal(path, selected):
            raise ValueError("selected path differs from recorded candidate")
        qc = path_qc(case, path, direction - 1)
        if qc["endpoint_winding_error"] > 5e-4 or qc["max_segment_to_mean"] > 20:
            raise ValueError("path winding/spacing failure")
        if stage == "final" and not final_ok:
            raise ValueError("final path not jointly converged")
        sample = path if replay_all else path[np.linspace(0, 400, 9, dtype=int)]
        checked = cpu_bodyx_pose_energies(case, sample)
        errors = np.abs(checked - sample[:, 6])
        if not (errors <= 1e-4 + 1e-7 * np.maximum(np.abs(checked), np.abs(sample[:, 6]))).all():
            raise ValueError("BODY-X triclinic CPU/native energy disagreement")
        capped_log = stable_logd(case, path, direction - 1, cap=True)
        diffusion = linear_value(capped_log)
        if diffusion is None:
            raise ValueError("linear capped diffusion unrepresentable")
        warnings = []
        if path[:, 6].max() >= 2000:
            warnings.append("historical_saddle_cap_used")
        if path[:, 6].max() >= 1e5:
            warnings.append("extreme_repulsive_path_needs_review")
        if stage == "initial":
            warnings.append("initial_candidate_convergence_not_certified")
        result.update(status="completed_with_review" if warnings else "completed",
                      D_m2_s=diffusion, logD=capped_log,
                      uncapped_logD=path_logd(case, path, direction - 1),
                      barrier_K=float(np.ptp(path[:, 6])), peak_K=float(path[:, 6].max()),
                      capped=bool(path[:, 6].max() >= 2000), stage=stage,
                      final_converged=final_ok, warnings=";".join(warnings),
                      optimizer_iterations=iterations,
                      framework_atoms=int(task["framework_atoms"]),
                      CPU_energy_checked_points=len(sample),
                      CPU_energy_max_abs_error_K=float(errors.max()),
                      path_sha256=control.sha(work / "string_path.dat"), path_qc=qc)
        if control.sha(source) != task["new_input_sha256"]:
            raise ValueError("prepared input changed during native run")
    except Exception as exc:
        result.update(status="failed", error=str(exc), warnings=str(exc))
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
    control.atomic_json(work / "result.json", result)
    return result


def pilot(prepared: Path, exe: Path, output: Path):
    tasks, binding = load_plan(prepared)
    native = control.native_binding(exe)
    uuid = control.allocated_uuid()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    atoms = {row["name"]: int(row["framework_atoms"]) for row in tasks[::6]}
    ordered = sorted(atoms, key=lambda name: (atoms[name], name))
    names = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    chosen = [row for name in names for row in tasks
              if row["name"] == name and int(row["direction"]) == 1]
    if len(chosen) != 6:
        raise ValueError("pilot requires three BODY-X materials and both gases")
    profiles = []
    lock = threading.Lock()
    for task in chosen:
        size = atoms[task["name"]]
        before = control.query(uuid)
        bound = control.source_allocation_budget(size)
        if before["free_MiB"] < RESERVE_MIB + bound:
            raise RuntimeError("pilot cannot preserve 1 GiB under source allocation budget")
        tracked = {task["task_id"]: dict(proc=None, abort=None)}
        observations = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(execute, task, output, exe, uuid, tracked, lock, True)
            while not future.done():
                card = control.query(uuid)
                observations.append(card)
                if card["free_MiB"] < RESERVE_MIB:
                    with lock:
                        tracked[task["task_id"]]["abort"] = "pilot GPU reserve breached"
                time.sleep(.15)
            result = future.result()
        if result["status"] == "failed":
            raise RuntimeError("BODY-X pilot failed: " + result.get("error", ""))
        increment = max((row["used_MiB"] for row in observations),
                        default=before["used_MiB"]) - before["used_MiB"]
        profiles.append(dict(atoms=size, sampled_peak_increment_MiB=increment,
                             source_allocation_budget_MiB=bound, result=result))
        control.atomic_json(output / "progress.json", dict(profiles=profiles))
    if control.native_binding(exe) != native:
        raise RuntimeError("native binary/source changed during pilot")
    if any(profile["sampled_peak_increment_MiB"] >
           control.source_allocation_budget(profile["atoms"]) for profile in profiles):
        raise RuntimeError("source allocation budget below measured pilot peak")
    control.atomic_json(output / "gate.json", dict(
        passed=True, manifest_sha256=binding, native_binding=native,
        runner_sha256=control.sha(__file__), reserve_MiB=RESERVE_MIB,
        gpu_uuid=uuid, profiles=profiles, materials=names,
        validation="six BODY-X/nonorthogonal String runs; all 401 path energies independently CPU-replayed",
        caveat="pilot validates the solver/FF/throughput, not global path optimality"))


def run(prepared: Path, exe: Path, gate_path: Path, shard: int,
        maximum: int = 25, stagger: float = 5.):
    tasks, binding = load_plan(prepared)
    native = control.native_binding(exe)
    gate = json.loads(gate_path.read_text())
    if (not gate.get("passed") or gate.get("manifest_sha256") != binding
            or gate.get("native_binding") != native
            or gate.get("runner_sha256") != control.sha(__file__)
            or gate.get("reserve_MiB") != RESERVE_MIB):
        raise ValueError("matching BODY-X pilot/runner/native/1-GiB gate required")
    if not 0 <= shard < 6 or not 1 <= maximum <= 25 or not 3 <= stagger <= 8:
        raise ValueError("invalid GPU shard/concurrency/stagger")
    job = os.environ["SLURM_JOB_ID"]
    uuid = control.allocated_uuid()
    out = BASE / "workers" / control.monitor_name(job, shard, uuid)
    out.mkdir(parents=True, exist_ok=False)
    host_budget = int(os.environ.get("SLURM_MEM_PER_NODE", "32768"))
    atoms = {row["name"]: int(row["framework_atoms"]) for row in tasks[::6]}
    lease = BASE / f"shard_{shard}.lock"
    with lease.open("a") as ownership:
        fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (BASE / "csv.lock").open("a") as snapshot_lock:
            fcntl.flock(snapshot_lock, fcntl.LOCK_EX)
            completed = control.read_completed(BASE)
        failed_before = {row["task_id"] for row in control.rows(BASE / "failures.csv")}
        queue = sorted((row for row in tasks if int(row["shard"]) == shard
                        and row["task_id"] not in completed),
                       key=lambda row: (row["task_id"] in failed_before,
                                        atoms[row["name"]], int(row["task_index"])))
        control.atomic_json(out / "context.json", dict(
            shard=shard, job_id=job, host=socket.gethostname(), gpu_uuid=uuid,
            manifest_sha256=binding, native_binding=native, maximum=maximum,
            stagger_seconds=stagger, reserve_MiB=RESERVE_MIB,
            host_budget_MiB=host_budget, remaining_tasks=len(queue),
            memory_policy="v7 source allocation bound calibrated by six BODY-X pilot runs"))
        tracked, active = {}, {}
        lock = threading.Lock()
        last = -float("inf")
        count = 0
        stopping = False
        empty_wait = None

        def stop(_signum, _frame):
            nonlocal stopping
            stopping = True
            with lock:
                for slot in tracked.values():
                    slot["abort"] = "job_stop_requested"

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        with (out / "gpu_memory.csv").open("x", newline="") as memory, \
                ThreadPoolExecutor(max_workers=maximum) as pool:
            fields = ["time", "job_id", "shard", "gpu_uuid", "used_MiB", "free_MiB",
                      "active", "pending_MiB", "queued", "next_launch_gap_seconds",
                      "initializing", "steady_allocating", "derivative_ready"]
            writer = csv.DictWriter(memory, fieldnames=fields)
            writer.writeheader()
            monitor_failures = 0
            while queue or active:
                for future, key in list(active.items()):
                    if future.done():
                        result = future.result()
                        control.publish(BASE, result, completed)
                        count += 1
                        del active[future]
                        with lock:
                            del tracked[key]
                if stopping and not active:
                    break
                try:
                    card = control.query(uuid)
                    allocated = control.process_memory(uuid)
                    monitor_failures = 0
                    with lock:
                        for key, slot in tracked.items():
                            proc = slot["proc"]
                            pid = proc.pid if proc else None
                            control.update_allocation_phase(slot, out / key / "native.stderr",
                                                            allocated.get(pid, 0.), control.rss(pid))
                        snapshots = [dict(slot) for slot in tracked.values()]
                        if card["free_MiB"] < RESERVE_MIB:
                            stopping = True
                            if snapshots:
                                tracked[next(reversed(tracked))]["abort"] = "GPU reserve breached"
                    pending = sum(max(0., slot["estimate"] - slot["allocated"])
                                  for slot in snapshots)
                    gap = control.launch_interval(card["used_MiB"], stagger)
                    phases = Counter(slot["phase"] for slot in snapshots)
                    writer.writerow(dict(time=time.time(), job_id=job, shard=shard,
                                         gpu_uuid=uuid, used_MiB=card["used_MiB"],
                                         free_MiB=card["free_MiB"], active=len(active),
                                         pending_MiB=pending, queued=len(queue),
                                         next_launch_gap_seconds=gap if gap is not None else "paused",
                                         initializing=phases["initial"],
                                         steady_allocating=phases["steady_allocating"],
                                         derivative_ready=phases["derivative_ready"]))
                    memory.flush()
                    control.atomic_json(out / "gpu_latest.json", dict(
                        **card, active=len(active), pending_MiB=pending,
                        queued=len(queue), next_launch_gap_seconds=gap, phases=dict(phases)))
                    if queue and not stopping and gap is not None and time.monotonic() - last >= gap:
                        for _ in range(control.launch_count(card["used_MiB"])):
                            if not queue:
                                break
                            task = queue[0]
                            budget = control.calibrated_start_budget(
                                atoms[task["name"]], gate["profiles"])
                            if not control.admit(card["free_MiB"], snapshots, budget,
                                                 RESERVE_MIB, maximum, host_budget):
                                if not active:
                                    empty_wait = empty_wait or time.monotonic()
                                    if time.monotonic() - empty_wait > 120:
                                        raise RuntimeError("task exceeds calibrated GPU/host budget")
                                break
                            queue.pop(0)
                            key = task["task_id"]
                            slot = dict(estimate=budget, allocated=0., rss=0.,
                                        atoms=atoms[task["name"]], phase="initial",
                                        host_estimate=budget * 1.5, proc=None, abort=None)
                            with lock:
                                tracked[key] = slot
                            snapshots.append(dict(slot))
                            active[pool.submit(execute, task, out, exe, uuid, tracked, lock)] = key
                            last = time.monotonic()
                            empty_wait = None
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    monitor_failures += 1
                    control.atomic_json(out / "monitor_failure.json", dict(
                        error=str(exc), consecutive_failures=monitor_failures))
                    if monitor_failures >= 3:
                        stop(None, None)
                time.sleep(2)
        if control.native_binding(exe) != native:
            raise RuntimeError("native binary/source changed during full run")
        with (BASE / "csv.lock").open("a") as csv_lock:
            fcntl.flock(csv_lock, fcntl.LOCK_EX)
            control.rebuild_summary(BASE)
            all_completed = control.read_completed(BASE)
        expected = {row["task_id"] for row in tasks if int(row["shard"]) == shard}
        missing = expected - set(all_completed)
        control.atomic_json(out / "report.json", dict(
            completed_this_job=count, stopped=stopping, queued=len(queue),
            shard=shard, expected_tasks=len(expected), missing_tasks=len(missing),
            first_missing_task_ids=sorted(missing)[:20],
            all_six_gas_directions_recorded=len(missing) == 0))
        if queue or stopping or missing:
            raise RuntimeError("dispatcher incomplete; successes retained for resume")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "run"), required=True)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--executable", type=Path, default=EXE)
    parser.add_argument("--output", type=Path, default=BASE / "pilot_v1")
    parser.add_argument("--gate", type=Path, default=BASE / "pilot_v1/gate.json")
    parser.add_argument("--shard", type=int)
    parser.add_argument("--max-concurrent", type=int, default=25)
    parser.add_argument("--stagger", type=float, default=5.)
    args = parser.parse_args()
    if args.mode == "pilot":
        pilot(args.prepared, args.executable, args.output)
    else:
        if args.shard is None:
            parser.error("run mode needs --shard")
        run(args.prepared, args.executable, args.gate, args.shard,
            args.max_concurrent, args.stagger)


if __name__ == "__main__":
    main()
