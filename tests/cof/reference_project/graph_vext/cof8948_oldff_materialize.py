"""Atomically materialize verified new-guest inputs without touching originals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from graph_vext.cof8948_oldff_prepare import ROOT, rewrite_input


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_task_path(prepared: Path, row: dict) -> Path:
    name, gas = row["name"], row["gas"]
    direction = int(row["direction"])
    if (not name or Path(name).name != name or "/" in name or "\\" in name
            or name in (".", "..") or gas not in ("C2H4", "C2H6")
            or direction not in (1, 2, 3)):
        raise ValueError("unsafe material/gas/direction in task manifest")
    return prepared / "inputs" / gas / name / f"dir{direction}" / "input.dat"


def _one(item):
    prepared, row = item
    source = Path(row["source_input_path"])
    raw = source.read_bytes()
    if sha(raw) != row["source_input_sha256"]:
        raise ValueError(f"source input hash mismatch: {row['task_id']}")
    converted = rewrite_input(raw, row["gas"], int(row["direction"]))
    if sha(converted) != row["new_input_sha256"] or sha(source.read_bytes()) != sha(raw):
        raise ValueError(f"new input hash mismatch: {row['task_id']}")
    destination = _safe_task_path(prepared, row)
    if destination.exists():
        if sha(destination.read_bytes()) != row["new_input_sha256"]:
            raise ValueError(f"existing input hash mismatch: {destination}")
        return "cached"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".input.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(converted)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if sha(destination.read_bytes()) != row["new_input_sha256"]:
                raise ValueError(f"concurrent input hash mismatch: {destination}")
            return "cached"
    finally:
        temporary.unlink(missing_ok=True)
    return "created"


def materialize(prepared: Path, workers: int = 8) -> dict:
    prepared = Path(prepared).resolve()
    prepared.relative_to(ROOT)
    if workers < 1:
        raise ValueError("positive workers required")
    source_report = json.loads((prepared / "report.json").read_text())
    task_path = prepared / "tasks.csv"
    manifest_hash = sha(task_path.read_bytes())
    if not source_report.get("passed") or source_report["task_csv_sha256"] != manifest_hash:
        raise ValueError("sealed complete task manifest required")
    with task_path.open(newline="") as handle:
        tasks = list(csv.DictReader(handle))
    if len(tasks) != source_report["tasks"] or len({r["task_id"] for r in tasks}) != len(tasks):
        raise ValueError("task coverage/identity mismatch")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        statuses = list(pool.map(_one, ((prepared, row) for row in tasks)))
    output = prepared / "inputs_report.json"
    if output.exists():
        prior = json.loads(output.read_text())
        if prior["task_csv_sha256"] != manifest_hash or prior["inputs"] != len(tasks):
            raise ValueError("existing materialization report mismatch")
        return prior
    report = dict(passed=True, inputs=len(tasks), materials=source_report["materials"],
                  created=statuses.count("created"), cached=statuses.count("cached"),
                  task_csv_sha256=manifest_hash,
                  guest_recipe=source_report["gases"],
                  policy="verified source/new SHA; atomic exclusive publish; originals unchanged")
    temporary = prepared / f".inputs_report.{os.getpid()}.tmp"
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(materialize(args.prepared, args.workers), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
