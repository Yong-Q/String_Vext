"""Single-writer initialization of the 53,688-task GPU String pool."""
from __future__ import annotations

import json
from pathlib import Path

from graph_vext import legacy8238_runner_v7 as control
from graph_vext import cof8948_string_runner as original
from graph_vext.cof8948_string_shared_v2 import (BASE, PILOT, POOL,
                                                   code_binding, task_order)
from graph_vext.string_shared_queue import SharedQueue


def main():
    tasks, binding = original.load_plan()
    gate = json.loads((PILOT / "gate.json").read_text())
    native = control.native_binding(original.EXE)
    if (not gate.get("passed") or gate.get("manifest_sha256") != binding
            or gate.get("native_binding") != native
            or gate.get("code_binding") != code_binding()):
        raise ValueError("shared String GPU pilot/source binding changed")
    previous = json.loads((BASE / "pilot_v1/gate.json").read_text())
    results = [profile["result"] for profile in previous["profiles"]]
    results.append(gate["result"])
    expected_ids = {row["task_id"] for row in tasks}
    if len(results) != len({row["task_id"] for row in results}) or \
            any(row["task_id"] not in expected_ids or row["status"] not in
                ("completed", "completed_with_review") for row in results):
        raise ValueError("pilot results cannot seed full pool")
    for result in results:
        record = Path(result["attempt_path"]) / "result.json"
        if json.loads(record.read_text())["task_id"] != result["task_id"]:
            raise ValueError(f"pilot result file changed: {record}")
        control.publish(BASE, result)
    queue = SharedQueue(POOL, task_order(tasks), binding)
    completed = set(control.read_completed(BASE))
    queue.bootstrap_completed(completed)
    report = {"passed": True, "manifest_sha256": binding,
              "native_binding": native, "code_binding": code_binding(),
              "task_count": len(tasks), "seeded_completed": len(completed),
              "pilot_result_ids": sorted(row["task_id"] for row in results),
              "pool": str(POOL)}
    path = POOL / "initialization_report.json"
    with path.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
