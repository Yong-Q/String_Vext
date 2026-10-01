"""Audit an exact, duplicate-free union of old and accelerated COF Vext."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from graph_vext.cof8948_oldff_prepare import GASES, ROOT
from graph_vext.cof8948_oldff_vext import code_hashes as old_code_hashes, digest, load_plan
from graph_vext.cof8948_oldff_vext_exact import NEW, OLD, code_hashes as exact_code_hashes


OUTPUT = ROOT / "inputs/cof8948_oldff_vext_union_v2"


def choose_source(old: bool, exact: bool) -> str:
    if old == exact:
        raise ValueError("each material needs exactly one old or accelerated Vext")
    return "old" if old else "exact_v2"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    tasks, binding = load_plan(expected_materials=8948)
    names = list(dict.fromkeys(row["name"] for row in tasks))
    if len(names) != 8948:
        raise ValueError("incomplete 8,948-material source manifest")
    expected_inputs = {}
    for task in tasks:
        expected_inputs.setdefault((task["name"], task["gas"]), []).append(
            task["new_input_sha256"])
    reports = []
    for shard in (0, 1):
        report = json.loads((NEW / f"shard{shard}" / "report.json").read_text())
        if (not report.get("passed") or report.get("expected") != 4474
                or report.get("source_manifest_sha256") != binding
                or report.get("code_hashes") != exact_code_hashes()):
            raise ValueError(f"exact shard{shard} completion gate failed")
        reports.append(report)
    rows = []
    for index, name in enumerate(names):
        shard = index % 2
        old_dir, new_dir = OLD / f"shard{shard}", NEW / f"shard{shard}"
        old_record = old_dir / "records" / f"{name}.json"
        new_record = new_dir / "records" / f"{name}.json"
        source = choose_source(old_record.is_file(), new_record.is_file())
        root, path = (old_dir, old_record) if source == "old" else (new_dir, new_record)
        data = root / "data" / f"{name}.npz"
        record = json.loads(path.read_text())
        expected_mode = "direct" if source == "old" else "exact_neighbor_reuse"
        expected_codes = old_code_hashes() if source == "old" else exact_code_hashes()
        if (record.get("name") != name or record.get("source_manifest_sha256") != binding
                or record.get("size") != 60 or record.get("orientations") != 64
                or record.get("energy_mode") != expected_mode
                or record.get("code_hashes") != expected_codes
                or not data.is_file() or record.get("sha256") != digest(data)):
            raise ValueError(f"Vext record/data provenance mismatch: {name}")
        for gas in GASES:
            gas_record = record["gases"][gas]
            if (gas_record["guest_sigma_A"] != GASES[gas]["sigma"]
                    or gas_record["guest_epsilon_K"] != GASES[gas]["epsilon"]
                    or gas_record["input_sha256"] != expected_inputs[name, gas]):
                raise ValueError(f"Vext gas FF/input mismatch: {name} {gas}")
        rows.append({"name": name, "source": source, "shard": shard,
                     "data_path": str(data), "record_path": str(path),
                     "data_sha256": record["sha256"],
                     "framework_atoms": record["framework_atoms"]})
        if (index + 1) % 500 == 0:
            print(json.dumps({"verified": index + 1}), flush=True)
    OUTPUT.mkdir(parents=True)
    temporary = OUTPUT / "index.csv.tmp"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, OUTPUT / "index.csv")
    temporary.unlink()
    report = {"passed": True, "materials": len(rows), "gas_fields": 2 * len(rows),
              "source_manifest_sha256": binding,
              "old_materials": sum(row["source"] == "old" for row in rows),
              "accelerated_materials": sum(row["source"] == "exact_v2" for row in rows),
              "index_sha256": digest(OUTPUT / "index.csv"),
              "shard_reports": [str(NEW / f"shard{shard}" / "report.json")
                                for shard in (0, 1)]}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
