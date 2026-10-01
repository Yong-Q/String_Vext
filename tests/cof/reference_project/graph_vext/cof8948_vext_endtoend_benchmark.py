"""End-to-end two-material DIRECT versus exact-native Vext timing."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from graph_vext.cof8948_oldff_prepare import GASES, ROOT
from graph_vext.cof8948_oldff_vext import (
    PREPARED, _group, build_one, code_hashes as direct_hashes, load_plan,
)
from graph_vext.cof8948_oldff_vext_exact import (
    build_one_exact, code_hashes as exact_hashes,
)


NAMES = ("bne+C56+C32+L14_COOH", "bne+C62+C32+L14_COOH")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tasks, binding = load_plan(PREPARED, expected_materials=8948)
    grouped = _group(tasks)
    args.output.mkdir(parents=True)
    cases = []
    for index, name in enumerate(NAMES):
        order = ("direct", "exact") if index == 0 else ("exact", "direct")
        seconds = {}
        paths = {}
        for method in order:
            destination = args.output / method / name
            started = time.perf_counter()
            if method == "direct":
                record = build_one(name, grouped[name], PREPARED, destination,
                                   60, 64, direct_hashes(), binding, False)
            else:
                record = build_one_exact(name, grouped[name], PREPARED, destination,
                                         60, 64, exact_hashes(), binding, False)
            seconds[method] = time.perf_counter() - started
            paths[method] = destination / "data" / f"{name}.npz"
            print(json.dumps({"name": name, "method": method,
                              "seconds": seconds[method], "sha256": record["sha256"]}), flush=True)
        with np.load(paths["direct"], allow_pickle=False) as old, \
                np.load(paths["exact"], allow_pickle=False) as new:
            if set(old.files) != set(new.files):
                raise ValueError(f"Vext archive channel mismatch: {name}")
            comparisons = {}
            for gas in GASES:
                direct = old[f"orientation_K_{gas}"].astype(float)
                candidate = new[f"orientation_K_{gas}"].astype(float)
                accessible = (direct < 2000) & (np.abs(direct) < 10000)
                delta = np.abs(candidate[accessible] - direct[accessible])
                marginal = np.abs(new[f"marginal_K_{gas}"].astype(float)
                                  - old[f"marginal_K_{gas}"].astype(float))
                site = np.abs(new[f"site_K_{gas}"].astype(float)
                              - old[f"site_K_{gas}"].astype(float))
                comparisons[gas] = {
                    "accessible_poses": int(accessible.sum()),
                    "orientation_max_abs_error_K": float(delta.max()),
                    "marginal_max_abs_error_K": float(marginal.max()),
                    "site_max_abs_error_K": float(site.max())}
                if max(comparisons[gas][key] for key in
                       ("orientation_max_abs_error_K", "marginal_max_abs_error_K",
                        "site_max_abs_error_K")) > 1e-3:
                    raise ValueError(f"end-to-end Vext accuracy failed: {name} {gas}")
        cases.append({"name": name, "order": order, "seconds": seconds,
                      "speedup": seconds["direct"] / seconds["exact"],
                      "comparisons": comparisons})
    report = {"passed": True, "materials": len(cases), "cases": cases,
              "source_manifest_sha256": binding, "direct_code_hashes": direct_hashes(),
              "exact_code_hashes": exact_hashes(),
              "note": "single-process full input-to-NPZ timing, alternating order; no multi-worker filesystem contention"}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "speedups": [row["speedup"] for row in cases]}), flush=True)


if __name__ == "__main__":
    main()
