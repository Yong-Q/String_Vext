"""Run the triclinic CUDA String solver on one native input.dat."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .input import parse_string_input


DEFAULT_SOLVER = Path(__file__).resolve().parent / "bin/string_triclinic_cuda"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="native String input.dat")
    parser.add_argument("--output-dir", type=Path, required=True, help="new run directory")
    parser.add_argument("--solver", type=Path, default=DEFAULT_SOLVER)
    parser.add_argument("--initial", type=Path, help="optional 7-column String warm start")
    parser.add_argument("--diffusivity-bin", type=Path, help="optional native diffusivity utility")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if args.output_dir.exists() or args.timeout < 1:
        raise ValueError("new output directory and positive timeout required")
    case = parse_string_input(args.input)
    args.output_dir.mkdir(parents=True)
    shutil.copyfile(args.input, args.output_dir / "input.dat")
    command = [str(args.solver.resolve()), "input.dat"]
    if args.initial is not None:
        shutil.copyfile(args.initial, args.output_dir / "initial.dat")
        command.append("initial.dat")
    command.append("string_path.dat")
    result = subprocess.run(command, cwd=args.output_dir, capture_output=True,
                            text=True, timeout=args.timeout)
    (args.output_dir / "string.stdout").write_text(result.stdout)
    (args.output_dir / "string.stderr").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"String solver exited {result.returncode}; logs retained")
    path = args.output_dir / "string_path.dat"
    values = np.loadtxt(path)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < 2 \
            or not np.isfinite(values).all():
        raise ValueError("String solver did not produce a finite 7-column path")
    if args.diffusivity_bin is not None:
        diffusion = subprocess.run([str(args.diffusivity_bin.resolve()),
                                    "input.dat", "string_path.dat"],
                                   cwd=args.output_dir, capture_output=True,
                                   text=True, timeout=60)
        (args.output_dir / "diffusivity.stdout").write_text(diffusion.stdout)
        (args.output_dir / "diffusivity.stderr").write_text(diffusion.stderr)
        if diffusion.returncode:
            raise RuntimeError("diffusivity utility failed; path and logs retained")
    record = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "solver_sha256": hashlib.sha256(args.solver.read_bytes()).hexdigest(),
              "path_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "path_points": len(values), "cell_lengths_A": case["cell_lengths"],
              "cell_angles_deg": case["cell_angles_deg"],
              "guest_sigma_A": case["guest_sigma"].tolist(),
              "guest_epsilon_K": case["guest_epsilon"].tolist(),
              "temperature_K": case["temperature"],
              "diffusivity_utility_used": args.diffusivity_bin is not None}
    (args.output_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
