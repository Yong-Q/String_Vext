"""Exact triclinic 64-orientation Vext field generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .input import parse_string_input
from .native import NativeReusePotential
from .orientation import regular_pose_grid
from .physics import TriclinicSitePotential, marginal_energy


DEFAULT_KERNEL = Path(__file__).resolve().parent / "bin/kernel.so"


def calculate_vext(case: dict, size: int = 60, orientations: int = 64,
                   kernel: Path = DEFAULT_KERNEL) -> dict[str, np.ndarray]:
    centers, axes = regular_pose_grid(size, orientations)
    potential = NativeReusePotential(case, kernel)
    energy = potential.evaluate(centers, axes)
    marginal = marginal_energy(energy, case["temperature"])
    site_potential = TriclinicSitePotential(case)
    site = site_potential.evaluate(centers @ case["cell"],
                                   case["guest_sigma"][0], case["guest_epsilon"][0])
    result = {
        "orientation_K": energy.T.reshape(orientations, size, size, size).astype(np.float32),
        "marginal_K": marginal.reshape(size, size, size).astype(np.float32),
        "site_K": np.stack((site, site)).reshape(2, size, size, size).astype(np.float32),
        "axes": axes.astype(np.float32),
        "cell_matrix": np.asarray(case["cell"], dtype=np.float64),
        "atom_fractional": np.asarray(case["frame_frac"], dtype=np.float64),
        "framework_sigma_A": np.asarray(case["frame_sigma"], dtype=np.float64),
        "framework_epsilon_K": np.asarray(case["frame_epsilon"], dtype=np.float64),
    }
    result["grid"] = np.arcsinh(result["marginal_K"] / 1000).astype(np.float32)
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("nonfinite Vext output")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="native String input.dat")
    parser.add_argument("--output", type=Path, required=True, help="new NPZ destination")
    parser.add_argument("--size", type=int, default=60, help="grid points per axis")
    parser.add_argument("--orientations", type=int, default=64)
    parser.add_argument("--kernel", type=Path, default=DEFAULT_KERNEL)
    args = parser.parse_args()
    record_path = args.output.with_suffix(".json")
    if args.output.exists() or record_path.exists():
        raise FileExistsError("Vext output or provenance record already exists")
    case = parse_string_input(args.input)
    fields = calculate_vext(case, args.size, args.orientations, args.kernel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".npz.tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **fields)
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    record = {"input": str(args.input.resolve()),
              "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "kernel_sha256": hashlib.sha256(args.kernel.read_bytes()).hexdigest(),
              "size": args.size, "orientations": args.orientations,
              "energy_mode": "exact_triclinic_LJ_neighbor_reuse",
              "cell_lengths_A": case["cell_lengths"],
              "cell_angles_deg": case["cell_angles_deg"],
              "guest_sigma_A": case["guest_sigma"].tolist(),
              "guest_epsilon_K": case["guest_epsilon"].tolist()}
    with record_path.open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
