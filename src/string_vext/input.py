"""Parse neutral two-site guests from the native String input format."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .cell import cell_matrix


def parse_string_input(path: str | Path) -> dict:
    lines = [line.strip() for line in Path(path).read_text().splitlines()]

    def after(prefix):
        return next(index + 1 for index, line in enumerate(lines)
                    if line.startswith(prefix))

    lengths = [float(value) for value in lines[after("La Lb Lc")].split()[:3]]
    angles = [float(value) for value in lines[after("Alpha Beta Gamma")].split()[:3]]
    cell = cell_matrix(lengths, angles)
    conditions = [float(value) for value in lines[after("cutoff(A)")].split()]
    guest_count_index = after("Number of sites")
    guest_count = int(lines[guest_count_index])
    guest = np.array([[float(value) for value in row.split()[:7]]
                      for row in lines[guest_count_index + 2:guest_count_index + 2 + guest_count]])
    frame_count_index = after("Number of atoms")
    frame_count = int(lines[frame_count_index])
    frame = np.array([[float(value) for value in row.split()[:8]]
                      for row in lines[frame_count_index + 2:frame_count_index + 2 + frame_count]])
    if (guest.shape != (2, 7) or frame.shape != (frame_count, 8)
            or len(conditions) < 4 or not np.isfinite(guest).all()
            or not np.isfinite(frame).all() or not np.isfinite(conditions).all()
            or not np.array_equal(frame[:, 0], np.arange(1, frame_count + 1))
            or (guest[:, 5] != 0).any() or (frame[:, 3] != 0).any()
            or (guest[:, 3:5] <= 0).any() or (guest[:, 6] <= 0).any()
            or (frame[:, 1:3] <= 0).any() or conditions[0] <= 0
            or conditions[3] <= 0):
        raise ValueError("invalid or charged two-site String input")
    mass = guest[:, 6]
    if not np.isclose(mass.sum(), conditions[2], rtol=1e-7):
        raise ValueError("guest site masses differ from declared total mass")
    body = guest[:, :3]
    separation = body[1] - body[0]
    bond = np.linalg.norm(separation)
    if bond <= 0:
        raise ValueError("two distinct guest sites required")
    director = separation / bond
    center = (body * mass[:, None]).sum(0) / conditions[2]
    canonical = np.zeros((2, 3))
    canonical[:, 0] = (body - center) @ director
    return {"cell": cell, "frame_frac": np.mod(frame[:, 5:8], 1),
            "frame_sigma": frame[:, 1], "frame_epsilon": frame[:, 2],
            "guest_xyz": canonical, "guest_sigma": guest[:, 4],
            "guest_epsilon": guest[:, 3], "guest_mass": mass,
            "total_mass": float(conditions[2]), "temperature": float(conditions[3]),
            "cutoff": float(conditions[0]), "cell_lengths": lengths,
            "cell_angles_deg": angles, "bond_length": float(bond)}
