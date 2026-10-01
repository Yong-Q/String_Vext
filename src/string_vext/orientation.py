"""Uniform cell-center grid and deterministic spherical directions."""
from __future__ import annotations

import numpy as np


def regular_pose_grid(size: int, orientations: int):
    if size < 2 or orientations < 4:
        raise ValueError("at least two grid cells and four orientations required")
    axis = (np.arange(size, dtype=float) + .5) / size
    centers = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    z = 1 - 2 * (np.arange(orientations) + .5) / orientations
    phi = np.arange(orientations) * np.pi * (3 - np.sqrt(5))
    directions = np.stack([np.sqrt(1 - z*z) * np.cos(phi),
                           np.sqrt(1 - z*z) * np.sin(phi), z], axis=1)
    return centers, directions
