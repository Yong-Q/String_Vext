"""Full triclinic fractional-to-Cartesian cell construction."""
from __future__ import annotations

import numpy as np


def cell_matrix(lengths, angles_deg):
    lengths = np.asarray(lengths, dtype=float)
    angles = np.asarray(angles_deg, dtype=float)
    if (lengths.shape != (3,) or angles.shape != (3,)
            or not np.isfinite(lengths).all() or not np.isfinite(angles).all()
            or (lengths <= 0).any() or (angles <= 0).any() or (angles >= 180).any()):
        raise ValueError("three positive lengths and three angles in (0, 180) required")
    alpha, beta, gamma = np.deg2rad(angles)
    a, b, c = lengths
    sin_gamma = np.sin(gamma)
    if abs(sin_gamma) < 1e-10:
        raise ValueError("singular triclinic cell")
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sin_gamma
    cz2 = c * c - cx * cx - cy * cy
    if cz2 <= 0:
        raise ValueError("nonphysical triclinic cell")
    matrix = np.array([[a, 0., 0.],
                       [b * np.cos(gamma), b * sin_gamma, 0.],
                       [cx, cy, np.sqrt(cz2)]])
    if not np.isfinite(matrix).all() or np.linalg.det(matrix) <= 0:
        raise ValueError("invalid cell determinant")
    return matrix
