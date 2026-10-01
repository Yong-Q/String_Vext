"""Center-based neighbor list for neutral symmetric two-site LJ guests."""
from __future__ import annotations

import numpy as np

from .physics import TriclinicSitePotential, body_offsets


class ReusePotential:
    def __init__(self, case: dict):
        self.case = case
        sigma = np.asarray(case["guest_sigma"], dtype=float)
        epsilon = np.asarray(case["guest_epsilon"], dtype=float)
        if (sigma.shape != (2,) or epsilon.shape != (2,)
                or sigma[0] != sigma[1] or epsilon[0] != epsilon[1]):
            raise ValueError("exact neighbor reuse requires identical two-site LJ parameters")
        self.offsets = body_offsets(case)
        self.cutoff = float(case["cutoff"])
        self.radius = self.cutoff + float(np.abs(self.offsets).max())
        self.potential = TriclinicSitePotential({**case, "cutoff": self.radius})
        mixed_sigma = (self.potential.sigma + sigma[0]) / 2
        mixed_epsilon = np.sqrt(self.potential.epsilon * epsilon[0])
        sigma6 = mixed_sigma**6
        self.coeff12 = 4 * mixed_epsilon * sigma6**2
        self.coeff6 = 4 * mixed_epsilon * sigma6
        cut6 = (mixed_sigma / self.cutoff)**6
        self.shift = -4 * mixed_epsilon * (cut6**2 - cut6)
        self.floor2 = (.1 * mixed_sigma)**2
        self.neighbor_queries = 0
        self.max_neighbors = 0
