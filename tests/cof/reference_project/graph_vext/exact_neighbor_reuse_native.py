"""Compiled CPU evaluation with the same triclinic neighbor/FF contract."""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from graph_vext.exact_neighbor_reuse_probe import ReusePotential


class NativeReusePotential(ReusePotential):
    def __init__(self, case: dict, library: Path, batch_size: int = 512):
        super().__init__(case)
        if batch_size < 1:
            raise ValueError("positive center batch required")
        self.batch_size = batch_size
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.kernel = self.library.evaluate_reused_neighbors
        self.kernel.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_double] + [ctypes.c_void_p] * 11
        self.kernel.restype = ctypes.c_int
        self.query_seconds = 0.
        self.compute_seconds = 0.

    def evaluate(self, centers: np.ndarray, axes: np.ndarray) -> np.ndarray:
        import time
        centers, axes = np.asarray(centers, dtype=float), np.asarray(axes, dtype=float)
        if (centers.ndim != 2 or centers.shape[1] != 3
                or axes.ndim != 2 or axes.shape[1] != 3
                or not np.isfinite(centers).all() or not np.isfinite(axes).all()
                or (np.linalg.norm(axes, axis=1) <= 0).any()):
            raise ValueError("finite fractional centers and nonzero directors required")
        axes = np.ascontiguousarray(axes / np.linalg.norm(axes, axis=1, keepdims=True))
        cart = np.ascontiguousarray(np.mod(centers, 1) @ self.potential.cell)
        images = np.ascontiguousarray(self.potential.images)
        offsets = np.ascontiguousarray(self.offsets)
        coeff12 = np.ascontiguousarray(self.coeff12)
        coeff6 = np.ascontiguousarray(self.coeff6)
        shift = np.ascontiguousarray(self.shift)
        floor2 = np.ascontiguousarray(self.floor2)
        output = np.empty((len(centers), len(axes)), dtype=np.float64)
        buffers = (images, offsets, coeff12, coeff6, shift, floor2)
        for start in range(0, len(cart), self.batch_size):
            end = min(start + self.batch_size, len(cart))
            stamp = time.perf_counter()
            groups = self.potential.tree.query_ball_point(cart[start:end], self.radius,
                                                           workers=1)
            lengths = np.fromiter((len(group) for group in groups), dtype=np.int64,
                                  count=len(groups))
            starts = np.empty(len(groups) + 1, dtype=np.int64)
            starts[0] = 0
            starts[1:] = np.cumsum(lengths)
            indices = np.concatenate([np.asarray(group, dtype=np.int32)
                                      for group in groups]) if starts[-1] else np.empty(0, np.int32)
            self.neighbor_queries += len(groups)
            self.max_neighbors = max(self.max_neighbors, int(lengths.max(initial=0)))
            self.query_seconds += time.perf_counter() - stamp
            stamp = time.perf_counter()
            arrays = (cart[start:end], axes, *buffers, starts, indices, output[start:end])
            result = self.kernel(len(groups), len(axes), len(images), self.cutoff**2,
                                 *(array.ctypes.data_as(ctypes.c_void_p) for array in arrays))
            if result:
                raise RuntimeError(f"native neighbor kernel failed with code {result}")
            self.compute_seconds += time.perf_counter() - stamp
        if not np.isfinite(output).all():
            raise ValueError("nonfinite native Vext energy")
        return output
