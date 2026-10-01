"""String LJ parameters with full triclinic image sums and COM pose fields.

No Coulomb term. Retain the source's0.1 mixed-sigma close-contact floor and
shifted cutoff, NOT its Cartesian-component image heuristic. No2000K or
60000K energy cap. Site-grid interpolation remains a documented approximation.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logsumexp

from graph_vext.orientation_oracle_fields import regular_pose_grid


def validate_case(case):
    cell = np.asarray(case['cell'], dtype=float)
    frame = np.asarray(case['frame_frac'], dtype=float)
    if (cell.shape != (3, 3) or not np.isfinite(cell).all() or np.linalg.det(cell) <= 0
            or frame.ndim != 2 or frame.shape[1] != 3 or not len(frame)
            or not np.isfinite(frame).all()):
        raise ValueError('invalid triclinic framework')
    for key in ('frame_sigma', 'frame_epsilon'):
        values = np.asarray(case[key], dtype=float)
        if values.shape != (len(frame),) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('invalid framework LJ parameters')
    for key in ('guest_sigma', 'guest_epsilon', 'guest_mass'):
        values = np.asarray(case[key], dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('invalid two-site guest')
    body = np.asarray(case['guest_xyz'], dtype=float)
    if body.shape != (2, 3) or not np.isfinite(body).all() or not np.allclose(body[:, 1:], 0):
        raise ValueError('guest must have a two-site body x-axis')
    if not np.isclose(sum(case['guest_mass']), case['total_mass'], rtol=1e-6):
        raise ValueError('guest mass mismatch')
    if not np.isfinite([case['cutoff'], case['temperature']]).all() or min(case['cutoff'], case['temperature']) <= 0:
        raise ValueError('invalid cutoff/temperature')


class TriclinicSitePotential:
    def __init__(self, case):
        validate_case(case)
        self.case = case
        self.cell = np.asarray(case['cell'], dtype=np.float64)
        self.inverse = np.linalg.inv(self.cell)
        self.cutoff = float(case['cutoff'])
        spacing = 1/np.linalg.norm(self.inverse, axis=0)
        # Wrapped fractional query/atom components differ by less than one.
        reach = np.ceil(self.cutoff/spacing).astype(int)+1
        shifts = np.array(list(itertools.product(*[range(-r, r+1) for r in reach])), dtype=float)
        atoms = np.mod(case['frame_frac'], 1)
        self.images = (atoms[None]+shifts[:, None]).reshape(-1, 3) @ self.cell
        self.sigma = np.tile(case['frame_sigma'], len(shifts))
        self.epsilon = np.tile(case['frame_epsilon'], len(shifts))
        self.tree = cKDTree(self.images)

    def evaluate(self, coordinates, guest_sigma, guest_epsilon, chunk=1024):
        query = np.asarray(coordinates, dtype=float)
        if query.ndim != 2 or query.shape[1] != 3 or not np.isfinite(query).all():
            raise ValueError('invalid Cartesian site coordinates')
        query = (np.mod(query @ self.inverse, 1)) @ self.cell
        mixed_sigma = (self.sigma+guest_sigma)/2
        mixed_epsilon = np.sqrt(self.epsilon*guest_epsilon)
        ratio_cut = (mixed_sigma/self.cutoff)**6
        shifted = -4*mixed_epsilon*(ratio_cut**2-ratio_cut)
        output = np.zeros(len(query), dtype=float)
        for start in range(0, len(query), chunk):
            positions = query[start:start+chunk]
            neighbors = self.tree.query_ball_point(positions, self.cutoff, workers=1)
            counts = np.fromiter((len(n) for n in neighbors), dtype=np.int64, count=len(positions))
            if not counts.any():
                continue
            indices = np.concatenate([np.asarray(n, dtype=np.int64) for n in neighbors])
            row = np.repeat(np.arange(len(positions)), counts)
            distances = np.linalg.norm(positions[row]-self.images[indices], axis=1)
            sigma = mixed_sigma[indices]
            ratio = (sigma/np.maximum(distances, .1*sigma))**6
            energy = 4*mixed_epsilon[indices]*(ratio**2-ratio)+shifted[indices]
            energy[distances >= self.cutoff] = 0
            output[start:start+len(positions)] = np.bincount(row, weights=energy, minlength=len(positions))
        if not np.isfinite(output).all():
            raise ValueError('nonfinite String-FF site energy')
        return output


def body_offsets(case):
    center = (case['guest_xyz']*case['guest_mass'][:, None]).sum(0)/case['total_mass']
    return case['guest_xyz'][:, 0]-center[0]


def direct_pose_energy(case, centers, axes, potential=None):
    potential = TriclinicSitePotential(case) if potential is None else potential
    centers, axes = np.asarray(centers, dtype=float), np.asarray(axes, dtype=float)
    if (centers.ndim != 2 or centers.shape[1] != 3 or axes.ndim != 2 or axes.shape[1] != 3
            or not np.isfinite(centers).all() or not np.isfinite(axes).all()
            or (np.linalg.norm(axes, axis=1) <= 0).any()):
        raise ValueError('invalid center/director queries')
    axes = axes/np.linalg.norm(axes, axis=1, keepdims=True)
    energy = np.zeros((len(centers), len(axes)))
    for site, offset in enumerate(body_offsets(case)):
        position = centers[:, None] @ case['cell']+offset*axes[None]
        value = potential.evaluate(position.reshape(-1, 3), case['guest_sigma'][site], case['guest_epsilon'][site])
        energy += value.reshape(energy.shape)
    return energy


def periodic_interpolate(volume, coordinates):
    volume, coordinates = np.asarray(volume), np.asarray(coordinates, dtype=float)
    size = np.array(volume.shape)
    if volume.ndim != 3 or coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError('three-dimensional periodic field required')
    position = np.mod(coordinates, 1)*size-.5
    lower = np.floor(position).astype(np.int64)
    fraction = position-lower
    result = np.zeros(len(coordinates), dtype=float)
    for offset in itertools.product((0, 1), repeat=3):
        offset = np.array(offset)
        indices = (lower+offset) % size
        weight = np.where(offset, fraction, 1-fraction).prod(1)
        result += volume[tuple(indices.T)]*weight
    return result


def marginal_energy(energies, temperature):
    energy = np.asarray(energies, dtype=float)
    if not np.isfinite(energy).all() or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('finite angular energies and positive temperature required')
    return -temperature*(logsumexp(-energy/temperature, axis=-1)-np.log(energy.shape[-1]))


def uniform_fields(case, size=60, orientations=64, energy_mode='direct'):
    if energy_mode not in ('direct', 'interpolate'):
        raise ValueError('unknown site-energy evaluation mode')
    potential = TriclinicSitePotential(case)
    centers, axes = regular_pose_grid(size, orientations)
    fields = {}
    site_fields = []
    for site in range(2):
        key = (float(case['guest_sigma'][site]), float(case['guest_epsilon'][site]))
        if key not in fields:
            fields[key] = potential.evaluate(centers @ case['cell'], *key).reshape(size, size, size)
        site_fields.append(fields[key])
    offsets = body_offsets(case)
    inverse = np.linalg.inv(case['cell'])
    channels = np.empty((orientations, size, size, size), dtype=np.float32)
    flat = channels.reshape(orientations, -1)
    marginal = np.empty(len(centers), dtype=np.float32)
    for start in range(0, len(centers), 2048):
        points = centers[start:start+2048]
        if energy_mode == 'direct':
            energy = direct_pose_energy(case, points, axes, potential=potential)
        else:
            energy = np.zeros((len(points), orientations))
            for site, offset in enumerate(offsets):
                fractional = points[:, None]+(offset*axes) @ inverse
                energy += periodic_interpolate(site_fields[site], fractional.reshape(-1, 3)).reshape(energy.shape)
        flat[:, start:start+len(points)] = energy.T
        marginal[start:start+len(points)] = marginal_energy(energy, case['temperature'])
    if not np.isfinite(channels).all() or not np.isfinite(marginal).all():
        raise ValueError('nonfinite uniform String-FF output')
    return dict(orientation_K=channels, marginal_K=marginal.reshape(size, size, size),
                site_K=np.stack(site_fields).astype(np.float32), axes=axes.astype(np.float32))
