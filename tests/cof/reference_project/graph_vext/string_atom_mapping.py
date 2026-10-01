"""Diagnostic back-mapping of saved String poses to per-framework-atom LJ terms.

Native mode mirrors the archived solver's extended-cell image selection, not
an assertion that its image selection is the exact triclinic minimum image.
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from graph_vext.dataset import GASES, load_manifest
from graph_vext.full_string_profile import select_corrected_cases


def parse_string_input(text: str) -> dict:
    lines = [line.strip() for line in text.splitlines()]
    def after(prefix):
        return next(i for i, line in enumerate(lines) if line.startswith(prefix)) + 1
    lengths = np.array([float(x) for x in lines[after('La Lb Lc')].split()[:3]])
    alpha, beta, gamma = np.radians([float(x) for x in lines[after('Alpha Beta Gamma')].split()])
    ca, cb, cg, sg = np.cos(alpha), np.cos(beta), np.cos(gamma), np.sin(gamma)
    volume_factor = np.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)
    a, b, c = lengths
    cell = np.array([[a, 0., 0.], [b * cg, b * sg, 0.],
                     [c * cb, c * (ca - cb * cg) / sg, c * volume_factor / sg]])
    conditions = lines[after('cutoff(A)')].split()
    si, fi = after('Number of sites'), after('Number of atoms')
    ns, nf = int(lines[si]), int(lines[fi])
    guest = np.array([[float(x) for x in line.split()[:7]] for line in lines[si + 2:si + 2 + ns]])
    frame_lines = [line.split() for line in lines[fi + 2:fi + 2 + nf]]
    frame = np.array([[float(x) for x in line[:8]] for line in frame_lines])
    if guest.shape != (ns, 7) or frame.shape != (nf, 8) or not np.isfinite(guest).all() or not np.isfinite(frame).all():
        raise ValueError('invalid String atom table')
    if (guest[:, 5] != 0).any() or (frame[:, 3] != 0).any():
        raise ValueError('charged case requires a separately verified electrostatic mapper')
    if (guest[:, 3:5] <= 0).any() or (frame[:, 1:3] <= 0).any():
        raise ValueError('nonpositive LJ parameters')
    return dict(cell=cell, cutoff=float(conditions[0]), fh_signal=int(conditions[1]),
                total_mass=float(conditions[2]), temperature=float(conditions[3]),
                guest_xyz=guest[:, :3], guest_epsilon=guest[:, 3], guest_sigma=guest[:, 4], guest_mass=guest[:, 6],
                frame_frac=frame[:, 5:8], frame_sigma=frame[:, 1], frame_epsilon=frame[:, 2],
                elements=[line[8] for line in frame_lines], atom_ids=frame[:, 0].astype(int))


def rotate_sites(angles: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """Source device_fxn.h convention: Rz(gamma) Ry(beta) Rx(alpha), radians."""
    alpha, beta, gamma = np.asarray(angles, dtype=float).T
    ca, sa, cb, sb, cg, sg = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta), np.cos(gamma), np.sin(gamma)
    matrix = np.stack([cg * cb, -sg * ca + sa * cg * sb, sg * sa + ca * cg * sb,
                       sg * cb, cg * ca + sg * sb * sa, -cg * sa + sg * sb * ca,
                       -sb, cb * sa, ca * cb], axis=-1).reshape(-1, 3, 3)
    return np.einsum('pij,sj->psi', matrix, sites)


def map_path(case: dict, path: np.ndarray) -> dict:
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 7 or not len(path) or not np.isfinite(path).all():
        raise ValueError('invalid raw String path')
    cell = case['cell']
    cutoff = case['cutoff']
    n_atoms = len(case['frame_frac'])
    # Reproduce pbc_expand and independent Cartesian-component image tests.
    times = (2 * cutoff / np.diag(cell)).astype(int) + 1
    shifts = np.array(list(itertools.product(*[range(n) for n in times])), dtype=float)
    frame = (case['frame_frac'][None] + shifts[:, None]).reshape(-1, 3)
    identities = np.tile(np.arange(n_atoms), len(shifts))
    frame_cart = frame @ cell
    extended_diagonal = np.diag(cell) * times
    center_of_mass = (case['guest_xyz'] * case['guest_mass'][:, None]).sum(0) / case['total_mass']
    relative_sites = case['guest_xyz'] - center_of_mass
    positions = path[:, None, :3] @ cell + rotate_sites(path[:, 3:6], relative_sites)
    sigma = (case['guest_sigma'][:, None] + case['frame_sigma'][identities][None]) / 2
    epsilon = np.sqrt(case['guest_epsilon'][:, None] * case['frame_epsilon'][identities][None])
    ratio_cut = (sigma / cutoff) ** 6
    shifted_constant = -4 * epsilon * (ratio_cut**2 - ratio_cut)
    result = {'atom_energy_K': np.zeros((len(path), n_atoms)),
              **{k: np.zeros(len(path)) for k in ('energy_K', 'repulsive_K', 'attractive_K', 'shift_K',
                                                  'minimum_distance_A', 'minimum_ratio', 'neighbor_count_cutoff')}}
    for start in range(0, len(path), 16):
        poses = positions[start:start + 16]
        raw_delta = poses[:, :, None] - frame_cart[None, None]
        modification = np.where(raw_delta > extended_diagonal / 2, times,
                                np.where(raw_delta < -extended_diagonal / 2, -times, 0))
        image_cart = (frame[None, None] + modification) @ cell
        distance = np.linalg.norm(poses[:, :, None] - image_cart, axis=-1)
        clipped = np.maximum(distance, sigma[None] * .1)
        mask = clipped < cutoff
        ratio6 = (sigma[None] / clipped) ** 6
        rep = np.where(mask, 4 * epsilon[None] * ratio6**2, 0.)
        attr = np.where(mask, -4 * epsilon[None] * ratio6, 0.)
        shift = np.where(mask, shifted_constant[None], 0.)
        terms = rep + attr + shift
        end = start + len(poses)
        result['atom_energy_K'][start:end] = terms.sum(1).reshape(len(poses), len(shifts), n_atoms).sum(1)
        for name, value in [('repulsive_K', rep), ('attractive_K', attr), ('shift_K', shift)]:
            result[name][start:end] = value.sum((1, 2))
        result['energy_K'][start:end] = result['atom_energy_K'][start:end].sum(1)
        result['minimum_distance_A'][start:end] = distance.min((1, 2))
        result['minimum_ratio'][start:end] = (distance / sigma[None]).min((1, 2))
        result['neighbor_count_cutoff'][start:end] = mask.sum((1, 2))
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError('nonfinite reconstructed energies')
    return result


def summarize_mapping(case: dict, path: np.ndarray, mapped: dict) -> dict:
    low, high = int(np.argmin(path[:, 6])), int(np.argmax(path[:, 6]))
    error = mapped['energy_K'] - path[:, 6]
    delta = mapped['atom_energy_K'][high] - mapped['atom_energy_K'][low]
    positives = np.maximum(delta, 0)
    order = np.argsort(np.abs(delta))[::-1][:10]
    top = [{'atom_index': int(i), 'element': case['elements'][i], 'delta_energy_K': float(delta[i])}
           for i in order]
    normalized_error = np.abs(error) / (.5 + 1e-4 * np.abs(path[:, 6]))
    return {'basin_index': low, 'bottleneck_index': high,
            'energy_reproduction_mae_K': float(np.mean(np.abs(error))),
            'energy_reproduction_max_abs_K': float(np.max(np.abs(error))),
            'energy_reproduction_normalized_max': float(normalized_error.max()),
            'energy_reproduction_passed': bool(normalized_error.max() <= 1),
            'reconstructed_barrier_K': float(mapped['energy_K'][high] - mapped['energy_K'][low]),
            'barrier_relative_path_min_K': float(path[high, 6] - path[low, 6]),
            'barrier_repulsive_delta_K': float(mapped['repulsive_K'][high] - mapped['repulsive_K'][low]),
            'barrier_attractive_delta_K': float(mapped['attractive_K'][high] - mapped['attractive_K'][low]),
            'barrier_shift_delta_K': float(mapped['shift_K'][high] - mapped['shift_K'][low]),
            'basin_minimum_distance_A': float(mapped['minimum_distance_A'][low]),
            'bottleneck_minimum_distance_A': float(mapped['minimum_distance_A'][high]),
            'bottleneck_minimum_ratio': float(mapped['minimum_ratio'][high]),
            'positive_barrier_top5_fraction': float(np.sort(positives)[-5:].sum() / max(positives.sum(), 1e-30)),
            'element_barrier_contributions_K': {e: float(delta[np.array(case['elements']) == e].sum()) for e in set(case['elements'])},
            'top_atoms': top}


def map_case(row: dict, root: Path, output_dir: Path, proxy: dict) -> dict:
    source = Path(row['path']).resolve()
    expected = root / f"string_runs_{row['gas']}"
    source.relative_to(expected.resolve())
    if source.parent.name != row['structure'] or source.name != f"dir{row['direction']}":
        raise ValueError('String case provenance mismatch')
    text = (source / 'input.dat').read_text()
    case = parse_string_input(text)
    path = np.loadtxt(source / 'string_path.dat')
    mapped = map_path(case, path)
    summary = summarize_mapping(case, path, mapped)
    alt = copy.deepcopy(case)
    alt['guest_sigma'][:] = proxy['site_sigma_A']
    alt['guest_epsilon'][:] = proxy['site_epsilon_K']
    separation = np.linalg.norm(alt['guest_xyz'][1] - alt['guest_xyz'][0])
    if len(alt['guest_xyz']) != 2 or separation <= 0:
        raise ValueError('proxy comparison requires two sites')
    com = (alt['guest_xyz'] * alt['guest_mass'][:, None]).sum(0) / alt['total_mass']
    alt['guest_xyz'] = com + (alt['guest_xyz'] - com) * proxy['bond_length_A'] / separation
    alternative = map_path(alt, path)
    low, high = summary['basin_index'], summary['bottleneck_index']
    summary.update(structure=row['structure'], gas=row['gas'], direction=int(row['direction']),
                   source_dir=str(source), logd=float(np.log10(float(row['new_value']))),
                   points=len(path), temperature_K=case['temperature'], cutoff_A=case['cutoff'],
                   fh_signal=case['fh_signal'], original_guest_sigma_A=case['guest_sigma'].tolist(),
                   original_guest_epsilon_K=case['guest_epsilon'].tolist(),
                   original_bond_length_A=float(separation),
                   proxy_energy_mae_vs_original_K=float(np.mean(np.abs(alternative['energy_K'] - mapped['energy_K']))),
                   proxy_relative_barrier_same_extrema_K=float(alternative['energy_K'][high] - alternative['energy_K'][low]),
                   proxy_energy_range_on_same_path_K=float(np.ptp(alternative['energy_K'])),
                   guest_axis_at_bottleneck=(rotate_sites(path[high:high + 1, 3:6], case['guest_xyz'] - com)[0, 1]
                                             - rotate_sites(path[high:high + 1, 3:6], case['guest_xyz'] - com)[0, 0]).tolist())
    filename = f"{row['structure']}__{row['gas']}__dir{row['direction']}"
    atom_delta = mapped['atom_energy_K'][high] - mapped['atom_energy_K'][low]
    for x in summary['top_atoms']:
        x['input_atom_id'] = int(case['atom_ids'][x['atom_index']])
        x['fractional_position'] = case['frame_frac'][x['atom_index']].tolist()
        x['sigma_A'] = float(case['frame_sigma'][x['atom_index']])
        x['epsilon_K'] = float(case['frame_epsilon'][x['atom_index']])
    temporary = output_dir / (filename + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, path=path, atom_energy_K=mapped['atom_energy_K'].astype(np.float32),
                            energy_K=mapped['energy_K'], proxy_energy_K=alternative['energy_K'],
                            minimum_distance_A=mapped['minimum_distance_A'], minimum_ratio=mapped['minimum_ratio'],
                            atom_delta_K=atom_delta, atom_frac=case['frame_frac'],
                            atom_sigma_A=case['frame_sigma'], atom_epsilon_K=case['frame_epsilon'],
                            elements=np.array(case['elements']), **{k: mapped[k] for k in ('repulsive_K', 'attractive_K', 'shift_K')})
    os.replace(temporary, output_dir / (filename + '.npz'))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source-root', type=Path, default=Path('/home/qiuyong/gcmc_agent/pormake_output/active_learning_cof30k_diffusion_test'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--structures', type=int, default=100)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest = load_manifest(args.project / 'runs/v2/manifest.csv')
    training = sorted([r for r in manifest if r['random_split'] == 'train'],
                      key=lambda r: (float(r['logd_c2h4']) + float(r['logd_c2h6'])) / 2)
    if not 0 < args.structures <= len(training):
        raise ValueError('invalid requested training sample size')
    rng = np.random.default_rng(42)
    chunks = np.array_split(np.arange(len(training)), min(10, args.structures))
    quota = [args.structures // len(chunks) + (i < args.structures % len(chunks)) for i in range(len(chunks))]
    selected_names = {training[int(j)]['name'] for indices, n in zip(chunks, quota)
                      for j in rng.choice(indices, int(n), replace=False)}
    selected_manifest = {r['name']: r for r in manifest if r['name'] in selected_names}
    with (args.source_root / 'diffusivity_corrected_rerun_values.csv').open() as handle:
        cases = select_corrected_cases(list(csv.DictReader(handle)), selected_manifest)
    proxy = json.loads((args.project / 'graph_vext/gas_forcefield.json').read_text())
    args.output_dir.mkdir(parents=True)
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(map_case, row, args.source_root, args.output_dir, proxy[row['gas']]): row
                   for row in cases.values()}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                row = futures[future]
                failures.append({'structure': row['structure'], 'gas': row['gas'], 'direction': row['direction'], 'error': str(error)})
            if (len(results) + len(failures)) % 50 == 0:
                print(json.dumps({'mapped_paths': len(results), 'failures': len(failures)}), flush=True)
    report = {'structures': len(selected_names), 'expected_paths': len(cases), 'mapped_paths': len(results),
              'failures': failures, 'energy_reproduction_failed_paths': sum(not r['energy_reproduction_passed'] for r in results),
              'sampling': 'random-split training structures, ten strata of mean gas logD, seed42',
              'caveats': ['Mapping uses measured training paths; this is not CIF-only test prediction.',
                          'LJ contributions mirror source image-selection conventions, not guaranteed exact triclinic images.',
                          'Energy tolerance is 0.5 K plus 1e-4 absolute saved energy, allowing six-decimal pose rounding.',
                          'Proxy comparison changes guest LJ/separation on fixed centers/orientations; retains String cutoff and geometry.',
                          'Relative barrier uses global path minimum, not independently identified adjacent-basin transition states.',
                          'Contributions explain the implemented potential, not independently proven dynamical causality.'],
              'gases': {}}
    for gas in GASES:
        rows = [r for r in results if r['gas'] == gas and r['energy_reproduction_passed']]
        fields = ('energy_reproduction_mae_K', 'barrier_relative_path_min_K', 'positive_barrier_top5_fraction',
                  'proxy_energy_mae_vs_original_K', 'bottleneck_minimum_ratio')
        stats = {k: {'median': float(np.median([r[k] for r in rows])),
                     'p90': float(np.percentile([r[k] for r in rows], 90))} for k in fields} if rows else {}
        stats['verified_paths'] = len(rows)
        stats['repulsion_delta_larger_than_attraction_delta_fraction'] = (float(np.mean([
            r['barrier_repulsive_delta_K'] > r['barrier_attractive_delta_K'] for r in rows])) if rows else None)
        stats['parameter_sets'] = sorted(set((tuple(r['original_guest_sigma_A']), tuple(r['original_guest_epsilon_K']),
                                              r['original_bond_length_A']) for r in rows))
        if len(rows) > 5:
            stats['descriptive_spearman_logD_vs_barrier'] = float(spearmanr([r['logd'] for r in rows], [r['barrier_relative_path_min_K'] for r in rows]).statistic)
        report['gases'][gas] = stats
    for filename, payload in [('cases.json', sorted(results, key=lambda r: (r['structure'], r['gas'], r['direction']))),
                              ('report.json', report)]:
        temporary = args.output_dir / (filename + '.tmp')
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, args.output_dir / filename)
    print(json.dumps(report), flush=True)
    if failures:
        raise RuntimeError('some mapping cases failed; failure evidence retained')


if __name__ == '__main__':
    main()
