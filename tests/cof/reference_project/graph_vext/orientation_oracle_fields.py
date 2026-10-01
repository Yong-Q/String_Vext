"""Uniform pose fields computed without reading optimized paths or D labels."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

from graph_vext.dataset import GASES, load_manifest
from graph_vext.string_atom_mapping import parse_string_input


def regular_pose_grid(size, orientations):
    axis = (np.arange(size, dtype=float) + .5) / size
    centers = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
    z = 1 - 2 * (np.arange(orientations) + .5) / orientations
    phi = np.arange(orientations) * np.pi * (3 - np.sqrt(5))
    directions = np.stack([np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z], axis=1)
    return centers, directions


def tensor_pose_energy(case, centers, axes, device, chunk_size=2048):
    """FP64 finite shifted LJ, matching archived String image-selection rules.

    The historical image rule is not claimed to be exact triclinic minimum
    image. No Coulomb term is added; parser rejects charged systems.
    """
    if len(case['guest_xyz']) != 2 or not np.allclose(case['guest_mass'], case['guest_mass'][0]):
        raise ValueError('requires a symmetric equal-mass two-site guest')
    if not np.allclose(case['guest_xyz'][:, 1:], 0) or not np.isfinite(case['cell']).all():
        raise ValueError('invalid cell or unsupported guest body axis')
    def tensor(value): return torch.as_tensor(value, dtype=torch.float64, device=device)
    cutoff, cell = case['cutoff'], tensor(case['cell'])
    times = (2 * cutoff / np.diag(case['cell'])).astype(int) + 1
    shifts = np.array(list(itertools.product(*[range(n) for n in times])))
    images = (case['frame_frac'][None] + shifts[:, None]).reshape(-1, 3)
    identities = np.tile(np.arange(len(case['frame_frac'])), len(shifts))
    frame = tensor(images)
    cart = frame @ cell
    diagonal = tensor(np.diag(case['cell']) * times)
    repeats = tensor(times)
    sigma = tensor((case['guest_sigma'][:, None] + case['frame_sigma'][identities][None]) / 2)
    epsilon = tensor(np.sqrt(case['guest_epsilon'][:, None] * case['frame_epsilon'][identities][None]))
    ratio_cut = (sigma / cutoff)**6
    shift = -4 * epsilon * (ratio_cut**2 - ratio_cut)
    com = (case['guest_xyz'] * case['guest_mass'][:, None]).sum(0) / case['total_mass']
    offsets = tensor(case['guest_xyz'][:, 0] - com[0])
    query_centers = np.repeat(centers, len(axes), axis=0)
    query_axes = np.tile(axes, (len(centers), 1))
    output = np.empty(len(query_centers))
    with torch.inference_mode():
        for start in range(0, len(output), chunk_size):
            position = tensor(query_centers[start:start+chunk_size]) @ cell
            pose = position[:, None] + tensor(query_axes[start:start+chunk_size])[:, None] * offsets[None, :, None]
            delta = pose[:, :, None] - cart[None, None]
            modifications = torch.where(delta > diagonal / 2, repeats,
                                        torch.where(delta < -diagonal / 2, -repeats, 0.))
            neighbor = (frame[None, None] + modifications) @ cell
            distance = torch.linalg.vector_norm(pose[:, :, None] - neighbor, dim=-1)
            distance = torch.maximum(distance, sigma[None] * .1)
            ratio6 = (sigma[None] / distance)**6
            value = torch.where(distance < cutoff,
                                4 * epsilon[None] * (ratio6**2 - ratio6) + shift[None], 0.)
            energy = value.sum((1, 2))
            output[start:start+len(energy)] = energy.cpu().numpy()
    if not np.isfinite(output).all(): raise ValueError('nonfinite uniform energies')
    return output.reshape(len(centers), len(axes))


def prepare_case(name, sources, output, size, orientations, device, chunk_size):
    centers, axes = regular_pose_grid(size, orientations)
    arrays, details = {}, {}
    for gas in GASES:
        # Source input is a material/force-field definition, not a path input.
        directory = Path(next(r['source'] for r in sources if r['gas']==gas and r['direction']==1))
        raw = (directory / 'input.dat').read_bytes()
        case = parse_string_input(raw.decode())
        energy = tensor_pose_energy(case, centers, axes, device, chunk_size)
        transformed = np.arcsinh(energy / case['temperature'])
        marginal = -case['temperature'] * (logsumexp(-energy / case['temperature'], axis=1) - np.log(orientations))
        arrays[f'orientation_{gas}'] = transformed.T.reshape(orientations, size, size, size).astype(np.float32)
        arrays[f'mean_{gas}'] = np.arcsinh(marginal / case['temperature']).reshape(1, size, size, size).astype(np.float32)
        details[gas] = {'input': str(directory / 'input.dat'), 'input_sha256': hashlib.sha256(raw).hexdigest(),
                        'temperature': case['temperature'], 'cutoff': case['cutoff']}
    arrays['axes'] = axes.astype(np.float32)
    temporary = output / (name + '.tmp')
    with temporary.open('wb') as handle: np.savez_compressed(handle, **arrays)
    os.replace(temporary, output / (name + '.npz'))
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('smoke','generate','collect'), required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=8)
    parser.add_argument('--size', type=int, default=16)
    parser.add_argument('--orientations', type=int, default=32)
    parser.add_argument('--chunk-size', type=int, default=2048)
    args = parser.parse_args()
    source = json.loads((args.root / 'inputs/full_pose_v4_v1/sources.json').read_text())
    names = [r['name'] for r in load_manifest(args.root / 'runs/v2/manifest.csv')]
    if set(source) != set(names): raise ValueError('material source coverage mismatch')
    if args.mode == 'collect':
        records = [json.loads((args.output_dir / f'shard_{i}.json').read_text()) for i in range(args.world_size)]
        observed = [name for r in records for name in r['completed']]
        failures = [x for r in records for x in r['failures']]
        passed = not failures and len(observed)==len(set(observed))==len(names) and set(observed)==set(names)
        report = {'passed': passed, 'structures': len(observed), 'expected_structures': len(names),
                  'failures': failures, 'size': args.size, 'orientations': args.orientations,
                  'independent_uniform_grid': True, 'uses_String_path_coordinates': False,
                  'energy_units': 'asinh(U_K/T_K)', 'reference': 'guest center of mass',
                  'grid': 'CIF_fractional_cell_centers', 'forcefield': 'String input, shifted finite LJ/no Coulomb',
                  'marginal': 'equal-area32 orientations, -T log mean exp(-U/T)',
                  'caveat': 'Finite grid/sampled angular approximation and archived String image rule; not exact continuous landscape.'}
        temp=args.output_dir/'report.tmp.json'; temp.write_text(json.dumps(report,indent=2)+'\n'); os.replace(temp,args.output_dir/'report.json')
        if not passed: raise RuntimeError('field preparation failed/incomplete')
        print(json.dumps(report),flush=True); return
    if not torch.cuda.is_available(): raise RuntimeError('GPU field calculation requires allocation')
    device = torch.device('cuda:0')
    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=args.mode=='generate')
    if args.mode == 'smoke':
        counts = json.loads((args.root / 'runs/v3/input_audit_full_v2/structures.json').read_text())
        selected = [r['name'] for r in sorted(counts,key=lambda r:r['atoms'],reverse=True)[:2]]
    else:
        selected = names[args.rank::args.world_size]
    completed, failures, started = [], [], time.time()
    for name in selected:
        destination = args.output_dir / (name + '.npz')
        if destination.exists(): raise FileExistsError(destination)
        try:
            prepare_case(name, source[name], args.output_dir, args.size, args.orientations, device, args.chunk_size)
            completed.append(name)
        except Exception as error:
            failures.append({'name': name, 'error': str(error)})
        if (len(completed)+len(failures))%20==0:
            print(json.dumps({'rank':args.rank,'completed':len(completed),'failed':len(failures),'seconds':time.time()-started}),flush=True)
    report = {'rank':args.rank,'world_size':args.world_size,'completed':completed,'failures':failures,
              'seconds':time.time()-started,'size':args.size,'orientations':args.orientations,
              'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3,'smoke_only':args.mode=='smoke'}
    (args.output_dir / f'shard_{args.rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if failures: raise RuntimeError('some uniform fields failed')


if __name__=='__main__': main()
