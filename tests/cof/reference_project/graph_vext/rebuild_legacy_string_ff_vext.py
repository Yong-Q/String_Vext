"""Old-framework adapter to the unchanged full-lattice DIRECT String-FF kernel.

Absent validated old String inputs, gas replacement is explicitly requested
standardization, not reproduction of old String or Vext calculations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from graph_vext.cif import Cell, load_uff_params, parse_cif
from graph_vext.string_ff_vext import direct_pose_energy, marginal_energy, uniform_fields, validate_case

ROOT = Path(__file__).resolve().parents[1]
LEGACY_BASE = Path('/home/qiuyong/TuTraSt_String/vext/grid_vext_Uni_MOF/Initialization_feature')
GASES = ('C2H4', 'C2H6')
MANIFESTS = {g: LEGACY_BASE/'runs'/d/'manifest.csv' for g, d in zip(GASES,
    ('111404.node01.hpc.local_c2h4_60', '111405.node01.hpc.local_c2h6_60'))}
UFF_REFERENCE = LEGACY_BASE/'forcefield/data_ff_UFF'
RECIPE_FILES = ('string_ff_vext.py', 'rebuild_legacy_string_ff_vext.py',
                'string_atom_mapping.py', 'orientation_oracle_fields.py', 'cif.py')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    return {name: file_hash(ROOT/'graph_vext'/name) for name in RECIPE_FILES}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.replace(temporary, path)


def parse_legacy_framework(input_path, cif_path, uff_path=UFF_REFERENCE):
    lines = [line.strip() for line in Path(input_path).read_text().splitlines()]
    def after(prefix):
        return next(i for i, line in enumerate(lines) if line.startswith(prefix))+1
    values = np.array([float(v) for v in lines[after('Lxa Lyb Lzc')].split()])
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError('source lengths, grid spacing and all three angles required')
    parameters = np.r_[values[:3], values[6:9]]
    if (parameters[:3] <= 0).any() or (parameters[3:] <= 0).any() or (parameters[3:] >= 180).any():
        raise ValueError('invalid source lattice parameters')
    cell = Cell(*parameters).matrix
    cif_cell, sites = parse_cif(Path(cif_path))
    if np.linalg.det(cell) <= 0 or not np.allclose(cell, cif_cell.matrix, rtol=0, atol=2e-5) \
            or not np.allclose(cell@cell.T, cif_cell.matrix@cif_cell.matrix.T, rtol=1e-6, atol=1e-4):
        raise ValueError('source/CIF triclinic metric mismatch')
    start = after('Number_of_solute_atoms')
    count = int(lines[start])
    rows = np.array([[float(v) for v in line.split()] for line in lines[start+2:start+2+count]])
    if rows.shape != (count, 7) or count != len(sites) or not np.isfinite(rows).all() \
            or not np.array_equal(rows[:, 0], np.arange(1, count+1)):
        raise ValueError('framework row IDs/counts do not preserve source CIF order')
    if (rows[:, 6] != 0).any() or any(site.charge != 0 for site in sites):
        raise ValueError('charged framework is not supported by the LJ-only recipe')
    elements = [site.element for site in sites]
    reference = load_uff_params(Path(uff_path))
    if any(element not in reference for element in elements):
        raise ValueError('CIF element missing from original UFF provenance')
    expected = np.array([reference[e] for e in elements])
    if not np.allclose(rows[:, 4:6], expected, rtol=0, atol=1e-6):
        raise ValueError('source UFF rows do not match CIF composition/order')
    fractional = rows[:, 1:4]@np.linalg.inv(cell)
    cif_fractional = np.array([[s.fract_x, s.fract_y, s.fract_z] for s in sites])
    difference = fractional-cif_fractional
    error = np.linalg.norm((difference-np.rint(difference))@cell, axis=1)
    if error.max() > 2e-4:
        raise ValueError('source Cartesian rows fail periodic CIF alignment')
    # Reject charged legacy guest definitions too, even when replacing the gas.
    start = after('Kind_of_solvent_atoms')
    kinds = int(lines[start])
    guest_types = np.array([[float(v) for v in line.split()] for line in lines[start+2:start+2+kinds]])
    if guest_types.shape != (kinds, 4) or not np.isfinite(guest_types).all() or (guest_types[:, 3] != 0).any():
        raise ValueError('charged or malformed legacy guest')
    conditions = np.array([float(v) for v in lines[after('Epsilon(K) Sigma(A)')].split()])
    frame = dict(cell=cell, frame_frac=np.mod(fractional, 1), frame_sigma=rows[:, 4].copy(),
                 frame_epsilon=rows[:, 5].copy(), elements=elements)
    audit = dict(atoms=count, composition=dict(Counter(elements)),
                 max_periodic_alignment_error_A=float(error.max()), cell_parameters=parameters.tolist(),
                 max_angle_deviation_deg=float(np.abs(parameters[3:]-90).max()),
                 legacy_guest_site_sigma_A=guest_types[:, 1].tolist(),
                 legacy_guest_site_epsilon_K=guest_types[:, 2].tolist(),
                 legacy_bulk_epsilon_K=float(conditions[0]), legacy_bulk_sigma_A=float(conditions[1]),
                 legacy_temperature_K=float(conditions[3]))
    return frame, audit


def verify_same_framework(first, second):
    for key in ('cell', 'frame_sigma', 'frame_epsilon'):
        if np.shape(first[key]) != np.shape(second[key]) or not np.allclose(first[key], second[key], atol=1e-6, rtol=0):
            raise ValueError('gas sources disagree on framework '+key)
    if first['elements'] != second['elements']:
        raise ValueError('gas sources disagree on framework composition')
    delta = first['frame_frac']-second['frame_frac']
    if np.shape(first['frame_frac']) != np.shape(second['frame_frac']) \
            or np.linalg.norm((delta-np.rint(delta))@first['cell'], axis=1).max() > 2e-4:
        raise ValueError('gas sources disagree on periodic framework coordinates')


def parse_string_definition(raw):
    """Numeric old/current String definitions; element tokens are optional."""
    lines = [line.strip() for line in raw.decode().splitlines()]
    def after(prefix):
        return next(i for i, line in enumerate(lines) if line.startswith(prefix))+1
    lengths = [float(x) for x in lines[after('La Lb Lc')].split()[:3]]
    angles = [float(x) for x in lines[after('Alpha Beta Gamma')].split()[:3]]
    cell = Cell(*lengths, *angles).matrix
    conditions = np.array([float(x) for x in lines[after('cutoff(A)')].split()])
    start = after('Number of sites')
    count = int(lines[start])
    guest = np.array([[float(x) for x in line.split()[:7]] for line in lines[start+2:start+2+count]])
    start = after('Number of atoms')
    atoms = int(lines[start])
    tokens = [line.split() for line in lines[start+2:start+2+atoms]]
    table = np.array([[float(x) for x in row[:8]] for row in tokens])
    if guest.shape != (2, 7) or table.shape != (atoms, 8) or not np.isfinite(guest).all() \
            or not np.isfinite(table).all() or (guest[:, 5] != 0).any() or (table[:, 3] != 0).any() \
            or not np.array_equal(table[:, 0], np.arange(1, atoms+1)):
        raise ValueError('unsupported/charged/malformed String definition')
    if len(conditions) < 4 or not np.isfinite(conditions).all() or not np.isfinite(cell).all():
        raise ValueError('invalid String conditions/lattice')
    body = guest[:, :3]
    mass = guest[:, 6]
    if (mass <= 0).any() or not np.isclose(mass.sum(), conditions[2], rtol=1e-7):
        raise ValueError('String guest mass mismatch')
    length = float(np.linalg.norm(body[1]-body[0]))
    if length <= 0:
        raise ValueError('two distinct linear guest sites required')
    center = (body*mass[:, None]).sum(0)/conditions[2]
    director = (body[1]-body[0])/length
    offsets = (body-center)@director
    canonical = np.zeros((2, 3))
    canonical[:, 0] = offsets
    return dict(cell=cell, frame_frac=np.mod(table[:, 5:8], 1), frame_sigma=table[:, 1],
                frame_epsilon=table[:, 2], frame_mass=table[:, 4],
                elements=[row[8] if len(row) > 8 else None for row in tokens],
                guest_sigma=guest[:, 4], guest_epsilon=guest[:, 3], guest_mass=mass,
                guest_xyz=canonical, original_body=body, original_COM=center,
                original_director=director, numeric_conditions=conditions,
                total_mass=float(conditions[2]), temperature=float(conditions[3]),
                cutoff=float(conditions[0]), fh_signal=int(conditions[1]))


def verify_geometry(frame, source):
    if np.shape(frame['frame_frac']) != np.shape(source['frame_frac']) \
            or not np.allclose(frame['cell'], source['cell'], atol=2e-5, rtol=0):
        raise ValueError('old String and CIF/source cell or atom count mismatch')
    delta = frame['frame_frac']-source['frame_frac']
    if np.linalg.norm((delta-np.rint(delta))@frame['cell'], axis=1).max() > 2e-4:
        raise ValueError('old String rows fail periodic CIF geometry alignment')
    if any(actual is not None and actual != expected for actual, expected in zip(source['elements'], frame['elements'])):
        raise ValueError('old String element tokens disagree with mapped CIF composition')
    # With no element token, CIF row association and positive source atom mass
    # provide an independent composition check, without overwriting source FF.
    from ase.data import atomic_masses, atomic_numbers
    expected_mass = np.array([atomic_masses[atomic_numbers[e]] for e in frame['elements']])
    if not np.allclose(source['frame_mass'], expected_mass, atol=.08, rtol=0):
        raise ValueError('old String atom masses disagree with CIF composition')


def attach_gas(frame, source_path, standardized):
    source_path = Path(source_path)
    raw = source_path.read_bytes()
    source = parse_string_definition(raw)
    if not standardized:
        verify_geometry(frame, source)
    guest = {key: copy.deepcopy(source[key]) for key in
             ('guest_sigma', 'guest_epsilon', 'guest_mass', 'guest_xyz', 'total_mass', 'temperature', 'cutoff')}
    case = {**copy.deepcopy(frame), **guest}
    differs = False
    if not standardized:
        differs = any(not np.allclose(frame[k], source[k], atol=1e-6, rtol=0)
                      for k in ('frame_sigma', 'frame_epsilon'))
        for key in ('frame_sigma', 'frame_epsilon', 'frame_frac', 'cell'):
            case[key] = source[key].copy()
    validate_case(case)
    if any(not np.allclose(case[k], case[k][0], rtol=0, atol=1e-7)
           for k in ('guest_sigma', 'guest_epsilon', 'guest_mass')):
        raise ValueError('equivalent symmetric two-site director guest required')
    metadata = dict(source_input=str(source_path), input_sha256=hashlib.sha256(raw).hexdigest(),
        definition='user_requested_current_String_gas_standardization' if standardized else 'validated_old_String_guest_and_framework',
        sigma_A=case['guest_sigma'].tolist(), epsilon_K=case['guest_epsilon'].tolist(),
        body_sites_A=case['guest_xyz'].tolist(), masses_g_mol=case['guest_mass'].tolist(),
        cutoff_A=float(case['cutoff']), temperature_K=float(case['temperature']),
        original_body_sites_A=source['original_body'].tolist(), original_COM_A=source['original_COM'].tolist(),
        original_body_director=source['original_director'].tolist(),
        canonicalization='mass-weighted COM-centered scalar offsets along centered BODYX; original bond norm and masses retained',
        fh_signal_recorded=source['fh_signal'],
        numeric_conditions=source['numeric_conditions'].tolist(),
        framework_FF_differs_from_archived_VEXT=differs,
        old_String_reproduction=False, geometry='full_triclinic_lattice_cutoff_sum')
    return case, metadata


def validate_directions(frame, paths):
    if len(paths) != 3:
        raise ValueError('all three old String directions required')
    first = None
    definitions = []
    for path in paths:
        case, metadata = attach_gas(frame, path, standardized=False)
        raw = parse_string_definition(Path(path).read_bytes())
        if first is not None:
            verify_same_framework(first, case)
            for key in ('guest_sigma', 'guest_epsilon', 'guest_mass', 'guest_xyz'):
                if not np.allclose(first[key], case[key], atol=1e-7, rtol=0):
                    raise ValueError('old gas/geometry differs between directions')
            if metadata['numeric_conditions'] != definitions[0]['numeric_conditions'] \
                    or not np.array_equal(raw['original_body'], original_body):
                raise ValueError('old numeric gas conditions differ between directions')
        else:
            first, original_body = case, raw['original_body']
        definitions.append(metadata)
    result = definitions[0].copy()
    result['direction_sources'] = [{'input': d['source_input'], 'sha256': d['input_sha256']} for d in definitions]
    result['three_direction_numeric_physics_match'] = True
    return first, result


def verify_existing(path, record, binding, codes, size, orientations):
    if record.get('binding') != binding or record.get('code_hashes') != codes \
            or record.get('energy_mode') != 'direct' or record.get('size') != size \
            or record.get('orientations') != orientations or record.get('sha256') != file_hash(path):
        raise ValueError('resumed source/code/definition/content mismatch')
    with np.load(path, allow_pickle=False) as arrays:
        for gas in GASES:
            for key, shape in [('orientation_K', (orientations, size, size, size)),
                               ('marginal_K', (size, size, size)), ('site_K', (2, size, size, size)),
                               ('axes', (orientations, 3))]:
                value = arrays[key+'_'+gas]
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError('resumed field shape/finiteness mismatch')


def load_inventory(directory, manifests=MANIFESTS, expected=8793):
    directory = Path(directory)
    entries = {}
    manifest_hashes = {}
    for gas, path in manifests.items():
        manifest_hashes[gas] = file_hash(path)
        with Path(path).open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        names = [row['name'] for row in rows]
        if len(names) != expected or len(set(names)) != expected \
                or any(Path(name).name != name or name in ('.', '..') for name in names):
            raise ValueError('legacy manifest count/uniqueness/name gate failed')
        for row in rows:
            if not Path(row['cif']).is_file() or not Path(row['input_dat']).is_file():
                raise ValueError('missing legacy CIF/input: '+row['name'])
            entry = entries.setdefault(row['name'], {'name': row['name'], 'cif': row['cif'], 'legacy': {}, 'sources': {}})
            if entry['cif'] != row['cif']:
                raise ValueError('paired gas manifests use different CIF paths')
            entry['legacy'][gas] = row['input_dat']
    if len(entries) != expected or any(set(e['legacy']) != set(GASES) for e in entries.values()):
        raise ValueError('legacy gas manifest names differ')
    source_csv = directory/'direction_sources.csv'
    # Only definition provenance is retained; collected D/path output columns
    # are never converted, passed to workers, or used for pilot selection.
    with source_csv.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['dataset'] != 'legacy':
                continue
            name, gas, direction = row['name'], row['gas_namespace'], int(row['direction'])
            if name not in entries or gas not in GASES or direction not in (1, 2, 3) \
                    or row['status'] != 'definition_read':
                raise ValueError('unvalidated legacy direction inventory row')
            selected = entries[name]['sources'].setdefault(gas, {})
            if direction in selected:
                raise ValueError('duplicate legacy direction')
            selected[direction] = {'path': row['input_path'], 'sha256': row['input_sha256'],
                                   'physics_sha256': row['physics_bytes_sha256']}
    for entry in entries.values():
        for gas in GASES:
            directions = entry['sources'].get(gas, {})
            if set(directions) != {1, 2, 3}:
                raise ValueError('all old String directions required; no silent global fallback')
            ordered = [directions[d] for d in (1, 2, 3)]
            if len({d['physics_sha256'] for d in ordered}) != 1 \
                    or any(not Path(d['path']).is_file() for d in ordered):
                raise ValueError('old direction physics differs or input is missing')
            entry['sources'][gas] = ordered
    report = json.loads((directory/'report.json').read_text())
    if not report.get('completed') or report.get('legacy_materials') != expected:
        raise ValueError('parent inventory report incomplete')
    binding = {'manifests': manifest_hashes, 'direction_sources_sha256': file_hash(source_csv),
               'parent_inventory_sha256': file_hash(directory/'report.json'),
               'original_UFF_reference_sha256': file_hash(UFF_REFERENCE)}
    return entries, binding


def prepare_entry(entry):
    frames, audits, cases, gases = {}, {}, {}, {}
    binding = {'cif': {'path': entry['cif'], 'sha256': file_hash(entry['cif'])}, 'legacy': {}, 'String': {}}
    for gas in GASES:
        path = entry['legacy'][gas]
        binding['legacy'][gas] = {'path': path, 'sha256': file_hash(path)}
        frames[gas], audits[gas] = parse_legacy_framework(path, entry['cif'])
        paths = []
        for source in entry['sources'][gas]:
            if file_hash(source['path']) != source['sha256']:
                raise ValueError('actual old String input no longer matches inventory: '+source['path'])
            paths.append(source['path'])
        cases[gas], gases[gas] = validate_directions(frames[gas], paths)
        binding['String'][gas] = gases[gas]['direction_sources']
    verify_same_framework(frames['C2H4'], frames['C2H6'])
    verify_same_framework(cases['C2H4'], cases['C2H6'])
    binding['UFF_reference_sha256'] = file_hash(UFF_REFERENCE)
    return cases, frames, audits, gases, binding


def build_one(task):
    entry, size, orientations = task['entry'], task['size'], task['orientations']
    output, name, codes = Path(task['output']), entry['name'], task['code_hashes']
    if code_hashes() != codes:
        raise ValueError('parent/adapter code changed after recipe gate')
    cases, frames, audits, gases, binding = prepare_entry(entry)
    destination, record_path = output/'data'/(name+'.npz'), output/'records'/(name+'.json')
    if destination.exists() or record_path.exists():
        if not task['resume']:
            raise FileExistsError(destination)
        if not destination.is_file() or not record_path.is_file():
            raise ValueError('orphan output retained; missing data/record pair: '+name)
        record = json.loads(record_path.read_text())
        verify_existing(destination, record, binding, codes, size, orientations)
        return record
    arrays, checks = {}, {}
    for gas in GASES:
        case = cases[gas]
        generated = uniform_fields(case, size=size, orientations=orientations, energy_mode='direct')
        for key, value in generated.items():
            arrays[key+'_'+gas] = value
        arrays['grid_'+gas] = np.arcsinh(generated['marginal_K']/1000.).astype(np.float32)
        # Check a few generated centers against a separate direct query path.
        center_indices = np.array([[0, 0, 0], [size//2]*3, [size-1]*3])
        centers = (center_indices+.5)/size
        direct = direct_pose_energy(case, centers, generated['axes'].astype(float))
        cached = generated['orientation_K'][:, center_indices[:, 0], center_indices[:, 1], center_indices[:, 2]].T
        error = float(np.max(np.abs(cached-direct)/np.maximum(1., np.abs(direct))))
        if error > 2e-4:
            raise ValueError('DIRECT field/sample mismatch')
        marginal = marginal_energy(direct, case['temperature'])
        actual_marginal = generated['marginal_K'][tuple(center_indices.T)]
        if not np.allclose(actual_marginal, marginal, rtol=2e-4, atol=.1):
            raise ValueError('DIRECT angular marginal mismatch')
        checks[gas] = {'sample_relative_error_max': error, 'checked_centers': center_indices.tolist()}
        if file_hash(entry['legacy'][gas]) != binding['legacy'][gas]['sha256'] \
                or any(file_hash(s['input']) != s['sha256'] for s in gases[gas]['direction_sources']):
            raise ValueError('definition sources changed during generation')
    common = cases['C2H4']
    arrays.update(cell_matrix=common['cell'], atom_fractional=common['frame_frac'],
        framework_sigma_A=common['frame_sigma'], framework_epsilon_K=common['frame_epsilon'],
        framework_elements=np.array(common['elements']),
        archived_VEXT_framework_sigma_A=frames['C2H4']['frame_sigma'],
        archived_VEXT_framework_epsilon_K=frames['C2H4']['frame_epsilon'])
    if code_hashes() != codes or file_hash(entry['cif']) != binding['cif']['sha256']:
        raise ValueError('code/CIF changed during generation')
    temporary = destination.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, destination)
    record = {'name': name, 'binding': binding, 'code_hashes': codes, 'size': size,
              'orientations': orientations, 'energy_mode': 'direct', 'sha256': file_hash(destination),
              'gases': gases, 'legacy_VEXT_provenance': audits, 'DIRECT_checks': checks,
              'ff_domains': ['legacy_C2H4', 'legacy_C2H6'], 'global_standardization_used': False}
    atomic_json(record_path, record)
    return record


def select_pilot(entries, count=6):
    names = sorted(entries)
    if len(names) < count:
        raise ValueError('insufficient pilot materials')
    # Include the first canonical old case; select remaining names independent
    # of diffusion labels or energies, with a recorded fixed seed.
    return sorted([names[0]]+np.random.default_rng(42).choice(names[1:], count-1, replace=False).tolist())


def verify_pilot(report, codes, sources):
    if not report.get('passed') or report.get('mode') != 'pilot' or report.get('size') != 16 \
            or report.get('orientations') != 64 or report.get('energy_mode') != 'direct' \
            or report.get('code_hashes') != codes or report.get('inventory_binding') != sources \
            or report.get('structures') != 6 or report.get('global_standardization_used') is not False:
        raise ValueError('successful six-material exact-recipe pilot required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('pilot', 'current'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, default=ROOT/'inputs/string_vext_rebuild_inventory_v1')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--pilot-report', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--guard-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.world_size < 1 or not 0 <= args.rank < args.world_size:
        parser.error('invalid worker/rank settings')
    entries, sources = load_inventory(args.inventory)
    codes = code_hashes()
    if args.guard_only:
        print(json.dumps({'passed': True, 'old_materials': len(entries), 'gas_definitions': 2*len(entries),
                          'direction_definitions': 6*len(entries), 'code_hashes': codes, 'inventory_binding': sources}), flush=True)
        return
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('prepare fields only inside a Slurm CPU allocation')
    allowed = ROOT/'inputs'/('string_ff_vext_legacy_pilot_v1' if args.mode == 'pilot' else 'string_ff_vext_legacy_current_v1')
    output = args.output.resolve()
    if output != allowed and allowed not in output.parents:
        raise ValueError('output must be inside the assigned new legacy output prefix')
    if args.mode == 'current':
        if args.pilot_report is None or args.world_size != 2:
            raise ValueError('coordinated full rebuild needs pilot report and two separate shard outputs')
        verify_pilot(json.loads(args.pilot_report.read_text()), codes, sources)
        names = sorted(entries)[args.rank::args.world_size]
        size = 60
    else:
        if args.rank != 0 or args.world_size != 1:
            raise ValueError('pilot is one six-material allocation')
        names, size = select_pilot(entries), 16
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    (output/'data').mkdir(parents=True, exist_ok=True)
    (output/'records').mkdir(exist_ok=True)
    atomic_json(output/'inventory.json', {'expected_old_materials': len(entries),
        'old_names': sorted(entries), 'selected_names': names, 'inventory_binding': sources,
        'code_hashes': codes, 'mode': args.mode, 'rank': args.rank, 'world_size': args.world_size,
        'source_policy': 'actual old String guest AND per-atom framework, all three directions checked; never global defaults'})
    started, records, failures = time.monotonic(), {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_one, {'entry': entries[n], 'output': str(output), 'size': size,
            'orientations': 64, 'code_hashes': codes, 'resume': args.resume}): n for n in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                records[name] = future.result()
            except Exception as error:
                failure = {'name': name, 'error': repr(error)}
                failures.append(failure)
                atomic_json(output/'records'/(name+'.failure.json'), failure)
            if args.mode == 'pilot' or (len(records)+len(failures)) % 100 == 0:
                print(json.dumps({'completed': len(records), 'failed': len(failures),
                                  'seconds': time.monotonic()-started}), flush=True)
    if code_hashes() != codes:
        failures.append({'error': 'recipe code changed during run'})
    report = {'passed': not failures and set(records) == set(names), 'mode': args.mode,
        'structures': len(records), 'expected': len(names), 'old_coverage_materials': len(entries),
        'legacy_direction_definitions': 6*len(entries), 'size': size, 'orientations': 64,
        'energy_mode': 'direct', 'inventory_binding': sources, 'code_hashes': codes,
        'records': records, 'failures': failures, 'selected_names': names,
        'rank': args.rank, 'world_size': args.world_size, 'seconds': time.monotonic()-started,
        'global_standardization_used': False, 'old_D_used': False, 'String_paths_used': False,
        'Coulomb': False, 'energy_caps': False, 'grid': 'fractional cell centers, molecular COM',
        'ff_policy': 'actual legacy String gas and framework FF; NOT archived VEXT UFF or current String gas',
        'old_String_numerical_reproduction': False,
        'limitations': ['Full triclinic image sums replace old Cartesian image rules.',
                       'Finite grid/64 directors are not continuous saddle certificates.',
                       'Source inventory binds definitions, not a historical solver binary/label provenance certificate.']}
    atomic_json(output/'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)
    if not report['passed']:
        raise RuntimeError('legacy rebuild failed; all failure evidence retained')


if __name__ == '__main__':
    main()
