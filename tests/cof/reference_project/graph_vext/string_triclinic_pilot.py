"""Prepared, gate-bound 10-material CUDA String comparison; no automatic submission."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GASES = ('C2H4', 'C2H6')
ARMS = ('current', 'old_sigma_epsilon_only')
CURRENT = {'C2H4': (3.70, 85.), 'C2H6': (3.75, 98.)}
OLD = {'C2H4': (3.68, 92.8), 'C2H6': (3.76, 108.)}
STATUS = 'provisional_saved_path_reference'
PREFIX = 'runs/generalization/string_triclinic_10_pilot_v1'
ERROR = re.compile(r'fatal\s+error|out of memory|illegal memory|cuda[_ ]error|\b(?:nan|inf|infinity)\b', re.I)


def digest(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    return digest(Path(path).read_bytes())


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def new_output(root, path):
    path = Path(path).resolve()
    prefix = (Path(root)/PREFIX).resolve()
    if prefix not in path.parents or path.exists():
        raise ValueError('provide a NEW unique directory beneath '+str(prefix))
    path.mkdir(parents=True, exist_ok=False)
    return path


def local_path(root, relative):
    path = (Path(root)/relative).resolve()
    if Path(relative).is_absolute() or Path(root).resolve() not in path.parents:
        raise ValueError('unsafe relative artifact path')
    return path


def recipe_hashes(root):
    return {name: file_hash(Path(root)/'graph_vext'/name) for name in
            ('string_triclinic_pilot.py', 'string_atom_mapping.py', 'rebuild_uncapped_labels.py')}


def input_case(raw):
    from graph_vext.string_atom_mapping import parse_string_input
    text = raw.decode()
    lines = [x.strip() for x in text.splitlines()]
    def after(prefix):
        return next(i for i, line in enumerate(lines) if line.startswith(prefix))+1
    conditions = lines[after('cutoff(A)')].split()
    images = lines[after('#_of_points')].split()
    direction = int(lines[after('Direction')])
    settings = dict(running_steps=int(conditions[4]), images=int(images[0]), direction=direction,
                    delta_frac=float(images[1]), delta_angle_degree=float(images[2]),
                    convergence_setting=lines[after('convergence_setting')])
    if settings['running_steps']!=10000 or settings['images']!=401 or direction not in (1, 2, 3):
        raise ValueError('source must retain 401 images, 10000 steps and a valid direction')
    case = parse_string_input(text)
    if len(case['guest_xyz'])!=2 or not np.isclose(case['guest_mass'].sum(), case['total_mass']):
        raise ValueError('two-site source/mass required')
    return case, settings


def semantics(case, settings):
    center = (case['guest_xyz']*case['guest_mass'][:, None]).sum(0)/case['total_mass']
    return dict(**settings, guest_sigma_A=case['guest_sigma'].tolist(),
                guest_epsilon_K=case['guest_epsilon'].tolist(), guest_mass=case['guest_mass'].tolist(),
                total_mass=case['total_mass'], body_sites_A=case['guest_xyz'].tolist(), body_COM_A=center.tolist(),
                bond_length_A=float(np.linalg.norm(case['guest_xyz'][1]-case['guest_xyz'][0])),
                temperature_K=case['temperature'], cutoff_A=case['cutoff'], fh_signal=case['fh_signal'],
                framework_atoms=len(case['frame_frac']), cell_matrix=case['cell'].tolist(),
                framework_definition='unchanged actual current source per-atom UFF table')


def arm_input(raw, gas, arm):
    if gas not in GASES or arm not in ARMS:
        raise ValueError('invalid gas/arm')
    case, _settings = input_case(raw)
    sigma, epsilon = CURRENT[gas]
    if not (np.allclose(case['guest_sigma'], sigma, rtol=0, atol=1e-7)
            and np.allclose(case['guest_epsilon'], epsilon, rtol=0, atol=1e-7)):
        raise ValueError('source is not the actual current gas definition')
    if arm=='current':
        return raw
    lines = raw.decode().splitlines(keepends=True)
    start = next(i for i, x in enumerate(lines) if x.strip().startswith('Number of sites'))
    count = int(lines[start+1])
    if count!=2:
        raise ValueError('two sites required')
    sigma, epsilon = OLD[gas]
    for index in range(start+3, start+3+count):
        matches = list(re.finditer(r'\S+', lines[index]))
        if len(matches)!=7:
            raise ValueError('exact seven-column site row required')
        # Replace tokens from right to left; all other bytes remain untouched.
        for column, value in ((4, sigma), (3, epsilon)):
            token = matches[column]
            lines[index] = lines[index][:token.start()]+f'{value:.6f}'+lines[index][token.end():]
    result = ''.join(lines).encode()
    new_case, new_settings = input_case(result)
    if _settings!=new_settings:
        raise ValueError('input controls changed')
    for key in ('cell', 'frame_frac', 'frame_sigma', 'frame_epsilon', 'guest_xyz', 'guest_mass'):
        if not np.array_equal(case[key], new_case[key]):
            raise ValueError('non-sigma/epsilon physics changed: '+key)
    return result


def select_materials(rows, descriptors, seed=42):
    groups = defaultdict(dict)
    for row in rows:
        name, key = row['name'], (row['gas'], row['direction'])
        if key in groups[name] or key[0] not in GASES or key[1] not in ('a', 'b', 'c'):
            raise ValueError('duplicate/invalid reference key')
        groups[name][key] = row['qc_status']
    metadata = {r['name']: r for r in descriptors}
    if len(metadata)!=len(descriptors):
        raise ValueError('duplicate atom/geometry descriptors')
    expected = {(g, d) for g in GASES for d in 'abc'}
    eligible = sorted(n for n, values in groups.items() if set(values)==expected
                      and all(v==STATUS for v in values.values()))
    if any(n not in metadata for n in eligible) or len(eligible)<10:
        raise ValueError('insufficient complete provisional cohort/descriptors')
    counts = np.array([metadata[n]['atoms'] for n in eligible])
    small_max, medium_max = map(int, np.quantile(counts, [.5, .8]))
    pools = defaultdict(list)
    for name in eligible:
        item = metadata[name]
        count = int(item['atoms'])
        band = 'small' if count<=small_max else ('medium' if count<=medium_max else 'large')
        pools[bool(item['nonorthogonal']), band].append(name)
    for key in pools:
        pools[key].sort(key=lambda n: digest(f'{seed}:{n}'.encode()))
    # Interleaved card schedules: each card gets 2/3 of both geometry and size classes.
    layout = [(False,'small'), (True,'medium'), (True,'medium'), (False,'small'),
              (False,'medium'), (True,'small'), (True,'small'), (False,'medium'),
              (False,'small'), (True,'medium')]
    selected, redistributed = [], []
    geometry_used, size_used = Counter(), Counter()
    available_counts = {str(key):len(values) for key,values in pools.items()}
    band_order = {'small':0, 'medium':1, 'large':2}
    for material_id, requested in enumerate(layout):
        key = requested
        if not pools[key]:
            # Keep the requested size where possible; use large cases only
            # when needed for coverage, without inventing geometry classes.
            available = [bucket for bucket,values in pools.items() if values]
            key = min(available, key=lambda bucket: (
                bucket[1]=='large', bucket[1]!=requested[1],
                geometry_used[bucket[0]], size_used[bucket[1]],
                bucket[0]!=requested[0], band_order[bucket[1]], bucket[0]))
            redistributed.append(dict(material_id=material_id,
                                      requested_geometry='skew' if requested[0] else 'orthogonal',
                                      requested_size=requested[1],
                                      actual_geometry='skew' if key[0] else 'orthogonal', actual_size=key[1]))
        name = pools[key].pop(0)
        geometry_used[key[0]] += 1
        size_used[key[1]] += 1
        selected.append(dict(material_id=material_id, name=name, atoms=int(metadata[name]['atoms']),
                             nonorthogonal=key[0], size_band=key[1]))
    return selected, dict(seed=seed, eligible_materials=len(eligible), small_max_atoms=small_max,
                         medium_max_atoms=medium_max,
                         geometry_counts={'orthogonal':geometry_used[False], 'skew':geometry_used[True]},
                         size_counts={band:size_used[band] for band in ('small','medium','large')},
                         available_bucket_counts=available_counts, quota_redistributed=bool(redistributed),
                         redistribution=redistributed,
                         selection_all_nonorthogonal_corpus=all(bool(metadata[n]['nonorthogonal']) for n in eligible),
                         uses_D_for_selection=False,
                         membership_policy='material IDs 0::2 on gpu, 1::2 on gpu2; both gases/all directions/both arms')


def convergence(stdout, stderr, returncode, running_steps, info_marker='info:', interval=200):
    result = dict(converged=False, iteration=None, expanded_atoms=None, reasons=[])
    if returncode!=0:
        result['reasons'].append('nonzero_exit_or_runner_abort')
    if ERROR.search(stdout+'\n'+stderr):
        result['reasons'].append('native_error_or_nonfinite_log')
    pattern = re.escape(info_marker)+r'\s*([01])\s+(\d+)\s+(\d+)'
    matches = re.findall(pattern, stdout)
    if len(matches)!=1:
        result['reasons'].append('missing_or_ambiguous_convergence_info')
    else:
        flag, iteration, atoms = map(int, matches[0])
        result.update(native_converged_flag=flag, iteration=iteration, expanded_atoms=atoms)
        if flag!=1:
            result['reasons'].append('native_not_converged')
        if not (0<=iteration<running_steps and atoms>0 and interval>0 and iteration % interval==0):
            result['reasons'].append('invalid_or_exhausted_step_counter')
    result['converged'] = not result['reasons']
    return result


def step_log_audit(text, final_iteration, contract):
    try:
        rows = list(csv.DictReader(io.StringIO(text)))
        iteration = [int(r[contract['iteration']]) for r in rows]
        coordinate = [int(r[contract['coordinate']]) for r in rows]
        orientation = [int(r[contract['orientation']]) for r in rows]
        passed = (bool(rows) and iteration==sorted(set(iteration)) and iteration[-1]==final_iteration
                  and coordinate[-1]==orientation[-1]==1
                  and all(x in (0, 1) for x in coordinate+orientation))
        return dict(passed=passed, rows=len(rows), final_iteration=iteration[-1] if rows else None)
    except (KeyError, ValueError, TypeError):
        return dict(passed=False, reason='malformed_step_log')


def score_result(case, array, stdout, stderr, returncode, direction, contract=None, step_text=None):
    from graph_vext.rebuild_uncapped_labels import path_logd, path_qc, linear_value
    contract = contract or dict(info_marker='info:', check_interval=200, step_log=None)
    audit = convergence(stdout, stderr, returncode, 10000, contract['info_marker'], contract['check_interval'])
    result = dict(status='nonconverged', logD=None, D_m2_s=None, path_barrier_K=None,
                  raw_min_energy_K=None, raw_max_energy_K=None, convergence=audit, flags=list(audit['reasons']))
    if not audit['converged']:
        return result
    if contract.get('step_log') is not None:
        check = step_log_audit(step_text or '', audit['iteration'], contract['step_log'])
        result['step_log_audit'] = check
        if not check['passed']:
            result.update(status='unknown_step_log', flags=['step_log_convergence_not_verified'])
            return result
    p = np.asarray(array, dtype=float)
    if p.shape!=(401, 7) or not np.isfinite(p).all():
        result.update(status='invalid_path', flags=['nonfinite_or_wrong_401x7_path'])
        return result
    try:
        qc = path_qc(case, p, direction-1)
        if qc['endpoint_winding_error']>5e-4 or qc['max_segment_to_mean']>20:
            raise ValueError('winding/step irregularity')
        value = path_logd(case, p, direction-1)
    except (ValueError, FloatingPointError) as exc:
        result.update(status='invalid_path', flags=['path_QC_or_logD_failed: '+str(exc)])
        return result
    linear = linear_value(value)
    result.update(status='valid', logD=value, D_m2_s=linear, path_barrier_K=float(np.ptp(p[:, 6])),
                  raw_min_energy_K=float(p[:, 6].min()), raw_max_energy_K=float(p[:, 6].max()),
                  path_qc=qc, flags=[] if linear is not None else ['linear_D_unrepresentable'],
                  logD_semantics='uncapped saved-final-path TST via rebuild_uncapped_labels.path_logd',
                  energy_source='new native final path, seventh column (K); not old cap utility')
    return result


def pair_values(new, reference):
    from graph_vext.rebuild_uncapped_labels import linear_value
    if new.get('status')!='valid' or reference.get('status')!='valid':
        return None
    delta = new['logD']-reference['logD']
    if not math.isfinite(delta):
        raise ValueError('nonfinite paired logD')
    return dict(reference_logD=reference['logD'], new_logD=new['logD'], delta_logD=delta,
                absolute_delta_logD=abs(delta), ratio_new_over_reference=linear_value(delta),
                absolute_fold_ratio=linear_value(abs(delta)),
                reference_path_barrier_K=reference.get('path_barrier_K'),
                new_path_barrier_K=new.get('path_barrier_K'))


def assigned_gpu(visible):
    token = visible.strip()
    if not token or ',' in token or not (token.isdigit() or token.startswith('GPU-')):
        raise ValueError('one explicitly assigned CUDA_VISIBLE_DEVICES card required')
    return token


def parse_gpu_query(text):
    rows = list(csv.reader(io.StringIO(text.strip())))
    if len(rows)!=1 or len(rows[0])!=4:
        raise ValueError('nvidia-smi must report exactly one assigned card')
    uuid, total, used, free = [s.strip() for s in rows[0]]
    values = list(map(float, (total, used, free)))
    if not uuid.startswith('GPU-') or not np.isfinite(values).all() or min(values)<0 or values[0]<=0:
        raise ValueError('invalid GPU memory query')
    return dict(uuid=uuid, total_MiB=values[0], used_MiB=values[1], free_MiB=values[2])


def gpu_safe(card, required_free_mib=20000, reserve_mib=2048):
    return card['free_MiB']>=max(required_free_mib, reserve_mib) and card['used_MiB']<=card['total_MiB']-reserve_mib


def gpu_query(token):
    proc = subprocess.run(['nvidia-smi', '-i', token, '--query-gpu=uuid,memory.total,memory.used,memory.free',
                           '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True)
    return parse_gpu_query(proc.stdout)


def validation_gate(root, executable, report_path):
    root, executable = Path(root).resolve(), Path(executable).resolve()
    native = root/'native/string_triclinic_v1'
    if (executable.parent!=native or not re.fullmatch(r'GPU_string_triclinic(?:_[A-Za-z0-9_]+)?', executable.name)
            or not executable.is_file() or not os.access(executable, os.X_OK)):
        raise ValueError('parent-provided actual/versioned executable under native/string_triclinic_v1 required')
    raw = Path(report_path).read_bytes()
    gate = json.loads(raw)
    checks = gate.get('checks', {})
    contract = gate.get('output_contract', {})
    if (gate.get('passed') is not True or gate.get('executable_sha256')!=file_hash(executable)
            or checks.get('actual_cuda_S0') is not True or checks.get('actual_cuda_gradient') is not True
            or contract.get('path_policy')!='final_converged_string'
            or contract.get('info_marker')!='info:' or contract.get('check_interval')!=200
            or 'step_log' not in contract or not gate.get('source_hashes')):
        raise ValueError('actual CUDA S0/gradient, final-output and code/binary validation gate failed')
    for name, expected in gate['source_hashes'].items():
        source = local_path(native, name)
        if not source.is_file() or file_hash(source)!=expected:
            raise ValueError('validated native source changed: '+name)
    log = contract['step_log']
    if log is not None:
        if not isinstance(log, dict) or not all(k in log for k in ('filename', 'iteration', 'coordinate', 'orientation')):
            raise ValueError('invalid required step-log contract')
        local_path(native, log['filename'])
    evidence = {}
    for name in ('kernel_report','solver_smoke_report'):
        if name in gate:
            if file_hash(gate[name])!=gate.get(name+'_sha256'):
                raise ValueError('validation evidence changed: '+name)
            evidence[name] = dict(path=gate[name],sha256=gate[name+'_sha256'])
    return gate, dict(validation_report=str(Path(report_path).resolve()), validation_sha256=digest(raw),
                      executable=str(executable), executable_sha256=gate['executable_sha256'],
                      source_hashes=gate['source_hashes'], evidence=evidence,
                      all_gradient_cases_strict=gate.get('all_gradient_cases_strict'),
                      strict_gradient_by_case=gate.get('strict_gradient_by_case'),
                      conditioning_warning=gate.get('conditioning_warning'))


def load_cohort(root):
    root = Path(root)
    files = dict(manifest='runs/v2/manifest.csv', descriptors='runs/v3/input_audit_full_v2/structures.json',
                 labels='inputs/uncapped_saved_path_labels_v1/directions.csv',
                 label_report='inputs/uncapped_saved_path_labels_v1/report.json',
                 sources='inputs/full_pose_v4_v1/sources.json', native_report='inputs/path_environments_v2/report.json')
    raw = {key: (root/path).read_bytes() for key, path in files.items()}
    manifest = list(csv.DictReader(io.StringIO(raw['manifest'].decode())))
    rows = list(csv.DictReader(io.StringIO(raw['labels'].decode())))
    names = {r['name'] for r in manifest}
    descriptors = json.loads(raw['descriptors'])
    label_report, native = json.loads(raw['label_report']), json.loads(raw['native_report'])
    sources = json.loads(raw['sources'])
    if (len(manifest)!=8948 or len(names)!=8948 or len(rows)!=53688
            or label_report.get('passed') is not True or native.get('passed') is not True
            or label_report.get('structures')!=8948 or label_report.get('directions')!=53688
            or label_report['source_sha256']!=digest(raw['sources'])
            or label_report['path_report_sha256']!=digest(raw['native_report'])
            or native['native_sources_sha256']!=digest(raw['sources'])
            or set(sources)!=names or set(native['records'])!=names
            or {r['name'] for r in rows}!=names or {r['name'] for r in descriptors}!=names):
        raise ValueError('current cohort/source/provisional report gate failed')
    for row in rows:
        if row['gas'] not in GASES or row['direction'] not in ('a','b','c'):
            raise ValueError('invalid reference direction')
        reference = native['records'][row['name']]['sources'][row['gas']]['abc'.index(row['direction'])]
        if any(row[k]!=reference[k] for k in ('input_sha256', 'path_sha256')):
            raise ValueError('reference direction hash binding failed')
    binding = {k: dict(path=files[k], sha256=digest(v)) for k,v in raw.items()}
    return rows, descriptors, sources, binding


def prepare(root, output, seed=42):
    rows, descriptors, sources, binding = load_cohort(root)
    selected, selection = select_materials(rows, descriptors, seed)
    references = {(r['name'],r['gas'],r['direction']): r for r in rows}
    output = new_output(root, output)
    cases = []
    for material in selected:
        name, ident = material['name'], material['material_id']
        for gas in GASES:
            for direction in (1, 2, 3):
                entry = next(e for e in sources[name] if e['gas']==gas and e['direction']==direction)
                original = Path(entry['source'])
                if original.name!=f'dir{direction}' or original.parent.name!=name:
                    raise ValueError('original material/direction source path mismatch')
                ref = references[name, gas, 'abc'[direction-1]]
                raw = (original/'input.dat').read_bytes()
                path_raw = (original/'string_path.dat').read_bytes()
                if digest(raw)!=ref['input_sha256'] or digest(path_raw)!=ref['path_sha256']:
                    raise ValueError('original input/path changed')
                source_case, controls = input_case(raw)
                if controls['direction']!=direction or len(source_case['frame_frac'])!=material['atoms']:
                    raise ValueError('source direction/atom count differs from selection descriptor')
                gram = source_case['cell'] @ source_case['cell'].T
                skew = not np.allclose(gram-np.diag(np.diag(gram)), 0, atol=1e-4, rtol=0)
                if skew!=material['nonorthogonal']:
                    raise ValueError('source geometry class differs from frozen selection')
                original_stdout_path = original/'string.stdout'
                original_stderr_path = original/'string.stderr'
                old_stdout = original_stdout_path.read_text() if original_stdout_path.is_file() else ''
                old_stderr = original_stderr_path.read_text() if original_stderr_path.is_file() else ''
                original_convergence = convergence(old_stdout, old_stderr, 0, controls['running_steps'])
                reference = dict(input_path=str(original/'input.dat'), input_sha256=digest(raw),
                    saved_path=str(original/'string_path.dat'), path_sha256=digest(path_raw),
                    original_stdout=str(original_stdout_path), stdout_sha256=digest(old_stdout.encode()),
                    original_convergence=original_convergence, qc_status=ref['qc_status'],
                    stored_raw_logD=float(ref['old_logD']),
                    stored_raw_D_tag='original read-only historical utility value; never recomputed/overwritten',
                    saved_path_uncapped_logD=float(ref['uncapped_logD_given_saved_path']),
                    saved_path_barrier_K=float(ref['raw_barrier_K']),
                    uncapped_D_tag='original provisional saved-path TST, not material ground truth',
                    reference_valid_for_comparison=original_convergence['converged'])
                for arm in ARMS:
                    content = arm_input(raw, gas, arm)
                    case, settings = input_case(content)
                    relative = f'inputs/{ident:02d}_{gas}_{direction}_{arm}.dat'
                    target = local_path(output, relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as handle:
                        handle.write(content)
                    metadata = dict(material_id=ident, name=name, gas=gas, direction=direction, arm=arm,
                        input_relative=relative, input_sha256=digest(content), reference=reference,
                        semantics=semantics(case, settings),
                        treatment='no source bytes changed' if arm=='current' else 'ONLY guest sigma/epsilon changed; not strict original old full FF',
                        actual_path_used_as_predictor_or_initial_input=False, cold_start=True)
                    meta_relative = relative+'.json'
                    write_json(local_path(output, meta_relative), metadata)
                    cases.append(dict(**metadata, metadata_relative=meta_relative,
                                      metadata_sha256=file_hash(local_path(output, meta_relative))))
    plan = dict(prepared=True, materials=selected, selection=selection, cases=cases,
                source_binding=binding, code_hashes=recipe_hashes(root), outputs=120,
                submits_jobs=False, original_binary_executed=False, original_data_modified=False)
    write_json(output/'plan.json', plan)
    print(json.dumps(dict(prepared=str(output), materials=selected, selection=selection, inputs=len(cases))), flush=True)
    return plan


def read_plan(root, directory):
    directory = Path(directory).resolve()
    if (Path(root)/PREFIX).resolve() not in directory.parents:
        raise ValueError('prepared artifacts outside dedicated project prefix')
    raw = (directory/'plan.json').read_bytes()
    plan = json.loads(raw)
    keys = {(r['material_id'],r['gas'],r['direction'],r['arm']) for r in plan['cases']}
    expected = {(i,g,d,a) for i in range(10) for g in GASES for d in (1,2,3) for a in ARMS}
    if (not plan.get('prepared') or keys!=expected or len(plan['cases'])!=120
            or [r['material_id'] for r in plan['materials']]!=list(range(10))
            or len({r['name'] for r in plan['materials']})!=10 or plan['code_hashes']!=recipe_hashes(root)):
        raise ValueError('prepared plan recipe/coverage invalid')
    for entry in plan['source_binding'].values():
        if file_hash(Path(root)/entry['path'])!=entry['sha256']:
            raise ValueError('cohort/provenance input changed since preparation')
    return plan, digest(raw)


def gpu_process_memory(token):
    proc = subprocess.run(['nvidia-smi','-i',token,'--query-compute-apps=pid,used_gpu_memory',
                           '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True)
    values = {}
    for row in csv.reader(io.StringIO(proc.stdout.strip())):
        if not row:
            continue
        if len(row)!=2:
            raise ValueError('invalid per-PID GPU memory query')
        pid, used = int(row[0].strip()),float(row[1].strip())
        if used<0 or not math.isfinite(used):
            raise ValueError('unavailable per-PID GPU memory')
        values[pid] = used
    return values


def process_rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return float(line.split()[1])/1024
    except OSError:
        pass
    return 0.


def native_process(executable, work, token, timeout, reserve, *, tracker=None, slot=None):
    started = time.monotonic()
    samples, abort = [], None
    with (work/'native.stdout').open('x') as stdout, (work/'native.stderr').open('x') as stderr:
        proc = subprocess.Popen([str(executable), 'input.dat', 'string_path.dat'], cwd=work,
                                stdout=stdout, stderr=stderr, env=os.environ.copy())
        if tracker is not None:
            with tracker['lock']:
                tracker['slots'][slot]['pid'] = proc.pid
        try:
            while proc.poll() is None:
                if time.monotonic()-started>timeout:
                    abort = 'wall_timeout'
                    break
                try:
                    card = gpu_query(token)
                    rss = process_rss(proc.pid)
                    sample = dict(seconds=time.monotonic()-started,host_rss_MiB=rss,**card)
                    if tracker is not None:
                        allocated = gpu_process_memory(token).get(proc.pid,0.)
                        sample['pid_gpu_MiB'] = allocated
                        with tracker['lock']:
                            tracker['slots'][slot].update(allocated_gpu_MiB=allocated,host_rss_MiB=rss)
                    samples.append(sample)
                    if card['free_MiB']<reserve:
                        abort = 'gpu_memory_reserve_abort'
                        break
                except (OSError, subprocess.SubprocessError, ValueError):
                    abort = 'gpu_memory_monitor_failed'
                    break
                time.sleep(2)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    return dict(returncode=proc.returncode,pid=proc.pid,abort=abort,seconds=time.monotonic()-started,gpu_samples=samples,
                observed_card_peak_MiB=max((s['used_MiB'] for s in samples),default=None),
                observed_pid_peak_MiB=max((s.get('pid_gpu_MiB',0) for s in samples),default=0),
                observed_host_rss_peak_MiB=max((s['host_rss_MiB'] for s in samples),default=0),
                memory_sampling_is_upper_bound=False)


def paired_tasks(tasks,rank):
    return sorted((r for r in tasks if r['material_id'] % 2==rank),
                  key=lambda r:(r['material_id'],GASES.index(r['gas']),r['direction'],ARMS.index(r['arm'])))


def valid_pair_count(rows):
    grouped = defaultdict(set)
    for row in rows:
        if row.get('status')=='valid' and row.get('logD') is not None and math.isfinite(row['logD']):
            grouped[row['material_id'],row['gas'],row['direction']].add(row['arm'])
    return sum(set(ARMS)<=arms for arms in grouped.values())


def load_memory_profile(path,binding):
    report = json.loads(Path(path).read_text())
    process = report.get('process',{})
    samples = process.get('gpu_samples',[])
    if (report.get('passed') is not True or report.get('actual_full_program') is not True
            or report['native_binding']['executable_sha256']!=binding['executable_sha256']
            or report['native_binding']['source_hashes']!=binding['source_hashes']
            or process.get('returncode')!=0 or process.get('abort') is not None or not samples):
        raise ValueError('matching passed actual full-solver memory profiling required')
    increment = max(s['used_MiB'] for s in samples)-report['gpu_before']['used_MiB']
    if increment<=0:
        raise ValueError('profile did not observe GPU allocations')
    return dict(report=str(Path(path).resolve()),sha256=file_hash(path),observed_increment_MiB=increment,
                observed_card_peak_MiB=max(s['used_MiB'] for s in samples),
                framework_atoms=report['task']['semantics']['framework_atoms'],safety_factor=2.,
                warning='Sampled single low-energy resident GPU usage is not an upper bound or total unified-memory footprint.')


def task_memory_estimate(task,profile):
    ratio = max(1.,task['semantics']['framework_atoms']/profile['framework_atoms'])
    gpu = max(1024,math.ceil(profile['safety_factor']*profile['observed_increment_MiB']*ratio+256))
    return dict(estimated_gpu_MiB=gpu,estimated_host_MiB=max(4096,2*gpu),
                allocated_gpu_MiB=0.,host_rss_MiB=0.,pid=None,
                estimate_not_proven_bound=True)


def memory_admission(free,active,candidate,reserve,max_concurrent,host_budget):
    future_gpu = sum(max(0.,r['estimated_gpu_MiB']-r.get('allocated_gpu_MiB',0.)) for r in active)
    host = sum(max(r['estimated_host_MiB'],r.get('host_rss_MiB',0.)) for r in active)
    return (len(active)<max_concurrent and free>=reserve+candidate['estimated_gpu_MiB']+future_gpu
            and host+candidate['estimated_host_MiB']+8192<=host_budget)


def host_available_mib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return float(line.split()[1])/1024
    raise ValueError('host available memory cannot be verified')


def atomic_progress(path,value):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def execute_case(task,root,prepared,output,executable,validation,plan,codes,binding,gate,
                 token,timeout,reserve,tracker,slot):
    work = output/f"{task['material_id']:02d}_{task['gas']}_{task['direction']}_{task['arm']}"
    work.mkdir(exist_ok=False)
    record = dict(material_id=task['material_id'],name=task['name'],gas=task['gas'],direction=task['direction'],
        arm=task['arm'],reference=task['reference'],work_relative=work.name,status='unknown',logD=None,
        D_m2_s=None,path_barrier_K=None,native_binding=binding,input_sha256=task['input_sha256'])
    try:
        _,current_binding = validation_gate(root,executable,validation)
        if current_binding!=binding or recipe_hashes(root)!=codes:
            raise ValueError('native/runner recipe changed during shard')
        path,meta = (local_path(prepared,task[key]) for key in ('input_relative','metadata_relative'))
        if file_hash(path)!=task['input_sha256'] or file_hash(meta)!=task['metadata_sha256']:
            raise ValueError('prepared input/semantics changed')
        reference = task['reference']
        raw = Path(reference['input_path']).read_bytes()
        if digest(raw)!=reference['input_sha256'] or file_hash(reference['saved_path'])!=reference['path_sha256']:
            raise ValueError('original source/path changed')
        content = path.read_bytes()
        if arm_input(raw,task['gas'],task['arm'])!=content:
            raise ValueError('not the exact controlled transformation')
        case,_ = input_case(content)
        with (work/'input.dat').open('xb') as handle:
            handle.write(content)
        write_json(work/'input.json',task)
        process = native_process(Path(executable).resolve(),work,token,timeout,reserve,tracker=tracker,slot=slot)
        record['process'] = process
        stdout,stderr = (work/'native.stdout').read_text(),(work/'native.stderr').read_text()
        produced = work/'string_path.dat'
        array = np.loadtxt(produced) if produced.is_file() else np.empty((0,7))
        contract = gate['output_contract']
        step_text = None
        if contract['step_log'] is not None:
            log = local_path(work,contract['step_log']['filename'])
            step_text = log.read_text() if log.is_file() else ''
        record.update(score_result(case,array,stdout,stderr,
            process['returncode'] if not process['abort'] else -1,task['direction'],contract,step_text))
        if process['abort']:
            record['flags'].append(process['abort'])
        if produced.is_file():
            record.update(output_path=work.name+'/string_path.dat',output_path_sha256=file_hash(produced))
        _,after = validation_gate(root,executable,validation)
        if (after!=binding or file_hash(work/'input.dat')!=task['input_sha256']
                or file_hash(reference['input_path'])!=reference['input_sha256']
                or file_hash(reference['saved_path'])!=reference['path_sha256']):
            raise ValueError('binary/source/input changed during execution')
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        record.update(status='failed_or_unknown',logD=None,D_m2_s=None,path_barrier_K=None,
            raw_min_energy_K=None,raw_max_energy_K=None,flags=['runner_guard_or_output_error: '+str(exc)])
    write_json(work/'outcome.json',record)
    return record


def maybe_early(root,prepared,parent,plan_sha):
    claim = Path(parent)/('early_claim_'+plan_sha[:16]+'.json')
    if claim.exists():
        return
    directories = []
    reports = []
    for path in sorted(Path(parent).glob('rank*/progress.json')):
        report = json.loads(path.read_text())
        if report.get('plan_sha256')==plan_sha:
            directories.append(path.parent)
            reports.append(report)
    if not reports or valid_pair_count([r for report in reports for r in report['outcomes']])<10:
        return
    try:
        with claim.open('x') as handle:
            json.dump(dict(pid=os.getpid(),shards=list(map(str,directories))),handle)
    except FileExistsError:
        return
    target = Path(parent)/f'early_comparison_{plan_sha[:12]}_{time.time_ns()}'
    try:
        collect(root,prepared,directories,target,early=True)
    except (OSError,ValueError,KeyError) as exc:
        write_json(Path(parent)/f'early_failure_{plan_sha[:12]}_{time.time_ns()}.json',dict(error=str(exc)))


def run(root, prepared, output, rank, executable, validation, timeout=3600, required_free=20000, reserve=4096,
        memory_profile=None,max_concurrent=8,stagger=4.,host_budget=49152):
    if rank not in (0, 1) or 'SLURM_JOB_ID' not in os.environ:
        raise ValueError('two-shard Slurm GPU allocation required')
    host = socket.gethostname().split('.')[0]
    expected_host = ('gpu', 'gpu2')[rank]
    if host!=expected_host:
        raise ValueError('rank '+str(rank)+' must run on '+expected_host+', not '+host)
    token = assigned_gpu(os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    if int(os.environ.get('SLURM_GPUS_ON_NODE', '1'))!=1:
        raise ValueError('exactly one allocated GPU per job required')
    gate, native_binding = validation_gate(root, executable, validation)
    plan, plan_sha = read_plan(root, prepared)
    if memory_profile is None or not 1<=max_concurrent<=16 or stagger<1 or not 2048<=reserve<=4096:
        raise ValueError('measured memory profile, max concurrency 1..16, stagger>=1s, reserve 2..4 GiB required')
    profile = load_memory_profile(memory_profile,native_binding)
    host_budget = min(host_budget,int(os.environ.get('SLURM_MEM_PER_NODE',host_budget)))
    if not gpu_safe(gpu_query(token),required_free,reserve):
        raise ValueError('insufficient initial free GPU memory')
    output = new_output(root, output)
    selected = plan['materials'][rank::2]
    write_json(output/'run_context.json', dict(rank=rank, host=host, gpu_token=token,
               material_ids=[r['material_id'] for r in selected], selected=selected,
               prepared=str(Path(prepared).resolve()), plan_sha256=plan_sha, native_binding=native_binding,
               max_string_processes_per_card=max_concurrent,required_free_MiB=required_free,reserve_MiB=reserve,
               memory_profile=profile,host_budget_MiB=host_budget,launch_stagger_seconds=stagger,
               timeout_seconds=timeout,node_local_scratch_used=False))
    outcomes,active = [],{}
    tracker = dict(lock=threading.Lock(),slots={})
    queue = paired_tasks(plan['cases'],rank)
    last_launch = -float('inf')
    stalled = None
    def publish(completed=False):
        report = dict(completed=completed,rank=rank,materials=selected,plan_sha256=plan_sha,native_binding=native_binding,
            outcomes=outcomes,status_counts=dict(Counter(r['status'] for r in outcomes)),valid_two_arm_pairs=valid_pair_count(outcomes),
            memory_profile=profile,max_concurrent=max_concurrent,total_shard_cases=60,total_experiment_cases=120,
            conditioning_warning=native_binding.get('conditioning_warning'))
        atomic_progress(output/'progress.json',report)
        return report
    publish()
    with ThreadPoolExecutor(max_workers=max_concurrent) as pool, ThreadPoolExecutor(max_workers=1) as comparisons:
        while queue or active:
            for future,slot in list(active.items()):
                if future.done():
                    record = future.result()
                    outcomes.append(record)
                    del active[future]
                    with tracker['lock']:
                        del tracker['slots'][slot]
                    publish()
                    print(json.dumps({k:record.get(k) for k in ('material_id','gas','direction','arm','status','logD')}),flush=True)
                    comparisons.submit(maybe_early,root,prepared,output.parent,plan_sha)
            if queue and time.monotonic()-last_launch>=stagger:
                card = gpu_query(token)
                allocated = gpu_process_memory(token)
                candidate = task_memory_estimate(queue[0],profile)
                with tracker['lock']:
                    for slot in tracker['slots'].values():
                        slot['allocated_gpu_MiB'] = allocated.get(slot.get('pid'),0.)
                        if slot.get('pid'):
                            slot['host_rss_MiB'] = process_rss(slot['pid'])
                    snapshots = [dict(r) for r in tracker['slots'].values()]
                    allowed = memory_admission(card['free_MiB'],snapshots,candidate,reserve,max_concurrent,
                                               min(host_budget,host_available_mib()))
                    if allowed:
                        task = queue.pop(0)
                        slot_id = task['input_relative']
                        tracker['slots'][slot_id] = candidate
                if allowed:
                    future = pool.submit(execute_case,task,root,prepared,output,executable,validation,plan,
                        plan['code_hashes'],native_binding,gate,token,timeout,reserve,tracker,slot_id)
                    active[future] = slot_id
                    last_launch = time.monotonic()
                    stalled = None
                elif not active:
                    stalled = stalled or time.monotonic()
                    if time.monotonic()-stalled>timeout:
                        raise RuntimeError('memory budget cannot admit next case; evidence retained, no oversized launch')
            if queue or active:
                time.sleep(.5)
    if len(outcomes)!=60:
        raise ValueError('shard coverage mismatch')
    report = publish(completed=True)
    write_json(output/'report.json', report)
    return report


def summarize_pairs(pairs):
    if not pairs:
        return dict(n=0, agreement_R2_logD=None)
    x = np.array([p['reference_logD'] for p in pairs])
    y = np.array([p['new_logD'] for p in pairs])
    denominator = float(((x-x.mean())**2).sum())
    return dict(n=len(pairs), agreement_R2_logD=1-float(((y-x)**2).sum())/denominator if denominator>0 and len(x)>1 else None,
                mae_logD=float(np.abs(y-x).mean()), median_absolute_delta_logD=float(np.median(np.abs(y-x))),
                median_absolute_fold_ratio=pair_values(dict(status='valid',logD=float(np.median(np.abs(y-x)))),
                                                      dict(status='valid',logD=0.))['absolute_fold_ratio'],
                R2_semantics='paired run/reference agreement, not ML prediction or generalization')


def periodic_vectors(fractional_delta, cell):
    """Exact minimum-image Cartesian vectors in a bounded triclinic search."""
    import itertools
    cell = np.asarray(cell,dtype=float)
    delta = np.asarray(fractional_delta,dtype=float)
    if (cell.shape!=(3,3) or not np.isfinite(cell).all() or np.linalg.det(cell)<=0
            or delta.shape[-1]!=3 or not np.isfinite(delta).all()):
        raise ValueError('invalid periodic comparison geometry')
    shape = delta.shape
    q = (delta-np.rint(delta)).reshape(-1,3)
    best = q @ cell
    best2 = np.einsum('ij,ij->i',best,best)
    def consider(shift):
        vector = (q-shift) @ cell
        norm2 = np.einsum('ij,ij->i',vector,vector)
        better = norm2<best2
        best[better] = vector[better]
        best2[better] = norm2[better]
    for shift in itertools.product((-1,0,1),repeat=3):
        consider(np.array(shift))
    # If |(q-n)C| <= R, then |q_i-n_i| <= R ||C^-1[:,i]||.
    # This bounds every potentially better integer image, unlike component wrapping.
    extent = np.ceil(.5+np.sqrt(best2.max())*np.linalg.norm(np.linalg.inv(cell),axis=0)).astype(int)
    if np.prod(2*extent+1)>20000:
        raise ValueError('cell too ill-conditioned for bounded exact path comparison')
    for shift in itertools.product(*(range(-n,n+1) for n in extent)):
        if max(map(abs,shift))>1:
            consider(np.array(shift))
    return best.reshape(shape)


def path_sampler(array, cell):
    p = np.asarray(array,dtype=float)
    if p.ndim!=2 or p.shape[1]!=7 or len(p)<3 or not np.isfinite(p).all():
        raise ValueError('finite native Nx7 path required for comparison')
    if np.linalg.norm(periodic_vectors(p[-1,:3]-p[0,:3],cell))>.05:
        raise ValueError('path is not a closed periodic channel')
    steps = periodic_vectors(np.diff(p[:,:3],axis=0),cell)
    lengths = np.linalg.norm(steps,axis=1)
    arc = np.r_[0.,np.cumsum(lengths)]
    if arc[-1]<=1e-10:
        raise ValueError('no translational path length')
    unwrapped = p[0,:3]+np.vstack([np.zeros(3),np.cumsum(steps @ np.linalg.inv(cell),axis=0)])
    beta,gamma = p[:,4],p[:,5]
    directors = np.column_stack([np.cos(gamma)*np.cos(beta),np.sin(gamma)*np.cos(beta),-np.sin(beta)])
    for i in range(1,len(directors)):
        if np.dot(directors[i],directors[i-1])<0:
            directors[i] *= -1
    keep = np.r_[True,np.diff(arc)>1e-12]
    parameter = arc[keep]/arc[-1]
    winding = unwrapped[-1]-unwrapped[0]
    def sample(u):
        u = np.asarray(u,dtype=float)
        wrapped = np.mod(u,1.)
        xyz = np.column_stack([np.interp(wrapped,parameter,unwrapped[keep,j]) for j in range(3)])
        xyz += np.floor(u)[:,None]*winding
        director = np.column_stack([np.interp(wrapped,parameter,directors[keep,j]) for j in range(3)])
        norm = np.linalg.norm(director,axis=1)
        if (norm<1e-8).any():
            raise ValueError('ambiguous interpolated guest director')
        energy = np.interp(wrapped,parameter,p[keep,6])
        return xyz,director/norm[:,None],energy
    return sample,dict(arclength_A=float(arc[-1]), zero_translation_segments=int((lengths<=1e-12).sum()),
                       energy_min_K=float(p[:,6].min()),energy_max_K=float(p[:,6].max()))


def compare_paths(new, reference, cell, samples=128):
    """Align by periodic channel geometry only; never fit energies or translate the cell."""
    from scipy.optimize import minimize_scalar
    if not 16<=samples<=512:
        raise ValueError('bounded 16..512 comparison samples required')
    new_sample,new_info = path_sampler(new,cell)
    ref_sample,ref_info = path_sampler(reference,cell)
    u = np.arange(samples)/samples
    ref_xyz,ref_axis,ref_energy = ref_sample(u)
    new_xyz,_,_ = new_sample(u)
    distance = periodic_vectors(new_xyz[:,None,:]-ref_xyz[None,:,:],cell)
    matrix = np.einsum('ijk,ijk->ij',distance,distance)
    candidates = []
    indices = np.arange(samples)
    for reversed_path in (False,True):
        sign = -1 if reversed_path else 1
        costs = [float(matrix[(sign*indices+k)%samples,indices].mean()) for k in range(samples)]
        coarse = int(np.argmin(costs))
        def objective(phase):
            xyz,_,_ = new_sample(sign*u+phase)
            v = periodic_vectors(xyz-ref_xyz,cell)
            return float(np.einsum('ij,ij->i',v,v).mean())
        optimum = minimize_scalar(objective,bounds=((coarse-1)/samples,(coarse+1)/samples),
                                   method='bounded',options={'xatol':1e-12})
        for phase in (coarse/samples,float(optimum.x)):
            candidates.append((objective(phase),reversed_path,phase))
    _,reversed_path,phase = min(candidates,key=lambda x:x[0])
    xyz,axis,energy = new_sample((-u if reversed_path else u)+phase)
    aligned = np.linalg.norm(periodic_vectors(xyz-ref_xyz,cell),axis=1)
    distances = np.linalg.norm(periodic_vectors(xyz[:,None,:]-ref_xyz[None,:,:],cell),axis=2)
    chamfer = np.r_[distances.min(0),distances.min(1)]
    cosine = np.clip(np.abs(np.einsum('ij,ij->i',axis,ref_axis)),0,1)
    angles = np.degrees(np.arccos(cosine))
    raw_delta = energy-ref_energy
    relative_delta = (energy-new_info['energy_min_K'])-(ref_energy-ref_info['energy_min_K'])
    return dict(path_comparison_status='valid',alignment_policy='geometry_only_periodic_cyclic_arclength',
        comparison_samples=samples,alignment_shift_fraction=float(phase % 1),alignment_reversed=reversed_path,
        arbitrary_rigid_translation_fitted=False, framework_or_cell_rotated=False,
        periodic_COM_rmsd_A=float(np.sqrt(np.mean(aligned**2))),
        periodic_COM_mean_A=float(aligned.mean()),periodic_COM_p95_A=float(np.quantile(aligned,.95)),
        periodic_COM_max_A=float(aligned.max()),periodic_chamfer_mean_A=float(chamfer.mean()),
        periodic_chamfer_p95_A=float(np.quantile(chamfer,.95)),
        reference_path_length_A=ref_info['arclength_A'],new_path_length_A=new_info['arclength_A'],
        path_length_ratio=new_info['arclength_A']/ref_info['arclength_A'],
        director_angle_mean_deg=float(angles.mean()),director_angle_p95_deg=float(np.quantile(angles,.95)),
        director_semantics='native radians -> body-X director, sign equivalent for symmetric two-site guests',
        raw_energy_profile_rmse_K=float(np.sqrt(np.mean(raw_delta**2))),
        raw_energy_profile_mae_K=float(np.mean(np.abs(raw_delta))),
        relative_energy_profile_rmse_K=float(np.sqrt(np.mean(relative_delta**2))),
        relative_energy_profile_mae_K=float(np.mean(np.abs(relative_delta))),
        minimum_energy_delta_K=new_info['energy_min_K']-ref_info['energy_min_K'],
        maximum_energy_delta_K=new_info['energy_max_K']-ref_info['energy_max_K'],
        barrier_delta_K=(new_info['energy_max_K']-new_info['energy_min_K'])-
                        (ref_info['energy_max_K']-ref_info['energy_min_K']),
        new_zero_translation_segments=new_info['zero_translation_segments'],
        reference_zero_translation_segments=ref_info['zero_translation_segments'],
        profile_semantics='native output energies linearly resampled on normalized translational arclength; relative profiles subtract each raw minimum',
        caveat='Finite sampling and translational resampling may miss localized/stationary rotations; raw extrema/barriers remain authoritative.')


def collect(root, prepared, shards, output, *, early=False,min_pairs=10):
    plan, plan_sha = read_plan(root, prepared)
    filename = 'progress.json' if early else 'report.json'
    reports = [json.loads((Path(p)/filename).read_text()) for p in shards]
    ranks = [r['rank'] for r in reports]
    if (not reports or len(set(ranks))!=len(ranks) or not set(ranks)<={0,1}
            or any(r['plan_sha256']!=plan_sha for r in reports)
            or (not early and (set(ranks)!={0,1} or any(not r['completed'] for r in reports)))):
        raise ValueError('completed matching shards required for final, matching partial progress for early')
    if any(r['native_binding']!=reports[0]['native_binding'] for r in reports):
        raise ValueError('GPU shards used different validated executable/source recipes')
    indexed, artifact_directories = {}, {}
    for directory, report in zip(shards, reports):
        for row in report['outcomes']:
            key = row['material_id'],row['gas'],row['direction'],row['arm']
            if key in indexed or row['material_id'] % 2!=report['rank']:
                raise ValueError('duplicate/wrong-shard case')
            evidence = local_path(directory, row['work_relative'])
            if json.loads((evidence/'outcome.json').read_text())!=row:
                raise ValueError('outcome/report evidence mismatch')
            if row.get('output_path') and file_hash(local_path(directory, row['output_path']))!=row['output_path_sha256']:
                raise ValueError('new output path changed')
            indexed[key] = row
            artifact_directories[key] = Path(directory)
    expected = {(r['material_id'],r['gas'],r['direction'],r['arm']) for r in plan['cases']}
    if not set(indexed)<=expected or (not early and set(indexed)!=expected):
        raise ValueError('missing cases in paired report')
    selected_pair_keys = {(i,g,d) for i,g,d,a in indexed}
    if early:
        ready = []
        for key in sorted(selected_pair_keys):
            pair = [indexed.get((*key,a),{}) for a in ARMS]
            if valid_pair_count(pair)==1:
                ready.append(key)
        if len(ready)<min_pairs:
            raise ValueError('early report needs ten COMPLETE valid two-arm pairs, not ten single arms')
        selected_pair_keys = set(ready[:min_pairs])
    comparisons = []
    tasks = {(r['material_id'],r['gas'],r['direction'],r['arm']):r for r in plan['cases']}
    for ident in range(10):
        for gas in GASES:
            for direction in (1,2,3):
                if (ident,gas,direction) not in selected_pair_keys:
                    continue
                baseline, old = (indexed[ident,gas,direction,a] for a in ARMS)
                ref = baseline['reference']
                original = dict(status='valid' if ref['reference_valid_for_comparison'] else 'unknown_original_convergence',
                                logD=ref['saved_path_uncapped_logD'], path_barrier_K=ref['saved_path_barrier_K'])
                for label, new, reference in [('current_vs_original_saved',baseline,original),
                                              ('old_sigma_epsilon_vs_current',old,baseline)]:
                    values = pair_values(new,reference)
                    diagnostics = dict(path_comparison_status='not_evaluated_invalid_pair')
                    if values is not None:
                        try:
                            new_key = ident,gas,direction,new['arm']
                            new_path = local_path(artifact_directories[new_key],new['output_path'])
                            if label=='current_vs_original_saved':
                                reference_path = Path(ref['saved_path'])
                                if file_hash(reference_path)!=ref['path_sha256']:
                                    raise ValueError('original saved-path source changed')
                            else:
                                reference_key = ident,gas,direction,baseline['arm']
                                reference_path = local_path(artifact_directories[reference_key],baseline['output_path'])
                            case_task = tasks[ident,gas,direction,new['arm']]
                            case_input = local_path(prepared,case_task['input_relative'])
                            if file_hash(case_input)!=case_task['input_sha256']:
                                raise ValueError('prepared comparison cell/input changed')
                            cell = input_case(case_input.read_bytes())[0]['cell']
                            diagnostics = compare_paths(np.loadtxt(new_path),np.loadtxt(reference_path),cell)
                        except (OSError,ValueError,KeyError) as exc:
                            diagnostics = dict(path_comparison_status='unavailable',path_comparison_reason=str(exc))
                    comparisons.append(dict(material_id=ident,name=baseline['name'],gas=gas,direction=direction,
                        comparison=label, paired_valid=values is not None, new_status=new['status'],
                        reference_status=reference['status'], **(values or {}), **diagnostics))
    summaries = {}
    for label in ('current_vs_original_saved','old_sigma_epsilon_vs_current'):
        selected = [p for p in comparisons if p['comparison']==label and p['paired_valid']]
        path_pairs = [p for p in selected if p['path_comparison_status']=='valid']
        summaries[label] = dict(overall=summarize_pairs(selected), gas_direction={g+'_'+'abc'[d-1]:
            summarize_pairs([p for p in selected if p['gas']==g and p['direction']==d]) for g in GASES for d in (1,2,3)},
            path_differences=dict(n=len(path_pairs),median_periodic_COM_rmsd_A=float(np.median(
                [p['periodic_COM_rmsd_A'] for p in path_pairs])) if path_pairs else None,
                median_relative_energy_profile_rmse_K=float(np.median(
                [p['relative_energy_profile_rmse_K'] for p in path_pairs])) if path_pairs else None))
    output = new_output(root, output)
    write_json(output/'report.json', dict(completed=True,full_experiment_completed=not early,early_snapshot=early,
        selected_pair_keys=len(selected_pair_keys),
        selected_valid_two_arm_pairs=sum(valid_pair_count([indexed.get((*key,a),{}) for a in ARMS])
                                         for key in selected_pair_keys),
        valid_paths_in_selected_pairs=sum(indexed.get((*key,a),{}).get('status')=='valid'
                                          for key in selected_pair_keys for a in ARMS),
        actual_observed_materials=len({k[0] for k in selected_pair_keys}),planned_materials=10,
        materials=plan['materials'], selection=plan['selection'],
        plan_sha256=plan_sha, native_binding=reports[0]['native_binding'], summaries=summaries,
        status_counts=dict(Counter(r['status'] for r in indexed.values())), comparisons=comparisons,
        limitations=['Ten selected provisional materials are not representative performance evidence.',
                     'R2 is descriptive paired logD agreement, not ML R2 or a promise of better D prediction.',
                     'Both new arms share corrected PBC; the old arm changes sigma/epsilon only, not full old FF.',
                     'Original saved paths used historical initialization/convergence/export logic; comparison is not proof of a sole PBC cause.',
                     'Convergence does not certify a globally optimal path or physical material diffusivity.',
                     'Path comparisons use exact triclinic minimum images and geometry-only cyclic phase/reversal alignment; not unwrapped-coordinate subtraction.',
                     'Missing, failed, nonfinite, nonconverged or invalid-QC cases never become numeric zero.']))
    keys = sorted(set().union(*(set(r) for r in comparisons)))
    with (output/'comparisons.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys)
        writer.writeheader()
        writer.writerows(comparisons)
    print(json.dumps(summaries),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--mode', choices=('prepare','run','collect'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepared-dir', type=Path)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--executable', type=Path, default=ROOT/'native/string_triclinic_v1/GPU_string_triclinic_final')
    parser.add_argument('--validation-report', type=Path)
    parser.add_argument('--shards', nargs=2, type=Path)
    parser.add_argument('--early',action='store_true')
    parser.add_argument('--memory-profile',type=Path)
    parser.add_argument('--max-concurrent',type=int,default=8)
    parser.add_argument('--launch-stagger-seconds',type=float,default=4.)
    parser.add_argument('--host-budget-mib',type=int,default=49152)
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    parser.add_argument('--required-free-mib', type=int, default=20000)
    parser.add_argument('--reserve-mib', type=int, default=4096)
    args = parser.parse_args()
    if args.timeout_seconds<=0 or args.required_free_mib<args.reserve_mib or args.reserve_mib<1024:
        parser.error('invalid process timeout/memory guard')
    if args.mode=='prepare':
        prepare(args.root,args.output_dir)
    elif args.mode=='run':
        if args.prepared_dir is None or args.validation_report is None or args.rank not in (0,1):
            parser.error('run requires prepared-dir, validation-report and rank 0/1')
        run(args.root,args.prepared_dir,args.output_dir,args.rank,args.executable,args.validation_report,
            args.timeout_seconds,args.required_free_mib,args.reserve_mib,args.memory_profile,
            args.max_concurrent,args.launch_stagger_seconds,args.host_budget_mib)
    else:
        if args.prepared_dir is None or args.shards is None:
            parser.error('collect requires prepared-dir and two shards')
        collect(args.root,args.prepared_dir,args.shards,args.output_dir,early=args.early)


if __name__=='__main__':
    main()
