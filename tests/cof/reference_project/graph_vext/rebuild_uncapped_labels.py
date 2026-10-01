"""Versioned uncapped log-domain TST on saved paths; NOT globally optimal diffusion."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import json
import numpy as np
from scipy.special import logsumexp

from graph_vext.audit_native_path_cap import stable_logd
from graph_vext.dataset import GASES, load_manifest
from graph_vext.prepare_spatial_pose_targets import string_parameters
from graph_vext.string_atom_mapping import parse_string_input
from graph_vext.train_spatial_transport_v3 import json_write


def _validate(case, pose, direction):
    p=np.asarray(pose,dtype=np.float64);cell=np.asarray(case['cell'],dtype=np.float64)
    if (p.ndim!=2 or p.shape[1]!=7 or len(p)<2 or not np.isfinite(p).all()
            or cell.shape!=(3,3) or not np.isfinite(cell).all() or np.linalg.det(cell)<=0
            or case['temperature']<=0 or not math.isfinite(case['temperature'])
            or case['total_mass']<=0 or not math.isfinite(case['total_mass']) or direction not in (0,1,2)):
        raise ValueError('invalid saved-path TST inputs')
    distance=np.linalg.norm(np.diff(p[:,:3]@cell,axis=0),axis=1)*1e-10
    if not np.isfinite(distance).all() or distance.sum()<=0:
        raise ValueError('nonpositive/invalid translational arc length')
    return p,cell,distance


def path_logd(case, pose, direction):
    p,cell,distance=_validate(case,pose,direction)
    temperature=case['temperature'];energy=p[:,6]
    relative=(energy-energy.min())/temperature
    if not np.isfinite(relative).all():raise ValueError('energy range not representable')
    positive=distance>0
    terms=np.log(distance[positive])+np.logaddexp(-relative[:-1][positive],-relative[1:][positive])-np.log(2)
    integral=logsumexp(terms)
    mass=case['total_mass']/1000/6.02214076e23
    length=np.linalg.norm(cell[direction])*1e-10
    logpref=np.log(.5)+.5*np.log(1.38e-23*temperature/(2*np.pi*mass))+2*np.log(length)
    result=float((logpref-relative.max()-integral)/np.log(10))
    if not math.isfinite(result):raise ValueError('nonfinite uncapped logD')
    return result


def linear_value(logd):
    if not math.isfinite(logd):raise ValueError('nonfinite logD')
    if logd<math.log10(np.nextafter(0.,1.)) or logd>math.log10(np.finfo(float).max):return None
    value=10.**logd
    return float(value) if value>0 and math.isfinite(value) else None


def mean_logd(values):
    y=np.asarray(values,dtype=float)
    if y.shape!=(3,) or not np.isfinite(y).all():raise ValueError('three finite directional logD required')
    anchor=y.max()
    return float(anchor+np.log10(np.exp((y-anchor)*np.log(10)).sum())-np.log10(3))


def path_qc(case, pose, direction):
    p,cell,distance=_validate(case,pose,direction)
    expected=np.eye(3)[direction]
    winding=p[-1,:3]-p[0,:3]
    arc=distance.sum()/1e-10;length=np.linalg.norm(cell[direction])
    return dict(arc_length_A=float(arc),hop_length_A=float(length),arc_to_hop_ratio=float(arc/length),
                max_segment_A=float(distance.max()/1e-10),max_segment_to_mean=float(distance.max()/distance.mean()),
                endpoint_winding_error=float(np.linalg.norm(winding-expected)),
                winding_a=float(winding[0]),winding_b=float(winding[1]),winding_c=float(winding[2]))


def _direction_header(raw):
    lines=[line.strip() for line in raw.decode().splitlines()]
    direction=int(lines[lines.index('Direction')+1])
    header=next(i for i,line in enumerate(lines) if line.startswith('#_of_points'))
    points=int(lines[header+1].split()[0])
    return direction,points


def rebuild_case(task):
    root,name,entries,record,parameters,old,membership=task;root=Path(root)
    cache_raw=(root/'inputs/full_pose_v4_v1'/(name+'.npz')).read_bytes()
    if hashlib.sha256(cache_raw).hexdigest()!=record['cached_native_sha256']:
        raise ValueError('native cache changed: '+name)
    rows=[]
    with np.load(io.BytesIO(cache_raw)) as data:
        for gas in GASES:
            native=data['raw_pose_'+gas]
            if native.shape!=(1203,7):raise ValueError('native path coverage mismatch')
            for axis in range(3):
                entry=next(r for r in entries if r['gas']==gas and r['direction']==axis+1)
                directory=Path(entry['source'])
                if directory.name!=f'dir{axis+1}' or directory.parent.name!=name:
                    raise ValueError('case source/direction mismatch')
                input_raw=(directory/'input.dat').read_bytes();path_raw=(directory/'string_path.dat').read_bytes()
                provenance=record['sources'][gas][axis]
                if hashlib.sha256(input_raw).hexdigest()!=provenance['input_sha256'] or hashlib.sha256(path_raw).hexdigest()!=entry['path_sha256']:
                    raise ValueError('raw material/path contents changed')
                header_direction,points=_direction_header(input_raw)
                if header_direction!=axis+1 or points!=401:raise ValueError('input header mismatch')
                case=parse_string_input(input_raw.decode())
                if not np.array_equal(string_parameters(case),np.array(parameters[gas],dtype=np.float32)):
                    raise ValueError('gas/physical conditions mismatch')
                pose=np.loadtxt(io.BytesIO(path_raw))
                if not np.array_equal(pose,native[axis*401:(axis+1)*401]):raise ValueError('cache/raw path mismatch')
                logd=path_logd(case,pose,axis);value=linear_value(logd);qc=path_qc(case,pose,axis)
                peak=float(pose[:,6].max());minimum=float(pose[:,6].min())
                historical=math.log10(float(old[gas+'_D_'+'abc'[axis]]))
                clipped=stable_logd(case,pose,axis,cap=True)
                flags=[]
                if peak>=2000:flags.append('historical_saddle_cap_affected')
                if peak>=1e5:flags.append('extreme_repulsive_peak_needs_path_review')
                if qc['endpoint_winding_error']>5e-4:flags.append('endpoint_winding_mismatch')
                if qc['max_segment_to_mean']>20:flags.append('irregular_spatial_steps')
                if abs(clipped-historical)>1e-4:flags.append('legacy_formula_reproduction_mismatch')
                if value is None:flags.append('linear_D_unrepresentable_keep_logD_not_zero')
                thermal=np.sqrt(1.38e-23*case['temperature']/(2*np.pi*(case['total_mass']/1000/6.02214076e23)))
                upper=np.log10(.5*thermal*(qc['hop_length_A']*1e-10)**2/(qc['arc_length_A']*1e-10))
                if logd>upper+1e-8:raise ValueError('uncapped TST bound violated')
                rows.append(dict(name=name,gas=gas,direction='abc'[axis],random_split=membership['random_split'],group_split=membership['group_split'],
                         source=str(directory),input_sha256=provenance['input_sha256'],path_sha256=entry['path_sha256'],
                         old_logD=historical,uncapped_logD_given_saved_path=logd,uncapped_D_m2_s='' if value is None else value,
                         uncapped_minus_old_logD=logd-historical,legacy_formula_minus_old_logD=clipped-historical,
                         saved_path_TST_logD_upper= float(upper),old_exceeds_saved_path_TST_bound=historical>upper+1e-4,
                         temperature_K=float(case['temperature']),raw_min_K=minimum,raw_max_K=peak,raw_barrier_K=peak-minimum,
                         historical_cap_affected=peak>=2000,linear_D_underflow=logd<math.log10(np.nextafter(0.,1.)),
                         qc_status='needs_path_review' if flags else 'provisional_saved_path_reference',qc_flags=';'.join(flags),
                         **qc))
    return rows


def atomic_csv(path,rows):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=96)
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest=load_manifest(args.root/'runs/v2/manifest.csv');names={r['name'] for r in manifest}
    source_path=args.root/'inputs/full_pose_v4_v1/sources.json';source_raw=source_path.read_bytes();sources=json.loads(source_raw)
    path_report=args.root/'inputs/path_environments_v2/report.json';path_raw=path_report.read_bytes();report=json.loads(path_raw)
    parameters=json.loads((args.root/'inputs/full_pose_v4_v1/string_parameters.json').read_text())
    label_path=args.root/'inputs/diffusivity_corrected_complete6_by_cif.csv'
    with label_path.open(newline='') as handle:old={r['cif_name']:r for r in csv.DictReader(handle)}
    if (not report['passed'] or report['native_sources_sha256']!=hashlib.sha256(source_raw).hexdigest()
            or any(set(table)!=names for table in (sources,parameters,report['records'],old))):
        raise ValueError('material/source provenance coverage mismatch')
    rows,failures=[],[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks={pool.submit(rebuild_case,(str(args.root),r['name'],sources[r['name']],report['records'][r['name']],
                         parameters[r['name']],old[r['name']],r)):r['name'] for r in manifest}
        for future in as_completed(tasks):
            try:rows.extend(future.result())
            except Exception as error:failures.append(dict(name=tasks[future],error=str(error)))
            if (len(rows)//6+len(failures))%500==0:print(json.dumps(dict(materials=len(rows)//6,failures=len(failures))),flush=True)
    rows.sort(key=lambda r:(r['name'],r['gas'],r['direction']))
    material_rows=[]
    by_name={n:[] for n in names}
    for r in rows:by_name[r['name']].append(r)
    for name,values in sorted(by_name.items()):
        if len(values)!=6:continue
        item=dict(name=name,random_split=values[0]['random_split'],group_split=values[0]['group_split'])
        for gas in GASES:
            group=sorted((r for r in values if r['gas']==gas),key=lambda r:r['direction'])
            average=mean_logd([r['uncapped_logD_given_saved_path'] for r in group])
            item[gas+'_uncapped_mean_logD']=average;item[gas+'_uncapped_mean_D_m2_s']=linear_value(average) or ''
            item[gas+'_cap_affected_directions']=sum(r['historical_cap_affected'] for r in group)
            item[gas+'_underflow_directions']=sum(r['linear_D_underflow'] for r in group)
            item[gas+'_review_directions']=sum(r['qc_status']=='needs_path_review' for r in group)
        material_rows.append(item)
    if rows:atomic_csv(args.output_dir/'directions.csv',rows)
    if material_rows:atomic_csv(args.output_dir/'materials.csv',material_rows)
    flag_counts=Counter(flag for r in rows for flag in r['qc_flags'].split(';') if flag)
    passed=not failures and len(rows)==len(manifest)*6 and len(material_rows)==len(manifest)
    result=dict(passed=passed,structures=len(material_rows),directions=len(rows),failures=failures,qc_flag_counts=dict(flag_counts),
                maximum_uncapped_minus_old_logD=float(max((r['uncapped_minus_old_logD'] for r in rows),default=0)),
                output_label_semantics='uncapped TST on saved trajectories, not optimal-path or material ground truth',
                original_inputs_preserved=True,training_manifest_replaced=False,
                underflow_policy='keep finite logD; blank linear D with flag, never manufacture zero',
                source_sha256=hashlib.sha256(source_raw).hexdigest(),path_report_sha256=hashlib.sha256(path_raw).hexdigest(),
                original_label_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest(),
                caveats=['Cap-affected/extreme energies do not classify materials as blocked.',
                         'Provisional uncapped references still need convergence/contact/path-optimality review.'])
    json_write(args.output_dir/'report.json',result);print(json.dumps(result),flush=True)
    if not passed:raise RuntimeError('uncapped reconstruction incomplete; failures preserved')


if __name__=='__main__':main()
