"""Read-only full native-energy counts and actual diffusivity binary replay."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from graph_vext.dataset import GASES, load_manifest
from graph_vext.m4_attentive_data import load_checked_directional_targets
from graph_vext.string_atom_mapping import parse_string_input
from graph_vext.train_spatial_transport_v3 import json_write


def stable_logd(case, pose, direction, cap=False):
    distance=np.linalg.norm(np.diff(pose[:,:3]@case['cell'],axis=0),axis=1)*1e-10
    T=case['temperature'];energy=pose[:,6]
    boltz=-np.minimum(energy/T,600.) if cap else -energy/T
    terms=np.log(distance[distance>0])+np.logaddexp(boltz[:-1][distance>0],boltz[1:][distance>0])-np.log(2.)
    integral=logsumexp(terms)
    saddle=min(float(energy.max()),2000.) if cap else float(energy.max())
    length=np.linalg.norm(case['cell'][direction])*1e-10
    mass=case['total_mass']/1000/6.02214076e23
    return float((np.log(.5)+.5*np.log(1.38e-23*T/(2*np.pi*mass))+2*np.log(length)-saddle/T-integral)/np.log(10.))


def summary_case(task):
    root,name,records=task;root=Path(root)
    raw=(root/'inputs/full_pose_v4_v1'/(name+'.npz')).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=records[name]['cached_native_sha256']:raise ValueError('native cache changed')
    rows=[]
    with np.load(io.BytesIO(raw)) as data:
        for gas in GASES:
            for i,energy in enumerate(data['raw_energy_K_'+gas].reshape(3,401)):
                rows.append(dict(name=name,gas=gas,direction=i,raw_min_K=float(energy.min()),raw_max_K=float(energy.max()),
                                 raw_barrier_K=float(energy.max()-energy.min()),points_ge2000=int((energy>=2000).sum())))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=32)
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    names=[r['name'] for r in load_manifest(args.root/'runs/v2/manifest.csv')]
    records=json.loads((args.root/'inputs/path_environments_v2/report.json').read_text())['records']
    sources=json.loads((args.root/'inputs/full_pose_v4_v1/sources.json').read_text())
    targets=load_checked_directional_targets(args.root/'inputs/diffusivity_corrected_complete6_by_cif.csv',load_manifest(args.root/'runs/v2/manifest.csv'))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pairs=list(pool.map(summary_case,[(str(args.root),n,records) for n in names]))
    rows=[r for group in pairs for r in group];maxima=np.array([r['raw_max_K'] for r in rows])
    result=dict(read_only=True,structures=len(names),paths=len(rows),counts={str(x):int((maxima>=x).sum()) for x in (2000,1e5,1e9)},
                materials_any_capped=sum(any(r['raw_max_K']>=2000 for r in group) for group in pairs),
                materials_all6_capped=sum(all(r['raw_max_K']>=2000 for r in group) for group in pairs),
                raw_max_K_quantiles=np.quantile(maxima,[0,.5,.9,.99,1]).tolist(),replays=[])
    binary=Path('/home/qiuyong/GaSSM/utility/cal_diffusivity');source=Path('/home/qiuyong/GaSSM/utility/diffusivity.c')
    result.update(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    sample=sorted(rows,key=lambda r:r['raw_max_K'],reverse=True)[:4]
    low=next(r for r in rows if r['raw_max_K']<2000);sample.append(low)
    for index,row in enumerate(sample):
        entry=next(s for s in sources[row['name']] if s['gas']==row['gas'] and s['direction']==row['direction']+1)
        directory=Path(entry['source']);input_path=directory/'input.dat';path=directory/'string_path.dat'
        path_raw=path.read_bytes()
        if hashlib.sha256(path_raw).hexdigest()!=entry['path_sha256']:raise ValueError('native replay input changed')
        case=parse_string_input(input_path.read_text());pose=np.loadtxt(io.BytesIO(path_raw))
        output=args.output_dir/f'original_binary_case_{index}';output.mkdir()
        proc=subprocess.run([str(binary),str(input_path),str(path)],cwd=output,capture_output=True,text=True,timeout=30)
        (output/'stdout.log').write_text(proc.stdout);(output/'stderr.log').write_text(proc.stderr)
        value=float((output/'1.dat').read_text().split()[0]) if proc.returncode==0 else None
        expected=10**float(targets[GASES.index(row['gas'])][row['name']][row['direction']])
        clipped=stable_logd(case,pose,row['direction'],cap=True);uncapped=stable_logd(case,pose,row['direction'],cap=False)
        shifted=pose.copy();shifted[:,6]+=500
        result['replays'].append(dict(**row,original_binary_returncode=proc.returncode,original_binary_D=value,
                    manifest_D=expected,stable_capped_logD=clipped,stable_uncapped_logD=uncapped,
                    binary_matches_capped_formula_rel=abs(value-10**clipped)/max(abs(value),1e-300) if value is not None else None,
                    cap_energy_zero_shift_delta_logD=stable_logd(case,shifted,row['direction'],cap=True)-clipped,
                    uncapped_energy_zero_shift_delta_logD=stable_logd(case,shifted,row['direction'],cap=False)-uncapped))
    result['interpretation']='Raw untruncated barriers and the implemented capped-saddle D are different targets; do not silently equate them.'
    json_write(args.output_dir/'report.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
