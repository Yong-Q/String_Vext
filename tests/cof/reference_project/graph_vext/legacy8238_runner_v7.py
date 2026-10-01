"""One allocated GPU per dispatcher, staggered admission and durable CSV results."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'runs/legacy8238_string_v1'
CSV_FIELDS = ['task_id','name','gas','direction','guest_sigma_A','guest_epsilon_K','guest_bond_A','guest_mass_g_mol',
              'status','D_m2_s','logD','uncapped_logD',
              'barrier_K','peak_K','capped','stage','final_converged','warnings','attempt_path','job_id','gpu_uuid']
_seen = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, payload):
    temp = Path(str(path)+'.tmp')
    with temp.open('w') as handle:
        json.dump(payload,handle,indent=2,allow_nan=False); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp,path)


def rows(path):
    if not Path(path).is_file(): return []
    with Path(path).open(newline='') as handle: return list(csv.DictReader(handle))


def read_completed(root):
    return {r['task_id']:r for r in rows(Path(root)/'directions.csv')}


def directional_mean(values):
    return sum(values[d] for d in (1,2,3))/3 if set(values)=={1,2,3} else None


def optimizer_iterations(text):
    match=re.search(r'info:\s*[01]\s+(\d+)\s+\d+',text)
    if not match or int(match.group(1))<200:
        raise ValueError('zero-step/short native optimizer run is not a reoptimized path')
    return int(match.group(1))


def append_csv(path, fields, record):
    fresh = not Path(path).exists()
    with Path(path).open('a',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        if fresh: writer.writeheader()
        writer.writerow(record);handle.flush();os.fsync(handle.fileno())


def rebuild_summary(root):
    grouped = defaultdict(dict)
    for r in rows(root/'directions.csv'):
        grouped[r['name'],r['gas']][int(r['direction'])] = r
    fields = ['name','gas','D_a','D_b','D_c','D_avg','directions_done','warnings']
    temp = root/'material_summary.csv.tmp'
    with temp.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for (name,gas), records in sorted(grouped.items()):
            values={d:float(r['D_m2_s']) for d,r in records.items()}
            mean=directional_mean(values)
            writer.writerow(dict(name=name,gas=gas,**{'D_'+a:values.get(d,'') for d,a in enumerate('abc',1)},
                D_avg='' if mean is None else mean,directions_done=len(values),
                warnings=';'.join(sorted({r.get('warnings','') for r in records.values()}))))
        handle.flush();os.fsync(handle.fileno())
    os.replace(temp,root/'material_summary.csv')
    (root/'.summary_time').write_text(str(time.time()))


def publish(root, record, completed=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'csv.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        known=completed if completed is not None else _seen.setdefault(str(root),read_completed(root))
        if record['task_id'] in known: return False
        success=record['status'] in ('completed','completed_with_review')
        append_csv(root/('directions.csv' if success else 'failures.csv'),CSV_FIELDS,record)
        if success: known[record['task_id']]=record
        stamp=root/'.summary_time'
        last=float(stamp.read_text()) if stamp.exists() else 0.
        if success and time.time()-last>30: rebuild_summary(root)
        return True


def monitor_name(job, shard, uuid):
    if not re.fullmatch(r'GPU-[A-Za-z0-9-]+',uuid): raise ValueError('invalid physical GPU UUID')
    return f'job_{job}_shard_{shard}_{uuid}'


def allocated_uuid():
    # CUDA device 0 is mapped by Slurm; PCI lookup avoids confusing local/global indices.
    runtime=ctypes.CDLL('libcudart.so')
    count=ctypes.c_int()
    if runtime.cudaGetDeviceCount(ctypes.byref(count))!=0 or count.value!=1:
        raise ValueError('exactly one visible CUDA device required')
    bus=ctypes.create_string_buffer(64)
    if runtime.cudaDeviceGetPCIBusId(bus,len(bus),0)!=0:
        raise ValueError('cannot resolve allocated CUDA device PCI bus')
    text=subprocess.check_output(['nvidia-smi','-i',bus.value.decode(),
        '--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).strip()
    if '\n' in text or not text.startswith('GPU-'): raise ValueError('ambiguous allocated GPU')
    return text


def query(uuid):
    from graph_vext.string_triclinic_pilot import gpu_query
    card=gpu_query(uuid)
    if card['uuid']!=uuid:raise ValueError('GPU UUID changed')
    return card


def process_memory(uuid):
    text=subprocess.check_output(['nvidia-smi','-i',uuid,
        '--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=15)
    values={}
    for row in csv.reader(text.splitlines()):
        if len(row)!=3 or row[0].strip()!=uuid:continue
        try: values[int(row[1])]=float(row[2])
        except ValueError: continue
    return values


def rss(pid):
    if not pid:return 0.
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):return float(line.split()[1])/1024
    except OSError:pass
    return 0.


def admit(free, active, estimate, reserve, maximum, host_budget):
    pending=sum(max(0.,x['estimate']-x.get('allocated',0.)) for x in active)
    host=sum(max(x.get('rss',0.),x.get('host_estimate',0.)) for x in active)
    return len(active)<maximum and free>=reserve+estimate+pending and host+estimate*1.5+4096<=host_budget


def launch_interval(used_mib, nominal=5.):
    if used_mib>=22528:return None
    if used_mib>=18432:return 8.
    if used_mib<15360:return 3.
    return float(nominal)


def launch_count(used_mib):
    if used_mib>=22528:return 0
    return 2 if used_mib<15360 else 1


def source_allocation_budget(atoms):
    if atoms<=0:raise ValueError('positive framework atom count required')
    # Straight-line initialization allocates seven dense arrays, each
    # 11^2*6*3*6 grid/angle states x two guest sites x framework atoms.
    dense_MiB=7*13068*2*8*atoms/1048576
    return max(1024,math.ceil(1.25*dense_MiB+768))


def calibrated_start_budget(atoms, profiles):
    if not profiles or atoms>max(p['atoms'] for p in profiles):
        raise ValueError('task exceeds calibrated framework size')
    if any(p['sampled_peak_increment_MiB']>source_allocation_budget(p['atoms']) for p in profiles):
        raise ValueError('measured GPU peak exceeds source allocation budget')
    return source_allocation_budget(atoms)


INIT_RELEASE_MARKER = 'LEGACY_DENSE_INIT_FREED_V1'
DERIV_READY_MARKER = 'LEGACY_DERIV_BUFFERS_READY_V1'


def steady_allocation_budget(atoms):
    # Derivative phase: seven stencil images x 401 points x two guest sites,
    # six FP64 arrays, four int32 index arrays and one FP64 energy array.
    dense_MiB=(7*401*2*atoms*(7*8+4*4))/1048576
    return max(1024,math.ceil(1.25*dense_MiB+768))


def update_allocation_phase(slot, stderr_path, allocated, host_rss):
    slot['allocated']=allocated
    slot['rss']=host_rss
    if slot.get('phase')=='derivative_ready':
        slot['estimate']=allocated+64
        slot['host_estimate']=max(512,host_rss+256)
        return
    try:
        text=Path(stderr_path).read_text()
    except OSError:
        text=''
    if DERIV_READY_MARKER in text:
        slot['phase']='derivative_ready'
        slot['estimate']=allocated+64
        slot['host_estimate']=max(512,host_rss+256)
    elif INIT_RELEASE_MARKER in text:
        slot['phase']='steady_allocating'
        slot['estimate']=steady_allocation_budget(slot['atoms'])
        slot['host_estimate']=max(512,host_rss+256)


def load_plan(prepared):
    report=json.loads((prepared/'manifest_report.json').read_text())
    valid=json.loads((prepared/'validation_report_recheck1.json').read_text())
    digest=sha(prepared/'tasks.csv')
    if report['materials']!=8238 or report['tasks']!=49428 or report['task_csv_sha256']!=digest \
            or not valid['passed'] or valid['manifest_sha256']!=digest:
        raise ValueError('8238-material full-input validation gate required')
    tasks=rows(prepared/'tasks.csv')
    if len(tasks)!=49428 or len({r['task_id'] for r in tasks})!=49428:raise ValueError('manifest duplicate/missing tasks')
    return tasks,digest


def native_binding(exe):
    record=json.loads(Path(str(exe)+'.build.json').read_text())
    if sha(exe)!=record.get('exe_sha256',record.get('binary_sha256')):raise ValueError('native binary hash mismatch')
    source_path=ROOT/record['source_manifest'] if not Path(record['source_manifest']).is_absolute() else Path(record['source_manifest'])
    for line in source_path.read_text().splitlines():
        digest,name=line.split(maxsplit=1)
        path=Path(name.strip());path=path if path.is_absolute() else ROOT/path
        if sha(path)!=digest:raise ValueError('native source hash mismatch')
    return dict(binary_sha256=sha(exe),build_sha256=sha(str(exe)+'.build.json'),source_manifest_sha256=sha(source_path))


def cpu_bodyz_pose_energies(case, pose):
    from graph_vext.string_atom_mapping import rotate_sites
    from graph_vext.string_ff_vext import TriclinicSitePotential
    p=np.asarray(pose,dtype=float)
    if p.ndim!=2 or p.shape[1]!=7 or not np.isfinite(p).all():
        raise ValueError('finite Nx7 poses required')
    center=(case['guest_xyz']*case['guest_mass'][:,None]).sum(0)/case['total_mass']
    body=case['guest_xyz']-center
    if not np.allclose(body[:,:2],0,atol=1e-10):
        raise ValueError('old cohort guest must be a BODY-Z two-site molecule')
    # The potential validates BODY-X metadata, but its site evaluator uses only
    # framework/FF parameters; actual guest site coordinates remain BODY-Z.
    validated={**case,'guest_xyz':np.column_stack((body[:,2],np.zeros((2,2))))}
    potential=TriclinicSitePotential(validated)
    cartesian=p[:,:3]@case['cell']
    sites=rotate_sites(p[:,3:6],body)
    result=np.zeros(len(p))
    for site in range(2):
        result+=potential.evaluate(cartesian+sites[:,site],case['guest_sigma'][site],case['guest_epsilon'][site])
    return result


def execute(task, attempt, exe, uuid, tracked, lock, replay_all=False, timeout=7200):
    from graph_vext.legacy8238_manifest import GASES, inspect_input
    from graph_vext.string_atom_mapping import parse_string_input
    from graph_vext.rebuild_uncapped_labels import path_logd,path_qc,linear_value
    from graph_vext.audit_native_path_cap import stable_logd
    work=attempt/task['task_id'];work.mkdir(exist_ok=False)
    sigma,epsilon,bond,mass=GASES[task['gas']]
    result=dict(task_id=task['task_id'],name=task['name'],gas=task['gas'],direction=int(task['direction']),
        guest_sigma_A=sigma,guest_epsilon_K=epsilon,guest_bond_A=bond,guest_mass_g_mol=mass,
        status='failed',attempt_path=str(work),job_id=os.environ.get('SLURM_JOB_ID'),gpu_uuid=uuid,warnings='')
    proc=None
    try:
        raw=Path(task['input_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=task['input_sha256']:raise ValueError('original input changed')
        definition=inspect_input(raw,task['gas'],int(task['direction']))
        with (work/'input.dat').open('xb') as handle:handle.write(raw)
        case=parse_string_input(raw.decode())
        command=[str(exe),'input.dat']
        seed=Path(task['saved_path'])
        if seed.is_file():
            seedraw=seed.read_bytes()
            try:
                candidate=np.loadtxt(seed)
                if candidate.shape==(401,7) and np.isfinite(candidate).all():
                    qc=path_qc(case,candidate,int(task['direction'])-1)
                    if qc['endpoint_winding_error']<=5e-4 and qc['max_segment_to_mean']<=20:
                        with (work/'initial.dat').open('xb') as handle:handle.write(seedraw)
                        command.append('initial.dat')
                        result['seed_sha256']=hashlib.sha256(seedraw).hexdigest()
            except (ValueError,OSError) as exc:
                result['seed_rejected_reason']=str(exc)
        command.append('string_path.dat')
        result['warm_started']='initial.dat' in command
        with (work/'native.stdout').open('x') as stdout,(work/'native.stderr').open('x') as stderr:
            proc=subprocess.Popen(command,cwd=work,stdout=stdout,stderr=stderr)
            with lock:tracked[task['task_id']]['proc']=proc
            started=time.monotonic()
            while proc.poll() is None:
                with lock:abort=tracked[task['task_id']].get('abort')
                if abort or time.monotonic()-started>timeout:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    raise RuntimeError(abort or 'case_timeout')
                time.sleep(1)
        if proc.returncode!=0:raise RuntimeError(f'native_exit_{proc.returncode}')
        text=(work/'native.stdout').read_text()
        stderr=(work/'native.stderr').read_text()
        if re.search(r'out of memory|illegal memory|fatal error',stderr,re.I):raise RuntimeError('native CUDA failure')
        iterations=optimizer_iterations(text)
        stage=re.search(r'legacy_selection:.*chosen_stage=(initial|final)',text)
        convergence=re.search(r'final_convergence:.*simultaneous=([01])',text)
        if not stage or not convergence:raise ValueError('missing native stage/convergence contract')
        stage=stage.group(1);final_ok=convergence.group(1)=='1'
        p=np.loadtxt(work/'string_path.dat')
        if p.shape!=(401,7) or not np.isfinite(p).all():raise ValueError('invalid selected 401x7 path')
        selected=np.loadtxt(work/('string_path.dat.'+stage))
        if not np.array_equal(p,selected):raise ValueError('selected candidate bytes differ')
        qc=path_qc(case,p,int(task['direction'])-1)
        if qc['endpoint_winding_error']>5e-4 or qc['max_segment_to_mean']>20:raise ValueError('path winding/spacing failure')
        if stage=='final' and not final_ok:raise ValueError('final path not jointly converged')
        # No saved-path energy is trusted: the new solver recomputes warm-start energies.
        sample=p if replay_all else p[np.linspace(0,400,9,dtype=int)]
        expected=cpu_bodyz_pose_energies(case,sample)
        errors=np.abs(expected-sample[:,6])
        if not (errors<=1e-4+1e-7*np.maximum(np.abs(expected),np.abs(sample[:,6]))).all():
            raise ValueError('CPU/native triclinic energy disagreement')
        capped_log=stable_logd(case,p,int(task['direction'])-1,cap=True)
        value=linear_value(capped_log)
        if value is None:raise ValueError('capped linear D unrepresentable; not invented zero')
        warnings=[]
        if p[:,6].max()>=2000:warnings.append('historical_saddle_cap_used')
        if p[:,6].max()>=1e5:warnings.append('extreme_repulsive_path_needs_review')
        if stage=='initial':warnings.append('initial_candidate_convergence_not_certified')
        result.update(status='completed_with_review' if warnings else 'completed',
            optimizer_iterations=iterations,
            D_m2_s=value,logD=capped_log,uncapped_logD=path_logd(case,p,int(task['direction'])-1),
            barrier_K=float(np.ptp(p[:,6])),peak_K=float(p[:,6].max()),capped=bool(p[:,6].max()>=2000),
            stage=stage,final_converged=final_ok,warnings=';'.join(warnings),
            framework_atoms=definition['atoms'],CPU_energy_checked_points=len(sample),
            CPU_energy_max_abs_error_K=float(errors.max()),path_sha256=sha(work/'string_path.dat'),path_qc=qc)
        if sha(task['input_path'])!=task['input_sha256']:raise ValueError('input changed during run')
    except Exception as exc:
        result.update(status='failed',error=str(exc),warnings=str(exc))
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
    atomic_json(work/'result.json',result)
    return result


def pilot(prepared,exe,output):
    tasks,digest=load_plan(prepared);binding=native_binding(exe);uuid=allocated_uuid()
    reserve=2048
    output.mkdir(parents=True,exist_ok=False)
    metadata={r['name']:json.loads(Path(r['material_json']).read_text()) for r in tasks[::6]}
    ordered=sorted(metadata,key=lambda n:metadata[n]['atoms'])
    names=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
    chosen=[r for n in names for r in tasks if r['name']==n and r['direction']=='1']
    profiles=[];lock=threading.Lock()
    for task in chosen:
        atoms=metadata[task['name']]['atoms'];before=query(uuid)
        bound=source_allocation_budget(atoms)
        if before['free_MiB']<reserve+bound:raise RuntimeError('pilot cannot preserve2GiB under source allocation budget')
        tracked={task['task_id']:dict(proc=None,abort=None)}
        samples=[]
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(execute,task,output,exe,uuid,tracked,lock,True)
            while not future.done():
                card=query(uuid);samples.append(card)
                if card['free_MiB']<reserve:
                    with lock:tracked[task['task_id']]['abort']='pilot reserve breached'
                time.sleep(.15)
            result=future.result()
        if result['status']=='failed':raise RuntimeError('pilot failed: '+result.get('error',''))
        increment=max((r['used_MiB'] for r in samples),default=before['used_MiB'])-before['used_MiB']
        profiles.append(dict(atoms=atoms,sampled_peak_increment_MiB=increment,
            source_allocation_budget_MiB=bound,result=result))
        atomic_json(output/'progress.json',dict(profiles=profiles))
    if native_binding(exe)!=binding:raise RuntimeError('native binding changed')
    if any(p['sampled_peak_increment_MiB']>source_allocation_budget(p['atoms']) for p in profiles):
        raise RuntimeError('source allocation budget below measured pilot peak')
    atomic_json(output/'gate.json',dict(passed=True,manifest_sha256=digest,native_binding=binding,profiles=profiles,
        runner_sha256=sha(__file__),gpu_uuid=uuid,reserve_MiB=reserve,
        calibration_policy='max(source array allocation budget, sampled transient peak); no zero-memory inference',
        cuda_allocation_model='seven dense initialization arrays: 11^2*6*3*6 states x 2 guest sites x framework atoms x 8-byte doubles; 1.25x plus768MiB context/scratch, checked against all six pilot peaks',
        sampled_zero_is_unobserved=any(p['sampled_peak_increment_MiB']<=0 for p in profiles),
        caveat='Numerical/operational acceptance only; no global optimality certificate.'))


def run(prepared,exe,gate_path,shard,maximum=20,stagger=8.):
    tasks,digest=load_plan(prepared);binding=native_binding(exe)
    gate=json.loads(gate_path.read_text())
    if not gate['passed'] or gate['manifest_sha256']!=digest or gate['native_binding']!=binding \
            or gate['runner_sha256']!=sha(__file__) or gate['reserve_MiB']!=2048:
        raise ValueError('pilot/runner/native/two-GiB-reserve gate mismatch')
    if not 0<=shard<6 or not 1<=maximum<=25 or not 3<=stagger<=8:raise ValueError('invalid shard/concurrency/stagger')
    job=os.environ['SLURM_JOB_ID'];uuid=allocated_uuid()
    out=BASE/'workers'/monitor_name(job,shard,uuid);out.mkdir(parents=True,exist_ok=False)
    reserve=2048;host_budget=int(os.environ.get('SLURM_MEM_PER_NODE','32768'))
    metadata={r['name']:json.loads(Path(r['material_json']).read_text()) for r in tasks[::6]}
    def estimate(t):
        atoms=metadata[t['name']]['atoms']
        return calibrated_start_budget(atoms,gate['profiles'])
    lease=BASE/f'shard_{shard}.lock'
    with lease.open('a') as ownership:
        fcntl.flock(ownership,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (BASE/'csv.lock').open('a') as snapshot_lock:
            fcntl.flock(snapshot_lock,fcntl.LOCK_EX)
            completed=read_completed(BASE)
        prior_failures={r['task_id'] for r in rows(BASE/'failures.csv')}
        queue=sorted((r for r in tasks if int(r['shard'])==shard and r['task_id'] not in completed),
            key=lambda r:(r['task_id'] in prior_failures,metadata[r['name']]['atoms'],int(r['task_index'])))
        atomic_json(out/'context.json',dict(shard=shard,job_id=job,host=socket.gethostname(),gpu_uuid=uuid,
            manifest_sha256=digest,native_binding=binding,maximum=maximum,stagger_seconds=stagger,
            reserve_MiB=reserve,host_budget_MiB=host_budget,remaining_tasks=len(queue),
            memory_policy='source allocation budget checked against all pilot peaks; no raw peak/atom extrapolation'))
        tracked={};active={};lock=threading.Lock();last=-float('inf');count=0;stopping=False;empty_wait=None
        def stop(signum,frame):
            nonlocal stopping
            stopping=True
            with lock:
                for slot in tracked.values():slot['abort']='job_stop_requested'
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        with (out/'gpu_memory.csv').open('x',newline='') as memory,ThreadPoolExecutor(max_workers=maximum) as pool:
            fields=['time','job_id','shard','gpu_uuid','used_MiB','free_MiB','active','pending_MiB','queued','next_launch_gap_seconds',
                'initializing','steady_allocating','derivative_ready']
            writer=csv.DictWriter(memory,fieldnames=fields);writer.writeheader();failures=0
            while queue or active:
                for future,key in list(active.items()):
                    if future.done():
                        result=future.result();publish(BASE,result,completed);count+=1
                        del active[future]
                        with lock:del tracked[key]
                if stopping and not active:break
                try:
                    card=query(uuid);allocated=process_memory(uuid);failures=0
                    with lock:
                        for key,slot in tracked.items():
                            proc=slot['proc'];pid=proc.pid if proc else None
                            update_allocation_phase(slot,out/key/'native.stderr',allocated.get(pid,0.),rss(pid))
                        snapshots=[dict(s) for s in tracked.values()]
                        if card['free_MiB']<1024:
                            stopping=True
                            # Emergency guard for unexpected external allocations.
                            if snapshots:tracked[next(reversed(tracked))]['abort']='GPU reserve breached; dispatcher stopped'
                    pending=sum(max(0.,s['estimate']-s['allocated']) for s in snapshots)
                    gap=launch_interval(card['used_MiB'],stagger)
                    phase_counts=Counter(s['phase'] for s in snapshots)
                    writer.writerow(dict(time=time.time(),job_id=job,shard=shard,gpu_uuid=uuid,
                        used_MiB=card['used_MiB'],free_MiB=card['free_MiB'],active=len(active),pending_MiB=pending,
                        queued=len(queue),next_launch_gap_seconds=gap if gap is not None else 'paused',
                        initializing=phase_counts['initial'],steady_allocating=phase_counts['steady_allocating'],
                        derivative_ready=phase_counts['derivative_ready']))
                    memory.flush()
                    atomic_json(out/'gpu_latest.json',dict(**card,active=len(active),pending_MiB=pending,
                        queued=len(queue),next_launch_gap_seconds=gap,phases=dict(phase_counts)))
                    if queue and not stopping and gap is not None and time.monotonic()-last>=gap:
                        for _ in range(launch_count(card['used_MiB'])):
                            if not queue:break
                            task=queue[0];budget=estimate(task)
                            if not admit(card['free_MiB'],snapshots,budget,reserve,maximum,host_budget):
                                if not active:
                                    empty_wait=empty_wait or time.monotonic()
                                    if time.monotonic()-empty_wait>120:
                                        raise RuntimeError('task exceeds memory budget; rerun shard after calibration review')
                                break
                            queue.pop(0);key=task['task_id']
                            slot=dict(estimate=budget,allocated=0.,rss=0.,
                                atoms=metadata[task['name']]['atoms'],phase='initial',
                                host_estimate=budget*1.5,proc=None,abort=None)
                            with lock:tracked[key]=slot
                            snapshots.append(dict(slot))
                            active[pool.submit(execute,task,out,exe,uuid,tracked,lock)]=key
                            last=time.monotonic();empty_wait=None
                except (OSError,subprocess.SubprocessError,ValueError) as exc:
                    failures+=1
                    atomic_json(out/'monitor_failure.json',dict(error=str(exc),consecutive_failures=failures))
                    if failures>=3:stop(None,None)
                time.sleep(2)
        if native_binding(exe)!=binding:raise RuntimeError('native source changed')
        with (BASE/'csv.lock').open('a') as csv_lock:
            fcntl.flock(csv_lock,fcntl.LOCK_EX)
            rebuild_summary(BASE)
            all_completed=read_completed(BASE)
        expected={r['task_id'] for r in tasks if int(r['shard'])==shard}
        missing=expected-set(all_completed)
        atomic_json(out/'report.json',dict(completed_this_job=count,stopped=stopping,
            queued=len(queue),shard=shard,expected_tasks=len(expected),missing_tasks=len(missing),
            first_missing_task_ids=sorted(missing)[:20],
            all_six_gas_directions_recorded=len(missing)==0))
        if queue or stopping or missing:
            raise RuntimeError('dispatcher incomplete; successes retained, missing cases require resume')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('pilot','run'),required=True)
    p.add_argument('--prepared',type=Path,default=BASE/'prepared_v2')
    p.add_argument('--executable',type=Path,default=ROOT/'native/string_legacy8238_v4/GPU_string_legacy8238_v4')
    p.add_argument('--output',type=Path)
    p.add_argument('--gate',type=Path)
    p.add_argument('--shard',type=int)
    p.add_argument('--max-concurrent',type=int,default=20)
    p.add_argument('--stagger',type=float,default=8.)
    a=p.parse_args()
    if a.mode=='pilot':pilot(a.prepared,a.executable,a.output)
    else:run(a.prepared,a.executable,a.gate,a.shard,a.max_concurrent,a.stagger)


if __name__=='__main__':main()
