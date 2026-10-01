"""Controlled-input, convergence, source-gate and paired-report contracts."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def pilot():
    from graph_vext import string_triclinic_pilot
    return string_triclinic_pilot


INPUT = '''Nmaxa Nmaxb Nmaxc:
70 70 70
La Lb Lc dL
30 31 32 0.5
Alpha Beta Gamma
90 110 90
cutoff(A) FH_signal Mass(g/mol) Tempearture(K) Running_steps
12.9 1 28.054 298 10000
---------String Calculation Settings---------
Direction
1
#_of_points delta_frac delta_angle_degree
401 0.000100 1.000000
convergence_setting
default
------------------Adsorbate------------------
Number of sites
2
x(A) y(A) z(A) Epsilon(K) Sigma(A) Charge(e) Mass(g/mol)
-0.670000 0 0 85.000000 3.700000 0 14.027000
0.670000 0 0 85.000000 3.700000 0 14.027000
------------------Adsorbent-----------------
Number of atoms
1
ID diameter(A) Epsilon(K) Charge(e) mass(g/mol) frac_x frac_y frac_z atom_name
1 3.430851 52.837743 0 12.011000 .25 .25 .25 C
'''


def cohort():
    rows, descriptors = [], []
    for skew in (False, True):
        for i in range(30):
            name = ('skew' if skew else 'ortho')+str(i)
            descriptors.append(dict(name=name, atoms=100+20*i, nonorthogonal=skew))
            for gas in ('C2H4', 'C2H6'):
                for direction in 'abc':
                    rows.append(dict(name=name, gas=gas, direction=direction,
                                     qc_status='provisional_saved_path_reference',
                                     uncapped_logD_given_saved_path='DO NOT READ FOR SELECTION'))
    return rows, descriptors


def pose():
    p = np.zeros((401, 7))
    p[:, 0] = np.linspace(0, 1, 401)
    p[:, 6] = np.sin(np.linspace(0, np.pi, 401))*100
    return p


class StringTriclinicPilotTests(unittest.TestCase):
    def test_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('graph_vext.string_triclinic_pilot'))

    def test_baseline_is_byte_identical_old_arm_changes_only_four_tokens(self):
        m = pilot()
        raw = INPUT.encode()
        self.assertEqual(m.arm_input(raw, 'C2H4', 'current'), raw)
        old = m.arm_input(raw, 'C2H4', 'old_sigma_epsilon_only')
        a, b = raw.decode().split(), old.decode().split()
        changes = [(x, y) for x, y in zip(a, b) if x != y]
        self.assertEqual(changes, [('85.000000', '92.800000'), ('3.700000', '3.680000')]*2)
        self.assertEqual(len(a), len(b))
        self.assertIn('1 3.430851 52.837743 0 12.011000', old.decode())
        self.assertIn('-0.670000 0 0', old.decode())
        self.assertIn('28.054 298 10000', old.decode())

    def test_ethane_keeps_mass_and_bond_and_uses_only_old_sigma_epsilon(self):
        m = pilot()
        raw = INPUT.replace('28.054', '30.070').replace('14.027000', '15.035000')
        raw = raw.replace('0.670000', '0.770000').replace('85.000000', '98.000000').replace('3.700000', '3.750000')
        changed = m.arm_input(raw.encode(), 'C2H6', 'old_sigma_epsilon_only').decode()
        self.assertEqual(changed.count('108.000000 3.760000'), 2)
        self.assertIn('30.070 298 10000', changed)
        self.assertIn('-0.770000 0 0', changed)
        self.assertEqual(changed.count('15.035000'), 2)

    def test_wrong_source_steps_points_or_current_guest_are_rejected(self):
        m = pilot()
        for text in (INPUT.replace('10000', '1000'), INPUT.replace('401 ', '64 '),
                     INPUT.replace('85.000000', '92.800000')):
            with self.assertRaises(ValueError):
                m.arm_input(text.encode(), 'C2H4', 'current')

    def test_selection_is_all_six_provisional_geometry_size_balanced_and_label_blind(self):
        m = pilot()
        rows, descriptors = cohort()
        rows[0]['qc_status'] = 'needs_path_review'
        selected, info = m.select_materials(rows, descriptors, seed=42)
        self.assertEqual(len(selected), 10)
        self.assertNotIn(rows[0]['name'], [r['name'] for r in selected])
        self.assertEqual(sum(r['nonorthogonal'] for r in selected), 5)
        self.assertEqual(sum(r['size_band']=='small' for r in selected), 5)
        for rank in (0, 1):
            shard = selected[rank::2]
            self.assertEqual(len(shard), 5)
            self.assertIn(sum(r['nonorthogonal'] for r in shard), (2, 3))
            self.assertIn(sum(r['size_band']=='small' for r in shard), (2, 3))
        altered = [{**r, 'uncapped_logD_given_saved_path': '-1e100'} for r in rows]
        self.assertEqual(selected, m.select_materials(altered, list(reversed(descriptors)), 42)[0])
        self.assertFalse(info['uses_D_for_selection'])

    def test_duplicate_or_missing_direction_never_qualifies(self):
        m = pilot()
        rows, descriptors = cohort()
        with self.assertRaises(ValueError):
            m.select_materials(rows+rows[:1], descriptors)
        missing = rows[0]['name']
        selected, _ = m.select_materials(rows[1:], descriptors)
        self.assertNotIn(missing, [r['name'] for r in selected])

    def test_all_nonorthogonal_still_selects_ten_size_balanced_without_fabrication(self):
        m = pilot()
        rows, descriptors = cohort()
        descriptors = [{**r, 'nonorthogonal':True} for r in descriptors]
        selected, info = m.select_materials(rows, descriptors)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len({r['name'] for r in selected}), 10)
        self.assertTrue(all(r['nonorthogonal'] for r in selected))
        self.assertEqual(sum(r['size_band']=='small' for r in selected), 5)
        self.assertEqual(sum(r['size_band']=='medium' for r in selected), 5)
        self.assertEqual(info['geometry_counts'], {'orthogonal':0, 'skew':10})
        self.assertTrue(info['selection_all_nonorthogonal_corpus'])
        self.assertEqual(selected, m.select_materials(list(reversed(rows)), list(reversed(descriptors)))[0])
        self.assertFalse(info['uses_D_for_selection'])

    def test_sparse_count_buckets_redistribute_instead_of_aborting(self):
        m = pilot()
        rows, descriptors = cohort()
        names = {r['name'] for r in descriptors[:10]}
        descriptors = [{**r, 'atoms':100, 'nonorthogonal':True} for r in descriptors[:10]]
        rows = [r for r in rows if r['name'] in names]
        selected, info = m.select_materials(rows, descriptors)
        self.assertEqual(len(selected), 10)
        self.assertEqual(info['size_counts'], {'small':10, 'medium':0, 'large':0})
        self.assertTrue(info['quota_redistributed'])

    def test_exit_zero_is_not_convergence_and_running_limit_cannot_pass(self):
        m = pilot()
        bad = ['info: 0 10000 1 2.5', 'finished successfully', 'info: 1 10000 1 2.5',
               'info: 1 1600 1\ninfo: 0 10000 1', 'fatal error!!!!!\ninfo: 1 1600 1']
        for text in bad:
            self.assertFalse(m.convergence(text, '', 0, 10000)['converged'])
        self.assertFalse(m.convergence('info: 1 1600 1', '', 1, 10000)['converged'])
        valid = m.convergence('long_coul_fft: 70 70 70 info: 1 1600 1 351.14', '', 0, 10000)
        self.assertTrue(valid['converged'])
        self.assertEqual(valid['iteration'], 1600)

    def test_nonconverged_nonfinite_and_bad_winding_have_no_numeric_outcome(self):
        m = pilot()
        case = m.input_case(INPUT.encode())[0]
        for array, stdout in ((pose(), 'info: 0 10000 1'), (pose()*np.nan, 'info: 1 1600 1'),
                              (pose()[:-1], 'info: 1 1600 1')):
            result = m.score_result(case, array, stdout, '', 0, direction=1)
            self.assertNotEqual(result['status'], 'valid')
            self.assertIsNone(result['logD'])
            self.assertIsNone(result['D_m2_s'])
        wrong = pose()
        wrong[-1, 0] = .5
        self.assertIsNone(m.score_result(case, wrong, 'info: 1 1600 1', '', 0, 1)['logD'])

    def test_valid_scoring_is_uncapped_energy_zero_invariant_and_underflow_not_zero(self):
        m = pilot()
        case = m.input_case(INPUT.encode())[0]
        p = pose()
        a = m.score_result(case, p, 'info: 1 1600 1', '', 0, 1)
        self.assertEqual(a['status'], 'valid')
        shifted = p.copy()
        shifted[:, 6] += 1e8
        b = m.score_result(case, shifted, 'info: 1 1600 1', '', 0, 1)
        self.assertAlmostEqual(a['logD'], b['logD'], places=7)
        p[200, 6] = 1e7
        c = m.score_result(case, p, 'info: 1 1600 1', '', 0, 1)
        self.assertEqual(c['status'], 'valid')
        self.assertLess(c['logD'], -1000)
        self.assertIsNone(c['D_m2_s'])
        self.assertIn('linear_D_unrepresentable', c['flags'])

    def test_step_log_requires_both_signals_consistent_with_stdout(self):
        m = pilot()
        contract = dict(iteration='iteration', coordinate='coordinate_converged', orientation='orientation_converged')
        good = 'iteration,coordinate_converged,orientation_converged\n0,0,0\n1600,1,1\n'
        self.assertTrue(m.step_log_audit(good, 1600, contract)['passed'])
        for text in (good.replace('1600,1,1', '1600,1,0'), good.replace('1600,1,1', '1800,1,1'),
                     'iteration,bad\n1600,1\n'):
            self.assertFalse(m.step_log_audit(text, 1600, contract)['passed'])

    def test_gpu_guard_requires_one_assigned_card_and_free_memory(self):
        m = pilot()
        parsed = m.parse_gpu_query('GPU-test, 24576, 512, 24064\n')
        self.assertTrue(m.gpu_safe(parsed, required_free_mib=20000, reserve_mib=2048))
        self.assertFalse(m.gpu_safe({**parsed, 'free_MiB': 1000}, 20000, 2048))
        with self.assertRaises(ValueError):
            m.parse_gpu_query('GPU-a, 24000, 0, 24000\nGPU-b, 24000, 0, 24000')
        for visible in ('', '0,1', '-1'):
            with self.assertRaises(ValueError):
                m.assigned_gpu(visible)
        self.assertEqual(m.assigned_gpu('GPU-test'), 'GPU-test')

    def test_validation_gate_binds_real_code_binary_and_output_contract(self):
        m = pilot()
        with tempfile.TemporaryDirectory(prefix='triclinic_pilot_test_', dir=ROOT) as directory:
            root = Path(directory)
            native = root/'native/string_triclinic_v1'
            native.mkdir(parents=True)
            exe = native/'GPU_string_triclinic'
            exe.write_bytes(b'not executed, source-gate fixture')
            exe.chmod(0o700)
            source = native/'RC_main.cu'
            source.write_text('fixture')
            gate = dict(passed=True, executable_sha256=m.digest(exe.read_bytes()),
                        source_hashes={'RC_main.cu': m.digest(source.read_bytes())},
                        checks={'actual_cuda_S0': True, 'actual_cuda_gradient': True},
                        output_contract={'path_policy': 'final_converged_string', 'info_marker': 'info:',
                                         'check_interval': 200, 'step_log': None})
            report = root/'validation.json'
            report.write_text(json.dumps(gate))
            m.validation_gate(root, exe, report)
            for mutated in ({**gate, 'passed': False}, {**gate, 'executable_sha256': '0'*64},
                            {**gate, 'checks': {'actual_cuda_S0': True, 'actual_cuda_gradient': False}},
                            {**gate, 'output_contract': {**gate['output_contract'], 'path_policy': 'legacy_D1_D2_choice'}}):
                report.write_text(json.dumps(mutated))
                with self.assertRaises(ValueError):
                    m.validation_gate(root, exe, report)
            report.write_text(json.dumps(gate))
            source.write_text('changed')
            with self.assertRaises(ValueError):
                m.validation_gate(root, exe, report)

    def test_versioned_executable_is_hash_bound_without_replacing_original(self):
        m = pilot()
        with tempfile.TemporaryDirectory(prefix='triclinic_pilot_test_', dir=ROOT) as directory:
            root = Path(directory)
            native = root/'native/string_triclinic_v1'
            native.mkdir(parents=True)
            exe = native/'GPU_string_triclinic_final_v2'
            exe.write_bytes(b'versioned test fixture, not executed')
            exe.chmod(0o700)
            source = native/'RC_main_final_v2.cu'
            source.write_text('fixture')
            gate = dict(passed=True, executable_sha256=m.digest(exe.read_bytes()),
                        source_hashes={'RC_main_final_v2.cu': m.digest(source.read_bytes())},
                        checks={'actual_cuda_S0':True, 'actual_cuda_gradient':True},
                        output_contract={'path_policy':'final_converged_string', 'info_marker':'info:',
                                         'check_interval':200, 'step_log':None})
            report = root/'validation.json'
            report.write_text(json.dumps(gate))
            m.validation_gate(root, exe, report)

    def test_paired_comparisons_exclude_failures_and_do_not_invent_ratios(self):
        m = pilot()
        a = dict(status='valid', logD=-8., D_m2_s=1e-8, path_barrier_K=100.)
        b = dict(status='valid', logD=-10., D_m2_s=1e-10, path_barrier_K=200.)
        values = m.pair_values(b, a)
        self.assertEqual(values['delta_logD'], -2.)
        self.assertAlmostEqual(values['ratio_new_over_reference'], .01)
        self.assertAlmostEqual(values['absolute_fold_ratio'], 100.)
        self.assertIsNone(m.pair_values({**b, 'status': 'nonconverged', 'logD': None}, a))
        huge = m.pair_values({**b, 'logD': -10000.}, a)
        self.assertIsNone(huge['ratio_new_over_reference'])
        self.assertIsNone(huge['absolute_fold_ratio'])

    def test_triclinic_minimum_image_is_not_fractional_component_rounding(self):
        m = pilot()
        cell = np.array([[10.,0,0],[9.9,.5,0],[0,0,10.]])
        delta = np.array([[.49,.49,0.]])
        vectors = m.periodic_vectors(delta,cell)
        brute = min(np.linalg.norm((delta[0]-[i,j,k])@cell)
                    for i in range(-3,4) for j in range(-3,4) for k in range(-1,2))
        self.assertAlmostEqual(float(np.linalg.norm(vectors[0])),brute,places=10)
        self.assertLess(float(np.linalg.norm(vectors[0])),.5)
        self.assertGreater(float(np.linalg.norm((delta-np.rint(delta))@cell)),9.)

    def test_periodic_path_comparison_handles_images_cyclic_phase_and_reversal(self):
        m = pilot()
        cell = np.diag([30.,31.,32.])
        p = pose()
        p[:,6] = 60+100*(1-np.cos(2*np.pi*p[:,0]))
        unique = np.roll(p[:-1],100,axis=0)[::-1].copy()
        unique[:,:3] = np.mod(unique[:,:3],1)+[2,-3,1]
        changed = np.vstack([unique,unique[0]])
        score = m.compare_paths(changed,p,cell,samples=32)
        self.assertLess(score['periodic_COM_rmsd_A'],1e-8)
        self.assertLess(score['relative_energy_profile_rmse_K'],1e-8)
        self.assertTrue(score['alignment_reversed'])
        self.assertEqual(score['alignment_policy'],'geometry_only_periodic_cyclic_arclength')

    def test_energy_zero_shift_does_not_look_like_barrier_or_relative_profile_change(self):
        m = pilot()
        p = pose()
        changed = p.copy()
        changed[:,6] += 500.
        score = m.compare_paths(changed,p,np.diag([30.,31.,32.]),samples=32)
        self.assertLess(score['periodic_COM_rmsd_A'],1e-8)
        self.assertLess(score['relative_energy_profile_rmse_K'],1e-8)
        self.assertAlmostEqual(score['raw_energy_profile_mae_K'],500.)
        self.assertAlmostEqual(score['minimum_energy_delta_K'],500.)
        self.assertAlmostEqual(score['barrier_delta_K'],0.)

    def test_real_transverse_path_change_is_not_fitted_away(self):
        m = pilot()
        p = pose()
        changed = p.copy()
        changed[:,1] += .1
        score = m.compare_paths(changed,p,np.diag([30.,31.,32.]),samples=32)
        self.assertAlmostEqual(score['periodic_COM_rmsd_A'],3.1,places=8)
        self.assertAlmostEqual(score['periodic_chamfer_mean_A'],3.1,places=8)

    def test_director_sign_equivalence_and_nonfinite_path_rejection(self):
        m = pilot()
        p = pose()
        changed = p.copy()
        changed[:,5] += np.pi
        score = m.compare_paths(changed,p,np.diag([30.,31.,32.]),samples=32)
        self.assertLess(score['director_angle_mean_deg'],1e-6)
        changed[4,6] = np.nan
        with self.assertRaises(ValueError):
            m.compare_paths(changed,p,np.diag([30.,31.,32.]))

    def test_pair_queue_keeps_same_material_gas_direction_arms_adjacent(self):
        m = pilot()
        tasks = [dict(material_id=i,gas=g,direction=d,arm=a) for i in range(10)
                 for g in ('C2H4','C2H6') for d in (1,2,3) for a in ('current','old_sigma_epsilon_only')]
        queued = m.paired_tasks(list(reversed(tasks)),0)
        self.assertEqual(len(queued),60)
        for first,second in zip(queued[::2],queued[1::2]):
            self.assertEqual((first['material_id'],first['gas'],first['direction']),
                             (second['material_id'],second['gas'],second['direction']))
            self.assertEqual((first['arm'],second['arm']),('current','old_sigma_epsilon_only'))
            self.assertEqual(first['material_id'] % 2,0)

    def test_memory_admission_reserves_not_yet_allocated_peak_and_host_UM_budget(self):
        m = pilot()
        active = [dict(estimated_gpu_MiB=2048,allocated_gpu_MiB=0,estimated_host_MiB=4096,host_rss_MiB=0)]
        candidate = dict(estimated_gpu_MiB=1024,estimated_host_MiB=4096)
        self.assertFalse(m.memory_admission(6000,active,candidate,4096,8,49152))
        active[0]['allocated_gpu_MiB'] = 2048
        self.assertTrue(m.memory_admission(6000,active,candidate,4096,8,49152))
        self.assertFalse(m.memory_admission(6000,active,candidate,4096,1,49152))
        self.assertFalse(m.memory_admission(6000,active,candidate,4096,8,10000))

    def test_memory_estimate_is_not_fixed_562_and_scales_with_atoms(self):
        m = pilot()
        profile = dict(observed_increment_MiB=561,framework_atoms=275,safety_factor=2.)
        small = m.task_memory_estimate(dict(semantics={'framework_atoms':275}),profile)
        bigger = m.task_memory_estimate(dict(semantics={'framework_atoms':543}),profile)
        self.assertGreaterEqual(small['estimated_gpu_MiB'],1024)
        self.assertGreater(bigger['estimated_gpu_MiB'],small['estimated_gpu_MiB'])
        self.assertGreaterEqual(bigger['estimated_host_MiB'],4096)

    def test_early_ten_means_ten_complete_valid_two_arm_pairs_not_ten_paths(self):
        m = pilot()
        rows = [dict(material_id=i,gas='C2H4',direction=1,arm='current',status='valid',logD=-8.) for i in range(10)]
        self.assertEqual(m.valid_pair_count(rows),0)
        rows += [{**r,'arm':'old_sigma_epsilon_only','logD':-9.} for r in rows]
        self.assertEqual(m.valid_pair_count(rows),10)
        rows[-1]['status']='nonconverged'
        self.assertEqual(m.valid_pair_count(rows),9)
        rows[-1]['status']='valid'; rows[-1]['logD']=None
        self.assertEqual(m.valid_pair_count(rows),9)

    def test_parallel_dispatch_keeps_running_all_sixty_after_early_threshold(self):
        import itertools
        m = pilot()
        tasks = [dict(material_id=i,name=f'n{i}',gas=g,direction=d,arm=a,
                      input_relative=f'{i}_{g}_{d}_{a}',semantics={'framework_atoms':275})
                 for i in range(10) for g in ('C2H4','C2H6') for d in (1,2,3) for a in m.ARMS]
        plan = dict(materials=[dict(material_id=i,name=f'n{i}') for i in range(10)],cases=tasks,code_hashes={})
        launched = []
        def worker(task,*_args):
            launched.append(task)
            return dict(material_id=task['material_id'],gas=task['gas'],direction=task['direction'],
                        arm=task['arm'],status='valid',logD=-8.)
        with tempfile.TemporaryDirectory(prefix='parallel_pilot_test_',dir=ROOT) as directory:
            root = Path(directory)
            output = root/m.PREFIX/'rank0_test'
            with patch.dict(m.os.environ,{'SLURM_JOB_ID':'test','CUDA_VISIBLE_DEVICES':'0','SLURM_GPUS_ON_NODE':'1'}), \
                 patch.object(m.socket,'gethostname',return_value='gpu'), \
                 patch.object(m,'validation_gate',return_value=({}, {'conditioning_warning':'retained'})), \
                 patch.object(m,'read_plan',return_value=(plan,'fixed')), \
                 patch.object(m,'load_memory_profile',return_value={'framework_atoms':275,'observed_increment_MiB':561,'safety_factor':2}), \
                 patch.object(m,'gpu_query',return_value={'free_MiB':24000,'used_MiB':0,'total_MiB':24576}), \
                 patch.object(m,'gpu_process_memory',return_value={}), \
                 patch.object(m,'execute_case',side_effect=worker), \
                 patch.object(m,'maybe_early') as early, \
                 patch('builtins.print'), \
                 patch.object(m.time,'sleep'), \
                 patch.object(m.time,'monotonic',side_effect=lambda:next(clock)):
                clock = itertools.count()
                report = m.run(root,root/'prepared',output,0,root/'exe',root/'gate',memory_profile=root/'profile',
                               max_concurrent=3,stagger=1)
            self.assertEqual(len(launched),60)
            self.assertEqual(len(report['outcomes']),60)
            self.assertEqual(report['valid_two_arm_pairs'],30)
            self.assertTrue(early.called)
            self.assertTrue(report['completed'])

    def test_early_collector_writes_twenty_valid_paths_without_waiting_for_complete_shards(self):
        m = pilot()
        with tempfile.TemporaryDirectory(prefix='early_pilot_test_',dir=ROOT) as directory:
            root = Path(directory)
            parent = root/m.PREFIX
            prepared = parent/'prepared'
            prepared.mkdir(parents=True)
            input_file = prepared/'input.dat'
            input_file.write_text(INPUT)
            original = parent/'original_path.dat'
            np.savetxt(original,pose())
            reference = dict(reference_valid_for_comparison=True,saved_path=str(original),
                path_sha256=m.file_hash(original),saved_path_uncapped_logD=-8.,saved_path_barrier_K=100.)
            tasks = [dict(material_id=i,gas=g,direction=d,arm=a,input_relative='input.dat',
                          input_sha256=m.file_hash(input_file)) for i in range(10) for g in m.GASES
                     for d in (1,2,3) for a in m.ARMS]
            plan = dict(cases=tasks,materials=[{'material_id':i} for i in range(10)],selection={})
            shards=[]
            for rank in (0,1):
                shard=parent/f'rank{rank}_test'; shard.mkdir(); outcomes=[]
                for i in range(rank,10,2):
                    for arm in m.ARMS:
                        work=shard/f'{i}_{arm}'; work.mkdir()
                        path=work/'string_path.dat'; np.savetxt(path,pose())
                        row=dict(material_id=i,name=f'n{i}',gas='C2H4',direction=1,arm=arm,status='valid',
                            logD=-8.1 if arm!=m.ARMS[0] else -8.,path_barrier_K=100.,reference=reference,
                            work_relative=work.name,output_path=work.name+'/string_path.dat',
                            output_path_sha256=m.file_hash(path))
                        m.write_json(work/'outcome.json',row); outcomes.append(row)
                m.write_json(shard/'progress.json',dict(rank=rank,completed=False,plan_sha256='frozen',
                    native_binding={'conditioning_warning':'not all strict'},outcomes=outcomes))
                shards.append(shard)
            output=parent/'early_comparison_test'
            with patch.object(m,'read_plan',return_value=(plan,'frozen')), \
                 patch.object(m,'compare_paths',return_value={'path_comparison_status':'valid',
                     'periodic_COM_rmsd_A':0.,'relative_energy_profile_rmse_K':0.}),patch('builtins.print'):
                m.collect(root,prepared,shards,output,early=True)
            report=json.loads((output/'report.json').read_text())
            self.assertTrue(report['early_snapshot'])
            self.assertFalse(report['full_experiment_completed'])
            self.assertEqual(report['selected_valid_two_arm_pairs'],10)
            self.assertEqual(report['valid_paths_in_selected_pairs'],20)
            self.assertEqual(report['planned_materials'],10)


if __name__=='__main__':
    unittest.main()
