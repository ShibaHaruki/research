"""Calibrate gains, record all material states, and export one debug trial per material.

python f_run/calibrate_liquid.py plan
python f_run/calibrate_liquid.py run --out-dir g_tactile_results/sequential_calibration
python f_run/calibrate_liquid.py run --out-dir ... --resume
python f_run/calibrate_liquid.py verify --out-dir ...
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'f_run'))
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_key, '1')

import numpy as np
from d_tools.activity import summarize_ei_activity

# ①–⑩の条件はここで管理する。通常実行のFIXED設定は書き換えない。
SETTINGS = {
    'neurons': 100, 'excitatory': 80, 'inhibitory': 20,
    'input_probability': .05,
    'probabilities': {'EE': .11, 'EI': .10, 'IE': .37, 'II': .10},
    'input_material': 'washi', 'recurrent_material': 'washi',
    'materials': ['Al_board', 'buta_omote', 'buta_ura', 'cork', 'denim',
                  'rubber_board', 'washi', 'wood_board'],
    'trials_per_material': 324, 'duration_s': .5, 'dt_s': .0001,
    'bin_ms': 25., 'tau_membrane_ms': 10., 'refractory_ms': 2.,
    'threshold_mV': -40., 'reset_mV': -65., 'bias_mV': -65.,
    'syn_tau_r_ms': 2., 'syn_tau_d_ms': 30.,
    'EE_EI_coarse_step': .01, 'EE_EI_fine_step': .001, 'EE_EI_max': 1.,
    'IE_grid': [0., .00001, .00002, .00005, .0001, .0002, .0005, .001,
                .002, .005, .01, .02, .05, .1, .2, .5, 1.],
    'IE_refinements': 2, 'debug_trial_id': 203,
    'state_policy': 'reset_each_trial',
}
REFERENCE = ROOT / 'c_configs/FIXED/calibration_reference.npz'


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def load_reference():
    fixed = {'neurons': 100, 'excitatory': 80, 'inhibitory': 20, 'input_probability': .05,
             'probabilities': {'EE': .11, 'EI': .10, 'IE': .37, 'II': .10},
             'trials_per_material': 324, 'duration_s': .5, 'dt_s': .0001, 'bin_ms': 25.,
             'tau_membrane_ms': 10., 'refractory_ms': 2., 'threshold_mV': -40.,
             'reset_mV': -65., 'bias_mV': -65., 'syn_tau_r_ms': 2., 'syn_tau_d_ms': 30.,
             'state_policy': 'reset_each_trial'}
    if any(SETTINGS[key] != value for key, value in fixed.items()):
        raise ValueError('Fixed topology/simulator conditions changed; update and validate the reference and accelerator first.')
    if len(SETTINGS['materials']) != 8:
        raise ValueError('Reference splits require the eight reference materials in their original order.')
    metadata = json.loads(REFERENCE.with_suffix('.json').read_text(encoding='utf-8'))
    if SETTINGS['materials'] != metadata['materials']:
        raise ValueError('Material order differs from reference holdout splits')
    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != metadata['sha256']:
        raise ValueError('Reference topology / splits checksum mismatch')
    with np.load(REFERENCE) as data:
        ref = {k: data[k].copy() for k in data.files}
    wi, wr = ref['input_unit_weights'], ref['recurrent_unit_weights']
    assert wi.shape == (6, 100) and wr.shape == (100, 100)
    assert np.all(wi[:, 80:] == 0) and np.all(wr[80:, 80:] == 0)
    for block in [wr[:80, :80], wr[:80, 80:], wr[80:, :80]]:
        assert np.all(np.count_nonzero(block, axis=0) > 0)
    assert np.all(np.diag(wr) == 0)
    return ref


def config(gains):
    """Apply this experiment's explicit conditions to an independent config copy."""
    from f_run.run_common import build_cfg
    cfg = deepcopy(build_cfg())
    common = cfg['common']
    if (common['NUM_SENSOR'] != 3 or not np.isclose(common['dt_ms'], .0001)
            or common['SLICE_END'] - common['SLICE_START'] != 5000
            or cfg['input_filter_map'] != {0: ['RI', 'SI'], 1: ['RI', 'SI'], 2: ['RI', 'SI']}):
        raise ValueError('This calibrated simulator requires 3 RI/SI sensors, 500 ms and dt=0.1 ms.')

    net = cfg['network']
    net['N_liq'] = [SETTINGS['neurons']]
    net['r_inh_liq'] = SETTINGS['inhibitory']/SETTINGS['neurons']
    # Set all six RI/SI routes explicitly; do not let the shared overlay replace them.
    if isinstance(net.get('SHARED_IN_ROUTE'), dict):
        net['SHARED_IN_ROUTE']['enabled'] = False
    for (_, filter_name), route in net['IN_ROUTE'].items():
        active = str(filter_name).lower() in ('ri', 'si')
        for layer in route.get('layers', {}).values():
            layer['p'] = {'E': SETTINGS['input_probability'] if active else 0., 'I': 0.}
            if active:
                layer['scale'] = {'E': float(gains['input_gain']), 'I': 0.}
    net['p_liq_intra_pairs'] = dict(SETTINGS['probabilities'])
    net['liq_intra_gain_pairs'] = {pair: gains.get(pair+'_gain', 0.) for pair in ['EE', 'EI', 'IE', 'II']}
    net['poisson_input']['enabled'] = False
    cfg['opt_filter_gain'].update(RI=1., SI=1.)
    cfg['neuron_models']['LIF'].update(
        tau_exc=SETTINGS['tau_membrane_ms'], tau_inh=SETTINGS['tau_membrane_ms'],
        ref_exc=SETTINGS['refractory_ms'], ref_inh=SETTINGS['refractory_ms'],
        bias=SETTINGS['bias_mV'], v_reset=SETTINGS['reset_mV'], v_thr=SETTINGS['threshold_mV'])
    cfg['models']['SYNAPSE_MODEL'] = 'double_exp'
    cfg['synapse_models']['double_exp'].update(tau_r=SETTINGS['syn_tau_r_ms'], tau_d=SETTINGS['syn_tau_d_ms'])
    cfg['run'].update(LIVE_PLOT_ENABLE=False, INTERNAL_STATE_PCA_ENABLE=False,
                      BRIAN_CODEGEN_TARGET='numpy')
    return cfg


def provenance():
    # Filter or input-window changes must not silently reuse cached currents/results.
    files = ['a_tactile/input_filters.py', 'a_tactile/experiment_sets.py',
             'c_configs/FIXED/cfg_filter.py', 'f_run/fast_liquid_calibration.pyx',
             'f_run/calibrate_liquid.py', 'f_run/run_common.py',
             'b_network/build_network.py', 'b_network/monitor.py', 'd_tools/activity.py',
             'd_tools/plotting.py', 'd_tools/weight_export.py',
             'c_configs/FIXED/calibration_reference.npz']
    cfg = config({'input_gain': 1.})
    return {'settings': SETTINGS,
            'source_sha256': {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
            'data_root': str(__import__('f_run.run_common', fromlist=['TACTILE_DATA_PATH']).TACTILE_DATA_PATH),
            'data_subdir': cfg['run']['TACTILE_DATA_DIR_NAME'],
            'base_seed': int(cfg['common']['BASE_SEED'])}


def filtered_trial(material, sid, cfg):
    import pandas as pd
    from f_run.run_common import build_input_current, load_tactile_data
    common = cfg['common']
    raw = pd.read_table(load_tactile_data(material, sid), header=None).to_numpy().T
    signal = raw[:3, common['SLICE_START']:common['SLICE_END']]
    if signal.shape != (3, 5000):
        raise ValueError(f'Unexpected signal shape: {material} sid{sid}: {signal.shape}')
    current = build_input_current(signal, np.arange(5000)*.0001, .0001,
        cfg['input_filter_map'], cfg['filter_funcs'], cfg['sensor_gain'],
        cfg['filter_gain'], cfg['opt_filter_gain'])
    if current.shape != (6, 5000) or not np.isfinite(current).all():
        raise ValueError(f'Invalid filtered current: {material} sid{sid}')
    return current.T


def currents(material, out):
    """Build/cache 324 filtered trials; source signals are not copied to results."""
    path = out / 'filtered_inputs' / f'{material}.npy'
    if path.exists():
        cached = np.load(path, mmap_mode='c')
        if cached.shape != (324, 5000, 6) or cached.dtype != np.float64:
            raise ValueError(f'Unexpected input-cache shape/dtype: {path}')
        return cached
    cfg = config({'input_gain': 1.})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.npy')
    x = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float64, shape=(324, 5000, 6))
    for sid in range(1, 325):
        x[sid-1] = filtered_trial(material, sid, cfg)
        if sid % 25 == 0 or sid == 324:
            print(f'[inputs] {material}: {sid}/324', flush=True)
    x.flush()
    del x
    temporary.replace(path)
    return np.load(path, mmap_mode='c')


def weights(ref, gains):
    wi = np.ascontiguousarray(ref['input_unit_weights'] * gains['input_gain'])
    wr = ref['recurrent_unit_weights'].copy()
    for pair, block in [('EE', (slice(0, 80), slice(0, 80))),
                        ('EI', (slice(0, 80), slice(80, 100))),
                        ('IE', (slice(80, 100), slice(0, 80)))]:
        wr[block] *= gains.get(pair + '_gain', 0.)
    return wi, np.ascontiguousarray(wr)


def simulate(x, ref, gains):
    from fast_liquid_calibration import simulate as fast_simulate
    wi, wr = weights(ref, gains)
    # The accelerator has fixed dimensions. Check before passing memoryviews.
    if x.ndim != 3 or x.shape[1:] != (5000, 6):
        raise ValueError('Expected trial x 5000 time steps x 6 input channels')
    return fast_simulate(np.ascontiguousarray(x), wi, wr, carry=False)


def input_gain(x, ref, out):
    """③ First-spike boundary is linear before any reset; verify adjacent integers."""
    from scipy.signal import lfilter
    ids = np.flatnonzero(np.any(ref['input_unit_weights'] != 0, axis=0))
    threshold = 0.
    critical = None
    for i, trial in enumerate(x):
        response = lfilter([.0001/.010], [1, -(1-.0001/.010)],
                          (trial @ ref['input_unit_weights'][:, ids]).T, axis=1)
        peaks = response.max(1)
        if np.any(peaks <= 0):
            raise ValueError('Some connected neurons have no finite positive first-spike threshold')
        required = 25./peaks
        j = int(np.argmax(required))
        if required[j] > threshold:
            threshold = float(required[j]); critical = {'sid': i+1, 'neuron_id': int(ids[j])}
    gain = int(np.ceil(threshold))
    counts = simulate(x, ref, {'input_gain': gain})
    if not np.all(counts.sum(2)[:, ids] > 0):
        raise AssertionError('Predicted first-spike gain failed full-trial verification')
    lower = simulate(x[critical['sid']-1:critical['sid']], ref, {'input_gain': gain-1})
    assert np.any(lower.sum(2)[:, ids] == 0), 'Lower integer unexpectedly passed'
    np.savez_compressed(out/'input_features.npz', counts_25ms=counts, connected_neuron_ids=ids)
    save(out/'input_summary.json', {'input_gain': gain, 'continuous_boundary': threshold,
         'critical': critical, 'connected_neurons': ids.tolist(), 'all_trials_all_connected_fired': True,
         'minimum_scope': 'integer gains, fixed topology, fresh states',
         **summarize_ei_activity(counts, n_excitatory=SETTINGS['excitatory'], bin_ms=SETTINGS['bin_ms'])})
    return gain


def recurrent_gains(x, ref, gains, out):
    """⑤⑦⑨ Sequential gain selection without assuming recurrent monotonicity."""
    import pandas as pd
    history = []
    def evaluate(stage, candidate):
        bins = simulate(x, ref, candidate)
        row = {'stage': stage, **candidate,
               **summarize_ei_activity(bins, n_excitatory=SETTINGS['excitatory'], bin_ms=SETTINGS['bin_ms'])}
        history.append(row)
        pd.DataFrame(history).to_csv(out/'search_history.csv', index=False)
        print(f'[{stage}] {candidate}: allE={row["all_E_trials"]}, '
              f'allI={row["all_I_trials"]}, EHz={row["mean_E_rate_hz"]:.3f}', flush=True)
        return row, bins
    for stage, target in [('EE', 'all_E_trials'), ('EI', 'all_I_trials')]:
        selected = None
        previous = 0.
        # Every candidate is evaluated over all 324 trials.
        # Retain the historical coarse .01 + local .001 refinement strategy.
        for tick in range(1, round(SETTINGS['EE_EI_max']/SETTINGS['EE_EI_coarse_step'])+1):
            coarse = round(tick*SETTINGS['EE_EI_coarse_step'], 9)
            row, _ = evaluate(stage, {**gains, stage+'_gain': coarse})
            if row[target] == 324:
                step = SETTINGS['EE_EI_fine_step']
                for fine_tick in range(round(previous/step)+1, round(coarse/step)+1):
                    candidate = {**gains, stage+'_gain': round(fine_tick*step, 9)}
                    result, bins = evaluate(stage, candidate)
                    if result[target] == 324:
                        selected = candidate
                        np.savez_compressed(out/f'{stage}_features.npz', counts_25ms=bins)
                        save(out/f'{stage}_summary.json', result)
                        break
                if selected is not None:
                    break
            previous = coarse
        if selected is None:
            raise RuntimeError(f'No passing {stage} gain in configured search range')
        gains = selected
    candidates = []
    for gain in SETTINGS['IE_grid']:
        row, _ = evaluate('IE', {**gains, 'IE_gain': gain})
        candidates.append(row)
    def best_candidate():
        feasible = [r for r in candidates if not r['never_active_E'] and not r['never_active_I']]
        if not feasible:
            raise RuntimeError('No feasible IE gain: no candidate activates all neurons across 324 trials')
        return min(feasible, key=lambda r: r['mean_E_rate_hz'])
    for _ in range(SETTINGS['IE_refinements']):
        best = best_candidate()
        tested = sorted({r['IE_gain'] for r in candidates})
        pos = tested.index(best['IE_gain'])
        lo, hi = tested[max(0, pos-1)], tested[min(len(tested)-1, pos+1)]
        for gain in np.linspace(lo, hi, 9):
            gain = round(float(gain), 9)
            if gain not in tested:
                row, _ = evaluate('IE', {**gains, 'IE_gain': gain})
                candidates.append(row)
    gains['IE_gain'] = best_candidate()['IE_gain']
    row, bins = evaluate('IE', gains)
    np.savez_compressed(out/'IE_features.npz', counts_25ms=bins)
    save(out/'IE_summary.json', row)
    save(out/'selected_parameters.json', gains)
    return gains


def verify_brian(x, ref, gains, out, *, debug_title=None):
    """Independent exact-count check of LIF/synaptic equations and topology."""
    from brian2 import prefs, start_scope, defaultclock, ms, second, seed, TimedArray
    from f_run.run_common import build_network_cfg
    from b_network.build_network import make_liquid_network
    from d_tools.weight_export import save_liquid_weight_matrices
    cfg = config(gains); nc = build_network_cfg(cfg)
    start_scope(); prefs.codegen.target = 'numpy'; defaultclock.dt = .1*ms
    base_seed = int(cfg['common']['BASE_SEED']); np.random.seed(base_seed); seed(base_seed)
    objects = make_liquid_network(nc, 6, np.random.default_rng(base_seed), cfg['run'])
    out.mkdir(parents=True, exist_ok=True)
    save_liquid_weight_matrices(out/'weights', objects)
    wi, wr = weights(ref, gains)
    assert np.allclose(np.load(out/'weights/w_in1_liq1.npy'), wi, atol=1e-12, rtol=1e-12)
    assert np.allclose(np.load(out/'weights/w_liq1_intra.npy'), wr, atol=1e-12, rtol=1e-12)
    for monitor in objects['M_liq_debug']:
        monitor.active = True
    if debug_title is not None:
        objects['V_liq'][0].active = True
    current = np.asarray(x[0]); ta = TimedArray(current, dt=.1*ms)
    objects['G_in'].namespace['input_ta'] = ta
    # A newly constructed network starts at t=0 even if the shared clock still
    # displays the previous verification run's time before Network.run().
    objects['G_in'].t_start = objects['net'].t
    objects['net'].run(.5*second, namespace={'input_ta': ta})
    m = objects['M_liq_debug'][0]
    actual = np.zeros((100, 20), np.int32)
    np.add.at(actual, (np.asarray(m.i, int), np.floor((np.asarray(m.t/ms)+1e-5)/25).astype(int)), 1)
    expected = simulate(x[:1], ref, gains)[0]
    assert np.array_equal(actual, expected), 'Accelerator / Brian2 spike-count mismatch'
    if debug_title is not None:
        from d_tools.plotting import save_liquid_debug
        voltage = objects['V_liq'][0]
        neuron_ids = np.asarray(objects['V_liq_indices'][0], dtype=int)
        debug_data = dict(spike_times_ms=np.asarray(m.t/ms), spike_neuron_ids=np.asarray(m.i, int),
                          voltage_times_ms=np.asarray(voltage.t/ms), voltage_mV=np.asarray(voltage.v),
                          voltage_neuron_ids=neuron_ids, counts_25ms=actual,
                          bin_ms=SETTINGS['bin_ms'], trial_id=SETTINGS['debug_trial_id'])
        np.savez_compressed(out/'debug.npz', **debug_data)
        save_liquid_debug(debug_data, out/'debug.png', title=debug_title,
                          n_excitatory=SETTINGS['excitatory'])
    result = {'exact_25ms_counts_match': True, 'gains': gains, 'n_trials_checked': 1}
    save(out/'verification.json', result)
    print('[verified] exact spike counts and weights match Brian2', flush=True)
    return result


def record_material_states(ref, gains, out):
    """⑩ 全素材の全試行を記録する。分類・PCA・素材別活動評価は行わない。"""
    sid = SETTINGS['debug_trial_id']
    if not 1 <= sid <= SETTINGS['trials_per_material']:
        raise ValueError('debug_trial_id must be between 1 and 324')
    records = []
    for material in SETTINGS['materials']:
        folder = out/'all_materials'/material
        file = folder/'features.npz'
        trial_ids = np.arange(1, SETTINGS['trials_per_material']+1)
        if file.exists():
            with np.load(file) as data:
                bins = data['counts_25ms'].copy()
                if not np.array_equal(data['trial_ids'], trial_ids):
                    raise ValueError(f'Unexpected trial IDs: {file}')
        else:
            bins = simulate(currents(material, out), ref, gains)
        if bins.shape != (324, 100, 20) or np.any(bins < 0):
            raise ValueError(f'Unexpected spike-count states: {material}: {bins.shape}')
        folder.mkdir(parents=True, exist_ok=True)
        # 既存のPCAノートブック用の counts_25ms（試行×ニューロン×時間）は保持。
        # states_hz は各試行を20時間ビン×100ニューロンとしてそのまま取り出せる。
        states_hz = bins.transpose(0, 2, 1) / (SETTINGS['bin_ms']/1000.)
        temporary = folder/'features.partial.npz'
        np.savez_compressed(temporary, states_hz=states_hz, counts_25ms=bins,
                            trial_ids=trial_ids, bin_ms=SETTINGS['bin_ms'],
                            neuron_ids=np.arange(100), neuron_types=np.array(['E']*80+['I']*20),
                            time_bin_edges_ms=np.arange(21)*SETTINGS['bin_ms'])
        temporary.replace(file)
        # debugは各素材の代表1試行のみ。全324試行の膜電位は記録しない。
        trial = filtered_trial(material, sid, config(gains))
        verify_brian(trial[None], ref, gains, folder/f'debug_sid{sid}',
                     debug_title=f'{material} | Trial {sid} | 100 liquid neurons')
        with np.load(folder/f'debug_sid{sid}'/'debug.npz') as debug_data:
            if not np.array_equal(bins[sid-1], debug_data['counts_25ms']):
                raise ValueError(f'Saved states differ from debug trial: {material}, sid{sid}')
        records.append({'material': material, 'file': str(file.relative_to(out)),
                        'states_shape': list(states_hz.shape), 'debug_trial_id': sid})
        print(f'[recorded] {material}: {states_hz.shape} (trial, time bin, neuron)', flush=True)
    save(out/'summary.json', {'status': 'complete', 'selected_parameters': gains,
         'n_trials': 2592, 'state_key': 'states_hz', 'state_unit': 'Hz',
         'state_axes': ['trial', 'time_bin', 'neuron'], 'bin_ms': SETTINGS['bin_ms'],
         'state_policy': SETTINGS['state_policy'], 'classification_performed': False,
         'PCA_performed': False, 'materials': records})


def plan():
    ref = load_reference()
    connected = np.flatnonzero(np.any(ref['input_unit_weights'] != 0, axis=0))
    print(json.dumps({'settings': SETTINGS, 'connected_E_ids': connected.tolist(),
         'expected_direct_fraction': 1-(1-.05)**6,
         'actual_direct_count': len(connected),
         'all_have_recurrent_input_probability': {'EE': (1-(1-.11)**79)**80,
              'EI': (1-(1-.10)**80)**20, 'IE': (1-(1-.37)**20)**80},
         'steps': ['1: fix 100 neurons, E80/I20 and LIF/synapse parameters',
                   '2: input p=.05; reuse fixed topology',
                   '3: integer input gain; no recurrence; all connected neurons in each trial',
                   '4: EE p=.11', '5: EE gain; EI/IE/II gains zero; E80 in each trial',
                   '6: EI p=.10', '7: EI gain; IE/II gains zero; I20 in each trial',
                   '8: IE p=.37', '9: IE gain; every E/I neuron across all 324 trials; minimize E rate',
                   '10: record 8 x 324 fresh states (20 time bins x 100 neurons); one debug trial per material'],
         'minimum_scope': 'fixed topology and screened gains; recurrent response is not assumed monotonic',
         'note': 'Washi is the selected calibration material; no claim it is both smallest and largest input.'},
          ensure_ascii=False, indent=2))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run', 'verify'])
    parser.add_argument('--out-dir', type=Path, default=ROOT/'g_tactile_results/sequential_calibration')
    parser.add_argument('--resume', action='store_true', help='Reuse completed stages with identical settings/source hashes')
    args = parser.parse_args()
    if args.action == 'plan':
        plan(); return
    out = args.out_dir.resolve(); ref = load_reference()
    if args.action == 'verify':
        out.mkdir(parents=True, exist_ok=True)
        gain_file = out/'selected_parameters.json'
        gains = json.loads(gain_file.read_text()) if gain_file.exists() else {
            'input_gain': 179., 'EE_gain': .150, 'EI_gain': .206, 'IE_gain': .01390625}
        trial = filtered_trial(SETTINGS['input_material'], 203, config(gains))
        verify_brian(trial[None], ref, gains, out/'verification/smoke')
        return
    manifest = provenance()
    if out.exists():
        if not args.resume:
            parser.error('Output directory exists. Use --resume with identical conditions or a new --out-dir.')
        if not (out/'plan.json').exists() or json.loads((out/'plan.json').read_text(encoding='utf-8')) != manifest:
            parser.error('Resume settings/source hashes do not match this run.')
    else:
        out.mkdir(parents=True)
        save(out/'plan.json', manifest)
    gain_file = out/'selected_parameters.json'
    if gain_file.exists():
        gains = json.loads(gain_file.read_text())
    else:
        x = currents(SETTINGS['input_material'], out)
        summary = out/'input_summary.json'
        gain = json.loads(summary.read_text())['input_gain'] if summary.exists() else input_gain(x, ref, out)
        verify_brian(x[202:203], ref, {'input_gain': gain}, out/'verification/input')
        x = currents(SETTINGS['recurrent_material'], out)
        gains = recurrent_gains(x, ref, {'input_gain': gain, 'II_gain': 0.}, out)
    # Verify even when gains were loaded after an interrupted run.
    x = currents(SETTINGS['recurrent_material'], out)
    for stage in ['EE', 'EI', 'IE']:
        selected = {**gains, 'IE_gain': gains['IE_gain'] if stage == 'IE' else 0.,
                    'EI_gain': gains['EI_gain'] if stage != 'EE' else 0.}
        verify_brian(x[202:203], ref, selected, out/'verification'/stage)
    record_material_states(ref, gains, out)
    print(f'[complete] {out}', flush=True)


if __name__ == '__main__':
    main()
