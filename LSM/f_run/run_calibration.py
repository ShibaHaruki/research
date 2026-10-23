"""Calibrate liquid/output gains, match population firing rates, and record all states.

python f_run/run_calibration.py plan
python f_run/run_calibration.py run  # saves to g_tactile_results/calibration
python f_run/run_calibration.py run --out-dir ... --resume
python f_run/run_calibration.py verify --out-dir ...
python f_run/run_calibration.py output --liquid-dir g_tactile_results/calibration --out-dir g_tactile_results/calibration_output
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

# リキッド層と出力層の校正条件はここで管理する。通常実行のFIXED設定は書き換えない。
SETTINGS = {
    'neurons': 100, 'excitatory': 80, 'inhibitory': 20,
    'input_probability': .05,
    'probabilities': {'EE': .11, 'EI': .10, 'IE': .37, 'II': .10},
    'input_material': 'Japanese_paper', 'recurrent_material': 'Japanese_paper',
    'materials': ['aluminum_board', 'outer_pigskin', 'back_pigskin', 'cork', 'denim',
                  'rubber_board', 'Japanese_paper', 'wood_board'],
    'trials_per_material': 324, 'duration_s': .5, 'dt_s': .0001,
    'bin_ms': 25., 'tau_membrane_ms': 10., 'refractory_ms': 2.,
    'threshold_mV': -40., 'reset_mV': -65., 'bias_mV': -65.,
    'syn_tau_r_ms': 2., 'syn_tau_d_ms': 30.,
    'EE_EI_coarse_step': .01, 'EE_EI_fine_step': .001, 'EE_EI_max': 1.,
    'IE_grid': [0., .00001, .00002, .00005, .0001, .0002, .0005, .001,
                .002, .005, .01, .02, .05, .1, .2, .5, 1.],
    'IE_refinements': 2, 'debug_trial_id': 203,
    'state_policy': 'reset_each_trial',
    'output_probability': .11, 'output_gain_step': .0001,
    'output_initial_gain': .8, 'output_max_gain': 2., 'output_rate_tolerance_hz': .1,
    'output_criterion': 'match_mean_liquid_E_and_output_rates_across_all_2592_trials',
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
        raise ValueError('Fixed topology/simulator conditions changed; update and validate the reference and calibration simulation first.')
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


def config(gains, *, include_output=False):
    """Apply this experiment's explicit conditions to an independent config copy."""
    from f_run.run_common import build_cfg
    cfg = deepcopy(build_cfg())
    common = cfg['common']
    if (common['NUM_SENSOR'] != 3 or not np.isclose(common['dt_ms'], .0001)
            or common['SLICE_END'] - common['SLICE_START'] != 5000
            or cfg['input_filter_map'] != {0: ['RI', 'SI'], 1: ['RI', 'SI'], 2: ['RI', 'SI']}):
        raise ValueError('This calibrated simulator requires 3 RI/SI sensors, 500 ms and dt=0.1 ms.')

    net = cfg['network']
    # リキッド校正では出力を無効にし、出力校正では明示的に有効にする。
    net['output_enabled'] = bool(include_output)
    if include_output:
        cfg['models']['LEARNING_RULE_MODEL'] = 'off'
        sizes = net['N_out']
        sizes = [int(sizes)] if np.isscalar(sizes) else list(sizes)
        if not sizes or any(int(n) != n or n < 1 for n in sizes):
            raise ValueError('Positive output layer sizes required')
        net['N_out'] = sizes
        net['p_liq_to_out_pairs'] = [dict(EE=SETTINGS['output_probability'], EI=0., IE=0., II=0.) for _ in sizes]
        net['gain_liq_to_out_pairs'] = [dict(EE=float(gains['output_gain']), EI=0., IE=0., II=0.) for _ in sizes]
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
             'c_configs/FIXED/cfg_filter.py',
             'f_run/run_calibration.py', 'f_run/run_common.py',
             'b_network/build_network.py', 'b_network/monitor.py', 'd_tools/activity.py',
             'd_tools/plotting.py', 'd_tools/debug.py', 'd_tools/weight_export.py',
             'c_configs/FIXED/calibration_reference.npz',
             'c_configs/FIXED/cfg_network.py', 'c_configs/FIXED/cfg_neuron_models.py',
             'b_network/weight_initialization.py', 'b_network/models/neuron_models.py',
             'b_network/models/synapse_models.py']
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


def _simulate_states(inputs, wi, wr, wo=None, gain=0.):
    """NumPyで複数試行をまとめて計算する。各試行は初期状態から開始。

    Eulerの更新順、RI/SIの加算順、発火元番号順のシナプス加算を維持する。
    ニューロンの式はBrian2による代表試行の検証と照合する。
    """
    x=np.asarray(inputs,dtype=np.float64)
    wi=np.asarray(wi,dtype=np.float64);wr=np.asarray(wr,dtype=np.float64)
    wo=np.zeros((100,0)) if wo is None else np.asarray(wo,dtype=np.float64)
    if x.ndim!=3 or not len(x) or x.shape[1:]!=(5000,6):
        raise ValueError('Expected nonempty trial x 5000 steps x 6 input channels')
    if wi.shape!=(6,100) or wr.shape!=(100,100) or wo.ndim!=2 or wo.shape[0]!=100:
        raise ValueError('Invalid input, recurrent, or output weight dimensions')
    if (not all(np.isfinite(a).all() for a in [x,wi,wr,wo])
            or not np.isfinite(gain) or gain<0 or np.any(wo<0) or np.any(wo[80:]!=0)):
        raise ValueError('Require finite inputs/weights, nonnegative output gain and E-only output')
    trials=len(x);outputs=wo.shape[1]
    liquid_bins=np.zeros((trials,100,20),np.int32)
    bins=np.zeros((trials,outputs,20),np.int32)
    peaks=np.zeros((trials,outputs))
    v=np.full((trials,100),-65.);last=np.full((trials,100),-100000,np.int32)
    ov=np.full((trials,outputs),-65.);olast=np.full((trials,outputs),-100000,np.int32)
    syn=np.zeros((trials,100+outputs));h=np.zeros_like(syn)
    probe=np.zeros((trials,outputs))
    # CSR形式で発火元ごとの接続先を持つ。add.atは同じ宛先への加算順を保つ。
    joined=np.concatenate([wr,wo],axis=1)
    pre,posts=np.nonzero(joined)
    increments=joined[pre,posts]/(.002*.030)
    degrees=np.bincount(pre,minlength=100)
    pointers=np.r_[0,np.cumsum(degrees)]
    for t in range(5000):
        old=syn
        syn=old+.0001*(-old/.030+h)
        h=h+.0001*(-h/.002)
        # CやBrian2と同じくRI3チャネル、SI3チャネルを別々に加算する。
        a=x[:,t,0,None]*wi[0]
        b=x[:,t,1,None]*wi[1]
        for channel in (1,2):
            a+=x[:,t,2*channel,None]*wi[2*channel]
            b+=x[:,t,2*channel+1,None]*wi[2*channel+1]
        drive=a+b
        active=t-last>20
        v=np.where(active,v+.0001*((-v-65.+drive+old[:,:100])/.010),v)
        fired=active & (v>=-40.)
        if outputs:
            old_output=old[:,100:]
            probe=probe+.0001*((-probe+old_output)/.010)
            np.maximum(peaks,probe,out=peaks)
            out_active=t-olast>20
            ov=np.where(out_active,ov+.0001*((-ov-65.+gain*old_output)/.010),ov)
            out_fired=out_active & (ov>=-40.)
            ov[out_fired]=-65.;olast[out_fired]=t
            bins[:,:,t//250]+=out_fired
        v[fired]=-65.;last[fired]=t
        liquid_bins[:,:,t//250]+=fired
        rows,neurons=np.nonzero(fired)
        sizes=degrees[neurons]
        number=int(sizes.sum())
        if number:
            starts=pointers[neurons]
            offsets=np.cumsum(sizes)-sizes
            edges=np.repeat(starts,sizes)+np.arange(number)-np.repeat(offsets,sizes)
            np.add.at(h,(np.repeat(rows,sizes),posts[edges]),increments[edges])
    return bins,peaks,liquid_bins


def simulate(x, ref, gains):
    wi,wr=weights(ref,gains)
    return _simulate_states(x,wi,wr)[2]


def simulate_output(inputs, wi, wr, wo, gain):
    """リキッド・出力の発火回数と、単位gainの発火前最大応答を返す。"""
    return _simulate_states(inputs,wi,wr,wo,gain)


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
    from d_tools.debug import execute_trial, save_debug_layer
    recorded = execute_trial(objects, x[0], dt_ms=.1, duration_ms=500., bin_ms=25.,
                             record_output=False, voltage=debug_title is not None)
    data = recorded['liquid'][0]
    actual = data['counts']
    expected = simulate(x[:1], ref, gains)[0]
    assert np.array_equal(actual, expected), 'Calibration / Brian2 spike-count mismatch'
    if debug_title is not None:
        save_debug_layer(data, out, title=debug_title,
                         n_excitatory=SETTINGS['excitatory'], trial_id=SETTINGS['debug_trial_id'])
    result = {'exact_25ms_counts_match': True, 'gains': gains, 'n_trials_checked': 1}
    save(out/'verification.json', result)
    print('[verified] exact spike counts and weights match Brian2', flush=True)
    return result


def make_output_network(gains):
    """固定リキッド＋Eから確率接続した出力を、学習なし・初期状態で構築する。"""
    from brian2 import start_scope, prefs, defaultclock, ms, seed
    from f_run.run_common import build_network_cfg
    from b_network.build_network import make_liquid_network
    cfg = config(gains, include_output=True)
    start_scope(); prefs.codegen.target = 'numpy'; defaultclock.dt = .1*ms
    base_seed = int(cfg['common']['BASE_SEED']); seed(base_seed); np.random.seed(base_seed)
    nc = build_network_cfg(cfg)
    objects = make_liquid_network(nc, 6, np.random.default_rng(base_seed), cfg['run'])
    # 校正計算の固定LIF/double-exp条件と実ネットワークを照合する。
    for group in objects['G_out']:
        for name, expected in [('tau_m', 10.), ('t_ref', 2.), ('tau_r', 2.), ('tau_d', 30.)]:
            if not np.allclose(np.asarray(getattr(group,name)/ms), expected):
                raise ValueError(f'Output {name} differs from calibrated simulation')
        for name, expected in [('v_thr', -40.), ('v_reset', -65.), ('bias', -65.)]:
            if not np.allclose(np.asarray(getattr(group,name)), expected):
                raise ValueError(f'Output {name} differs from calibrated simulation')
    return cfg, objects


def output_unit_weights(objects, *, require_positive=True):
    """gain=1で構築した出力の重みを、100液体ニューロン×全出力へ並べる。"""
    from d_tools.weight_export import _dense_w_matrix
    sizes = [len(group) for group in objects['G_out']]
    offsets = np.cumsum([0, *sizes])
    matrix = np.zeros((100, sum(sizes)))
    for meta in objects['output_connection_meta']:
        if meta['liquid_index'] != 0 or meta['pre_offset'] != 0:
            raise AssertionError('Output calibration requires liquid E only')
        syn = meta['synapses']; index = meta['output_index']
        indegree = np.bincount(np.asarray(syn.j[:],int),minlength=sizes[index])
        if not np.all(indegree > 0):
            raise ValueError('Some output neurons have no E input; inspect fixed topology/seed before calibrating')
        matrix[:, offsets[index]:offsets[index+1]] += _dense_w_matrix(syn,100,sizes[index])
    if (require_positive and not np.all(np.any(matrix[:80] > 0,axis=0))) or np.any(matrix < 0) or np.any(matrix[80:] != 0):
        raise AssertionError('Each output must have positive E input; I-to-output absent')
    return np.ascontiguousarray(matrix), sizes


def calibrate_output(ref, gains, out, *, input_dir):
    """全2592試行の平均差（リキッドE−出力）を符号で挟み込み、gainを調整・全状態保存する。"""
    from brian2 import ms
    from d_tools.debug import execute_trial, save_network_debug
    from d_tools.weight_export import save_liquid_weight_matrices
    from f_run.run_common import encode, snapshot
    import pandas as pd
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    step = float(SETTINGS['output_gain_step'])
    maximum = float(SETTINGS['output_max_gain'])
    if not np.isfinite(step) or step <= 0 or not np.isfinite(maximum) or maximum < step:
        raise ValueError('Require finite positive gain resolution and search maximum')
    cfg, objects = make_output_network({**gains, 'output_gain': 1.})
    unit, sizes = output_unit_weights(objects)
    objects['net'].run(0*ms,namespace={'input_ta':objects['G_in'].namespace['input_ta']})
    wi, wr = weights(ref,gains)
    materials = SETTINGS['materials']
    inputs = {material:currents(material,Path(input_dir)) for material in materials}
    liquid_counts = {}; liquid_rates = {}
    for material in materials:
        liquid_counts[material] = simulate(inputs[material],ref,gains)
        liquid_rates[material] = liquid_counts[material][:,:80].sum(axis=2).mean(axis=1)/.5
    target = float(np.concatenate(list(liquid_rates.values())).mean())
    if target <= 0:
        raise ValueError('Cannot match output rate to an inactive liquid population')
    history = {}; rate_cache = {}
    def evaluate(tick):
        if tick in history:return history[tick]
        gain = round(tick*step,12)
        output_rates = {}
        for material in materials:
            if tick == 0:
                output_rates[material] = np.zeros(324)
            else:
                counts,_,liquid = simulate_output(inputs[material],wi,wr,unit,gain)
                assert np.array_equal(liquid,liquid_counts[material]), 'Liquid counts changed during output search'
                output_rates[material] = counts.sum(axis=2).mean(axis=1)/.5
        difference = np.concatenate([liquid_rates[m]-output_rates[m] for m in materials])
        output_mean = float(np.concatenate(list(output_rates.values())).mean())
        row = {'output_gain':gain,'liquid_E_rate_hz':target,'output_rate_hz':output_mean,
               'mean_difference_hz':float(difference.mean()),
               'absolute_mean_difference_hz':float(abs(difference.mean())),
               'trial_rmse_hz':float(np.sqrt(np.mean(difference**2))),
               'trial_mae_hz':float(np.mean(np.abs(difference))), 'n_trials':2592}
        history[tick]=row;rate_cache[tick]=output_rates
        pd.DataFrame(list(history.values())).to_csv(out/'search_history.csv',index=False)
        print(f'[output rate] gain={gain:.4f}, liquid E={target:.4f} Hz, '
              f'output={output_mean:.4f} Hz, E-output={row["mean_difference_hz"]:+.4f} Hz',flush=True)
        return row
    low=0;evaluate(low)
    max_tick=int(np.floor(maximum/step))
    high=min(max_tick,max(1,int(np.ceil(SETTINGS['output_initial_gain']/step))))
    while evaluate(high)['mean_difference_hz']>0:
        if high==max_tick:
            raise RuntimeError('Output rate remains below liquid E at output_max_gain; increase search limit')
        low=high;high=min(max_tick,high*2)
    # 差が正ならgainを上げ、負なら下げる。調整幅を半分ずつ狭める。
    while high-low>1:
        middle=(low+high)//2
        if evaluate(middle)['mean_difference_hz']>0:low=middle
        else:high=middle
    # 平均差の絶対値が最も小さい評価済み候補を採用する（同値なら小さいgain）。
    best=min(history,key=lambda tick:(history[tick]['absolute_mean_difference_hz'],tick))
    gain=history[best]['output_gain'];selected_gains={**gains,'output_gain':gain}
    comparisons=[];records=[];checks=[]
    for material in materials:
        counts,_,liquid=simulate_output(inputs[material],wi,wr,unit,gain)
        assert np.array_equal(liquid,liquid_counts[material])
        assert np.array_equal(counts.sum(axis=2).mean(axis=1)/.5,rate_cache[best][material])
        folder=out/'all_materials'/material
        for name,bins,types,layer_sizes in [
                ('liquid',liquid,np.array(['E']*80+['I']*20),[100]),
                ('output',counts,np.array(['E']*sum(sizes)),sizes)]:
            destination=folder/name;destination.mkdir(parents=True,exist_ok=True)
            temporary=destination/'features.partial.npz'
            np.savez_compressed(temporary,counts_25ms=bins,states_hz=bins.transpose(0,2,1)/.025,
                trial_ids=np.arange(1,325),bin_ms=25.,neuron_ids=np.arange(bins.shape[1]),
                neuron_types=types,layer_sizes=np.asarray(layer_sizes),time_bin_edges_ms=np.arange(21)*25.)
            temporary.replace(destination/'features.npz')
        for index in range(324):
            comparisons.append({'material':material,'trial_id':index+1,
                'liquid_E_rate_hz':float(liquid_rates[material][index]),
                'output_rate_hz':float(rate_cache[best][material][index]),
                'difference_hz':float(liquid_rates[material][index]-rate_cache[best][material][index])})
        sid=SETTINGS['debug_trial_id'];_,network=make_output_network(selected_gains)
        initial=[np.asarray(s.w[:]).copy() for s in network['S_all']]
        recorded=execute_trial(network,inputs[material][sid-1],dt_ms=.1,duration_ms=500.,voltage=True)
        actual=np.concatenate([data['counts'] for data in recorded['output']])
        assert np.array_equal(actual,counts[sid-1]), 'Brian2/output calibration count mismatch'
        assert np.array_equal(recorded['liquid'][0]['counts'],liquid[sid-1])
        assert all(np.array_equal(s.w[:],w) for s,w in zip(network['S_all'],initial))
        actual_w,_=output_unit_weights(network)
        assert np.allclose(actual_w,unit*gain,rtol=1e-12,atol=1e-12)
        save_network_debug(recorded,network,folder/f'debug_sid{sid}',material=material,trial_id=sid)
        if material==materials[0]:save_liquid_weight_matrices(out/'weights',network)
        checks.append({'material':material,'trial_id':sid,'exact_output_counts_match':True,
                       'exact_liquid_counts_match':True,'weights_unchanged':True})
        records.append({'material':material,'trials':324,'liquid_states_shape':[324,20,100],
            'output_states_shape':[324,20,sum(sizes)],
            'never_active_output':int(np.count_nonzero(counts.sum(axis=(0,2))==0)),
            'liquid_E_rate_hz':float(liquid_rates[material].mean()),
            'output_rate_hz':float(rate_cache[best][material].mean())})
        print(f'[output recorded] {material}: 324 trials and debug sid{sid}',flush=True)
    pd.DataFrame(comparisons).to_csv(out/'trial_rate_comparison.csv',index=False)
    pd.DataFrame(records).to_csv(out/'material_rate_comparison.csv',index=False)
    result={**history[best],'output_probability':SETTINGS['output_probability'],'output_layer_sizes':sizes,
        'actual_output_connections':int(np.count_nonzero(unit)),
        'actual_indegree_per_output':np.count_nonzero(unit[:80],axis=0).tolist(),
        'all_outputs_have_input_probability':float((1-(1-SETTINGS['output_probability'])**80)**sum(sizes)),
        'criterion':SETTINGS['output_criterion'],'selection_metric':'absolute_mean_difference_hz',
        'difference_definition':'mean_liquid_E_rate - mean_output_rate',
        'positive_difference_action':'increase output gain','negative_difference_action':'decrease output gain',
        'gain_step':step,'rate_tolerance_hz':SETTINGS['output_rate_tolerance_hz'],
        'within_tolerance':history[best]['absolute_mean_difference_hz']<=SETTINGS['output_rate_tolerance_hz'],
        'search_scope':'sign-guided bracket refinement on configured grid; one common gain for all trials',
        'bracket':[history[low],history[high]],'state_policy':'reset_each_trial','learning':False,
        'I_to_output':False,'state_axes':['trial','time_bin','neuron'],'bin_ms':25.,
        'verification':checks,'materials':records}
    save(out/'output_summary.json',result)
    save(out/'config_snapshot.json',encode(snapshot(config(selected_gains,include_output=True))))
    return selected_gains


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
                   '10: output EE p=.11; match liquid E/output mean firing rates across 8 x 324 trials; no learning',
                   '11: record 8 x 324 fresh states (20 time bins x 100 neurons); one debug trial per material'],
         'minimum_scope': 'fixed topology and screened gains; recurrent response is not assumed monotonic',
         'note': 'Japanese_paper is the selected calibration material; no claim it is both smallest and largest input.'},
          ensure_ascii=False, indent=2))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run', 'verify', 'output'])
    parser.add_argument('--out-dir', type=Path, default=ROOT/'g_tactile_results/calibration',
                        help='保存先（既定: LSM/g_tactile_results/calibration）')
    parser.add_argument('--liquid-dir',type=Path,default=ROOT/'g_tactile_results/calibration',
                        help='For output action: directory containing calibrated liquid gains')
    parser.add_argument('--resume', action='store_true', help='Reuse completed stages with identical settings/source hashes')
    args = parser.parse_args()
    if args.action == 'plan':
        plan(); return
    out = args.out_dir.resolve(); ref = load_reference()
    if args.action == 'output':
        source = args.liquid_dir.resolve()
        if out.exists():
            parser.error('Choose a new --out-dir for output calibration; existing liquid results are preserved.')
        gains = json.loads((source/'selected_parameters.json').read_text(encoding='utf-8'))
        if not all(key in gains for key in ['input_gain','EE_gain','EI_gain','IE_gain']):
            parser.error('Liquid calibration is incomplete')
        out.mkdir(parents=True)
        save(out/'plan.json',{'provenance':provenance(),'liquid_source':str(source),'liquid_gains':gains})
        selected = calibrate_output(ref,gains,out/'output_calibration',input_dir=source)
        save(out/'selected_parameters.json',selected)
        print(f'[complete output] {out}',flush=True)
        return

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
    gains = calibrate_output(ref,gains,out/'output_calibration',input_dir=out)
    save(out/'selected_parameters.json',gains)
    record_material_states(ref, gains, out)
    print(f'[complete] {out}', flush=True)


if __name__ == '__main__':
    main()
