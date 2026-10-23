"""各素材1試行のdebugに共通する実行・発火集計・膜電位抽出・保存。

入力電流とネットワークを受け取るので、f_runの実行入口には依存しない。
通常の試行記録でもexecute_trialを利用でき、debug時だけ膜電位を記録する。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from brian2 import ms, TimedArray
from d_tools.plotting import save_liquid_debug


def _offset(monitor):
    if hasattr(monitor, 'exc') and hasattr(monitor, 'inh'):
        return len(monitor.exc.t), len(monitor.inh.t)
    return len(monitor.t)


def _events(monitor, offset, start):
    if isinstance(offset, tuple):
        times = np.concatenate([np.asarray((monitor.exc.t[offset[0]:]-start)/ms),
                                np.asarray((monitor.inh.t[offset[1]:]-start)/ms)])
        ids = np.concatenate([np.asarray(monitor.exc.i[offset[0]:], int),
                              np.asarray(monitor.inh.i[offset[1]:], int)+monitor.n_exc])
    else:
        times = np.asarray((monitor.t[offset:]-start)/ms)
        ids = np.asarray(monitor.i[offset:], int)
    order = np.argsort(times, kind='stable')
    return times[order], ids[order]


def bin_spikes(times_ms, neuron_ids, n_neurons, *, bin_ms, duration_ms):
    """試行相対時刻をニューロン×時間ビンの発火回数へ変換する。"""
    counts = np.zeros((n_neurons, round(duration_ms/bin_ms)), np.int32)
    bins = np.floor((np.asarray(times_ms)+1e-5)/bin_ms).astype(int)
    valid = (bins >= 0) & (bins < counts.shape[1])
    np.add.at(counts, (np.asarray(neuron_ids, int)[valid], bins[valid]), 1)
    return counts


def execute_trial(objects, current, *, dt_ms, duration_ms, bin_ms=25.,
                  record_liquid=True, record_output=True, voltage=False):
    """1試行を実行し、liquid/outputの層別記録を返す。状態初期化は呼出し元で行う。

    voltage=Trueなら選択ニューロンの膜電位も返す。モニタの有効状態は終了時に戻す。
    """
    current = np.asarray(current, dtype=float)
    if (current.ndim != 2 or current.shape[1] != len(objects['G_in'])
            or not np.isfinite(current).all()):
        raise ValueError('Expected finite time-step x input-channel currents')
    if (not np.isfinite(dt_ms) or dt_ms <= 0 or not np.isfinite(bin_ms) or bin_ms <= 0
            or not np.isfinite(duration_ms) or duration_ms <= 0
            or not np.isclose(duration_ms/bin_ms, round(duration_ms/bin_ms))
            or duration_ms > len(current)*dt_ms+1e-7):
        raise ValueError('Invalid time step, trial duration, or bin width')
    if not np.isclose(float(objects['G_in'].clock.dt/ms), dt_ms):
        raise ValueError('Input time step differs from network clock')
    start = objects['net'].t
    selections = []
    for population, enabled, group_key, spike_key, voltage_key, ids_key in [
            ('liquid', record_liquid, 'G_liq', 'M_liq_debug', 'V_liq', 'V_liq_indices'),
            ('output', record_output, 'G_out', 'M_out', 'V_out', 'V_out_indices')]:
        if not enabled:
            continue
        groups = objects.get(group_key, [])
        for index, group in enumerate(groups):
            spike = objects[spike_key][index]
            vmonitor = objects[voltage_key][index]
            selections.append(dict(population=population, group=group, spike=spike,
                voltage=vmonitor, ids=objects[ids_key][index], spike_offset=_offset(spike),
                voltage_offset=len(vmonitor.t), old_spike_active=spike.active,
                old_voltage_active=vmonitor.active))
    try:
        for item in selections:
            item['spike'].active = True
            item['voltage'].active = bool(voltage)
        input_ta = TimedArray(current, dt=dt_ms*ms)
        objects['G_in'].namespace['input_ta'] = input_ta
        objects['G_in'].t_start = start
        objects['net'].run(duration_ms*ms, namespace={'input_ta': input_ta})
        result = {'liquid': [], 'output': []}
        for item in selections:
            times, ids = _events(item['spike'], item['spike_offset'], start)
            counts = bin_spikes(times, ids, len(item['group']), bin_ms=bin_ms, duration_ms=duration_ms)
            vmonitor = item['voltage']; offset = item['voltage_offset']
            data = dict(spike_times_ms=times, spike_neuron_ids=ids, counts=counts,
                        counts_25ms=counts, bin_ms=float(bin_ms))
            # counts_25msは既存描画API用のキー。実際のビン幅はbin_msを参照する。
            data.update(voltage_times_ms=np.asarray((vmonitor.t[offset:]-start)/ms) if voltage else np.empty(0),
                        voltage_mV=np.asarray(vmonitor.v)[:, offset:] if voltage else np.empty((0,0)),
                        voltage_neuron_ids=np.asarray(item['ids'], int) if voltage else np.empty(0,int))
            result[item['population']].append(data)
        return result
    finally:
        for item in selections:
            item['spike'].active = item['old_spike_active']
            item['voltage'].active = item['old_voltage_active']


def save_debug_layer(data, folder, *, title, n_excitatory, trial_id):
    """共通の3段debug画像と元データを保存する。"""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    payload = {**data, 'trial_id': int(trial_id)}
    np.savez_compressed(folder/'debug.npz', **payload)
    save_liquid_debug(payload, folder/'debug.png', title=title, n_excitatory=n_excitatory)


def save_network_debug(result, objects, folder, *, material, trial_id):
    """リキッド各層と出力各層のdebugを同じ規則で保存する。"""
    folder = Path(folder)
    for population, group_key in [('liquid', 'G_liq'), ('output', 'G_out')]:
        for index, data in enumerate(result[population]):
            group = objects[group_key][index]
            n_exc = len(group.exc) if population == 'liquid' else len(group)
            save_debug_layer(data, folder/f'{population}_layer_{index+1}',
                title=f'{material} | Trial {trial_id} | {population.title()} layer {index+1}',
                n_excitatory=n_exc, trial_id=trial_id)


def run_material_debug(materials, *, trial_id, make_network, load_current,
                       dt_ms, duration_ms, bin_ms, out_dir):
    """各素材1回、独立したネットワークでdebugする。学習無効のfactoryを渡す。

    学習・テスト本体とは別ネットワークを使用し、debug中に重みが変化したら停止する。
    make_network()とload_current(material, sid)は呼出し元から与える。
    """
    reports = []
    for material in materials:
        objects = make_network()
        before = [np.asarray(s.w[:]).copy() for s in objects['S_all'] if 'w' in s.variables]
        result = execute_trial(objects, load_current(material, trial_id), dt_ms=dt_ms,
            duration_ms=duration_ms, bin_ms=bin_ms, voltage=True)
        after = [np.asarray(s.w[:]) for s in objects['S_all'] if 'w' in s.variables]
        if any(not np.array_equal(a,b) for a,b in zip(before,after)):
            raise AssertionError('Debug must use a network with plasticity disabled')
        folder = Path(out_dir)/material/f'sid{trial_id}'
        save_network_debug(result, objects, folder, material=material, trial_id=trial_id)
        reports.append({'material': material, 'trial_id': int(trial_id),
            'liquid_spikes': [int(data['counts'].sum()) for data in result['liquid']],
            'output_spikes': [int(data['counts'].sum()) for data in result['output']],
            'weights_unchanged': True})
        print(f'[debug] {material}: trial {trial_id}', flush=True)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
    return reports
