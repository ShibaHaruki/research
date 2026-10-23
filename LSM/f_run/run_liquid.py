"""gain探索・学習・PCAを行わず、現在の設定でリキッド層だけを動かして状態を記録する。

python f_run/run_liquid.py --samples 324
python f_run/run_liquid.py --materials aluminum_board Japanese_paper --trial-ids 203
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
from brian2 import prefs, start_scope, defaultclock, ms, seed
from f_run.run_common import build_cfg, build_network_cfg
from b_network.build_network import make_liquid_network
from d_tools.debug import execute_trial, save_debug_layer
from f_run.run_calibration import filtered_trial
from f_run.run_common import first, snapshot, write_json
from d_tools.weight_export import save_liquid_weight_matrices

LIQUID_RESULT_DIR = ROOT/'g_tactile_results/liquid_run'


def make_liquid_output_dir(cfg, net_cfg=None):
    return Path(cfg['run'].get('LIQUID_OUTPUT_DIR', LIQUID_RESULT_DIR/'recording')).resolve()


def run_liquid(cfg=None, *, out_dir=None, trial_ids=None):
    cfg = deepcopy(build_cfg() if cfg is None else cfg)
    cfg['network']['output_enabled'] = False
    cfg['models']['LEARNING_RULE_MODEL'] = 'off'
    cfg['run']['INTERNAL_STATE_PCA_ENABLE'] = False
    materials = list(cfg['liquid']['LIQUID_MAT'])
    from f_run.run_calibration import SETTINGS
    if not materials or len(set(materials)) != len(materials) or not set(materials) <= set(SETTINGS['materials']):
        raise ValueError('Require a nonempty, unique list of valid materials')
    bin_ms = float(cfg['run'].get('INTERNAL_STATE_BIN_MS', 25.))
    duration_ms = float(cfg['run'].get('LIQUID_DURATION_MS', 500.))
    state_policy = cfg['run'].get('LIQUID_STATE_POLICY', 'carry')
    if state_policy not in ('carry', 'reset_each_trial'):
        raise ValueError('LIQUID_STATE_POLICY must be carry or reset_each_trial')
    if cfg['run'].get('MATERIAL_SWITCH_WASHOUT_ENABLE', False):
        raise ValueError('Disable MATERIAL_SWITCH_WASHOUT_ENABLE for this recording workflow')
    if not np.isfinite(bin_ms) or bin_ms <= 0 or not 0 < duration_ms <= 500 or not np.isclose(duration_ms/bin_ms, round(duration_ms/bin_ms)):
        raise ValueError('Duration must be <=500 ms and divisible by bin width')
    base_seed = int(cfg['common']['BASE_SEED'])
    if trial_ids is None:
        count = int(first(cfg['liquid']['NUM_LIQUID_SAMPLE']))
        if not 1 <= count <= 324: raise ValueError('NUM_LIQUID_SAMPLE must be 1..324')
        trial_ids = np.random.default_rng(base_seed).permutation(np.arange(1,325))[:count]
    else:
        trial_ids = np.asarray(trial_ids)
        if trial_ids.ndim != 1 or not len(trial_ids) or not np.issubdtype(trial_ids.dtype, np.integer):
            raise ValueError('Trial IDs must be a nonempty 1D integer list')
        if len(np.unique(trial_ids)) != len(trial_ids) or np.any((trial_ids < 1) | (trial_ids > 324)):
            raise ValueError('Trial IDs must be unique values in 1..324')
    cfg['liquid']['NUM_LIQUID_SAMPLE'] = [len(trial_ids)]
    folder = Path(out_dir).resolve() if out_dir is not None else make_liquid_output_dir(cfg)
    if folder.exists(): raise FileExistsError(f'Output exists; choose a new --out-dir: {folder}')
    start_scope(); prefs.codegen.target = 'numpy'; defaultclock.dt = float(cfg['common']['dt_ms'])*1000*ms
    seed(base_seed); np.random.seed(base_seed)
    net_cfg = build_network_cfg(cfg)
    objects = make_liquid_network(net_cfg, len(net_cfg['INPUT_ROWS']), np.random.default_rng(base_seed), cfg['run'])
    assert not objects['G_out']
    objects['net'].store('liquid_initial')
    initial_weights = [np.asarray(s.w[:]).copy() for s in objects['S_all']]
    folder.mkdir(parents=True)
    write_json(folder/'config_snapshot.json', snapshot(cfg))
    np.save(folder/'sample_seq_liquid.npy', trial_ids)
    save_liquid_weight_matrices(folder/'weights', objects)
    layer_sizes = [len(layer) for layer in objects['G_liq']]
    neuron_types = np.concatenate([np.array(['E']*len(layer.exc)+['I']*len(layer.inh)) for layer in objects['G_liq']])
    debug_enabled = cfg['run'].get('LIQUID_DEBUG_ENABLE', True)
    for material in materials:
        all_counts = []
        for sample, sid in enumerate(trial_ids):
            if state_policy == 'reset_each_trial':
                objects['net'].restore('liquid_initial')
            debug = debug_enabled and sample == 0
            result = execute_trial(objects, filtered_trial(material, int(sid), cfg),
                dt_ms=float(cfg['common']['dt_ms'])*1000, bin_ms=bin_ms,
                duration_ms=duration_ms, record_output=False, voltage=debug)
            all_counts.append(np.concatenate([data['counts'] for data in result['liquid']]))
            if debug:
                for layer_index, data in enumerate(result['liquid']):
                    debug_folder = folder/'all_materials'/material/f'debug_sid{sid}'/f'layer_{layer_index+1}'
                    save_debug_layer(data, debug_folder,
                        title=f'{material} | Trial {sid} | Liquid layer {layer_index+1}',
                        n_excitatory=len(objects['G_liq'][layer_index].exc), trial_id=int(sid))
            print(f'[liquid] {material}: {sample+1}/{len(trial_ids)} (sid{sid})', flush=True)
        all_counts = np.stack(all_counts)
        data = dict(counts=all_counts, states_hz=all_counts.transpose(0,2,1)/(bin_ms/1000),
            trial_ids=trial_ids, bin_ms=bin_ms, neuron_ids=np.arange(sum(layer_sizes)),
            neuron_types=neuron_types, layer_sizes=np.asarray(layer_sizes),
            time_bin_edges_ms=np.arange(all_counts.shape[2]+1)*bin_ms)
        if np.isclose(bin_ms,25.): data['counts_25ms']=all_counts
        output = folder/'all_materials'/material; output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output/'features.npz', **data)
    for syn, initial in zip(objects['S_all'], initial_weights):
        if not np.array_equal(syn.w[:], initial): raise AssertionError('Liquid weights changed')
    write_json(folder/'summary.json', {'status':'complete','materials':materials,
        'trials_per_material':len(trial_ids),'trial_ids':trial_ids.tolist(),
        'state_shape_per_material':[len(trial_ids),round(duration_ms/bin_ms),sum(layer_sizes)],
        'state_axes':['trial','time_bin','neuron'],'state_unit':'Hz','bin_ms':bin_ms,
        'state_policy':state_policy,'output_layer':False,'learning':False,'automatic_PCA':False,
        'debug_trials_per_material':1 if debug_enabled else 0})
    return f'[liquid] saved {len(materials)} materials x {len(trial_ids)} trials: {folder}'


def run_liquid_worker():
    return run_liquid()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=None)
    parser.add_argument('--trial-ids',type=int,nargs='+',default=None)
    parser.add_argument('--materials',nargs='+',default=None)
    parser.add_argument('--state-policy',choices=['carry','reset_each_trial'],default=None)
    parser.add_argument('--out-dir',type=Path,default=None)
    args=parser.parse_args()
    if args.samples is not None and args.trial_ids is not None:
        parser.error('Choose either --samples or --trial-ids')
    cfg=deepcopy(build_cfg())
    if args.samples is not None: cfg['liquid']['NUM_LIQUID_SAMPLE']=[args.samples]
    if args.materials is not None: cfg['liquid']['LIQUID_MAT']=args.materials
    if args.state_policy is not None: cfg['run']['LIQUID_STATE_POLICY']=args.state_policy
    print(run_liquid(cfg,out_dir=args.out_dir,trial_ids=args.trial_ids))

if __name__=='__main__': main()
