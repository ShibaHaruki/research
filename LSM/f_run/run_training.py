"""現在のネットワークで出力E結合を学習し、学習済み重みを保存する入口。"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from a_tactile.paths import training_directory
from f_run.run_common import (validate_cfg, sample_split, snapshot, encode,
    debug_readout_materials, write_json, build_runtime, run_trial, add_training_arguments, cfg_from_args)
from d_tools.weight_export import save_weights, weight_summary

def run_training(rep: int, cfg: dict) -> str:
    cfg = deepcopy(cfg)
    materials, n_train, _, epochs, _, _ = validate_cfg(cfg)
    folder = training_directory(cfg)
    final = folder/'weights/final'/f'rep{rep}'
    if final.exists(): raise FileExistsError(f'Trained result already exists: {final}')
    import json
    cfg_file = folder/'config_snapshot.json'
    if cfg_file.exists() and json.loads(cfg_file.read_text(encoding='utf-8')) != encode(snapshot(cfg)):
        raise ValueError('Existing training directory has a different configuration; choose --out-dir')
    folder.mkdir(parents=True, exist_ok=True)
    write_json(cfg_file, snapshot(cfg))
    train_ids, test_ids = sample_split(cfg, rep)
    np.savez_compressed(folder/f'split_rep{rep}.npz', train_ids=train_ids, test_ids=test_ids)
    objects = build_runtime(cfg, learning=True)
    save_weights(folder/'weights/initial'/f'rep{rep}', objects)
    initial = [np.asarray(s.w[:]).copy() for s in objects['S_out']]
    schedule_rng = np.random.default_rng(int(cfg['common']['BASE_SEED'])+rep)
    history = []
    schedule = [(material, int(sid)) for material in materials for sid in train_ids]
    for epoch in range(1, epochs+1):
        for order_index in schedule_rng.permutation(len(schedule)):
            material, sid = schedule[int(order_index)]
            run_trial(objects, cfg, material, sid, record=False)
            history.append({'epoch': epoch, 'material': material, 'trial_id': sid, **weight_summary(objects)})
            print(f'[training rep{rep}] {len(history)}/{epochs*len(schedule)}: {material} sid{sid}', flush=True)
    save_weights(final, objects)
    pd.DataFrame(history).to_csv(folder/f'training_history_rep{rep}.csv', index=False)
    delta = sum(float(np.sum((np.asarray(s.w[:])-before)**2)) for s,before in zip(objects['S_out'],initial))**.5
    write_json(folder/f'training_summary_rep{rep}.json', {'rep': rep,
        'learning_rule': cfg['models'].get('LEARNING_RULE_MODEL', 'off'),
        'state_policy': cfg['run'].get('READOUT_STATE_POLICY', 'carry'),
        'epochs': epochs, 'n_training_presentations': len(history), 'output_weight_change_norm': delta,
        'train_trial_ids': train_ids.tolist(), 'test_trial_ids': test_ids.tolist(),
        'plastic_connections': 'liquid E to output E only; inhibitory connections fixed'})
    debug_readout_materials(cfg, final, folder/'debug'/f'training_rep{rep}')
    return f'[training rep{rep}] weights saved: {final}; weight change norm={delta:.6g}'


def main():
    parser=argparse.ArgumentParser(description=__doc__);add_training_arguments(parser)
    args=parser.parse_args();cfg=cfg_from_args(args)
    from f_run.run_common import repetition_numbers
    for rep in repetition_numbers(cfg,args.rep):
        print(run_training(rep,cfg))

if __name__=='__main__': main()
