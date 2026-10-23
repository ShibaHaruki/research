"""学習時の設定と重みを読み込み、学習と重複しない試行の出力を記録する。"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from f_run.run_common import (load_cfg, training_directory, test_directory, build_runtime,
    debug_readout_materials, run_trial, validate_cfg, write_json, snapshot)
from d_tools.weight_export import load_weights
from f_run.run_common import build_cfg

TEST_RESULT_DIR=ROOT/'g_tactile_results/test_run'
training_output_dir_for_cfg=training_directory
test_output_dir_for_training_dir=test_directory


def run_test(rep: int, train_dir: Path | None=None) -> str:
    folder=Path(train_dir) if train_dir is not None else training_directory(build_cfg())
    cfg=load_cfg(folder)
    materials, _, n_test, _, bin_ms, duration=validate_cfg(cfg)
    output=test_directory(folder)
    file=output/f'sout_rec_rep{rep}.npy'
    if file.exists(): raise FileExistsError(f'Test result already exists: {file}')
    with np.load(folder/f'split_rep{rep}.npz') as split:
        train_ids=split['train_ids']; test_ids=split['test_ids']
    if len(test_ids)!=n_test or np.intersect1d(train_ids,test_ids).size:
        raise ValueError('Saved training/test split overlaps or has the wrong length')
    objects=build_runtime(cfg,learning=False)
    load_weights(folder/'weights/final'/f'rep{rep}',objects)
    before=[np.asarray(s.w[:]).copy() for s in objects['S_all']]
    counts=np.zeros((len(materials),n_test,sum(len(g) for g in objects['G_out']),round(duration/bin_ms)),np.int32)
    for m,material in enumerate(materials):
        for sample,sid in enumerate(test_ids):
            counts[m,sample]=run_trial(objects,cfg,material,int(sid),record=True)
            print(f'[test rep{rep}] {material}: {sample+1}/{n_test}',flush=True)
    for syn,initial in zip(objects['S_all'],before):
        if not np.array_equal(syn.w[:],initial): raise AssertionError('Weights changed during testing')
    output.mkdir(parents=True,exist_ok=True)
    np.save(file,counts)
    np.save(output/f'test_trial_ids_rep{rep}.npy',test_ids)
    metadata=snapshot(cfg)
    metadata.update({'bin_width_ms':bin_ms,'duration_ms':duration,'materials':materials,
        'learning_during_test':False,'source_training_directory':str(folder.resolve())})
    write_json(output/'test_config_snapshot.json',metadata)
    debug_readout_materials(cfg, folder/'weights/final'/f'rep{rep}', output/'debug'/f'test_rep{rep}')
    return f'[test rep{rep}] saved {file}; shape={counts.shape}; weights fixed'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir',type=Path,default=None);parser.add_argument('--rep',type=int,default=None)
    args=parser.parse_args()
    from f_run.run_common import repetition_numbers
    source=args.train_dir if args.train_dir is not None else training_directory(build_cfg())
    cfg=load_cfg(source)
    for rep in repetition_numbers(cfg,args.rep):
        print(run_test(rep,source))

if __name__=='__main__': main()
