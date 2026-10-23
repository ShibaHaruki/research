"""c_configs/EXPERIMENTSの実験定義を現在のLSMで実行する。

python run_experiment.py exp_001_base --dry-run
python run_experiment.py exp_002_gain_sweep_example --trial liq_out_gain_0p5
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
import traceback

from f_run.run_common import build_cfg, build_network_cfg
from d_tools.experiments import (apply_overrides, expand_experiment, safe_stem, jsonable,
    now_text, append_manifest_row, write_experiment_trial_json)
from f_run.run_common import validate_cfg, repetition_numbers

ROOT = Path(__file__).resolve().parent
TARGETS = {'training','liquid','test','train_test','train_test_eval','classify'}


def load_experiment(name_or_path):
    path = Path(name_or_path)
    if path.suffix == '.py':
        if not path.is_file(): raise FileNotFoundError(path)
        spec = importlib.util.spec_from_file_location('_lsm_experiment_'+safe_stem(path.stem),path.resolve())
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    else:
        name = str(name_or_path)
        if '.' not in name: name='c_configs.EXPERIMENTS.'+name
        module=importlib.import_module(name)
    if not isinstance(getattr(module,'EXPERIMENT',None),dict):
        raise ValueError('Experiment module must define an EXPERIMENT dictionary')
    experiment=deepcopy(module.EXPERIMENT)
    experiment.setdefault('name',path.stem)
    return experiment


def normalized_target(value):
    value=str(value).replace('-','_')
    value={'training_test':'train_test','training_test_eval':'train_test_eval'}.get(value,value)
    if value not in TARGETS: raise ValueError(f'Unknown experiment target: {value}')
    return value


def execute_target(target,cfg):
    from f_run.run_common import training_directory, test_directory
    if target=='liquid':
        from f_run.run_liquid import run_liquid
        return run_liquid(cfg)
    if target=='train_test_eval':
        from f_run.run_train_test_eval import run_train_test_eval
        return run_train_test_eval(cfg)
    if target=='classify':
        from d_tools.evaluation import classify_dataset
        rows=classify_dataset(test_directory(training_directory(cfg)),
            n_folds=cfg['run'].get('EVAL_N_FOLDS',10),
            T_n_list=cfg['run'].get('EVAL_TN_MS',[25]),base_seed=cfg['run'].get('EVAL_BASE_SEED',1))
        return f'[classification] {len(rows)} summary rows saved'
    messages=[]
    for rep in repetition_numbers(cfg):
        if target=='training':
            from f_run.run_training import run_training
            messages.append(run_training(rep,cfg))
        elif target=='test':
            from f_run.run_test import run_test
            messages.append(run_test(rep,training_directory(cfg)))
        else:
            from f_run.run_train_test import run_train_test
            messages.append(run_train_test(rep,cfg))
    return ' | '.join(messages)


def run_experiment(experiment, *, out_dir=None,target=None,trial_id=None,dry_run=False):
    trials=expand_experiment(experiment)
    if trial_id is not None: trials=[trial for trial in trials if trial['id']==trial_id]
    if not trials: raise ValueError('No experiment trials selected')
    ids=[trial['id'] for trial in trials]
    if len(set(ids))!=len(ids): raise ValueError('Experiment trial IDs must be unique')
    root=Path(out_dir).resolve() if out_dir is not None else ROOT/'g_tactile_results/experiments'/safe_stem(experiment['name'])
    manifest=root/'experiment_manifest.csv';results=[]
    # 全trialのキーと基本設定を先に検査し、後半の誤記で前半だけ実行されるのを防ぐ。
    prepared=[]
    for trial in trials:
        current_target=normalized_target(target or trial.get('target') or experiment.get('target','training'))
        cfg=apply_overrides(build_cfg(),trial['overrides'])
        build_network_cfg(cfg)
        if current_target not in ('liquid','test','classify'):validate_cfg(cfg)
        case=root/safe_stem(trial['id'])
        case.relative_to(root)
        cfg['run']['READOUT_TRAIN_DIR']=str(case/'training')
        cfg['run']['LIQUID_OUTPUT_DIR']=str(case/'liquid')
        prepared.append((trial,current_target,cfg,case))
    for trial,current_target,cfg,case in prepared:
        payload={'experiment_name':experiment['name'],'trial_id':trial['id'],'target':current_target,
                 'memo':trial.get('memo',''),'overrides':trial['overrides'],
                 'training_directory':str(case/'training'),'liquid_directory':str(case/'liquid')}
        plan=case/f'experiment_trial_{current_target}.json'
        if plan.exists() and json.loads(plan.read_text(encoding='utf-8'))!=jsonable(payload):
            raise FileExistsError(f'Existing experiment plan differs; use a new --out-dir: {plan}')
        write_experiment_trial_json(plan,payload)
        row={'experiment_name':experiment['name'],'trial_index':trial['index'],'trial_id':trial['id'],
             'target':current_target,'started_at':now_text(),'result_dir':str(case),
             'overrides_json':json.dumps(jsonable(trial['overrides']),ensure_ascii=False)}
        try:
            if dry_run:
                message=f'[dry-run] {trial["id"]}: target={current_target}, rule={cfg["models"].get("LEARNING_RULE_MODEL","off")}, result={case}'
                row['status']='dry_run'
            else:
                message=execute_target(current_target,cfg);row['status']='done'
            row['message']=message
        except Exception as exc:
            row.update(status='failed',message=f'{type(exc).__name__}: {exc}')
            (case/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        row['finished_at']=now_text();append_manifest_row(manifest,row)
        results.append(row);print(row['message'],flush=True)
        if row['status']=='failed':raise RuntimeError(row['message'])
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',nargs='?',default='exp_001_base')
    parser.add_argument('--target',default=None,choices=sorted(TARGETS|{'train-test','train-test-eval','training_test','training_test_eval'}))
    parser.add_argument('--trial',default=None)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--out-dir',type=Path,default=None)
    args=parser.parse_args()
    run_experiment(load_experiment(args.experiment),out_dir=args.out_dir,target=args.target,
                   trial_id=args.trial,dry_run=args.dry_run)

if __name__=='__main__':main()
