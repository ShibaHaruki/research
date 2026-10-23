"""分類指標の集計と、保存済み出力発火記録の交差検証・結果保存。"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def confusion_percentages(counts) -> np.ndarray:
    """Normalize each true-class row to percentages; empty rows remain zero."""
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1] or counts.size == 0:
        raise ValueError('Expected a nonempty square confusion matrix')
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError('Confusion counts must be finite and nonnegative')
    totals = counts.sum(1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros(counts.shape, dtype=float), where=totals != 0)*100


def evaluate_classification(true_labels, predicted_labels, *, labels=None) -> dict:
    """Return overall/per-class accuracy (0..1), counts and row percentages."""
    true = np.asarray(true_labels); predicted = np.asarray(predicted_labels)
    if true.ndim != 1 or predicted.shape != true.shape or not len(true):
        raise ValueError('Expected equal-length, nonempty 1D true and predicted labels')
    classes = np.unique(np.concatenate([true, predicted])) if labels is None else np.asarray(labels)
    if classes.ndim != 1 or not len(classes) or len(np.unique(classes)) != len(classes):
        raise ValueError('labels must be a nonempty sequence of unique classes')
    if not np.isin(true, classes).all() or not np.isin(predicted, classes).all():
        raise ValueError('labels must include every true and predicted class')
    cm = confusion_matrix(true, predicted, labels=classes)
    percentages = confusion_percentages(cm)
    return {'accuracy': float(np.mean(predicted == true)), 'labels': classes,
            'confusion_matrix': cm, 'confusion_percent': percentages,
            'per_class_accuracy': np.diag(percentages)/100,
            'class_support': cm.sum(1)}


def extract_features(sout_rec, T_n_ms, *, bin_width_ms):
    """素材×試行×出力ニューロン×時間ビンの発火回数を、指定時間幅の発火率へ変換する。"""
    values=np.asarray(sout_rec)
    if values.ndim!=4 or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('Expected finite, nonnegative material x trial x output x bin spike counts')
    ratio=float(T_n_ms)/float(bin_width_ms)
    step=round(ratio)
    if step<1 or not np.isclose(step,ratio) or values.shape[-1]%step:
        raise ValueError('Aggregation window must be a multiple of recording bin and divide the trial duration')
    grouped=values.reshape(*values.shape[:-1],values.shape[-1]//step,step).sum(axis=-1)
    return (grouped/(float(T_n_ms)/1000.)).reshape(values.shape[0],values.shape[1],-1)


def classify_dataset(dataset_source, *, rep_start=1,rep_end=None,n_folds=10,T_n_list=None,base_seed=1):
    """保存済み出力記録を標準化なしで分類し、正答率・混同行列を保存する。

    出力層の可塑性学習には使わなかった記録内で、分類器の層別交差検証を行う。
    ネットワークの実行や重みの更新は行わない。
    """
    # 基本の分類指標だけを使う場合は、記録読込・保存用の依存を読み込まない。
    import json
    import re
    from pathlib import Path
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    from threadpoolctl import threadpool_limits
    from d_tools.classification import classify
    from d_tools.plotting import save_confusion_matrix
    from f_run.run_common import decode, write_json

    folder=Path(dataset_source['dataset_dir'] if isinstance(dataset_source,dict) else dataset_source)
    metadata=decode(json.loads((folder/'test_config_snapshot.json').read_text(encoding='utf-8')))
    materials=list(metadata.get('materials',metadata['test']['TEST_MAT']))
    bin_width=float(metadata['bin_width_ms'])
    files=sorted(folder.glob('sout_rec_rep*.npy'))
    selected=[]
    for file in files:
        match=re.fullmatch(r'sout_rec_rep(\d+)\.npy',file.name)
        if match and int(match[1])>=rep_start and (rep_end is None or int(match[1])<=rep_end):
            selected.append((int(match[1]),file))
    if not selected: raise FileNotFoundError(f'No test recordings selected in {folder}')
    output=folder/'results_10fold';summaries=[];all_details=[]
    with threadpool_limits(limits=4):
        for rep,file in selected:
            counts=np.load(file)
            if counts.ndim!=4 or counts.shape[0]!=len(materials): raise ValueError('Material count or record shape mismatch')
            n_samples=counts.shape[1]
            if n_folds<2 or n_samples<n_folds or n_samples-(n_samples+n_folds-1)//n_folds<2:
                raise ValueError('Insufficient trials: each fold needs at least 2 classifier-training trials per class')
            labels=np.repeat(np.arange(len(materials)),n_samples)
            for window in (T_n_list or [25]):
                features=extract_features(counts,window,bin_width_ms=bin_width).reshape(len(labels),-1)
                cm=np.zeros((len(materials),len(materials)),np.int64);fold_rows=[]
                cv=StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=base_seed+rep-1)
                for fold,(train,test) in enumerate(cv.split(features,labels),1):
                    active=np.any(features[train]!=0,axis=0)
                    if not active.any(): raise ValueError('No output spikes in classifier training trials; adjust output gain')
                    predicted=classify(features[train][:,active],labels[train],features[test][:,active])
                    metrics=evaluate_classification(labels[test],predicted,labels=np.arange(len(materials)))
                    cm+=metrics['confusion_matrix']
                    row={'rep':rep,'T_n_ms':window,'fold':fold,'accuracy':metrics['accuracy'],
                         'n_train':len(train),'n_test':len(test)}
                    fold_rows.append(row);all_details.append(row)
                # 各テスト試行は、分類CVの評価側へ1回ずつ現れる。
                assert int(cm.sum())==len(labels)
                summary={'rep':rep,'T_n_ms':window,'folds':n_folds,'accuracy':float(np.trace(cm)/cm.sum()),
                         'mean_fold_accuracy':float(np.mean([r['accuracy'] for r in fold_rows])),
                         'std_fold_accuracy':float(np.std([r['accuracy'] for r in fold_rows],ddof=1))}
                summaries.append(summary)
                output.mkdir(parents=True,exist_ok=True)
                stem=f'rep{rep}_tn{window:g}ms'
                pd.DataFrame(cm,index=materials,columns=materials).to_csv(output/f'confusion_{stem}.csv')
                save_confusion_matrix(cm,materials,output/f'confusion_{stem}.png',
                    title=f'Output layer | {counts.shape[2]} neurons | {window:g} ms bins\n'
                          f'Accuracy: {summary["accuracy"]:.2%}; {n_folds}-fold classification')
                print(f'[classification] rep{rep}, {window:g} ms: {summary["accuracy"]:.2%}',flush=True)
    pd.DataFrame(summaries).to_csv(output/'accuracy_summary.csv',index=False)
    pd.DataFrame(all_details).to_csv(output/'accuracy_by_fold.csv',index=False)
    write_json(output/'evaluation.json',{'materials':materials,'standardization':False,
        'classifier':'class-specific regularized Mahalanobis','regularization':1e-6,
        'folds':n_folds,'base_seed':base_seed,'T_n_ms':T_n_list or [25],
        'split_scope':'Stratified CV within held-out output recordings; liquid/readout plastic training trials excluded',
        'summary':summaries})
    return summaries
