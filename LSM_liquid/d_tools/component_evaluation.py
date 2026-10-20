"""Evaluate PC1..PCk and original features on identical held-out trials."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from d_tools.pca import fit_pca, transform_pca
from d_tools.classification import classify
from d_tools.evaluation import evaluate_classification


def evaluate_component_counts(states, labels, splits, *, n_components=40,
                              regularization=1e-6, threads=4, verbose=True) -> pd.DataFrame:
    """同じ分割で全特徴とPC1〜PCkの分類正答率を計算する。

    statesは「試行×ニューロン×時間ビン」または「試行×特徴」。
    splitsは(train_indices, test_indices)のリストで、添字はstatesの試行番号。
    各分割で学習試行だけから無活動特徴の除外とPCAを行う。標準化なし。
    返り値のaccuracyは0〜1。PCAは1分割につき1回だけ求め、上位k列を使う。
    """
    data = np.asarray(states, dtype=np.float64)
    y = np.asarray(labels)
    if (data.ndim not in (2, 3) or any(size == 0 for size in data.shape)
            or y.ndim != 1 or len(y) != len(data) or not np.isfinite(data).all()):
        raise ValueError('Expected finite trial states and one label per trial')
    if not isinstance(n_components, (int, np.integer)) or isinstance(n_components, bool) or n_components < 1:
        raise ValueError('n_components must be a positive integer')
    splits = list(splits)
    if not splits:
        raise ValueError('At least one training/evaluation split is required')
    X = data.reshape(len(data), -1)
    rows = []
    with threadpool_limits(limits=threads):
        for fold, (training_indices, test_indices) in enumerate(splits, 1):
            indices = []
            for values in [training_indices, test_indices]:
                values = np.asarray(values)
                if (values.ndim != 1 or not len(values) or not np.issubdtype(values.dtype, np.integer)
                        or np.any(values < 0) or np.any(values >= len(X)) or len(np.unique(values)) != len(values)):
                    raise ValueError('Split indices must be nonempty, unique valid integers')
                indices.append(values)
            tridx, teidx = indices
            if np.intersect1d(tridx, teidx).size:
                raise ValueError('Training and evaluation trials must not overlap')
            if not np.isin(y[teidx], y[tridx]).all():
                raise ValueError('Each evaluation class must occur in training')
            start = time.perf_counter()
            train, test = X[tridx], X[teidx]
            active = np.any(train != 0, axis=0)
            train, test = train[:, active], test[:, active]
            model = fit_pca(train, n_components=n_components)
            train_pc, test_pc = transform_pca(train, model), transform_pca(test, model)
            common = {'fold': fold, 'n_train': len(tridx), 'n_test': len(teidx),
                      'n_active_features': int(active.sum())}
            predicted = classify(train, y[tridx], test, regularization=regularization)
            full = evaluate_classification(y[teidx], predicted)['accuracy']
            rows.append({**common, 'representation': 'full_features', 'n_pcs': 0, 'accuracy': full})
            for k in range(1, n_components+1):
                predicted = classify(train_pc[:, :k], y[tridx], test_pc[:, :k], regularization=regularization)
                score = evaluate_classification(y[teidx], predicted)['accuracy']
                rows.append({**common, 'representation': 'PCA', 'n_pcs': k, 'accuracy': score})
            if verbose:
                elapsed = time.perf_counter()-start
                print(f'Split {fold}/{len(splits)} | Full: {full:.2%} | PC1-PC{n_components}: '
                      f'{rows[-1]["accuracy"]:.2%} | {elapsed:.1f} s', flush=True)
    return pd.DataFrame(rows)
