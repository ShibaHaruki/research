# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


SCRIPT_DIR = Path(__file__).resolve().parent
DIR_NAME = [
    "Al_board", "buta_omote", "buta_ura", "cork",
    "denim", "rubber_board", "washi", "wood_board",
]


class MahalanobisClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.means_ = []
        self.cov_inv_ = []

        for cls in self.classes_:
            x_cls = x[y == cls]
            mean = x_cls.mean(axis=0)
            cov = np.cov(x_cls.T)
            if np.ndim(cov) == 0:
                cov = np.array([[float(cov)]])
            cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
            cov = cov + self.ridge * np.eye(cov.shape[0], dtype=np.float64)
            self.means_.append(mean)
            self.cov_inv_.append(np.linalg.pinv(cov))

        self.means_ = np.stack(self.means_, axis=0)
        self.cov_inv_ = np.stack(self.cov_inv_, axis=0)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        distances = np.zeros((x.shape[0], len(self.classes_)), dtype=np.float64)
        for i, cls in enumerate(self.classes_):
            diff = x - self.means_[i]
            val = np.einsum("ij,jk,ik->i", diff, self.cov_inv_[i], diff)
            distances[:, i] = np.sqrt(np.maximum(val, 0.0))
        return self.classes_[np.argmin(distances, axis=1)]


def load_sout(dataset: str, rep: int):
    path = SCRIPT_DIR / f"{dataset}_sout_rec_rep{rep}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    sout = np.load(path)
    if sout.ndim != 4:
        raise ValueError(f"expected 4D sout_rec, got {sout.shape}")
    return path, sout


def extract_rate_features(sout_rec: np.ndarray, t_n: int):
    n_class, n_sample, n_neuron, t = sout_rec.shape
    if t % t_n != 0:
        raise ValueError(f"T={t} is not divisible by T_n={t_n}")
    n_interval = t // t_n
    x = sout_rec.reshape(n_class, n_sample, n_neuron, n_interval, t_n).sum(axis=-1)
    x = x / (t_n / 1000.0)
    x = x.reshape(n_class, n_sample, n_neuron * n_interval).astype(np.float32, copy=False)
    y = np.repeat(np.arange(n_class), n_sample)
    x = x.reshape(n_class * n_sample, -1)
    return x, y, n_class, n_sample


def make_fold_indices(n_sample: int, n_folds: int, seed: int):
    indices = np.arange(n_sample)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return np.array_split(indices, n_folds)


def flatten_class_sample_indices(sample_indices: np.ndarray, n_class: int, n_sample: int):
    out = []
    for c in range(n_class):
        out.extend(c * n_sample + sample_indices)
    return np.asarray(out, dtype=int)


def make_models(random_state: int, pca_dims: list[int]):
    models = {
        "original_mahalanobis": MahalanobisClassifier(ridge=1e-6),
    }
    for dim in pca_dims:
        models[f"pca{dim}_mahalanobis"] = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=dim, svd_solver="randomized", random_state=random_state)),
            ("clf", MahalanobisClassifier(ridge=1e-3)),
        ])
        models[f"pca{dim}_linear_svm"] = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=dim, svd_solver="randomized", random_state=random_state)),
            ("clf", LinearSVC(C=1.0, dual="auto", max_iter=20000, random_state=random_state)),
        ])
        models[f"pca{dim}_rbf_svm"] = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=dim, svd_solver="randomized", random_state=random_state)),
            ("clf", SVC(C=10.0, gamma="scale", kernel="rbf", random_state=random_state)),
        ])

    return models


def eval_models(x, y, n_class: int, n_sample: int, n_folds: int, seed: int, pca_dims: list[int]):
    folds = make_fold_indices(n_sample, n_folds, seed)
    models = make_models(seed, pca_dims)
    rows = []
    conf_total = {name: np.zeros((n_class, n_class), dtype=int) for name in models}

    for name, model in models.items():
        fold_acc = []
        status = "ok"
        error = ""
        for fold_id, test_sample_idx in enumerate(folds):
            train_sample_idx = np.setdiff1d(np.arange(n_sample), test_sample_idx)
            train_idx = flatten_class_sample_indices(train_sample_idx, n_class, n_sample)
            test_idx = flatten_class_sample_indices(test_sample_idx, n_class, n_sample)

            estimator = clone(model)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    estimator.fit(x[train_idx], y[train_idx])
                    pred = estimator.predict(x[test_idx])
                acc = accuracy_score(y[test_idx], pred)
                fold_acc.append(acc)
                conf_total[name] += confusion_matrix(y[test_idx], pred, labels=np.arange(n_class))
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
                break

        if fold_acc:
            rows.append({
                "model": name,
                "status": status,
                "acc_mean": float(np.mean(fold_acc)),
                "acc_std": float(np.std(fold_acc)),
                "acc_min": float(np.min(fold_acc)),
                "acc_max": float(np.max(fold_acc)),
                "n_success_folds": int(len(fold_acc)),
                "error": error,
            })
        else:
            rows.append({
                "model": name,
                "status": status,
                "acc_mean": np.nan,
                "acc_std": np.nan,
                "acc_min": np.nan,
                "acc_max": np.nan,
                "n_success_folds": 0,
                "error": error,
            })

        print(f"[{status}] {name}: {rows[-1]['acc_mean']}")

    return pd.DataFrame(rows).sort_values("acc_mean", ascending=False), conf_total


def save_results(summary_df, conf_total, dataset: str, rep: int, t_n: int):
    out_dir = SCRIPT_DIR / "decoder_compare_results" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / f"{dataset}_rep{rep:02d}_Tn{t_n}_decoder_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")

    xlsx_path = out_dir / f"{dataset}_rep{rep:02d}_Tn{t_n}_decoder_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        for _, row in summary_df.iterrows():
            model = row["model"]
            sheet = str(model)[:31]
            pd.DataFrame(conf_total[model], index=DIR_NAME, columns=DIR_NAME).to_excel(writer, sheet_name=sheet)
    print(f"[saved] {xlsx_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="liquid")
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--t-n", type=int, default=500)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pca-dims", default="100,150,200,300")
    return parser.parse_args()


def main():
    args = parse_args()
    pca_dims = [int(v) for v in args.pca_dims.split(",") if v.strip()]
    path, sout = load_sout(args.dataset, args.rep)
    print(f"[loaded] {path} shape={sout.shape}")
    x, y, n_class, n_sample = extract_rate_features(sout, args.t_n)
    max_pca_dim = min(x.shape[0] - n_class, x.shape[1], n_sample * (n_class - 1))
    pca_dims = [d for d in pca_dims if 1 <= d <= max_pca_dim]
    print(f"[features] X={x.shape}, y={y.shape}, pca_dims={pca_dims}")

    summary_df, conf_total = eval_models(
        x=x,
        y=y,
        n_class=n_class,
        n_sample=n_sample,
        n_folds=args.n_folds,
        seed=args.seed,
        pca_dims=pca_dims,
    )
    save_results(summary_df, conf_total, args.dataset, args.rep, args.t_n)


if __name__ == "__main__":
    main()
