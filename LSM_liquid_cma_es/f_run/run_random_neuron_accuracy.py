"""Evaluate material classification accuracy from random liquid neurons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from c_configs.FIXED import cfg_run
from d_tools.run_paths import jsonable
from d_tools.separation_metrics import discover_internal_state_files, scatter_metrics


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})
RESULTS_PATH = PROJECT_ROOT / RUN_CFG["RESULTS_DIR"]
LIQUID_RESULT_DIR = RESULTS_PATH / RUN_CFG["LIQUID_RESULT_DIR"]
INTERNAL_STATE_DIR_NAME = str(RUN_CFG.get("INTERNAL_STATE_DIR", "internal_states"))
DEFAULT_EXPERIMENT_ID = "exp_classify_random100_quick__nliq1000_samples100"
DEFAULT_SAMPLES_PER_CLASS = 100
DEFAULT_SELECTED_NEURONS = 1000
DIR_NAME = [
    "Al_board",
    "buta_omote",
    "buta_ura",
    "cork",
    "denim",
    "rubber_board",
    "washi",
    "wood_board",
]
RIDGE_LAMBDA = 1e-6
RIDGE_LAMBDA_RETRY = 1e-3
NAN_FILL_VALUE = 0.0


def find_latest_internal_state_dir(root: Path = LIQUID_RESULT_DIR) -> Path:
    candidates = [
        path
        for path in Path(root).rglob(INTERNAL_STATE_DIR_NAME)
        if path.is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"No {INTERNAL_STATE_DIR_NAME} directory found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_default_internal_state_dir(root: Path = LIQUID_RESULT_DIR) -> Path:
    preferred = [
        path
        for path in Path(root).rglob(INTERNAL_STATE_DIR_NAME)
        if path.is_dir() and path.parent.name == DEFAULT_EXPERIMENT_ID
    ]
    if preferred:
        return max(preferred, key=lambda path: path.stat().st_mtime)
    return find_latest_internal_state_dir(root)


def _window_mask(
    t_ms: np.ndarray,
    *,
    window_start_ms: float | None = None,
    window_end_ms: float | None = None,
) -> np.ndarray | None:
    if window_start_ms is None and window_end_ms is None:
        return None

    mask = np.ones(t_ms.shape, dtype=bool)
    if window_start_ms is not None:
        mask &= t_ms >= float(window_start_ms)
    if window_end_ms is not None:
        mask &= t_ms <= float(window_end_ms)
    if not np.any(mask):
        raise ValueError("selected internal-state time window is empty.")
    return mask


def load_liquid_states_by_material(
    internal_state_dir: Path,
    *,
    max_samples_per_class: int | None = None,
    window_start_ms: float | None = None,
    window_end_ms: float | None = None,
) -> tuple[np.ndarray, list[str], list[list[str]], float]:
    material_to_files = discover_internal_state_files(internal_state_dir)
    materials = [material for material in DIR_NAME if material in material_to_files]
    if not materials:
        materials = sorted(material_to_files)

    states_by_material = []
    files_by_material = []
    min_neurons: int | None = None
    min_time: int | None = None
    min_samples: int | None = None
    bin_ms_ref: float | None = None

    for material in materials:
        files = material_to_files[material]
        if max_samples_per_class is not None:
            files = files[: int(max_samples_per_class)]
        if not files:
            continue
        material_states = []
        material_files = []
        for fp in files:
            with np.load(fp) as data:
                x_state = np.asarray(data["x_state"], dtype=np.float32)
                t_ms = np.asarray(data["t_ms"], dtype=np.float64)
            if bin_ms_ref is None and t_ms.size >= 2:
                bin_ms_ref = float(np.median(np.diff(t_ms)))
            mask = _window_mask(
                t_ms,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
            )
            if mask is not None:
                x_state = x_state[:, mask]
            min_neurons = x_state.shape[0] if min_neurons is None else min(min_neurons, x_state.shape[0])
            min_time = x_state.shape[1] if min_time is None else min(min_time, x_state.shape[1])
            material_states.append(x_state)
            material_files.append(str(fp))
        min_samples = len(material_states) if min_samples is None else min(min_samples, len(material_states))
        states_by_material.append(material_states)
        files_by_material.append(material_files)

    if not states_by_material:
        raise FileNotFoundError(f"No internal state files found under {internal_state_dir}")
    if (
        min_neurons is None
        or min_time is None
        or min_samples is None
        or min_neurons <= 0
        or min_time <= 0
        or min_samples <= 0
    ):
        raise ValueError("internal states are empty.")

    aligned = np.stack(
        [
            np.stack(
                [state[:min_neurons, :min_time] for state in material_states[:min_samples]],
                axis=0,
            )
            for material_states in states_by_material
        ],
        axis=0,
    )
    aligned_files = [material_files[:min_samples] for material_files in files_by_material]
    return aligned, materials, aligned_files, float(bin_ms_ref if bin_ms_ref else 1.0)


def extract_eval_features(
    states: np.ndarray,
    neuron_indices: np.ndarray,
    *,
    t_n: int,
) -> np.ndarray:
    selected = np.asarray(states[:, :, neuron_indices, :], dtype=np.float64)
    n_sozai, n_sample, n_neurons, n_time = selected.shape
    t_n = int(t_n)
    if n_time % t_n != 0:
        raise ValueError(f"T={n_time} is not divisible by T_n={t_n}")
    n_interval = n_time // t_n
    x = selected.reshape(n_sozai, n_sample, n_neurons, n_interval, t_n)
    state_sum = x.sum(axis=-1)
    rate = state_sum / (t_n / 1000.0)
    features = rate.reshape(n_sozai, n_sample, n_neurons * n_interval).astype(
        np.float64,
        copy=False,
    )
    if not np.isfinite(features).all():
        features = np.nan_to_num(
            features,
            nan=NAN_FILL_VALUE,
            posinf=NAN_FILL_VALUE,
            neginf=NAN_FILL_VALUE,
        )
    return features


def safe_pinv_cov(cov: np.ndarray, ridge: float) -> np.ndarray:
    d = cov.shape[0]
    cov2 = cov + ridge * np.eye(d, dtype=cov.dtype)
    if not np.isfinite(cov2).all():
        cov2 = np.nan_to_num(cov2, nan=0.0, posinf=0.0, neginf=0.0)
    return np.linalg.pinv(cov2)


def fit_ridge_mahalanobis_model(train_data: np.ndarray, ridge: float) -> dict[str, np.ndarray | float]:
    train = np.asarray(train_data, dtype=np.float64)
    if not np.isfinite(train).all():
        train = np.nan_to_num(
            train,
            nan=NAN_FILL_VALUE,
            posinf=NAN_FILL_VALUE,
            neginf=NAN_FILL_VALUE,
        )
    mean = np.mean(train, axis=0)
    centered = train - mean
    denom = max(train.shape[0] - 1, 1)
    u = centered.T / np.sqrt(float(denom))
    gram = np.eye(u.shape[1], dtype=np.float64) + (u.T @ u) / float(ridge)
    try:
        gram_inv = np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        gram_inv = np.linalg.pinv(gram)
    return {"mean": mean, "u": u, "gram_inv": gram_inv, "ridge": float(ridge)}

#マハラノビス距離の2乗を計算、計算量を減らすため Woodbury の恒等式を利用
def mahalanobis_sq_woodbury(x: np.ndarray, model: dict[str, np.ndarray | float]) -> float:
    diff = np.asarray(x, dtype=np.float64) - np.asarray(model["mean"], dtype=np.float64) #素材平均との差を求める
    u = np.asarray(model["u"], dtype=np.float64)
    gram_inv = np.asarray(model["gram_inv"], dtype=np.float64)
    ridge = float(model["ridge"])
    projected = u.T @ diff
    value = (diff @ diff) / ridge - (
        projected.reshape(1, -1) @ gram_inv @ projected.reshape(-1, 1)
    ).item() / (ridge * ridge)
    return float(max(value, 0.0))


def fold_8_to_3(conf_8_fold: np.ndarray) -> np.ndarray:
    mtrx1 = np.zeros((8, 3))
    mtrx1[:, 0] = conf_8_fold[:, 0] + conf_8_fold[:, 5] + conf_8_fold[:, 7]
    mtrx1[:, 1] = conf_8_fold[:, 3] + conf_8_fold[:, 4] + conf_8_fold[:, 6]
    mtrx1[:, 2] = conf_8_fold[:, 1] + conf_8_fold[:, 2]

    mtrx2 = np.zeros((3, 3))
    mtrx2[0, :] = mtrx1[0, :] + mtrx1[5, :] + mtrx1[7, :]
    mtrx2[1, :] = mtrx1[3, :] + mtrx1[4, :] + mtrx1[6, :]
    mtrx2[2, :] = mtrx1[1, :] + mtrx1[2, :]
    return mtrx2


def eval_10fold_like_eval_py(
    features: np.ndarray,
    rng: np.random.Generator,
    n_folds: int,
    *,
    test_size: float | None = None,
    n_repeats: int = 1,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, int, list[float], list[float]]:
    n_sozai, n_sample, dim = features.shape
    if test_size is None and n_sample < n_folds:
        raise ValueError(f"Need at least {n_folds} samples per class, but n_sample={n_sample}")

    if test_size is not None:
        test_size = float(test_size)
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1.")
        if int(n_repeats) <= 0:
            raise ValueError("n_repeats must be at least 1.")
        n_test = max(1, min(n_sample - 1, int(np.ceil(n_sample * test_size))))
        split_indices = []
        for _ in range(int(n_repeats)):
            shuffled = np.arange(n_sample)
            rng.shuffle(shuffled)
            split_indices.append(np.asarray(shuffled[:n_test], dtype=int))
    else:
        all_indices = np.arange(n_sample)
        rng.shuffle(all_indices)
        split_indices = [np.asarray(x, dtype=int) for x in np.array_split(all_indices, n_folds)]

    conf_8_total = np.zeros((n_sozai, n_sozai))
    conf_3_total = np.zeros((3, 3))
    acc_list8 = []
    acc_list3 = []

    for test_idx in split_indices:
        all_indices = np.arange(n_sample)
        train_idx = np.setdiff1d(all_indices, test_idx)

        models = [
            fit_ridge_mahalanobis_model(features[c, train_idx, :], RIDGE_LAMBDA)
            for c in range(n_sozai)
        ]

        conf_8_fold = np.zeros((n_sozai, n_sozai))
        for true_c in range(n_sozai):
            for idx in test_idx:
                x = features[true_c, idx, :]
                if not np.isfinite(x).all():
                    x = np.nan_to_num(
                        x,
                        nan=NAN_FILL_VALUE,
                        posinf=NAN_FILL_VALUE,
                        neginf=NAN_FILL_VALUE,
                    )
                distances = np.asarray(
                    [np.sqrt(mahalanobis_sq_woodbury(x, model)) for model in models],
                    dtype=np.float64,
                )
                pred_c = int(np.argmin(distances))
                conf_8_fold[true_c, pred_c] += 1

        total_samples_fold = np.sum(conf_8_fold)
        acc_list8.append(float(np.trace(conf_8_fold) / total_samples_fold))

        conf_3_fold = fold_8_to_3(conf_8_fold)
        correct_fold_3 = conf_3_fold[0, 0] + conf_3_fold[1, 1] + conf_3_fold[2, 2]
        acc_list3.append(float(correct_fold_3 / total_samples_fold))

        conf_8_total += conf_8_fold
        conf_3_total += conf_3_fold

    total_samples = np.sum(conf_8_total)
    accuracy8_overall = float(np.trace(conf_8_total) / total_samples)
    accuracy8_mean = float(np.mean(acc_list8))
    correct_3_total = conf_3_total[0, 0] + conf_3_total[1, 1] + conf_3_total[2, 2]
    accuracy3_overall = float(correct_3_total / total_samples)
    accuracy3_mean = float(np.mean(acc_list3))

    return (
        conf_8_total,
        conf_3_total,
        accuracy8_overall,
        accuracy8_mean,
        accuracy3_overall,
        accuracy3_mean,
        int(total_samples),
        acc_list8,
        acc_list3,
    )


def evaluate_random_neuron_accuracy(
    internal_state_dir: Path,
    *,
    n_neurons: int | float | None = 100,
    n_repeats: int = 20,
    n_folds: int = 10,
    test_size: float | None = None,
    hold: int = 1,
    seed_value: int = 0,
    t_n_ms: float = 25.0,
    max_samples_per_class: int | None = None,
    window_start_ms: float | None = None,
    window_end_ms: float | None = None,
    out_dir: Path | None = None,
) -> dict:
    if int(n_repeats) <= 0:
        raise ValueError(
            "n_repeats must be at least 1. Use n_neurons=0 to evaluate all available neurons."
        )

    states, materials, source_files, bin_ms = load_liquid_states_by_material(
        internal_state_dir,
        max_samples_per_class=max_samples_per_class,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
    )
    n_available_neurons = int(states.shape[2])
    if n_neurons is None:
        n_select = n_available_neurons
    elif isinstance(n_neurons, float) and not n_neurons.is_integer():
        if not 0.0 < n_neurons <= 1.0:
            raise ValueError("n_neurons ratio must be in the range (0, 1].")
        n_select = max(1, int(np.ceil(n_available_neurons * n_neurons)))
    else:
        requested_neurons = int(n_neurons)
        n_select = (
            n_available_neurons
            if requested_neurons <= 0
            else min(requested_neurons, n_available_neurons)
        )
    n_sozai, n_sample, _, n_time = states.shape
    t_n_bins_float = float(t_n_ms) / float(bin_ms)
    t_n_bins = int(round(t_n_bins_float))
    if not np.isclose(t_n_bins_float, t_n_bins, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"T_n={t_n_ms:g} ms cannot be represented exactly with internal-state "
            f"bin_ms={bin_ms:g} ms. Rerun liquid with INTERNAL_STATE_BIN_MS that "
            f"divides {t_n_ms:g}, e.g. 1.0 or 5.0."
        )
    if n_time % t_n_bins != 0:
        raise ValueError(
            f"T={n_time} bins is not divisible by T_n={t_n_bins} bins "
            f"({t_n_ms:g} ms at bin_ms={bin_ms:g} ms)."
        )
    effective_folds = int(n_folds)
    if test_size is None and n_sample < effective_folds:
        raise ValueError(
            f"eval.py style {effective_folds}-fold needs at least {effective_folds} "
            f"samples per class, but n_sample={n_sample}."
        )
    if n_sozai != 8:
        missing = [name for name in DIR_NAME if name not in materials]
        missing_text = f"; missing: {missing}" if missing else ""
        raise ValueError(
            "eval.py style 3-class aggregation expects 8 classes, "
            f"got {n_sozai}{missing_text}."
        )
    if test_size is None and effective_folds < 2:
        raise ValueError("Need at least 2 samples per class for k-fold accuracy.")
    if test_size is not None and int(hold) <= 0:
        raise ValueError("hold must be at least 1.")

    out_dir = Path(out_dir) if out_dir is not None else Path(internal_state_dir).parent / "random_neuron_accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed_value))
    rows = []
    prediction_rows = []
    fisher_ratio_dr_values = []

    repeat_iter = tqdm(range(1, int(n_repeats) + 1), desc=f"[accuracy] random{n_select}")
    for repeat_index in repeat_iter:
        neuron_indices = np.sort(
            rng.choice(n_available_neurons, size=n_select, replace=False)
        )
        features = extract_eval_features(states, neuron_indices, t_n=t_n_bins)
        fisher_metrics = scatter_metrics(features)
        fisher_ratio_dr_values.append(float(fisher_metrics["DR"]))
        repeat_rng = np.random.default_rng(int(seed_value) + repeat_index)
        (
            conf8,
            conf3,
            acc8_overall,
            acc8_mean,
            acc3_overall,
            acc3_mean,
            total_samples,
            acc_list8,
            acc_list3,
        ) = eval_10fold_like_eval_py(
            features,
            repeat_rng,
            effective_folds,
            test_size=test_size,
            n_repeats=int(hold),
        )

        conf8_fp = out_dir / f"conf_8cls_repeat{repeat_index:03d}.csv"
        conf3_fp = out_dir / f"conf_3cls_repeat{repeat_index:03d}.csv"
        pd.DataFrame(conf8, index=materials, columns=materials).to_csv(conf8_fp)
        pd.DataFrame(conf3).to_csv(conf3_fp, index=False)

        rows.append(
            {
                "repeat": repeat_index,
                "accuracy8_overall": acc8_overall,
                "accuracy8_mean": acc8_mean,
                "accuracy3_overall": acc3_overall,
                "accuracy3_mean": acc3_mean,
                "accuracy8_fold_variance": float(
                    np.var(acc_list8, ddof=1 if len(acc_list8) > 1 else 0)
                ),
                "accuracy3_fold_variance": float(
                    np.var(acc_list3, ddof=1 if len(acc_list3) > 1 else 0)
                ),
                "fisher_ratio_DR": float(fisher_metrics["DR"]),
                "trace_Sb": float(fisher_metrics["trace_Sb"]),
                "trace_Sw": float(fisher_metrics["trace_Sw"]),
                "fold_accuracies8": json.dumps(acc_list8),
                "fold_accuracies3": json.dumps(acc_list3),
                "n_selected_neurons": int(n_select),
                "selected_neurons": json.dumps(neuron_indices.tolist()),
                "conf8_csv": str(conf8_fp),
                "conf3_csv": str(conf3_fp),
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "random_neuron_accuracy_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    holdout_test_count = (
        max(1, min(n_sample - 1, int(np.ceil(n_sample * float(test_size)))))
        if test_size is not None
        else None
    )

    payload = {
        "internal_state_dir": str(Path(internal_state_dir)),
        "out_dir": str(out_dir),
        "classifier": (
            "Mahalanobis repeated holdout"
            if test_size is not None
            else "eval.py Mahalanobis 10-fold"
        ),
        "feature_extraction": "eval.py rate features: reshape by T_n, sum over interval, divide by T_n/1000",
        "n_classes": int(n_sozai),
        "n_sample_per_class": int(n_sample),
        "n_test_per_class": holdout_test_count,
        "n_train_per_class": (
            None if holdout_test_count is None else int(n_sample - holdout_test_count)
        ),
        "n_time_bins": int(n_time),
        "bin_ms": float(bin_ms),
        "T_n_ms": float(t_n_ms),
        "T_n_bins": int(t_n_bins),
        "materials": materials,
        "class_counts": {str(material): int(n_sample) for material in materials},
        "n_available_neurons": n_available_neurons,
        "n_selected_neurons": int(n_select),
        "n_repeats": int(n_repeats),
        "n_folds": int(effective_folds),
        "evaluation_method": (
            "repeated_holdout" if test_size is not None else "k_fold"
        ),
        "test_size": None if test_size is None else float(test_size),
        "train_size": None if test_size is None else float(1.0 - test_size),
        "hold": int(hold),
        "seed": int(seed_value),
        "window_start_ms": window_start_ms,
        "window_end_ms": window_end_ms,
        "accuracy8_overall_mean": float(summary_df["accuracy8_overall"].mean()),
        # Report the standard deviation corresponding to the fold variance
        # used by CMA-ES. The neuron-selection repeat spread is retained
        # separately for diagnostics.
        "accuracy8_overall_std": float(
            np.sqrt(max(float(summary_df["accuracy8_fold_variance"].mean()), 0.0))
        ),
        "accuracy8_neuron_selection_std": float(
            summary_df["accuracy8_overall"].std(ddof=1 if len(summary_df) > 1 else 0)
        ),
        "accuracy8_fold_variance_mean": float(summary_df["accuracy8_fold_variance"].mean()),
        "accuracy8_mean_mean": float(summary_df["accuracy8_mean"].mean()),
        "accuracy3_overall_mean": float(summary_df["accuracy3_overall"].mean()),
        "accuracy3_overall_std": float(
            np.sqrt(max(float(summary_df["accuracy3_fold_variance"].mean()), 0.0))
        ),
        "accuracy3_neuron_selection_std": float(
            summary_df["accuracy3_overall"].std(ddof=1 if len(summary_df) > 1 else 0)
        ),
        "accuracy3_fold_variance_mean": float(summary_df["accuracy3_fold_variance"].mean()),
        "accuracy3_mean_mean": float(summary_df["accuracy3_mean"].mean()),
        "fisher_ratio_DR_mean": float(summary_df["fisher_ratio_DR"].mean()),
        "fisher_ratio_DR_std": float(summary_df["fisher_ratio_DR"].std(ddof=1 if len(summary_df) > 1 else 0)),
        "trace_Sb_mean": float(summary_df["trace_Sb"].mean()),
        "trace_Sw_mean": float(summary_df["trace_Sw"].mean()),
        "summary_csv": str(summary_csv),
    }
    summary_json = out_dir / "random_neuron_accuracy_summary.json"
    summary_json.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    payload["summary_json"] = str(summary_json)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate accuracy from random liquid-neuron subsets."
    )
    parser.add_argument(
        "--internal-state-dir",
        type=Path,
        default=None,
        help="Directory containing material subfolders of *_liquid_internal_state_all.npz files.",
    )
    parser.add_argument("--neurons", type=int, default=DEFAULT_SELECTED_NEURONS)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Use repeated holdout with this test fraction instead of k-fold.",
    )
    parser.add_argument("--hold", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--t-n-ms",
        type=float,
        default=25.0,
        help="eval.py-style interval length in milliseconds.",
    )
    parser.add_argument(
        "--t-n",
        type=float,
        default=None,
        help="Alias for --t-n-ms kept for compatibility.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help="Maximum samples per material class. Default is 100.",
    )
    parser.add_argument("--window-start-ms", type=float, default=None)
    parser.add_argument("--window-end-ms", type=float, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    internal_state_dir = (
        Path(args.internal_state_dir)
        if args.internal_state_dir is not None
        else find_default_internal_state_dir()
    )
    result = evaluate_random_neuron_accuracy(
        internal_state_dir,
        n_neurons=args.neurons,
        n_repeats=args.repeats,
        n_folds=args.folds,
        test_size=args.test_size,
        hold=args.hold,
        seed_value=args.seed,
        t_n_ms=args.t_n if args.t_n is not None else args.t_n_ms,
        max_samples_per_class=args.max_samples_per_class,
        window_start_ms=args.window_start_ms,
        window_end_ms=args.window_end_ms,
        out_dir=args.out_dir,
    )
    print(
        "[accuracy] "
        f"acc8_overall_mean={result['accuracy8_overall_mean']:.4f} "
        f"acc8_overall_std={result['accuracy8_overall_std']:.4f} "
        f"acc3_overall_mean={result['accuracy3_overall_mean']:.4f} "
        f"acc3_overall_std={result['accuracy3_overall_std']:.4f}"
    )
    print(f"[accuracy] saved to {result['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
