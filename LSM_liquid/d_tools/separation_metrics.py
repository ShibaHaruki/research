"""DR、Sb、Sw、ペアワイズ分離、線形分離などの内部状態評価指標。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .plotting import try_import_pyplot


EPS = 1e-12


def _as_2d_samples(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def class_feature_matrices(
    features_by_class: np.ndarray | Sequence[np.ndarray],
    *,
    nan_fill_value: float = 0.0,
) -> list[np.ndarray]:
    """Return a list of ``(n_samples, n_features)`` arrays, one per class."""

    if isinstance(features_by_class, np.ndarray) and features_by_class.dtype != object:
        arr = np.asarray(features_by_class, dtype=np.float64)
        if arr.ndim < 3:
            raise ValueError(
                "features_by_class must have shape "
                "(n_classes, n_samples, ...features)."
            )
        classes = [_as_2d_samples(arr[class_index]) for class_index in range(arr.shape[0])]
    else:
        classes = [_as_2d_samples(np.asarray(item, dtype=np.float64)) for item in features_by_class]

    if not classes:
        raise ValueError("At least one class is required.")

    n_features = classes[0].shape[1]
    for class_index, class_features in enumerate(classes):
        if class_features.ndim != 2:
            raise ValueError(f"class {class_index} is not 2D after flattening.")
        if class_features.shape[0] == 0:
            raise ValueError(f"class {class_index} has no samples.")
        if class_features.shape[1] != n_features:
            raise ValueError(
                f"feature dimension mismatch at class {class_index}: "
                f"{class_features.shape[1]} != {n_features}"
            )

    return [
        np.nan_to_num(
            class_features,
            nan=nan_fill_value,
            posinf=nan_fill_value,
            neginf=nan_fill_value,
        )
        for class_features in classes
    ]


def features_from_labels(
    features: np.ndarray,
    labels: Sequence,
    *,
    label_order: Sequence | None = None,
    nan_fill_value: float = 0.0,
) -> tuple[list[np.ndarray], list]:
    """Convert ``(n_samples, ...features)`` plus labels into class feature matrices."""

    x = _as_2d_samples(np.asarray(features, dtype=np.float64))
    y = np.asarray(labels)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"features and labels length mismatch: {x.shape[0]} != {y.shape[0]}")

    labels_unique = list(label_order) if label_order is not None else list(dict.fromkeys(y.tolist()))
    classes = [x[y == label] for label in labels_unique]
    return class_feature_matrices(classes, nan_fill_value=nan_fill_value), labels_unique


def class_priors_from_counts(
    counts: Sequence[int],
    priors: Sequence[float] | None = None,
) -> np.ndarray:
    counts_arr = np.asarray(counts, dtype=np.float64)
    if priors is None:
        total = float(np.sum(counts_arr))
        if total <= 0:
            raise ValueError("total sample count must be positive.")
        return counts_arr / total

    priors_arr = np.asarray(priors, dtype=np.float64).reshape(-1)
    if priors_arr.shape[0] != counts_arr.shape[0]:
        raise ValueError("class_priors length must match number of classes.")
    total = float(np.sum(priors_arr))
    if total <= 0:
        raise ValueError("class_priors must sum to a positive value.")
    return priors_arr / total


def scatter_metrics(
    features_by_class: np.ndarray | Sequence[np.ndarray],
    *,
    class_priors: Sequence[float] | None = None,
    ddof: int = 1,
    return_matrices: bool = False,
    nan_fill_value: float = 0.0,
    eps: float = EPS,
) -> dict:
    """Compute Sw, Sb and DR used in Wijesinghe et al. (2019).

    Sw = sum_i P(w_i) Sigma_i
    Sb = sum_i P(w_i) (mu_i - mu_g)(mu_i - mu_g)^T
    DR = tr(Sb) / tr(Sw)
    """

    # 素材ごとの特徴を使い、クラス内の広がり Sw とクラス間の広がり Sb を計算する。
    classes = class_feature_matrices(features_by_class, nan_fill_value=nan_fill_value)
    counts = np.asarray([class_features.shape[0] for class_features in classes], dtype=int)
    priors = class_priors_from_counts(counts, class_priors)
    means = np.vstack([np.mean(class_features, axis=0) for class_features in classes])
    global_mean = np.sum(means * priors[:, None], axis=0)

    trace_sw = 0.0
    trace_sb = 0.0
    sw = None
    sb = None
    n_features = classes[0].shape[1]
    if return_matrices:
        sw = np.zeros((n_features, n_features), dtype=np.float64)
        sb = np.zeros((n_features, n_features), dtype=np.float64)

    for class_index, class_features in enumerate(classes):
        centered = class_features - means[class_index]
        denom = max(int(class_features.shape[0]) - int(ddof), 1)
        variances = np.sum(centered * centered, axis=0) / float(denom)
        trace_sw += float(priors[class_index] * np.sum(variances))
        if return_matrices:
            sw += priors[class_index] * (centered.T @ centered) / float(denom)

        diff = means[class_index] - global_mean
        trace_sb += float(priors[class_index] * np.dot(diff, diff))
        if return_matrices:
            sb += priors[class_index] * np.outer(diff, diff)

    if trace_sw <= eps:
        dr = np.inf if trace_sb > eps else 0.0
    else:
        dr = trace_sb / trace_sw

    result = {
        "DR": float(dr),
        "trace_Sb": float(trace_sb),
        "trace_Sw": float(trace_sw),
        "class_priors": priors,
        "class_means": means,
        "global_mean": global_mean,
        "n_classes": int(len(classes)),
        "n_samples_total": int(np.sum(counts)),
        "n_features": int(n_features),
        "class_counts": counts,
    }
    if return_matrices:
        result["Sb"] = sb
        result["Sw"] = sw
    return result


def discriminant_ratio(
    features_by_class: np.ndarray | Sequence[np.ndarray],
    **kwargs,
) -> float:
    return float(scatter_metrics(features_by_class, **kwargs)["DR"])


def _trajectory_to_time_feature_matrix(states: np.ndarray, *, time_axis: int = -1) -> np.ndarray:
    arr = np.asarray(states, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError("trajectory states must have at least feature and time axes.")
    arr = np.moveaxis(arr, time_axis, 0)
    return arr.reshape(arr.shape[0], -1)


def pairwise_separation_property(
    states_u: np.ndarray,
    states_v: np.ndarray,
    *,
    time_axis: int = -1,
    nan_fill_value: float = 0.0,
) -> dict:
    """Compute SPpw = average_t ||x_u(t) - x_v(t)|| for two trajectories."""

    u = _trajectory_to_time_feature_matrix(states_u, time_axis=time_axis)
    v = _trajectory_to_time_feature_matrix(states_v, time_axis=time_axis)
    n_time = min(u.shape[0], v.shape[0])
    if n_time <= 0:
        raise ValueError("trajectories must have at least one time point.")
    if u.shape[1] != v.shape[1]:
        raise ValueError(f"feature dimension mismatch: {u.shape[1]} != {v.shape[1]}")

    diff = np.nan_to_num(
        u[:n_time] - v[:n_time],
        nan=nan_fill_value,
        posinf=nan_fill_value,
        neginf=nan_fill_value,
    )
    distances = np.linalg.norm(diff, axis=1)
    return {
        "SPpw": float(np.mean(distances)),
        "distances": distances,
        "n_time": int(n_time),
        "n_features": int(u.shape[1]),
    }


def _mean_pairwise_distance(a: np.ndarray, b: np.ndarray, *, batch_size: int = 256) -> float:
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"feature dimension mismatch: {a.shape[1]} != {b.shape[1]}")
    total = 0.0
    count = 0
    for start in range(0, a.shape[0], batch_size):
        chunk = a[start : start + batch_size]
        distances = np.sqrt(np.sum((chunk[:, None, :] - b[None, :, :]) ** 2, axis=2))
        total += float(np.sum(distances))
        count += int(distances.size)
    return total / max(count, 1)


def _mean_within_class_distance(a: np.ndarray, *, batch_size: int = 256) -> float:
    if a.shape[0] <= 1:
        return 0.0
    total = 0.0
    count = 0
    for start in range(0, a.shape[0], batch_size):
        chunk = a[start : start + batch_size]
        distances = np.sqrt(np.sum((chunk[:, None, :] - a[None, :, :]) ** 2, axis=2))
        row_indices = np.arange(start, start + chunk.shape[0])
        distances[np.arange(chunk.shape[0]), row_indices] = np.nan
        valid = np.isfinite(distances)
        total += float(np.sum(distances[valid]))
        count += int(np.sum(valid))
    return total / max(count, 1)


def pairwise_separation_matrix(
    features_by_class: np.ndarray | Sequence[np.ndarray],
    *,
    batch_size: int = 256,
    nan_fill_value: float = 0.0,
) -> dict:
    """Aggregate pairwise Euclidean separations for class-organized state vectors."""

    # 各素材内・素材間の平均ユークリッド距離を計算し、ペアワイズ分離の強さを見る。
    classes = class_feature_matrices(features_by_class, nan_fill_value=nan_fill_value)
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        matrix[i, i] = _mean_within_class_distance(classes[i], batch_size=batch_size)
        for j in range(i + 1, n_classes):
            value = _mean_pairwise_distance(classes[i], classes[j], batch_size=batch_size)
            matrix[i, j] = value
            matrix[j, i] = value

    if n_classes > 1:
        mask = ~np.eye(n_classes, dtype=bool)
        overall_between = float(np.mean(matrix[mask]))
    else:
        overall_between = 0.0
    overall_within = float(np.mean(np.diag(matrix)))
    return {
        "SPpw_between_mean": overall_between,
        "SPpw_within_mean": overall_within,
        "pairwise_matrix": matrix,
        "n_classes": int(n_classes),
    }


def _trajectory_samples(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr.reshape(1, arr.shape[0], arr.shape[1])
    if arr.ndim != 3:
        raise ValueError(f"trajectory samples must be 3D, got shape={arr.shape}")
    return arr


def _trajectory_pair_distance(a: np.ndarray, b: np.ndarray) -> float:
    n_features = min(a.shape[0], b.shape[0])
    n_time = min(a.shape[1], b.shape[1])
    if n_features <= 0 or n_time <= 0:
        return 0.0
    diff = a[:n_features, :n_time] - b[:n_features, :n_time]
    return float(np.mean(np.sqrt(np.sum(diff * diff, axis=0))))


def _selected_pair_indices(
    n_a: int,
    n_b: int,
    *,
    include_self: bool,
    max_pairs: int | None,
    seed: int,
) -> list[tuple[int, int]]:
    pairs = [
        (i, j)
        for i in range(n_a)
        for j in range(n_b)
        if include_self or i != j
    ]
    if max_pairs is not None and len(pairs) > int(max_pairs):
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(pairs), size=int(max_pairs), replace=False)
        pairs = [pairs[int(index)] for index in selected]
    return pairs


def pairwise_trajectory_separation_matrix(
    trajectories_by_class: Sequence[np.ndarray],
    *,
    max_pairs_per_class_pair: int | None = None,
    seed: int = 0,
    nan_fill_value: float = 0.0,
) -> dict:
    """Compute SPpw by averaging trajectory distances across sample pairs.

    Each class entry is expected to have shape ``(n_samples, n_neurons, n_time)``.
    """

    classes = [
        np.nan_to_num(
            _trajectory_samples(np.asarray(item, dtype=np.float64)),
            nan=nan_fill_value,
            posinf=nan_fill_value,
            neginf=nan_fill_value,
        )
        for item in trajectories_by_class
    ]
    if not classes:
        raise ValueError("At least one class is required.")

    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=np.float64)
    counts = np.zeros((n_classes, n_classes), dtype=np.int32)

    for i in range(n_classes):
        for j in range(i, n_classes):
            pairs = _selected_pair_indices(
                classes[i].shape[0],
                classes[j].shape[0],
                include_self=(i != j),
                max_pairs=max_pairs_per_class_pair,
                seed=seed + i * 1009 + j,
            )
            if not pairs:
                value = 0.0
            else:
                value = float(
                    np.mean([
                        _trajectory_pair_distance(classes[i][a], classes[j][b])
                        for a, b in pairs
                    ])
                )
            matrix[i, j] = value
            matrix[j, i] = value
            counts[i, j] = len(pairs)
            counts[j, i] = len(pairs)

    if n_classes > 1:
        mask = ~np.eye(n_classes, dtype=bool)
        overall_between = float(np.mean(matrix[mask]))
    else:
        overall_between = 0.0
    overall_within = float(np.mean(np.diag(matrix)))
    return {
        "SPpw_between_mean": overall_between,
        "SPpw_within_mean": overall_within,
        "pairwise_matrix": matrix,
        "pair_counts": counts,
        "n_classes": int(n_classes),
    }


def _aligned_trajectory_classes(
    trajectories_by_class: Sequence[np.ndarray],
    *,
    nan_fill_value: float = 0.0,
) -> list[np.ndarray]:
    classes = [
        np.nan_to_num(
            _trajectory_samples(np.asarray(item, dtype=np.float64)),
            nan=nan_fill_value,
            posinf=nan_fill_value,
            neginf=nan_fill_value,
        )
        for item in trajectories_by_class
    ]
    if not classes:
        raise ValueError("At least one class is required.")

    min_features = min(item.shape[1] for item in classes)
    min_time = min(item.shape[2] for item in classes)
    if min_features <= 0 or min_time <= 0:
        raise ValueError("trajectory classes must have non-empty feature and time axes.")
    return [item[:, :min_features, :min_time] for item in classes]


def temporal_separation_metrics(
    trajectories_by_class: Sequence[np.ndarray],
    *,
    t_ms: np.ndarray | None = None,
    class_priors: Sequence[float] | None = None,
    ddof: int = 1,
    rank_tol: float | None = None,
    nan_fill_value: float = 0.0,
    eps: float = EPS,
) -> dict:
    """Compute DR, tr(Sb), tr(Sw), and SPlin at each internal-state time point."""

    # 時間点ごとに特徴を切り出し、DR/Sb/Sw/線形分離がどの時刻で高いかを見る。
    classes = _aligned_trajectory_classes(
        trajectories_by_class,
        nan_fill_value=nan_fill_value,
    )
    counts = np.asarray([item.shape[0] for item in classes], dtype=int)
    priors = class_priors_from_counts(counts, class_priors)
    n_classes = len(classes)
    n_features = int(classes[0].shape[1])
    n_time = int(classes[0].shape[2])

    trace_sb = np.zeros(n_time, dtype=np.float64)
    trace_sw = np.zeros(n_time, dtype=np.float64)
    dr = np.zeros(n_time, dtype=np.float64)
    splin = np.zeros(n_time, dtype=np.int32)

    for time_index in range(n_time):
        means = np.vstack([
            np.mean(class_values[:, :, time_index], axis=0)
            for class_values in classes
        ])
        global_mean = np.sum(means * priors[:, None], axis=0)

        sw_t = 0.0
        sb_t = 0.0
        features_at_t = []
        for class_index, class_values in enumerate(classes):
            x_t = class_values[:, :, time_index]
            features_at_t.append(x_t)

            centered = x_t - means[class_index]
            denom = max(int(x_t.shape[0]) - int(ddof), 1)
            variances = np.sum(centered * centered, axis=0) / float(denom)
            sw_t += float(priors[class_index] * np.sum(variances))

            diff = means[class_index] - global_mean
            sb_t += float(priors[class_index] * np.dot(diff, diff))

        trace_sw[time_index] = sw_t
        trace_sb[time_index] = sb_t
        if sw_t <= eps:
            dr[time_index] = np.inf if sb_t > eps else 0.0
        else:
            dr[time_index] = sb_t / sw_t

        stacked = np.vstack(features_at_t)
        splin[time_index] = int(np.linalg.matrix_rank(stacked.T, tol=rank_tol))

    max_rank = int(min(int(np.sum(counts)), n_features))
    if t_ms is None:
        t_ms_out = np.arange(n_time, dtype=np.float64)
    else:
        t_ms_out = np.asarray(t_ms, dtype=np.float64).reshape(-1)[:n_time]
        if t_ms_out.size < n_time:
            t_ms_out = np.arange(n_time, dtype=np.float64)

    return {
        "t_ms": t_ms_out,
        "DR": dr,
        "trace_Sb": trace_sb,
        "trace_Sw": trace_sw,
        "SPlin": splin,
        "SPlin_normalized": splin.astype(np.float64) / max(max_rank, 1),
        "n_classes": int(n_classes),
        "n_samples_total": int(np.sum(counts)),
        "n_features": int(n_features),
        "n_time": int(n_time),
        "max_rank": max_rank,
        "class_counts": counts,
    }


def save_temporal_separation_metrics(
    out_dir: Path,
    metrics: dict,
    *,
    stem: str = "temporal_separation",
) -> tuple[Path, Path | None]:
    """Save time-series DR/Sb/Sw/SPlin metrics as CSV and a summary plot."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_fp = out_dir / f"{stem}.csv"
    df = pd.DataFrame(
        {
            "t_ms": metrics["t_ms"],
            "DR": metrics["DR"],
            "trace_Sb": metrics["trace_Sb"],
            "trace_Sw": metrics["trace_Sw"],
            "SPlin": metrics["SPlin"],
            "SPlin_normalized": metrics["SPlin_normalized"],
        }
    )
    df.to_csv(csv_fp, index=False)

    plt = try_import_pyplot()
    if plt is None:
        return csv_fp, None

    plot_fp = out_dir / f"{stem}.png"
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    t_ms = metrics["t_ms"]
    series = (
        ("DR", metrics["DR"]),
        ("trace_Sb", metrics["trace_Sb"]),
        ("trace_Sw", metrics["trace_Sw"]),
        ("SPlin", metrics["SPlin"]),
    )
    for ax, (ylabel, values) in zip(axes, series):
        ax.plot(t_ms, values, linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time [ms]")
    fig.suptitle("Temporal internal-state separation metrics")
    fig.tight_layout()
    fig.savefig(plot_fp, dpi=150)
    plt.close(fig)
    return csv_fp, plot_fp


def linear_separation_property(
    states: np.ndarray | Sequence[np.ndarray],
    *,
    tol: float | None = None,
    nan_fill_value: float = 0.0,
) -> dict:
    """Compute SPlin = rank(Ms), where columns of Ms are liquid state vectors."""

    # 状態ベクトルが張る空間のランクを使って、線形に区別できる自由度を見る。
    if isinstance(states, np.ndarray) and states.dtype != object and states.ndim == 2:
        x = _as_2d_samples(states)
    else:
        classes = class_feature_matrices(states, nan_fill_value=nan_fill_value)
        x = np.vstack(classes)

    x = np.nan_to_num(x, nan=nan_fill_value, posinf=nan_fill_value, neginf=nan_fill_value)
    rank = int(np.linalg.matrix_rank(x.T, tol=tol))
    max_rank = int(min(x.shape[0], x.shape[1]))
    return {
        "SPlin": rank,
        "rank": rank,
        "normalized_rank": float(rank / max(max_rank, 1)),
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "max_rank": max_rank,
    }


def linear_separation_by_time(
    trajectories: np.ndarray,
    *,
    time_axis: int = -1,
    tol: float | None = None,
    nan_fill_value: float = 0.0,
) -> dict:
    """Compute SPlin at every sampled time point for trajectories.

    Expected input is ``(n_classes, n_samples, ...features, n_time)`` by default.
    """

    arr = np.asarray(trajectories, dtype=np.float64)
    if arr.ndim < 4:
        raise ValueError(
            "trajectories must have shape "
            "(n_classes, n_samples, ...features, n_time)."
        )
    arr = np.moveaxis(arr, time_axis, -1)
    n_classes, n_samples, *feature_shape, n_time = arr.shape
    x = arr.reshape(n_classes * n_samples, int(np.prod(feature_shape)), n_time)
    ranks = np.zeros(n_time, dtype=np.int32)
    for time_index in range(n_time):
        ranks[time_index] = linear_separation_property(
            x[:, :, time_index],
            tol=tol,
            nan_fill_value=nan_fill_value,
        )["rank"]
    max_rank = int(min(n_classes * n_samples, int(np.prod(feature_shape))))
    return {
        "SPlin_by_time": ranks,
        "SPlin_mean": float(np.mean(ranks)),
        "SPlin_max": int(np.max(ranks)),
        "normalized_rank_mean": float(np.mean(ranks / max(max_rank, 1))),
        "n_time": int(n_time),
        "max_rank": max_rank,
    }


def sout_rec_to_features(
    sout_rec: np.ndarray,
    *,
    window_bins: int | None = None,
    T_n_ms: float | None = None,
    bin_width_ms: float | None = None,
    rate: bool = True,
    nan_fill_value: float = 0.0,
) -> np.ndarray:
    """Convert ``sout_rec[class, sample, neuron, bin]`` to class feature vectors."""

    arr = np.asarray(sout_rec, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"sout_rec must be 4D, got shape={arr.shape}")

    if window_bins is None:
        if T_n_ms is not None:
            if bin_width_ms is None:
                raise ValueError("bin_width_ms is required when T_n_ms is provided.")
            window_bins = max(1, int(round(float(T_n_ms) / float(bin_width_ms))))
        else:
            window_bins = 1
    window_bins = max(1, int(window_bins))

    n_classes, n_samples, n_units, n_bins = arr.shape
    n_windows = n_bins // window_bins
    if n_windows <= 0:
        raise ValueError(f"window_bins={window_bins} is larger than n_bins={n_bins}.")

    arr = arr[:, :, :, : n_windows * window_bins]
    arr = arr.reshape(n_classes, n_samples, n_units, n_windows, window_bins)
    values = np.sum(arr, axis=-1)
    if rate:
        if bin_width_ms is None:
            duration_s = float(window_bins)
        else:
            duration_s = float(window_bins) * float(bin_width_ms) / 1000.0
        values = values / max(duration_s, EPS)

    features = values.reshape(n_classes, n_samples, n_units * n_windows)
    return np.nan_to_num(
        features,
        nan=nan_fill_value,
        posinf=nan_fill_value,
        neginf=nan_fill_value,
    )


def internal_state_to_feature(
    x_state: np.ndarray,
    *,
    mode: str = "final",
    window_start_ms: float | None = None,
    window_end_ms: float | None = None,
    t_ms: np.ndarray | None = None,
) -> np.ndarray:
    """Convert one saved liquid internal state ``(n_neurons, n_time)`` to a feature vector."""

    # final/mean/max/sum/flatten のどれで時間軸を特徴量へ畳み込むかを選ぶ。
    x = np.asarray(x_state, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"x_state must be 2D, got shape={x.shape}")

    if window_start_ms is not None or window_end_ms is not None:
        if t_ms is None:
            raise ValueError("t_ms is required when using a time window.")
        t_arr = np.asarray(t_ms, dtype=np.float64)
        mask = np.ones(t_arr.shape, dtype=bool)
        if window_start_ms is not None:
            mask &= t_arr >= float(window_start_ms)
        if window_end_ms is not None:
            mask &= t_arr <= float(window_end_ms)
        if not np.any(mask):
            raise ValueError("selected internal-state time window is empty.")
        x = x[:, mask]

    mode = str(mode).lower()
    if mode == "final":
        return x[:, -1]
    if mode == "mean":
        return np.mean(x, axis=1)
    if mode == "max":
        return np.max(x, axis=1)
    if mode == "sum":
        return np.sum(x, axis=1)
    if mode == "flatten":
        return x.reshape(-1)
    raise ValueError(f"Unknown internal state feature mode: {mode}")


def discover_internal_state_files(
    internal_state_dir: Path,
    *,
    file_glob: str = "*_liquid_internal_state_all.npz",
) -> dict[str, list[Path]]:
    root = Path(internal_state_dir)
    material_to_files: dict[str, list[Path]] = {}
    if not root.exists():
        raise FileNotFoundError(root)

    for material_dir in sorted(child for child in root.iterdir() if child.is_dir()):
        files = sorted(material_dir.glob(file_glob))
        if files:
            material_to_files[material_dir.name] = files
    if not material_to_files:
        raise FileNotFoundError(f"No internal state files found under {root}")
    return material_to_files


def load_internal_state_dataset(
    internal_state_dir: Path,
    *,
    feature_mode: str = "final",
    materials: Sequence[str] | None = None,
    max_samples_per_class: int | None = None,
    window_start_ms: float | None = None,
    window_end_ms: float | None = None,
    load_trajectories: bool = True,
    file_glob: str = "*_liquid_internal_state_all.npz",
) -> dict:
    """Load run_liquid internal states grouped by material."""

    # internal_states/<素材名>/*.npz を素材ラベルごとに読み、指標計算用に整形する。
    material_to_files = discover_internal_state_files(internal_state_dir, file_glob=file_glob)
    material_names = list(materials) if materials is not None else sorted(material_to_files)

    features_by_class = []
    trajectories_by_class = []
    files_by_class = {}
    t_ms_ref = None

    for material in material_names:
        files = material_to_files.get(str(material), [])
        if max_samples_per_class is not None:
            files = files[: int(max_samples_per_class)]
        if not files:
            continue

        features = []
        trajectories = []
        for fp in files:
            with np.load(fp) as data:
                x_state = np.asarray(data["x_state"], dtype=np.float64)
                t_ms = np.asarray(data["t_ms"], dtype=np.float64)
            if t_ms_ref is None:
                t_ms_ref = t_ms
            features.append(
                internal_state_to_feature(
                    x_state,
                    mode=feature_mode,
                    window_start_ms=window_start_ms,
                    window_end_ms=window_end_ms,
                    t_ms=t_ms,
                )
            )
            if load_trajectories:
                if window_start_ms is not None or window_end_ms is not None:
                    mask = np.ones(t_ms.shape, dtype=bool)
                    if window_start_ms is not None:
                        mask &= t_ms >= float(window_start_ms)
                    if window_end_ms is not None:
                        mask &= t_ms <= float(window_end_ms)
                    trajectories.append(x_state[:, mask])
                else:
                    trajectories.append(x_state)

        features_by_class.append(np.vstack(features))
        if load_trajectories:
            min_neurons = min(item.shape[0] for item in trajectories)
            min_time = min(item.shape[1] for item in trajectories)
            trajectories_by_class.append(
                np.stack([item[:min_neurons, :min_time] for item in trajectories], axis=0)
            )
        files_by_class[str(material)] = [str(fp) for fp in files]

    if not features_by_class:
        raise FileNotFoundError(f"No selected internal state files found under {internal_state_dir}")

    return {
        "materials": [name for name in material_names if str(name) in files_by_class],
        "features_by_class": features_by_class,
        "trajectories_by_class": trajectories_by_class,
        "files_by_class": files_by_class,
        "t_ms": t_ms_ref,
    }


def separation_summary(
    features_by_class: np.ndarray | Sequence[np.ndarray],
    *,
    class_priors: Sequence[float] | None = None,
    rank_tol: float | None = None,
    batch_size: int = 256,
    nan_fill_value: float = 0.0,
) -> dict:
    # 代表的な分離指標をまとめて計算する高レベル入口。
    scatter = scatter_metrics(
        features_by_class,
        class_priors=class_priors,
        return_matrices=False,
        nan_fill_value=nan_fill_value,
    )
    linear = linear_separation_property(
        features_by_class,
        tol=rank_tol,
        nan_fill_value=nan_fill_value,
    )
    pairwise = pairwise_separation_matrix(
        features_by_class,
        batch_size=batch_size,
        nan_fill_value=nan_fill_value,
    )
    return {
        "DR": scatter["DR"],
        "trace_Sb": scatter["trace_Sb"],
        "trace_Sw": scatter["trace_Sw"],
        "SPlin": linear["SPlin"],
        "SPlin_normalized": linear["normalized_rank"],
        "SPpw_between_mean": pairwise["SPpw_between_mean"],
        "SPpw_within_mean": pairwise["SPpw_within_mean"],
        "n_classes": scatter["n_classes"],
        "n_samples_total": scatter["n_samples_total"],
        "n_features": scatter["n_features"],
    }


def save_scatter_matrices_npz(out_dir: Path, metrics: dict, *, stem: str) -> Path:
    if "Sb" not in metrics or "Sw" not in metrics:
        raise ValueError("metrics must be computed with return_matrices=True.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"{stem}_scatter_matrices.npz"
    np.savez_compressed(
        out_fp,
        Sb=metrics["Sb"],
        Sw=metrics["Sw"],
        class_means=metrics["class_means"],
        global_mean=metrics["global_mean"],
        class_priors=metrics["class_priors"],
    )
    return out_fp


def save_pairwise_matrix_csv(
    out_dir: Path,
    matrix: np.ndarray,
    *,
    stem: str,
    labels: Sequence[str] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(labels) if labels is not None else [f"class_{i}" for i in range(matrix.shape[0])]
    out_fp = out_dir / f"{stem}_pairwise_separation.csv"
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(out_fp)
    return out_fp
