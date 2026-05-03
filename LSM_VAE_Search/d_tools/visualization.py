"""スパイクラスタ、膜電位、重み分布などのデバッグ図を保存する処理。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms


def safe_stem(value: object) -> str:
    text = str(value)
    stem = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    return stem or "item"


def select_record_indices(group, n_record: int, rng: np.random.Generator | None = None) -> list[int]:
    n_neurons = len(group)
    n_record = min(max(int(n_record), 0), n_neurons)
    if n_record == 0:
        return []
    if rng is None or n_record == n_neurons:
        return list(range(n_record))
    return sorted(int(idx) for idx in rng.choice(n_neurons, size=n_record, replace=False))


def _slice_bounds(total: int, start_index: int | None = None, end_index: int | None = None) -> tuple[int, int]:
    start = 0 if start_index is None else max(0, int(start_index))
    end = total if end_index is None else min(total, int(end_index))
    return start, max(start, end)


def _as_ms(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value / ms)
    except Exception:
        return float(value)


def _trial_time_ms_and_mask(
    time_values,
    *,
    start_time=None,
    duration_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    raw_t_ms = np.asarray(time_values / ms)
    keep_all = np.ones(raw_t_ms.shape, dtype=bool)
    if raw_t_ms.size == 0:
        return raw_t_ms, keep_all

    start_ms = _as_ms(start_time)
    duration = float(duration_ms) if duration_ms is not None else None
    if duration is None or duration <= 0:
        if start_ms is None:
            return raw_t_ms, keep_all
        shifted = raw_t_ms - start_ms
        return shifted if np.min(shifted) >= -1e-6 else raw_t_ms, keep_all

    candidates = []
    if start_ms is not None:
        candidates.append(raw_t_ms - start_ms)
    candidates.append(raw_t_ms)
    candidates.append(raw_t_ms - float(np.min(raw_t_ms)))

    best_t = candidates[0]
    best_mask = (best_t >= -1e-6) & (best_t <= duration + 1e-6)
    best_count = int(np.count_nonzero(best_mask))
    for candidate in candidates[1:]:
        mask = (candidate >= -1e-6) & (candidate <= duration + 1e-6)
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_t = candidate
            best_mask = mask
            best_count = count

    return best_t, best_mask


def save_spike_raster(
    out_fp: Path,
    monitor,
    title: str,
    *,
    start_index: int | None = None,
    end_index: int | None = None,
    start_time=None,
    duration_ms: float | None = None,
) -> None:
    # SpikeMonitor の時刻を試行内 0 ms 始まりに直し、ラスタープロットとして保存する。
    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    start, end = _slice_bounds(len(monitor.t), start_index, end_index)
    spike_t = monitor.t[start:end]
    indices = np.asarray(monitor.i[start:end])
    t_ms, mask = _trial_time_ms_and_mask(
        spike_t,
        start_time=start_time,
        duration_ms=duration_ms,
    )
    if t_ms.size:
        t_ms = t_ms[mask]
        indices = indices[mask]

    plt.figure(figsize=(9, 4))
    if t_ms.size:
        plt.scatter(t_ms, indices, s=3, alpha=0.75)
    else:
        plt.text(
            0.5,
            0.5,
            "No spikes",
            transform=plt.gca().transAxes,
            ha="center",
            va="center",
            color="0.35",
        )
    plt.xlabel("Time [ms]")
    plt.ylabel("Neuron index")
    plt.title(f"{title} | spikes={int(indices.size)}")
    if duration_ms is not None and float(duration_ms) > 0:
        plt.xlim(0.0, float(duration_ms))
    plt.tight_layout()
    plt.savefig(out_fp, dpi=150)
    plt.close()


def save_voltage_plot(
    out_fp: Path,
    monitor,
    record_indices: Sequence[int],
    title: str,
    *,
    start_index: int | None = None,
    end_index: int | None = None,
    start_time=None,
    duration_ms: float | None = None,
    spike_monitor=None,
    spike_start_index: int | None = None,
    spike_time=None,
    spike_y: float | None = None,
) -> None:
    # StateMonitor の膜電位を保存する。必要ならスパイク時刻も閾値位置へ重ねて表示する。
    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    start, end = _slice_bounds(len(monitor.t), start_index, end_index)
    voltage_t = monitor.t[start:end]
    voltage = np.asarray(monitor.v[:, start:end])
    t_ms, mask = _trial_time_ms_and_mask(
        voltage_t,
        start_time=start_time,
        duration_ms=duration_ms,
    )
    if t_ms.size:
        t_ms = t_ms[mask]
        voltage = voltage[:, mask]

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for row, neuron_index in enumerate(record_indices):
        if row >= voltage.shape[0]:
            break
        ax.plot(t_ms, voltage[row], linewidth=1.0, label=f"n{int(neuron_index)}")
        plotted = True
    if not plotted or t_ms.size == 0:
        ax.text(
            0.5,
            0.5,
            "No voltage samples",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="0.35",
        )

    if spike_monitor is not None:
        spike_start, spike_end = _slice_bounds(
            len(spike_monitor.t),
            spike_start_index,
            None,
        )
        spike_t = spike_monitor.t[spike_start:spike_end]
        spike_ids = np.asarray(spike_monitor.i[spike_start:spike_end])
        spike_t_ms, spike_mask = _trial_time_ms_and_mask(
            spike_t,
            start_time=spike_time if spike_time is not None else start_time,
            duration_ms=duration_ms,
        )
        if spike_t_ms.size:
            spike_t_ms = spike_t_ms[spike_mask]
            spike_ids = spike_ids[spike_mask]
        marker_y = float(spike_y) if spike_y is not None else None
        if marker_y is None:
            if voltage.size:
                marker_y = float(np.nanmax(voltage))
            else:
                marker_y = 0.0
        for neuron_index in record_indices:
            neuron_spikes = spike_t_ms[spike_ids == int(neuron_index)]
            if neuron_spikes.size:
                ax.scatter(
                    neuron_spikes,
                    np.full(neuron_spikes.shape, marker_y),
                    marker="|",
                    s=40,
                    color="black",
                    alpha=0.45,
                )

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("v")
    ax.set_title(title)
    if duration_ms is not None and float(duration_ms) > 0:
        ax.set_xlim(0.0, float(duration_ms))
    if len(record_indices) <= 12:
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150)
    plt.close(fig)


def save_liquid_3d_plot(out_fp: Path, group, title: str) -> None:
    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(group.x)
    y = np.asarray(group.y)
    z = np.asarray(group.z)
    typ = np.asarray(group.typ)
    exc = typ == 1
    inh = typ == -1

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    if np.any(exc):
        ax.scatter(x[exc], y[exc], z[exc], s=9, alpha=0.65, label="E")
    if np.any(inh):
        ax.scatter(x[inh], y[inh], z[inh], s=14, alpha=0.85, label="I")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150)
    plt.close(fig)


def save_count_bar(out_fp: Path, counts: np.ndarray, title: str) -> None:
    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    counts = np.asarray(counts)
    plt.figure(figsize=(9, 4))
    plt.bar(np.arange(len(counts)), counts)
    plt.xlabel("Neuron index")
    plt.ylabel("Spike count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_fp, dpi=150)
    plt.close()


def _synapse_weights(synapses) -> np.ndarray | None:
    if not hasattr(synapses, "variables") or "w" not in synapses.variables:
        return None
    return np.asarray(synapses.w, dtype=float).reshape(-1)


def _weight_summary(weights: np.ndarray) -> dict[str, float | int]:
    if weights.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(weights.size),
        "mean": float(np.mean(weights)),
        "std": float(np.std(weights)),
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
    }


def save_weight_histogram(out_fp: Path, weights: np.ndarray, title: str, bins: int = 80) -> None:
    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    weights = np.asarray(weights, dtype=float).reshape(-1)
    plt.figure(figsize=(7, 4))
    if weights.size:
        plt.hist(weights, bins=bins)
    plt.xlabel("w")
    plt.ylabel("count")
    plt.title(f"{title} | n={weights.size}")
    plt.tight_layout()
    plt.savefig(out_fp, dpi=150)
    plt.close()


def save_weight_distributions(
    out_dir: Path,
    synapse_groups: dict[str, Sequence],
    tag: str,
    bins: int = 80,
) -> None:
    # Synapses から重みを取り出し、層ごとのヒストグラムと summary を保存する。
    weight_dir = Path(out_dir) / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    rows = []
    text_lines = [f"[{tag}] weight summary\n"]

    for group_name, synapses_list in synapse_groups.items():
        group_weights = []

        for synapses in synapses_list:
            weights = _synapse_weights(synapses)
            if weights is None:
                continue

            group_weights.append(weights)
            key = safe_stem(f"{tag}__{group_name}__{synapses.name}")
            arrays[key] = weights
            save_weight_histogram(
                weight_dir / f"hist_{key}.png",
                weights,
                f"{tag} | {group_name} | {synapses.name}",
                bins=bins,
            )

            summary = _weight_summary(weights)
            rows.append({"tag": tag, "group": group_name, "name": synapses.name, **summary})

        weights_all = np.concatenate(group_weights) if group_weights else np.array([], dtype=float)
        group_key = safe_stem(f"{tag}__{group_name}__all")
        arrays[group_key] = weights_all
        save_weight_histogram(
            weight_dir / f"hist_{group_key}.png",
            weights_all,
            f"{tag} | {group_name} | all",
            bins=bins,
        )

        summary = _weight_summary(weights_all)
        rows.append({"tag": tag, "group": group_name, "name": "all", **summary})
        text_lines.append(
            f"{group_name}: n={summary['n']} mean={summary['mean']:.6g} "
            f"std={summary['std']:.6g} min={summary['min']:.6g} max={summary['max']:.6g}\n"
        )

    np.savez_compressed(weight_dir / f"weights_{safe_stem(tag)}.npz", **arrays)

    fieldnames = ["tag", "group", "name", "n", "mean", "std", "min", "max"]
    with (weight_dir / f"weights_summary_{safe_stem(tag)}.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (weight_dir / f"weights_summary_{safe_stem(tag)}.txt").write_text(
        "".join(text_lines),
        encoding="utf-8",
    )
