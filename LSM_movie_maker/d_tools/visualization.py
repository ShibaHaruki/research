"""Plotting helpers for spikes, voltages, weights, and movies."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch
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


def _group_scalar_value(group, name: str, default: float | None = None) -> float | None:
    if not hasattr(group, name):
        return default
    value = getattr(group, name)
    for expr in (
        lambda: value[:],
        lambda: value[0],
        lambda: value,
    ):
        try:
            arr = np.asarray(expr(), dtype=float).reshape(-1)
            if arr.size:
                return float(arr[0])
        except Exception:
            pass
    return default


def _monitor_record_labels(monitor, fallback: Sequence[int] | None = None) -> list[int]:
    try:
        labels = np.asarray(monitor.record, dtype=int).reshape(-1)
        if labels.size:
            return [int(label) for label in labels]
    except Exception:
        pass
    return [int(label) for label in (fallback or [])]


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

    if start_ms is not None:
        shifted = raw_t_ms - start_ms
        mask = (shifted >= -1e-6) & (shifted <= duration + 1e-6)
        if np.any(mask):
            return shifted, mask

    candidates = []
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
    # SpikeMonitor 縺ｮ譎ょ綾繧定ｩｦ陦悟・ 0 ms 蟋九∪繧翫↓逶ｴ縺励√Λ繧ｹ繧ｿ繝ｼ繝励Ο繝・ヨ縺ｨ縺励※菫晏ｭ倥☆繧九・    out_fp = Path(out_fp)
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
    # StateMonitor 縺ｮ閹憺崕菴阪ｒ菫晏ｭ倥☆繧九ょｿ・ｦ√↑繧峨せ繝代う繧ｯ譎ょ綾繧る明蛟､菴咲ｽｮ縺ｸ驥阪・縺ｦ陦ｨ遉ｺ縺吶ｋ縲・    out_fp = Path(out_fp)
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
    line_colors: dict[int, str] = {}
    record_labels = _monitor_record_labels(monitor, record_indices)
    for row, neuron_index in enumerate(record_labels):
        if row >= voltage.shape[0]:
            break
        (line,) = ax.plot(t_ms, voltage[row], linewidth=1.0, label=f"n{int(neuron_index)}")
        line_colors[int(neuron_index)] = line.get_color()
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


    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("v")
    ax.set_title(title)
    if duration_ms is not None and float(duration_ms) > 0:
        ax.set_xlim(0.0, float(duration_ms))
    if len(record_labels) <= 12:
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150)
    plt.close(fig)


def _monitor_window_arrays(
    monitor,
    *,
    start_index: int | None,
    start_time=None,
    duration_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    start, end = _slice_bounds(len(monitor.t), start_index, None)
    t_ms, mask = _trial_time_ms_and_mask(
        monitor.t[start:end],
        start_time=start_time,
        duration_ms=duration_ms,
    )
    values = np.asarray(monitor.v[:, start:end])
    if t_ms.size:
        t_ms = t_ms[mask]
        values = values[:, mask]
    return t_ms, values


def _spike_window_arrays(
    monitor,
    *,
    start_index: int | None,
    start_time=None,
    duration_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    start, end = _slice_bounds(len(monitor.t), start_index, None)
    t_ms, mask = _trial_time_ms_and_mask(
        monitor.t[start:end],
        start_time=start_time,
        duration_ms=duration_ms,
    )
    indices = np.asarray(monitor.i[start:end])
    if t_ms.size:
        t_ms = t_ms[mask]
        indices = indices[mask]
    return t_ms, indices


def save_liquid_process_movie(
    out_fp: Path,
    *,
    mat: str,
    sid: int,
    raw_input: np.ndarray,
    t_array: np.ndarray,
    input_current: np.ndarray,
    input_filter_map: dict[int, list[str]],
    groups: Sequence,
    spike_monitors: Sequence,
    voltage_monitors: Sequence,
    voltage_indices: Sequence[Sequence[int]],
    output_groups: Sequence | None = None,
    output_spike_monitors: Sequence | None = None,
    output_voltage_monitors: Sequence | None = None,
    output_voltage_indices: Sequence[Sequence[int]] | None = None,
    duration_ms: float,
    spike_start_indices: Sequence[int] | None = None,
    voltage_start_indices: Sequence[int] | None = None,
    output_spike_start_indices: Sequence[int] | None = None,
    output_voltage_start_indices: Sequence[int] | None = None,
    start_time=None,
    fps: float = 12.0,
    seconds: float = 8.0,
    max_raw_traces: int = 4,
    max_filter_traces: int = 8,
    max_voltage_traces: int = 5,
    select_voltage_by_spike_count: bool = False,
) -> Path:
    # Slide-friendly overview: raw input, filtered input, voltage, and spike raster.
    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    raw_input = np.asarray(raw_input, dtype=float)
    t_input_ms = np.asarray(t_array, dtype=float) * 1000.0
    input_current = np.asarray(input_current, dtype=float)
    duration_ms = float(duration_ms)

    n_frames = max(2, int(round(max(float(fps), 1.0) * max(float(seconds), 0.1))))
    frame_times = np.linspace(0.0, duration_ms, n_frames)

    output_groups = list(output_groups or [])
    output_spike_monitors = list(output_spike_monitors or [])
    output_voltage_monitors = list(output_voltage_monitors or [])
    output_voltage_indices = list(output_voltage_indices or [])
    has_output = bool(output_groups and output_spike_monitors and output_voltage_monitors)

    n_panels = 6 if has_output else 4
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(11, 10 if has_output else 8),
        sharex=True,
    )
    raw_ax, filtered_ax, voltage_ax, spike_ax = axes[:4]
    output_voltage_ax = axes[4] if has_output else None
    output_spike_ax = axes[5] if has_output else None
    fig.suptitle(f"{mat} sid{sid} liquid processing", fontsize=13)
    trial_start_ms = _as_ms(start_time)
    v_threshold = None
    if groups:
        v_threshold = _group_scalar_value(groups[0], "v_thr", None)

    raw_lines = []
    for row in range(min(max_raw_traces, raw_input.shape[0])):
        (line,) = raw_ax.plot([], [], linewidth=1.0, label=f"sensor {row}")
        raw_lines.append((line, row))
    raw_ax.set_ylabel("raw")
    raw_ax.grid(True, alpha=0.25)
    if raw_lines:
        raw_ax.legend(ncol=4, fontsize=8, loc="upper right")

    channels = sorted(input_filter_map)
    filter_labels = []
    for channel in channels:
        for filter_name in input_filter_map[channel]:
            filter_labels.append(f"ch{channel}:{filter_name}")
    filtered_lines = []
    for row in range(min(max_filter_traces, input_current.shape[0])):
        label = filter_labels[row] if row < len(filter_labels) else f"filter {row}"
        (line,) = filtered_ax.plot([], [], linewidth=0.9, label=label)
        filtered_lines.append((line, row))
    filtered_ax.set_ylabel("filtered")
    filtered_ax.grid(True, alpha=0.25)
    if filtered_lines:
        filtered_ax.legend(ncol=2, fontsize=7, loc="upper right")

    voltage_t = np.array([], dtype=float)
    voltage_values = np.empty((0, 0), dtype=float)
    voltage_labels: Sequence[int] = []
    voltage_source_rows: list[int] = []
    if voltage_monitors:
        voltage_record_labels = _monitor_record_labels(
            voltage_monitors[0],
            voltage_indices[0] if voltage_indices else None,
        )
        voltage_t, voltage_values = _monitor_window_arrays(
            voltage_monitors[0],
            start_index=(
                voltage_start_indices[0]
                if voltage_start_indices is not None and len(voltage_start_indices)
                else None
            ),
            start_time=start_time,
            duration_ms=duration_ms,
        )
        voltage_labels = voltage_record_labels
        voltage_source_rows = list(range(voltage_values.shape[0]))
    if select_voltage_by_spike_count and groups and spike_monitors and voltage_values.size:
        spike_start = (
            spike_start_indices[0]
            if spike_start_indices is not None and len(spike_start_indices)
            else None
        )
        layer_t_for_selection, layer_i_for_selection = _spike_window_arrays(
            spike_monitors[0],
            start_index=spike_start,
            start_time=start_time,
            duration_ms=duration_ms,
        )
        n_layer_neurons = int(len(groups[0]))
        if layer_i_for_selection.size and n_layer_neurons > 0:
            counts = np.bincount(
                np.asarray(layer_i_for_selection, dtype=int),
                minlength=n_layer_neurons,
            )
            row_by_neuron = {
                int(neuron_index): int(row)
                for row, neuron_index in enumerate(voltage_labels)
            }
            active_order = np.argsort(counts)[::-1]
            top_indices = [
                int(index)
                for index in active_order
                if int(index) in row_by_neuron and counts[int(index)] > 0
            ][:max_voltage_traces]
            if top_indices:
                top_rows = [row_by_neuron[int(index)] for index in top_indices]
                voltage_values = voltage_values[top_rows, :]
                voltage_labels = top_indices
                voltage_source_rows = top_rows
    voltage_lines = []
    voltage_line_colors = []
    for row in range(min(max_voltage_traces, voltage_values.shape[0])):
        label_index = voltage_labels[row] if row < len(voltage_labels) else row
        (line,) = voltage_ax.plot([], [], linewidth=1.0, label=f"n{int(label_index)}")
        voltage_lines.append((line, row))
        voltage_line_colors.append(line.get_color())
    voltage_ax.set_ylabel("v")
    voltage_ax.grid(True, alpha=0.25)
    if voltage_lines:
        voltage_ax.legend(ncol=5, fontsize=8, loc="upper right")
        if v_threshold is not None:
            voltage_ax.axhline(
                v_threshold,
                color="black",
                linestyle="--",
                linewidth=0.9,
                alpha=0.55,
            )
    else:
        voltage_ax.text(
            0.5,
            0.5,
            "No voltage samples",
            transform=voltage_ax.transAxes,
            ha="center",
            va="center",
            color="0.35",
        )

    threshold_times_by_neuron = {}
    if v_threshold is not None and voltage_t.size:
        for row, neuron_index in enumerate(voltage_labels[:max_voltage_traces]):
            if row >= voltage_values.shape[0]:
                continue
            trace = np.asarray(voltage_values[row])
            over = trace >= float(v_threshold)
            if over.size:
                prev_over = np.concatenate(([False], over[:-1]))
                crossing_mask = over & ~prev_over
                threshold_times_by_neuron[int(neuron_index)] = voltage_t[crossing_mask]

    spike_t_parts = []
    spike_i_parts = []
    selected_spikes_by_neuron = {}
    monitor_spikes_by_neuron = {}
    offset = 0
    y_ticks = []
    y_labels = []
    for layer_index, (group, monitor) in enumerate(zip(groups, spike_monitors), start=1):
        start_index = (
            spike_start_indices[layer_index - 1]
            if spike_start_indices is not None and layer_index - 1 < len(spike_start_indices)
            else None
        )
        layer_t, layer_i = _spike_window_arrays(
            monitor,
            start_index=start_index,
            start_time=start_time,
            duration_ms=duration_ms,
        )
        if layer_t.size:
            spike_t_parts.append(layer_t)
            spike_i_parts.append(layer_i + offset)
            if layer_index == 1 and voltage_labels and voltage_line_colors:
                for color_index, neuron_index in enumerate(voltage_labels[:max_voltage_traces]):
                    selected_mask = layer_i == int(neuron_index)
                    monitor_times = layer_t[selected_mask]
                    threshold_times = np.asarray(
                        threshold_times_by_neuron.get(int(neuron_index), ()),
                        dtype=float,
                    )
                    monitor_spikes_by_neuron[int(neuron_index)] = monitor_times
                    display_times = monitor_times
                    selected_spikes_by_neuron[int(neuron_index)] = display_times
        y_ticks.append(offset + max(0, len(group) - 1) / 2.0)
        y_labels.append(f"L{layer_index}")
        offset += int(len(group))
    spike_t = np.concatenate(spike_t_parts) if spike_t_parts else np.array([], dtype=float)
    spike_i = np.concatenate(spike_i_parts) if spike_i_parts else np.array([], dtype=float)
    spike_scatter = spike_ax.scatter([], [], s=3, alpha=0.18, color="0.25", label="all")
    voltage_spike_scatters = []
    for color_index, neuron_index in enumerate(voltage_labels[:max_voltage_traces]):
        spike_times = np.asarray(selected_spikes_by_neuron.get(int(neuron_index), ()), dtype=float)
        if not spike_times.size or color_index >= len(voltage_line_colors):
            continue
        marker_y = (
            float(v_threshold)
            if v_threshold is not None
            else float(np.nanmax(voltage_values[color_index]))
        )
        scatter = voltage_ax.scatter(
            [],
            [],
            marker="|",
            s=40,
            color=voltage_line_colors[color_index],
            alpha=0.75,
        )
        voltage_spike_scatters.append((scatter, spike_times, marker_y))
    spike_ax.set_ylabel("spikes")
    spike_ax.set_xlabel("Time [ms]")
    spike_ax.set_yticks(y_ticks)
    spike_ax.set_yticklabels(y_labels)
    spike_ax.set_ylim(-0.5, max(0.5, offset - 0.5))
    spike_ax.grid(True, alpha=0.2)

    output_voltage_t = np.array([], dtype=float)
    output_voltage_values = np.empty((0, 0), dtype=float)
    output_voltage_labels: Sequence[int] = []
    output_voltage_lines = []
    output_spike_scatter = None
    output_spike_t = np.array([], dtype=float)
    output_spike_i = np.array([], dtype=float)
    if has_output and output_voltage_ax is not None and output_spike_ax is not None:
        output_voltage_t, output_voltage_values = _monitor_window_arrays(
            output_voltage_monitors[0],
            start_index=(
                output_voltage_start_indices[0]
                if output_voltage_start_indices is not None and len(output_voltage_start_indices)
                else None
            ),
            start_time=start_time,
            duration_ms=duration_ms,
        )
        output_voltage_labels = _monitor_record_labels(
            output_voltage_monitors[0],
            output_voltage_indices[0] if output_voltage_indices else None,
        )
        for row in range(min(max_voltage_traces, output_voltage_values.shape[0])):
            label_index = (
                output_voltage_labels[row]
                if row < len(output_voltage_labels)
                else row
            )
            (line,) = output_voltage_ax.plot([], [], linewidth=1.0, label=f"o{int(label_index)}")
            output_voltage_lines.append((line, row))
        output_voltage_ax.set_ylabel("out v")
        output_voltage_ax.grid(True, alpha=0.25)
        if output_voltage_lines:
            output_voltage_ax.legend(ncol=5, fontsize=8, loc="upper right")
            output_threshold = _group_scalar_value(output_groups[0], "v_thr", None)
            if output_threshold is not None:
                output_voltage_ax.axhline(
                    output_threshold,
                    color="black",
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.55,
                )
        else:
            output_voltage_ax.text(
                0.5,
                0.5,
                "No output voltage samples",
                transform=output_voltage_ax.transAxes,
                ha="center",
                va="center",
                color="0.35",
            )

        output_spike_t_parts = []
        output_spike_i_parts = []
        output_offset = 0
        output_y_ticks = []
        output_y_labels = []
        for layer_index, (group, monitor) in enumerate(
            zip(output_groups, output_spike_monitors),
            start=1,
        ):
            start_index = (
                output_spike_start_indices[layer_index - 1]
                if output_spike_start_indices is not None
                and layer_index - 1 < len(output_spike_start_indices)
                else None
            )
            layer_t, layer_i = _spike_window_arrays(
                monitor,
                start_index=start_index,
                start_time=start_time,
                duration_ms=duration_ms,
            )
            if layer_t.size:
                output_spike_t_parts.append(layer_t)
                output_spike_i_parts.append(layer_i + output_offset)
            output_y_ticks.append(output_offset + max(0, len(group) - 1) / 2.0)
            output_y_labels.append(f"O{layer_index}")
            output_offset += int(len(group))
        output_spike_t = (
            np.concatenate(output_spike_t_parts)
            if output_spike_t_parts
            else np.array([], dtype=float)
        )
        output_spike_i = (
            np.concatenate(output_spike_i_parts)
            if output_spike_i_parts
            else np.array([], dtype=float)
        )
        output_spike_scatter = output_spike_ax.scatter(
            [],
            [],
            s=6,
            alpha=0.35,
            color="0.25",
        )
        output_spike_ax.set_ylabel("out spikes")
        output_spike_ax.set_xlabel("Time [ms]")
        output_spike_ax.set_yticks(output_y_ticks)
        output_spike_ax.set_yticklabels(output_y_labels)
        output_spike_ax.set_ylim(-0.5, max(0.5, output_offset - 0.5))
        output_spike_ax.grid(True, alpha=0.2)

    diagnostic_rows = []
    for row, neuron_index in enumerate(voltage_labels[:max_voltage_traces]):
        if row >= voltage_values.shape[0]:
            continue
        neuron_index = int(neuron_index)
        trace = voltage_values[row]
        display_spike_times = np.asarray(selected_spikes_by_neuron.get(neuron_index, ()), dtype=float)
        monitor_spike_times = np.asarray(monitor_spikes_by_neuron.get(neuron_index, ()), dtype=float)
        crossing_times = np.asarray(threshold_times_by_neuron.get(neuron_index, ()), dtype=float)
        reset_drop_times = np.array([], dtype=float)
        if trace.size > 1 and voltage_t.size == trace.size:
            reset_drop_mask = np.diff(trace) < -5.0
            reset_drop_times = voltage_t[1:][reset_drop_mask]
        last_monitor_spike = float(np.max(monitor_spike_times)) if monitor_spike_times.size else None
        if last_monitor_spike is None:
            reset_after_last_monitor = reset_drop_times
        else:
            reset_after_last_monitor = reset_drop_times[
                reset_drop_times > last_monitor_spike + 1e-6
            ]
        diagnostic_rows.append(
            {
                "neuron_index": neuron_index,
                "trial_start_abs_ms": float(trial_start_ms) if trial_start_ms is not None else "",
                "voltage_row": int(voltage_source_rows[row]) if row < len(voltage_source_rows) else int(row),
                "spike_marker_source": "SpikeMonitor",
                "display_spike_count": int(display_spike_times.size),
                "last_display_spike_ms": (
                    float(np.max(display_spike_times)) if display_spike_times.size else ""
                ),
                "monitor_spike_count": int(monitor_spike_times.size),
                "last_monitor_spike_ms": (
                    float(np.max(monitor_spike_times)) if monitor_spike_times.size else ""
                ),
                "threshold_cross_sample_count": int(crossing_times.size),
                "last_threshold_cross_sample_ms": (
                    float(np.max(crossing_times)) if crossing_times.size else ""
                ),
                "voltage_reset_drop_count": int(reset_drop_times.size),
                "last_voltage_reset_drop_ms": (
                    float(np.max(reset_drop_times)) if reset_drop_times.size else ""
                ),
                "reset_drop_after_last_monitor_spike_count": int(
                    reset_after_last_monitor.size
                ),
                "last_reset_drop_after_last_monitor_spike_ms": (
                    float(np.max(reset_after_last_monitor))
                    if reset_after_last_monitor.size
                    else ""
                ),
                "last_voltage_sample_ms": float(np.max(voltage_t)) if voltage_t.size else "",
                "voltage_max": float(np.nanmax(trace)) if trace.size else "",
                "v_threshold": float(v_threshold) if v_threshold is not None else "",
            }
        )
    if diagnostic_rows:
        diagnostic_fp = out_fp.with_name(out_fp.stem + "_diagnostic.csv")
        with diagnostic_fp.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(diagnostic_rows[0]))
            writer.writeheader()
            writer.writerows(diagnostic_rows)

    for ax in axes:
        ax.set_xlim(0.0, duration_ms)
    progress_lines = [ax.axvline(0.0, color="#c53030", linewidth=1.1) for ax in axes]
    time_text = raw_ax.text(
        0.01,
        0.95,
        "",
        transform=raw_ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    def _set_ylim_from_data(axis, values: np.ndarray) -> None:
        if not values.size:
            return
        finite = values[np.isfinite(values)]
        if not finite.size:
            return
        y_min = float(np.min(finite))
        y_max = float(np.max(finite))
        margin = max(1e-9, 0.1 * (y_max - y_min + 1e-9))
        axis.set_ylim(y_min - margin, y_max + margin)

    _set_ylim_from_data(raw_ax, raw_input[:max_raw_traces])
    _set_ylim_from_data(filtered_ax, input_current[:max_filter_traces])
    _set_ylim_from_data(voltage_ax, voltage_values[:max_voltage_traces])
    if has_output and output_voltage_ax is not None:
        _set_ylim_from_data(
            output_voltage_ax,
            output_voltage_values[:max_voltage_traces],
        )
    fig.tight_layout()

    def update(current_time_ms: float):
        input_mask = t_input_ms <= current_time_ms
        for line, row in raw_lines:
            line.set_data(t_input_ms[input_mask], raw_input[row, input_mask])
        for line, row in filtered_lines:
            line.set_data(t_input_ms[input_mask], input_current[row, input_mask])

        voltage_mask = voltage_t <= current_time_ms
        for line, row in voltage_lines:
            line.set_data(voltage_t[voltage_mask], voltage_values[row, voltage_mask])
        for scatter, spike_times, marker_y in voltage_spike_scatters:
            visible_spikes = spike_times[spike_times <= current_time_ms]
            if visible_spikes.size:
                scatter.set_offsets(
                    np.column_stack(
                        (visible_spikes, np.full(visible_spikes.shape, marker_y))
                    )
                )
            else:
                scatter.set_offsets(np.empty((0, 2), dtype=float))

        spike_mask = spike_t <= current_time_ms
        if np.any(spike_mask):
            spike_scatter.set_offsets(np.column_stack((spike_t[spike_mask], spike_i[spike_mask])))
        else:
            spike_scatter.set_offsets(np.empty((0, 2), dtype=float))
        if has_output:
            output_voltage_mask = output_voltage_t <= current_time_ms
            for line, row in output_voltage_lines:
                line.set_data(
                    output_voltage_t[output_voltage_mask],
                    output_voltage_values[row, output_voltage_mask],
                )
            if output_spike_scatter is not None:
                output_spike_mask = output_spike_t <= current_time_ms
                if np.any(output_spike_mask):
                    output_spike_scatter.set_offsets(
                        np.column_stack(
                            (
                                output_spike_t[output_spike_mask],
                                output_spike_i[output_spike_mask],
                            )
                        )
                    )
                else:
                    output_spike_scatter.set_offsets(np.empty((0, 2), dtype=float))
        for progress_line in progress_lines:
            progress_line.set_xdata([current_time_ms, current_time_ms])
        time_text.set_text(f"{current_time_ms:.1f} ms")
        artists = [line for line, _ in raw_lines + filtered_lines + voltage_lines]
        artists.extend(line for line, _ in output_voltage_lines)
        artists.extend(
            [
            spike_scatter,
            *(scatter for scatter, _, _ in voltage_spike_scatters),
            *progress_lines,
            time_text,
            ]
        )
        if output_spike_scatter is not None:
            artists.append(output_spike_scatter)
        return artists

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frame_times,
        interval=1000.0 / max(float(fps), 1.0),
        blit=False,
    )
    suffix = out_fp.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=max(1, int(round(fps))))
    elif suffix == ".mp4":
        writer = animation.FFMpegWriter(
            fps=max(1, int(round(fps))),
            codec="libx264",
            bitrate=2000,
            extra_args=["-pix_fmt", "yuv420p"],
        )
    else:
        plt.close(fig)
        raise ValueError(f"Unsupported movie extension: {out_fp.suffix}. Use .gif or .mp4.")
    anim.save(out_fp, writer=writer, dpi=140)
    plt.close(fig)
    return out_fp


def save_side_by_side_gif(
    out_fp: Path,
    left_fp: Path,
    right_fp: Path,
    *,
    gap_px: int = 0,
    background: str = "white",
    max_width_px: int = 1600,
    max_height_px: int = 900,
    crop_padding_px: int = 6,
) -> Path:
    out_fp = Path(out_fp)
    left_fp = Path(left_fp)
    right_fp = Path(right_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    from PIL import Image, ImageSequence
    resample_lanczos = getattr(
        getattr(Image, "Resampling", Image),
        "LANCZOS",
        getattr(Image, "LANCZOS", 1),
    )
    palette_adaptive = getattr(
        getattr(Image, "Palette", Image),
        "ADAPTIVE",
        getattr(Image, "ADAPTIVE", 1),
    )

    def _load_frames(path: Path):
        with Image.open(path) as image:
            duration = int(image.info.get("duration", 100))
            frames = [frame.convert("RGBA") for frame in ImageSequence.Iterator(image)]
        return frames, duration

    left_frames, left_duration = _load_frames(left_fp)
    right_frames, right_duration = _load_frames(right_fp)
    if not left_frames or not right_frames:
        raise ValueError("Cannot combine empty GIFs.")

    def _content_bbox(frames: list[Image.Image]) -> tuple[int, int, int, int]:
        x0 = y0 = 10**9
        x1 = y1 = -1
        for frame in frames:
            rgb = np.asarray(frame.convert("RGB"), dtype=np.uint8)
            mask = np.any(rgb < 248, axis=2)
            if not np.any(mask):
                continue
            ys, xs = np.where(mask)
            x0 = min(x0, int(xs.min()))
            y0 = min(y0, int(ys.min()))
            x1 = max(x1, int(xs.max()) + 1)
            y1 = max(y1, int(ys.max()) + 1)
        if x1 < x0 or y1 < y0:
            first = frames[0]
            return (0, 0, first.width, first.height)
        padding = max(0, int(crop_padding_px))
        first = frames[0]
        return (
            max(0, x0 - padding),
            max(0, y0 - padding),
            min(first.width, x1 + padding),
            min(first.height, y1 + padding),
        )

    def _crop_frames(frames: list[Image.Image]) -> list[Image.Image]:
        bbox = _content_bbox(frames)
        return [frame.crop(bbox) for frame in frames]

    left_frames = _crop_frames(left_frames)
    right_frames = _crop_frames(right_frames)

    left_aspect = max(frame.width / max(frame.height, 1) for frame in left_frames)
    right_aspect = max(frame.width / max(frame.height, 1) for frame in right_frames)
    height_by_width = int(
        (max_width_px - gap_px) / max(left_aspect + right_aspect, 1e-9)
    )
    base_height = max(
        1,
        min(
            max_height_px,
            height_by_width,
            max(frame.height for frame in left_frames + right_frames),
        ),
    )

    def _resize_to_height(frame: Image.Image) -> Image.Image:
        if frame.height == base_height:
            return frame
        width = max(1, int(round(frame.width * base_height / frame.height)))
        return frame.resize((width, base_height), resample_lanczos)

    n_frames = max(len(left_frames), len(right_frames))
    combined_frames = []
    for frame_index in range(n_frames):
        left = _resize_to_height(left_frames[min(frame_index, len(left_frames) - 1)])
        right = _resize_to_height(right_frames[min(frame_index, len(right_frames) - 1)])
        canvas = Image.new(
            "RGBA",
            (left.width + gap_px + right.width, base_height),
            background,
        )
        canvas.alpha_composite(left, (0, 0))
        canvas.alpha_composite(right, (left.width + gap_px, 0))
        combined_frames.append(canvas.convert("P", palette=palette_adaptive))

    duration = min(left_duration, right_duration)
    combined_frames[0].save(
        out_fp,
        save_all=True,
        append_images=combined_frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return out_fp


def save_gif_sequence(
    out_fp: Path,
    gif_paths: Sequence[Path],
    *,
    background: str = "white",
    max_width_px: int = 1600,
    max_height_px: int = 900,
) -> Path:
    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    gif_paths = [Path(path) for path in gif_paths if Path(path).exists()]
    if not gif_paths:
        raise ValueError("No GIFs to concatenate.")

    from PIL import Image, ImageSequence

    resample_lanczos = getattr(
        getattr(Image, "Resampling", Image),
        "LANCZOS",
        getattr(Image, "LANCZOS", 1),
    )
    palette_adaptive = getattr(
        getattr(Image, "Palette", Image),
        "ADAPTIVE",
        getattr(Image, "ADAPTIVE", 1),
    )

    frames: list[Image.Image] = []
    durations: list[int] = []
    for path in gif_paths:
        with Image.open(path) as image:
            for frame in ImageSequence.Iterator(image):
                frames.append(frame.convert("RGBA"))
                durations.append(int(frame.info.get("duration", image.info.get("duration", 100))))

    if not frames:
        raise ValueError("No GIF frames to concatenate.")

    scale = min(
        1.0,
        max_width_px / max(frame.width for frame in frames),
        max_height_px / max(frame.height for frame in frames),
    )
    target_width = max(1, int(round(max(frame.width for frame in frames) * scale)))
    target_height = max(1, int(round(max(frame.height for frame in frames) * scale)))

    normalized_frames = []
    for frame in frames:
        frame_scale = min(
            target_width / max(frame.width, 1),
            target_height / max(frame.height, 1),
        )
        width = max(1, int(round(frame.width * frame_scale)))
        height = max(1, int(round(frame.height * frame_scale)))
        resized = frame.resize((width, height), resample_lanczos)
        canvas = Image.new("RGBA", (target_width, target_height), background)
        canvas.alpha_composite(
            resized,
            ((target_width - width) // 2, (target_height - height) // 2),
        )
        normalized_frames.append(canvas.convert("P", palette=palette_adaptive))

    normalized_frames[0].save(
        out_fp,
        save_all=True,
        append_images=normalized_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return out_fp


def save_liquid_network_activity_movie(
    out_fp: Path,
    *,
    group,
    synapses: Sequence,
    spike_monitor,
    output_group=None,
    output_synapses: Sequence | None = None,
    output_spike_monitor=None,
    title: str,
    duration_ms: float,
    spike_start_index: int | None = None,
    output_spike_start_index: int | None = None,
    start_time=None,
    fps: float = 12.0,
    seconds: float = 8.0,
    max_edges: int = 1500,
    spike_window_ms: float = 5.0,
    seed: int = 0,
) -> Path:
    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(group.x, dtype=float)
    y = np.asarray(group.y, dtype=float)
    typ = np.asarray(group.typ, dtype=int)
    n_neurons = int(len(group))
    positions = np.column_stack((x, y))
    has_output = output_group is not None and output_spike_monitor is not None
    n_output = int(len(output_group)) if has_output else 0
    if has_output:
        output_cols = max(1, int(np.ceil(n_output / 20.0)))
        output_rows = int(np.ceil(n_output / output_cols))
        output_y_grid = np.linspace(
            float(np.max(y)),
            float(np.min(y)),
            max(output_rows, 1),
        )
        output_x_grid = float(np.max(x)) + 0.20 + 0.055 * np.arange(output_cols)
        output_x = np.array(
            [output_x_grid[index // output_rows] for index in range(n_output)],
            dtype=float,
        )
        output_y = np.array(
            [output_y_grid[index % output_rows] for index in range(n_output)],
            dtype=float,
        )
        output_positions = np.column_stack((output_x, output_y))
    else:
        output_positions = np.empty((0, 2), dtype=float)

    edge_pre_parts = []
    edge_post_parts = []
    for syn in synapses:
        try:
            pre = np.asarray(syn.i[:], dtype=int)
            post = np.asarray(syn.j[:], dtype=int)
        except Exception:
            continue
        valid = (
            (pre >= 0)
            & (pre < n_neurons)
            & (post >= 0)
            & (post < n_neurons)
            & (pre != post)
        )
        if np.any(valid):
            edge_pre_parts.append(pre[valid])
            edge_post_parts.append(post[valid])

    if edge_pre_parts:
        edge_pre = np.concatenate(edge_pre_parts)
        edge_post = np.concatenate(edge_post_parts)
        if edge_pre.size > max_edges:
            rng = np.random.default_rng(seed)
            keep = rng.choice(edge_pre.size, size=max_edges, replace=False)
            edge_pre = edge_pre[keep]
            edge_post = edge_post[keep]
        edge_segments = np.stack((positions[edge_pre], positions[edge_post]), axis=1)
        edge_colors = np.where(typ[edge_pre] == 1, "#5b8cc0", "#c96f53")
    else:
        edge_segments = np.empty((0, 2, 2), dtype=float)
        edge_colors = np.array([], dtype=str)

    output_edge_segments = np.empty((0, 2, 2), dtype=float)
    if has_output and output_synapses:
        out_pre_parts = []
        out_post_parts = []
        for syn in output_synapses:
            try:
                pre = np.asarray(syn.i[:], dtype=int)
                post = np.asarray(syn.j[:], dtype=int)
            except Exception:
                continue
            valid = (
                (pre >= 0)
                & (pre < n_neurons)
                & (post >= 0)
                & (post < n_output)
            )
            if np.any(valid):
                out_pre_parts.append(pre[valid])
                out_post_parts.append(post[valid])
        if out_pre_parts:
            out_pre = np.concatenate(out_pre_parts)
            out_post = np.concatenate(out_post_parts)
            output_edge_limit = max(1, int(max_edges // 2))
            if out_pre.size > output_edge_limit:
                rng = np.random.default_rng(seed + 1)
                keep = rng.choice(out_pre.size, size=output_edge_limit, replace=False)
                out_pre = out_pre[keep]
                out_post = out_post[keep]
            output_edge_segments = np.stack(
                (positions[out_pre], output_positions[out_post]),
                axis=1,
            )

    spike_t, spike_i = _spike_window_arrays(
        spike_monitor,
        start_index=spike_start_index,
        start_time=start_time,
        duration_ms=duration_ms,
    )
    if has_output:
        output_spike_t, output_spike_i = _spike_window_arrays(
            output_spike_monitor,
            start_index=output_spike_start_index,
            start_time=start_time,
            duration_ms=duration_ms,
        )
    else:
        output_spike_t = np.array([], dtype=float)
        output_spike_i = np.array([], dtype=float)

    n_frames = max(2, int(round(max(float(fps), 1.0) * max(float(seconds), 0.1))))
    frame_times = np.linspace(0.0, float(duration_ms), n_frames)

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    x_right = (
        float(np.max(output_positions[:, 0])) + 0.05
        if has_output and output_positions.size
        else float(np.max(x)) + 0.03
    )
    ax.set_xlim(float(np.min(x)) - 0.03, x_right)
    ax.set_ylim(float(np.min(y)) - 0.03, float(np.max(y)) + 0.03)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.15)

    if edge_segments.size:
        edges = LineCollection(
            edge_segments,
            colors=edge_colors,
            linewidths=0.25,
            alpha=0.12,
            zorder=1,
        )
        ax.add_collection(edges)
    if output_edge_segments.size:
        output_edges = LineCollection(
            output_edge_segments,
            colors="red",
            linewidths=0.35,
            alpha=0.2,
            zorder=1,
        )
        ax.add_collection(output_edges)

    exc = typ == 1
    inh = typ == -1
    if np.any(exc):
        ax.scatter(x[exc], y[exc], s=14, c="#8fb9df", alpha=0.7, edgecolors="none", label="E", zorder=2)
    if np.any(inh):
        ax.scatter(x[inh], y[inh], s=18, c="#e5a27f", alpha=0.8, edgecolors="none", label="I", zorder=2)
    if has_output and output_positions.size:
        ax.scatter(
            output_positions[:, 0],
            output_positions[:, 1],
            s=28,
            c="#6bc7ee",
            alpha=0.9,
            edgecolors="none",
            label="Output",
            zorder=3,
        )
        ax.text(
            float(np.mean(output_positions[:, 0])),
            float(np.max(y)) + 0.02,
            "Output",
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.25",
        )
    active_scatter = ax.scatter([], [], s=48, c="#d62728", alpha=0.95, edgecolors="white", linewidths=0.4, zorder=4)
    active_output_scatter = ax.scatter([], [], s=70, c="#d62728", alpha=0.95, edgecolors="white", linewidths=0.5, zorder=5)
    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.legend(loc="upper right", fontsize=8)

    window = max(float(spike_window_ms), 0.0)

    def update(current_time_ms: float):
        if spike_t.size:
            active_mask = (
                (spike_t > current_time_ms - window)
                & (spike_t <= current_time_ms)
            )
            active_ids = np.unique(spike_i[active_mask].astype(int))
            active_ids = active_ids[(active_ids >= 0) & (active_ids < n_neurons)]
        else:
            active_ids = np.array([], dtype=int)
        if active_ids.size:
            active_scatter.set_offsets(positions[active_ids])
        else:
            active_scatter.set_offsets(np.empty((0, 2), dtype=float))
        if output_spike_t.size:
            output_active_mask = (
                (output_spike_t > current_time_ms - window)
                & (output_spike_t <= current_time_ms)
            )
            output_active_ids = np.unique(output_spike_i[output_active_mask].astype(int))
            output_active_ids = output_active_ids[
                (output_active_ids >= 0) & (output_active_ids < n_output)
            ]
        else:
            output_active_ids = np.array([], dtype=int)
        if output_active_ids.size:
            active_output_scatter.set_offsets(output_positions[output_active_ids])
        else:
            active_output_scatter.set_offsets(np.empty((0, 2), dtype=float))
        time_text.set_text(
            f"{current_time_ms:.1f} ms | liquid active={active_ids.size} | output active={output_active_ids.size}"
        )
        return [active_scatter, active_output_scatter, time_text]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frame_times,
        interval=1000.0 / max(float(fps), 1.0),
        blit=False,
    )
    suffix = out_fp.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=max(1, int(round(fps))))
    elif suffix == ".mp4":
        writer = animation.FFMpegWriter(
            fps=max(1, int(round(fps))),
            codec="libx264",
            bitrate=2000,
            extra_args=["-pix_fmt", "yuv420p"],
        )
    else:
        plt.close(fig)
        raise ValueError(f"Unsupported movie extension: {out_fp.suffix}. Use .gif or .mp4.")
    anim.save(out_fp, writer=writer, dpi=140)
    plt.close(fig)
    return out_fp


def save_lsm_schematic_activity_movie(
    out_fp: Path,
    *,
    mat: str,
    sid: int,
    input_current: np.ndarray,
    t_array: np.ndarray,
    liquid_group,
    output_group,
    liquid_spike_monitor,
    output_spike_monitor,
    input_to_liquid_synapses: Sequence,
    liquid_to_output_synapses: Sequence,
    duration_ms: float,
    liquid_spike_start_index: int | None = None,
    output_spike_start_index: int | None = None,
    start_time=None,
    fps: float = 12.0,
    seconds: float = 8.0,
    max_input_nodes: int = 2,
    max_liquid_nodes: int | None = None,
    max_output_nodes: int | None = None,
    max_edges: int = 18,
    spike_window_ms: float = 8.0,
    seed: int = 0,
) -> Path:
    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    input_current = np.asarray(input_current, dtype=float)
    t_input_ms = np.asarray(t_array, dtype=float) * 1000.0
    n_input_total = int(input_current.shape[0]) if input_current.ndim == 2 else 0
    n_liquid = int(len(liquid_group))
    n_output = int(len(output_group))
    rng = np.random.default_rng(seed)

    n_input = min(max_input_nodes, n_input_total)
    input_ids = np.arange(n_input, dtype=int)
    n_liquid_show = n_liquid if max_liquid_nodes is None or max_liquid_nodes <= 0 else min(max_liquid_nodes, n_liquid)
    n_output_show = n_output if max_output_nodes is None or max_output_nodes <= 0 else min(max_output_nodes, n_output)
    liquid_ids = (
        np.arange(n_liquid, dtype=int)
        if n_liquid_show >= n_liquid
        else np.sort(rng.choice(n_liquid, size=n_liquid_show, replace=False))
    )
    output_ids = (
        np.arange(n_output, dtype=int)
        if n_output_show >= n_output
        else np.sort(rng.choice(n_output, size=n_output_show, replace=False))
    )
    liquid_set = set(int(i) for i in liquid_ids)
    output_set = set(int(i) for i in output_ids)

    liquid_t, liquid_i = _spike_window_arrays(
        liquid_spike_monitor,
        start_index=liquid_spike_start_index,
        start_time=start_time,
        duration_ms=duration_ms,
    )
    output_t, output_i = _spike_window_arrays(
        output_spike_monitor,
        start_index=output_spike_start_index,
        start_time=start_time,
        duration_ms=duration_ms,
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"{mat} sid{sid} LSM schematic activity", color="black", fontsize=14, pad=12)

    input_y = np.linspace(0.66, 0.34, max(n_input, 2))[:n_input] if n_input else np.array([])
    input_pos = {int(idx): np.array([0.20, float(input_y[row])]) for row, idx in enumerate(input_ids)}

    center = np.array([0.56, 0.52])
    radius = 0.22
    if liquid_ids.size:
        raw_x = np.asarray(liquid_group.x, dtype=float)[liquid_ids]
        raw_y = np.asarray(liquid_group.y, dtype=float)[liquid_ids]
        x_span = max(float(np.max(raw_x) - np.min(raw_x)), 1e-9)
        y_span = max(float(np.max(raw_y) - np.min(raw_y)), 1e-9)
        scaled = np.column_stack(
            (
                (raw_x - float(np.min(raw_x))) / x_span - 0.5,
                (raw_y - float(np.min(raw_y))) / y_span - 0.5,
            )
        )
        liquid_xy_all = center + scaled * (radius * 1.35)
    else:
        liquid_xy_all = np.empty((0, 2), dtype=float)
    liquid_pos = {
        int(idx): liquid_xy_all[row]
        for row, idx in enumerate(liquid_ids)
    }

    output_columns = max(1, int(np.ceil(n_output_show / 12))) if n_output_show else 1
    output_rows = max(1, int(np.ceil(n_output_show / output_columns))) if n_output_show else 1
    output_pos = {}
    for row, idx in enumerate(output_ids):
        col = row // output_rows
        local_row = row % output_rows
        x_pos = 0.84 + 0.032 * col
        y_pos = 0.82 - (0.64 * local_row / max(output_rows - 1, 1))
        output_pos[int(idx)] = np.array([x_pos, y_pos])

    arrow = FancyArrowPatch(
        (0.04, 0.50),
        (0.13, 0.50),
        arrowstyle="simple",
        mutation_scale=72,
        facecolor="#00b85b",
        edgecolor="#00b85b",
        alpha=0.95,
        zorder=1,
    )
    ax.add_patch(arrow)
    ax.text(0.04, 0.58, "input", color="#087a38", fontsize=10, ha="left")

    reservoir = Circle(
        center,
        radius,
        edgecolor="#0b3d56",
        facecolor="none",
        linewidth=2.0,
        alpha=0.95,
        zorder=1,
    )
    ax.add_patch(reservoir)
    ax.text(center[0], center[1] + radius + 0.035, "liquid", color="#005f86", fontsize=10, ha="center")
    ax.text(0.89, 0.88, "output", color="#005f86", fontsize=10, ha="center")

    def _sample_edges(pairs: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
        if len(pairs) <= limit:
            return pairs
        keep = rng.choice(len(pairs), size=limit, replace=False)
        return [pairs[int(i)] for i in keep]

    in_liq_pairs: list[tuple[int, int]] = []
    for syn in input_to_liquid_synapses:
        try:
            pre = np.asarray(syn.i[:], dtype=int)
            post = np.asarray(syn.j[:], dtype=int)
        except Exception:
            continue
        for src, dst in zip(pre, post):
            if int(src) in input_pos and int(dst) in liquid_set:
                in_liq_pairs.append((int(src), int(dst)))
    in_liq_pairs = _sample_edges(in_liq_pairs, max_edges)

    liq_out_pairs: list[tuple[int, int]] = []
    for syn in liquid_to_output_synapses:
        try:
            pre = np.asarray(syn.i[:], dtype=int)
            post = np.asarray(syn.j[:], dtype=int)
        except Exception:
            continue
        for src, dst in zip(pre, post):
            if int(src) in liquid_set and int(dst) in output_set:
                liq_out_pairs.append((int(src), int(dst)))
    liq_out_pairs = _sample_edges(liq_out_pairs, max_edges)

    def _segments_from_pairs(pairs, left_pos, right_pos):
        segments = []
        for src, dst in pairs:
            if src in left_pos and dst in right_pos:
                segments.append([left_pos[src], right_pos[dst]])
        return np.asarray(segments, dtype=float) if segments else np.empty((0, 2, 2), dtype=float)

    in_liq_segments = _segments_from_pairs(in_liq_pairs, input_pos, liquid_pos)
    liq_out_segments = _segments_from_pairs(liq_out_pairs, liquid_pos, output_pos)
    if in_liq_segments.size:
        ax.add_collection(LineCollection(in_liq_segments, colors="#cf3eec", linewidths=1.0, linestyles="dotted", alpha=0.55, zorder=2))
    if liq_out_segments.size:
        ax.add_collection(LineCollection(liq_out_segments, colors="red", linewidths=1.4, linestyles="dotted", alpha=0.85, zorder=2))

    input_scatter = ax.scatter([], [], s=1050, c="#dc8cdc", edgecolors="none", zorder=3)
    liquid_size = 18 if n_liquid_show > 300 else 80 if n_liquid_show > 80 else 420
    output_size = 115 if n_output_show > 20 else 420
    active_liquid_size = max(liquid_size * 2.4, 42)
    active_output_size = max(output_size * 1.6, 120)
    liquid_scatter = ax.scatter([], [], s=liquid_size, c="#5ec6ee", edgecolors="none", alpha=0.72, zorder=3)
    output_scatter = ax.scatter([], [], s=output_size, c="#5ec6ee", edgecolors="none", alpha=0.9, zorder=3)
    active_liquid_scatter = ax.scatter([], [], s=active_liquid_size, c="#ff3838", edgecolors="white", linewidths=0.45, zorder=4)
    active_output_scatter = ax.scatter([], [], s=active_output_size, c="#ff3838", edgecolors="white", linewidths=0.65, zorder=4)

    if input_pos:
        input_xy = np.asarray([input_pos[int(i)] for i in input_ids])
        input_scatter.set_offsets(input_xy)
    liquid_xy = np.asarray([liquid_pos[int(i)] for i in liquid_ids]) if liquid_ids.size else np.empty((0, 2))
    output_xy = np.asarray([output_pos[int(i)] for i in output_ids]) if output_ids.size else np.empty((0, 2))
    liquid_scatter.set_offsets(liquid_xy)
    output_scatter.set_offsets(output_xy)

    for row, idx in enumerate(input_ids):
        ax.text(input_pos[int(idx)][0], input_pos[int(idx)][1], f"I{row+1}", color="black", fontsize=9, ha="center", va="center", zorder=5)
    if liquid_ids.size <= 40:
        for idx in liquid_ids:
            ax.text(liquid_pos[int(idx)][0], liquid_pos[int(idx)][1], str(int(idx)), color="black", fontsize=7, ha="center", va="center", zorder=5)
    if output_ids.size <= 60:
        for idx in output_ids:
            ax.text(output_pos[int(idx)][0], output_pos[int(idx)][1], str(int(idx)), color="black", fontsize=6, ha="center", va="center", zorder=5)

    time_text = ax.text(0.04, 0.08, "", color="black", fontsize=11, ha="left")
    n_frames = max(2, int(round(max(float(fps), 1.0) * max(float(seconds), 0.1))))
    frame_times = np.linspace(0.0, float(duration_ms), n_frames)
    window = max(float(spike_window_ms), 0.0)

    def update(current_time_ms: float):
        if liquid_t.size:
            active_liq = np.unique(
                liquid_i[
                    (liquid_t > current_time_ms - window)
                    & (liquid_t <= current_time_ms)
                ].astype(int)
            )
            active_liq = [idx for idx in active_liq if idx in liquid_pos]
        else:
            active_liq = []
        if output_t.size:
            active_out = np.unique(
                output_i[
                    (output_t > current_time_ms - window)
                    & (output_t <= current_time_ms)
                ].astype(int)
            )
            active_out = [idx for idx in active_out if idx in output_pos]
        else:
            active_out = []
        active_liquid_scatter.set_offsets(
            np.asarray([liquid_pos[idx] for idx in active_liq], dtype=float)
            if active_liq
            else np.empty((0, 2), dtype=float)
        )
        active_output_scatter.set_offsets(
            np.asarray([output_pos[idx] for idx in active_out], dtype=float)
            if active_out
            else np.empty((0, 2), dtype=float)
        )
        time_text.set_text(
            f"{current_time_ms:.1f} ms   liquid spikes={len(active_liq)}   output spikes={len(active_out)}"
        )
        return [active_liquid_scatter, active_output_scatter, time_text]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frame_times,
        interval=1000.0 / max(float(fps), 1.0),
        blit=False,
    )
    suffix = out_fp.suffix.lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=max(1, int(round(fps))))
    elif suffix == ".mp4":
        writer = animation.FFMpegWriter(
            fps=max(1, int(round(fps))),
            codec="libx264",
            bitrate=2000,
            extra_args=["-pix_fmt", "yuv420p"],
        )
    else:
        plt.close(fig)
        raise ValueError(f"Unsupported movie extension: {out_fp.suffix}. Use .gif or .mp4.")
    anim.save(out_fp, writer=writer, dpi=140)
    plt.close(fig)
    return out_fp


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
    # Synapses 縺九ｉ驥阪∩繧貞叙繧雁・縺励∝ｱ､縺斐→縺ｮ繝偵せ繝医げ繝ｩ繝縺ｨ summary 繧剃ｿ晏ｭ倥☆繧九・    weight_dir = Path(out_dir) / "weights"
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

