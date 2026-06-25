"""Live visualization helpers for liquid-only runs."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms


def live_plot_enabled(run_cfg: dict) -> bool:
    if not bool(run_cfg.get("LIVE_PLOT_ENABLE", False)):
        return False
    backend = str(plt.get_backend()).lower()
    if backend == "agg":
        target_backend = str(run_cfg.get("LIVE_PLOT_BACKEND", "TkAgg"))
        try:
            plt.switch_backend(target_backend)
            backend = str(plt.get_backend()).lower()
        except Exception as exc:
            print(
                f"[live-plot] failed to switch backend to {target_backend}: "
                f"{type(exc).__name__}: {exc}"
            )
            return False
    print(f"[live-plot] backend={plt.get_backend()}")
    if backend == "agg" or "backend_inline" in backend:
        print(f"[live-plot] non-GUI backend detected: {plt.get_backend()}")
        return False
    return True


def live_plot_chunk_steps(run_cfg: dict, dt_s: float) -> int:
    update_ms = float(run_cfg.get("LIVE_PLOT_UPDATE_MS", 20.0))
    dt_ms = float(dt_s) * 1000.0
    return max(1, int(round(update_ms / dt_ms)))


def _relative_time_ms(time_values, start_time_ms: float) -> np.ndarray:
    t_ms = np.asarray(time_values / ms, dtype=np.float32)
    if t_ms.size == 0 or start_time_ms == 0.0:
        return t_ms

    shifted = t_ms - np.float32(start_time_ms)
    if shifted.size and float(np.min(shifted)) < -1e-6:
        return t_ms
    return shifted


class LiquidLiveViewer:
    def __init__(
        self,
        *,
        groups: Sequence,
        spike_monitors: Sequence,
        voltage_monitors: Sequence,
        voltage_indices: Sequence[Sequence[int]],
        run_cfg: dict,
        spike_start_indices: Sequence[int] | None = None,
        voltage_start_indices: Sequence[int] | None = None,
        start_time=None,
    ) -> None:
        self.groups = groups
        self.spike_monitors = spike_monitors
        self.voltage_monitors = voltage_monitors
        self.voltage_indices = voltage_indices
        self.window_ms = float(run_cfg.get("LIVE_PLOT_WINDOW_MS", 500.0))
        self.n_layers = len(groups)
        self.spike_start_indices = list(spike_start_indices or [0] * len(spike_monitors))
        self.voltage_start_indices = list(voltage_start_indices or [0] * len(voltage_monitors))
        self.start_time_ms = float(start_time / ms) if start_time is not None else 0.0

        self.fig, axes = plt.subplots(
            self.n_layers,
            2,
            figsize=(13, 4 * max(self.n_layers, 1)),
            squeeze=False,
        )
        self.axes = axes
        self.raster_scatters = []
        self.voltage_lines = []
        self.raster_empty_texts = []
        self.voltage_empty_texts = []

        for layer_index, group in enumerate(groups):
            raster_ax = axes[layer_index, 0]
            voltage_ax = axes[layer_index, 1]

            raster_ax.set_title(f"Liquid L{layer_index + 1} spikes")
            raster_ax.set_xlabel("Time [ms]")
            raster_ax.set_ylabel("Neuron index")
            raster_ax.set_ylim(-0.5, len(group) - 0.5)

            scatter = raster_ax.scatter([], [], s=3, alpha=0.75)
            self.raster_scatters.append(scatter)
            self.raster_empty_texts.append(
                raster_ax.text(
                    0.5,
                    0.5,
                    "No spikes",
                    transform=raster_ax.transAxes,
                    ha="center",
                    va="center",
                    color="0.35",
                )
            )

            voltage_ax.set_title(f"Liquid L{layer_index + 1} membrane voltage")
            voltage_ax.set_xlabel("Time [ms]")
            voltage_ax.set_ylabel("v")
            layer_lines = []
            for neuron_index in voltage_indices[layer_index]:
                (line,) = voltage_ax.plot([], [], linewidth=1.0, label=f"n{int(neuron_index)}")
                layer_lines.append(line)
            if len(layer_lines) <= 12 and layer_lines:
                voltage_ax.legend(ncol=2, fontsize=8)
            self.voltage_lines.append(layer_lines)
            self.voltage_empty_texts.append(
                voltage_ax.text(
                    0.5,
                    0.5,
                    "No voltage samples",
                    transform=voltage_ax.transAxes,
                    ha="center",
                    va="center",
                    color="0.35",
                )
            )

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        plt.show(block=False)

    def update(self, *, current_time_ms: float) -> None:
        x_min = max(0.0, float(current_time_ms) - self.window_ms)
        x_max = max(self.window_ms, float(current_time_ms))

        for layer_index, (group, monitor) in enumerate(zip(self.groups, self.spike_monitors)):
            raster_ax = self.axes[layer_index, 0]
            start = self.spike_start_indices[layer_index]
            t_ms = _relative_time_ms(monitor.t[start:], self.start_time_ms)
            neuron_idx = np.asarray(monitor.i[start:], dtype=np.int32)
            mask = t_ms >= x_min
            offsets = (
                np.column_stack((t_ms[mask], neuron_idx[mask]))
                if np.any(mask)
                else np.empty((0, 2), dtype=np.float32)
            )
            self.raster_scatters[layer_index].set_offsets(offsets)
            self.raster_empty_texts[layer_index].set_visible(offsets.shape[0] == 0)
            raster_ax.set_xlim(x_min, x_max)

        for layer_index, monitor in enumerate(self.voltage_monitors):
            voltage_ax = self.axes[layer_index, 1]
            start = self.voltage_start_indices[layer_index]
            t_ms = _relative_time_ms(monitor.t[start:], self.start_time_ms)
            voltage = np.asarray(monitor.v[:, start:], dtype=np.float32)
            mask = t_ms >= x_min

            y_min = None
            y_max = None
            for row, line in enumerate(self.voltage_lines[layer_index]):
                if row >= voltage.shape[0]:
                    continue
                line_t = t_ms[mask] if np.any(mask) else t_ms
                line_v = voltage[row, mask] if np.any(mask) else voltage[row]
                line.set_data(line_t, line_v)
                if line_v.size:
                    local_min = float(np.min(line_v))
                    local_max = float(np.max(line_v))
                    y_min = local_min if y_min is None else min(y_min, local_min)
                    y_max = local_max if y_max is None else max(y_max, local_max)

            voltage_ax.set_xlim(x_min, x_max)
            if y_min is not None and y_max is not None:
                margin = max(1.0, 0.1 * (y_max - y_min + 1e-6))
                voltage_ax.set_ylim(y_min - margin, y_max + margin)
            self.voltage_empty_texts[layer_index].set_visible(y_min is None)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
