"""学習中・実行中に膜電位、ラスタ、平均重み変動をリアルタイム表示する処理。"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from brian2 import ms


def live_plot_enabled(run_cfg: dict) -> bool:
    # 設定値と matplotlib backend を見て、リアルタイム表示できるか判定する。
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
    # run_liquid 用。リキッド層のラスタと膜電位を試行中に更新表示する。
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
            if np.any(mask):
                offsets = np.column_stack((t_ms[mask], neuron_idx[mask]))
            else:
                offsets = np.empty((0, 2), dtype=np.float32)
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


class SpikeRasterLiveViewer:
    # run_training 用。リキッド層と出力層のラスタを試行をまたいでつなげて表示する。
    def __init__(
        self,
        *,
        liquid_groups: Sequence,
        liquid_monitors: Sequence,
        output_groups: Sequence,
        output_monitors: Sequence,
        run_cfg: dict,
        liquid_start_indices: Sequence[int] | None = None,
        output_start_indices: Sequence[int] | None = None,
    ) -> None:
        self.window_ms = float(run_cfg.get("LIVE_PLOT_WINDOW_MS", 500.0))
        self.sections = []
        for layer_index, (group, monitor) in enumerate(zip(liquid_groups, liquid_monitors), start=1):
            start_index = (
                liquid_start_indices[layer_index - 1]
                if liquid_start_indices is not None and layer_index - 1 < len(liquid_start_indices)
                else 0
            )
            self.sections.append(
                {
                    "title": f"Liquid L{layer_index} spikes",
                    "group": group,
                    "monitor": monitor,
                    "start_index": int(start_index),
                }
            )
        for layer_index, (group, monitor) in enumerate(zip(output_groups, output_monitors), start=1):
            start_index = (
                output_start_indices[layer_index - 1]
                if output_start_indices is not None and layer_index - 1 < len(output_start_indices)
                else 0
            )
            self.sections.append(
                {
                    "title": f"Output O{layer_index} spikes",
                    "group": group,
                    "monitor": monitor,
                    "start_index": int(start_index),
                }
            )

        n_rows = max(1, len(self.sections))
        self.fig, axes = plt.subplots(
            n_rows,
            1,
            figsize=(10, 3.0 * n_rows),
            squeeze=False,
        )
        self.axes = axes[:, 0]
        self.scatters = []
        self.empty_texts = []

        if not self.sections:
            self.axes[0].set_title("Spike raster")
            self.axes[0].set_xlabel("Time [ms]")
            self.axes[0].set_ylabel("Neuron index")
            self.scatters.append(self.axes[0].scatter([], [], s=3, alpha=0.75))
            self.empty_texts.append(
                self.axes[0].text(
                    0.5,
                    0.5,
                    "No spike monitors",
                    transform=self.axes[0].transAxes,
                    ha="center",
                    va="center",
                    color="0.35",
                )
            )
        else:
            for axis, section in zip(self.axes, self.sections):
                axis.set_title(section["title"])
                axis.set_xlabel("Time [ms]")
                axis.set_ylabel("Neuron index")
                axis.set_ylim(-0.5, len(section["group"]) - 0.5)
                self.scatters.append(axis.scatter([], [], s=3, alpha=0.75))
                self.empty_texts.append(
                    axis.text(
                        0.5,
                        0.5,
                        "No spikes in window",
                        transform=axis.transAxes,
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

        for section_index, (axis, scatter, section) in enumerate(
            zip(self.axes, self.scatters, self.sections)
        ):
            monitor = section["monitor"]
            start = int(section["start_index"])
            t_ms = np.asarray(monitor.t[start:] / ms, dtype=np.float32)
            neuron_idx = np.asarray(monitor.i[start:], dtype=np.int32)
            mask = (t_ms >= x_min) & (t_ms <= x_max)
            if np.any(mask):
                offsets = np.column_stack((t_ms[mask], neuron_idx[mask]))
            else:
                offsets = np.empty((0, 2), dtype=np.float32)
            scatter.set_offsets(offsets)
            self.empty_texts[section_index].set_visible(offsets.shape[0] == 0)
            axis.set_xlim(x_min, x_max)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


class WeightChangeLiveViewer:
    # run_training 用。リキッド層・出力層の平均重み変動をリアルタイムに表示する。
    def __init__(
        self,
        *,
        liquid_layers: Sequence[int],
        output_layers: Sequence[int],
        run_cfg: dict,
    ) -> None:
        self.window_ms = float(run_cfg.get("LIVE_PLOT_WINDOW_MS", 500.0))
        self.liquid_layers = [int(layer) for layer in liquid_layers]
        self.output_layers = [int(layer) for layer in output_layers]
        self.history = {
            "liquid": {layer: {"t": [], "v": []} for layer in self.liquid_layers},
            "output": {layer: {"t": [], "v": []} for layer in self.output_layers},
        }

        self.fig, axes = plt.subplots(2, 1, figsize=(10, 7), squeeze=False)
        self.axes = axes[:, 0]
        self.lines = {"liquid": {}, "output": {}}
        self.empty_texts = {}

        liquid_ax = self.axes[0]
        liquid_ax.set_title("Mean weight delta by liquid layer")
        liquid_ax.set_xlabel("Time [ms]")
        liquid_ax.set_ylabel("Mean delta w")
        liquid_ax.grid(True, alpha=0.3)
        self.empty_texts["liquid"] = liquid_ax.text(
            0.5,
            0.5,
            "No weight data yet",
            transform=liquid_ax.transAxes,
            ha="center",
            va="center",
            color="0.35",
        )
        for layer in self.liquid_layers:
            (line,) = liquid_ax.plot([], [], linewidth=1.2, label=f"L{layer}")
            self.lines["liquid"][layer] = line
        if self.liquid_layers:
            liquid_ax.legend(fontsize=8)

        output_ax = self.axes[1]
        output_ax.set_title("Mean weight delta by output layer")
        output_ax.set_xlabel("Time [ms]")
        output_ax.set_ylabel("Mean delta w")
        output_ax.grid(True, alpha=0.3)
        self.empty_texts["output"] = output_ax.text(
            0.5,
            0.5,
            "No weight data yet",
            transform=output_ax.transAxes,
            ha="center",
            va="center",
            color="0.35",
        )
        for layer in self.output_layers:
            (line,) = output_ax.plot([], [], linewidth=1.2, label=f"O{layer}")
            self.lines["output"][layer] = line
        if self.output_layers:
            output_ax.legend(fontsize=8)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        plt.show(block=False)

    def _update_axis(
        self,
        axis_key: str,
        *,
        current_time_ms: float,
        values: dict[int, float],
    ) -> None:
        ax = self.axes[0] if axis_key == "liquid" else self.axes[1]
        x_min = max(0.0, float(current_time_ms) - self.window_ms)
        x_max = max(self.window_ms, float(current_time_ms))
        y_min = None
        y_max = None

        for layer, line in self.lines[axis_key].items():
            history = self.history[axis_key][layer]
            if layer in values:
                history["t"].append(float(current_time_ms))
                history["v"].append(float(values[layer]))

            line_t = np.asarray(history["t"], dtype=np.float32)
            line_v = np.asarray(history["v"], dtype=np.float32)
            mask = line_t >= x_min
            if np.any(mask):
                line_t = line_t[mask]
                line_v = line_v[mask]
                history["t"] = line_t.tolist()
                history["v"] = line_v.tolist()
            else:
                history["t"] = []
                history["v"] = []
                line_t = np.array([], dtype=np.float32)
                line_v = np.array([], dtype=np.float32)

            line.set_data(line_t, line_v)
            if line_v.size:
                local_min = float(np.min(line_v))
                local_max = float(np.max(line_v))
                y_min = local_min if y_min is None else min(y_min, local_min)
                y_max = local_max if y_max is None else max(y_max, local_max)

        ax.set_xlim(x_min, x_max)
        if y_min is not None and y_max is not None:
            margin = max(1e-6, 0.1 * (y_max - y_min + 1e-9))
            ax.set_ylim(y_min - margin, y_max + margin)
        self.empty_texts[axis_key].set_visible(y_min is None)

    def update(
        self,
        *,
        current_time_ms: float,
        liquid_values: dict[int, float],
        output_values: dict[int, float],
    ) -> None:
        self._update_axis("liquid", current_time_ms=current_time_ms, values=liquid_values)
        self._update_axis("output", current_time_ms=current_time_ms, values=output_values)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
