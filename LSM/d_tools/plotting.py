"""Figure export separated from simulation, classification, and evaluation."""

from __future__ import annotations

from a_experiment.experiment_sets import RUNTIME_BACKEND_SETTINGS
from pathlib import Path
import numpy as np


def save_weight_convergence(presentations, squared_updates, path, *, title, epochs=None):
    """全実結合の重み変化二乗和と、各保存時点のエポック区切りを表示する。"""
    import matplotlib.pyplot as plt

    presentations = np.asarray(presentations, dtype=float)
    squared_updates = np.asarray(squared_updates, dtype=float)
    if presentations.ndim != 1 or squared_updates.shape != presentations.shape:
        raise ValueError("Expected matching 1D trial counts and squared updates")
    epoch_values = None if epochs is None else np.asarray(epochs)
    if epoch_values is not None:
        if (
            epoch_values.shape != presentations.shape
            or not len(epoch_values)
            or not np.isfinite(epoch_values).all()
            or np.any(epoch_values < 1)
            or not np.equal(epoch_values, np.floor(epoch_values)).all()
            or np.any(np.diff(epoch_values) < 0)
            or not np.isfinite(presentations).all()
            or presentations[0] <= 0
            or np.any(np.diff(presentations) <= 0)
        ):
            raise ValueError("Expected increasing trial counts and ordered positive epoch numbers")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    try:
        ax.plot(presentations, squared_updates, "o-", markersize=3, linewidth=1)
        ax.set(
            xlabel="Training trials completed", ylabel="Sum of squared weight changes", title=title
        )
        if epoch_values is not None:
            # エポック最後の保存時点を区切りにし、上側の軸には区間中央へ番号を置く。
            centers, labels = [], []
            left = 0.0
            for index, epoch in enumerate(np.unique(epoch_values)):
                right = float(presentations[epoch_values == epoch][-1])
                if index % 2 == 0:
                    ax.axvspan(left, right, color="gray", alpha=0.06, zorder=0)
                if index:
                    ax.axvline(left, color="gray", linestyle="--", linewidth=1, alpha=0.8)
                centers.append((left + right) / 2)
                labels.append(str(int(epoch)))
                left = right
            ax.set_xlim(0, presentations[-1])
            epoch_axis = ax.secondary_xaxis("top")
            epoch_axis.set_xticks(centers, labels=labels)
            epoch_axis.set_xlabel("Training epoch")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
        ax.grid(alpha=0.25)
        fig.savefig(path)
    finally:
        plt.close(fig)
    return path


def save_input_current(data, path, *, title, n_excitatory=80):
    """RI・SI等のフィルタ別に、各センサーのフィルタ後波形を重ねて表示する。"""
    import matplotlib.pyplot as plt

    times = np.asarray(data["input_times_ms"])
    filtered = np.asarray(data["filtered_input"])
    labels = [str(label) for label in data["input_labels"]]
    kinds = [
        str(value)
        for value in data.get(
            "input_filters", [label.rsplit(":", 1)[-1].strip() for label in labels]
        )
    ]
    sensors = [
        str(value)
        for value in data.get(
            "input_sensors", [label.split(":", 1)[0].replace("Sensor ", "") for label in labels]
        )
    ]
    if (
        filtered.ndim != 2
        or filtered.shape != (len(kinds), len(times))
        or len(sensors) != len(kinds)
    ):
        raise ValueError("Filtered input shape must match time, sensor and filter labels")
    filters = list(dict.fromkeys(kinds))
    sensor_order = list(dict.fromkeys(sensors))
    if not filters:
        raise ValueError("No input filters to plot")
    colors = {sensor: plt.get_cmap("tab10")(i % 10) for i, sensor in enumerate(sensor_order)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = float(data["bin_ms"]) * np.asarray(data["counts"]).shape[1]
    fig, axes = plt.subplots(
        len(filters),
        1,
        figsize=(12, 3.5 * len(filters) + 1),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    try:
        for ax, kind in zip(axes[:, 0], filters):
            for index, (sensor, filter_name) in enumerate(zip(sensors, kinds)):
                if filter_name == kind:
                    ax.plot(
                        times,
                        filtered[index],
                        label=f"Sensor {sensor}",
                        color=colors[sensor],
                        linewidth=1,
                    )
            ax.set(
                title=f"{kind} filtered inputs",
                ylabel="Filtered input (model units)",
                xlim=(0, duration),
            )
            ax.grid(alpha=0.2)
            ax.legend(ncol=3, fontsize=9, loc="upper right")
        axes[-1, 0].set_xlabel("Time within trial (ms)")
        fig.suptitle(f"{title}\nFiltered input waveforms by filter")
        fig.savefig(path)
    finally:
        plt.close(fig)
    return path


def save_weight_changes(times_ms, weights, connections, path, *, title):
    """代表学習試行の実測重みと、その試行開始時からの変化を表示する。"""
    import matplotlib.pyplot as plt

    times_ms = np.asarray(times_ms)
    weights = np.asarray(weights)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    try:
        for i, connection in enumerate(connections):
            label = f'L{connection.get("liquid_layer",1)}:{connection["pre_id"]} -> R{connection.get("readout_layer",1)}:{connection["post_id"]}'
            axes[0].plot(times_ms, weights[i], label=label, linewidth=1)
            axes[1].plot(times_ms, weights[i] - weights[i, 0], linewidth=1)
        if not len(connections):
            axes[0].text(
                0.5,
                0.5,
                "No Liquid-to-Readout connections",
                transform=axes[0].transAxes,
                ha="center",
            )
        else:
            axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        axes[0].set_ylabel("Synaptic weight")
        axes[1].set(ylabel="Change from trial start", xlabel="Time within training trial (ms)")
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
        fig.suptitle(title)
        fig.savefig(path)
    finally:
        plt.close(fig)
    return path


def save_weight_distribution(values, path, *, title):
    """実在する結合だけを数えた重み分布をPDFへ保存する。ゼロ重みも含む。"""
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=float).ravel()
    if not np.isfinite(values).all():
        raise ValueError("Synaptic weights must be finite")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    try:
        if values.size:
            ax.hist(values, bins=40, color="steelblue", edgecolor="white")
            ax.axvline(values.mean(), color="darkorange", linestyle="--", label="Mean")
            stats = (
                f"Connections: {values.size:,}\nMean: {values.mean():.6g}\n"
                f"SD: {values.std():.6g}\nMin: {values.min():.6g}\nMax: {values.max():.6g}"
            )
            ax.text(
                0.98,
                0.95,
                stats,
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgray"),
            )
            ax.legend(loc="upper left")
        else:
            ax.text(0.5, 0.5, "No connections", transform=ax.transAxes, ha="center")
        ax.set(xlabel="Stored synaptic weight", ylabel="Number of connections", title=title)
        fig.savefig(path)
    finally:
        plt.close(fig)
    return path


def save_confusion_matrix(counts, labels, path, *, title: str, dpi=150) -> Path:
    """Save counts and true-class percentages using a fixed 0–100% color bar."""
    import matplotlib

    matplotlib.use(RUNTIME_BACKEND_SETTINGS["MATPLOTLIB_BACKEND"])
    import matplotlib.pyplot as plt
    from d_tools.evaluation import confusion_percentages

    cm = np.asarray(counts)
    pct = confusion_percentages(cm)
    labels = list(labels)
    if len(labels) != len(cm):
        raise ValueError("One label is required for each confusion-matrix row")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")
    try:
        im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    f"{cm[i,j]}\n{pct[i,j]:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if pct[i, j] > 50 else "black",
                    fontsize=8,
                )
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set(xlabel="Predicted material", ylabel="True material", title=title)
        fig.colorbar(im, ax=ax, label="Percentage of true class (%)")
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)
    return path


def save_debug_raster(data, path, *, title, n_excitatory=80):
    """層全体の発火時刻を、独立したラスタープロットPDFとして保存する。"""
    import matplotlib

    matplotlib.use(RUNTIME_BACKEND_SETTINGS["MATPLOTLIB_BACKEND"])
    import matplotlib.pyplot as plt

    counts = np.asarray(data["counts_25ms"])
    times = np.asarray(data["spike_times_ms"])
    ids = np.asarray(data["spike_neuron_ids"])
    if counts.ndim != 2 or times.ndim != 1 or ids.shape != times.shape:
        raise ValueError("Invalid spike arrays for debug raster")
    duration = counts.shape[1] * float(data["bin_ms"])
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    try:
        for mask, color, label, size in [
            (ids < n_excitatory, "tab:blue", "Excitatory", min(n_excitatory, len(counts))),
            (ids >= n_excitatory, "tab:red", "Inhibitory", max(0, len(counts) - n_excitatory)),
        ]:
            if size:
                ax.scatter(
                    times[mask],
                    ids[mask],
                    marker="|",
                    s=13,
                    linewidths=0.6,
                    color=color,
                    label=label,
                )
        if 0 < n_excitatory < len(counts):
            ax.axhline(n_excitatory - 0.5, color="gray", linewidth=0.7, linestyle="--")
        ax.set(
            xlabel="Time within trial (ms)",
            ylabel="Neuron ID",
            xlim=(0, duration),
            ylim=(-0.5, len(counts) - 0.5),
            title=f"{title}\nSpike raster",
        )
        ax.grid(axis="x", alpha=0.2)
        ax.legend(loc="upper right")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, metadata={"Title": f"{title} - Spike raster"})
    finally:
        plt.close(fig)
    return path


def save_debug_voltage(data, path, *, title, n_excitatory=80, max_neurons=10):
    """選択した約10ニューロンの膜電位を、ニューロン別の小図でPDF保存する。"""
    import matplotlib

    matplotlib.use(RUNTIME_BACKEND_SETTINGS["MATPLOTLIB_BACKEND"])
    import matplotlib.pyplot as plt

    time = np.asarray(data["voltage_times_ms"])
    ids = np.asarray(data["voltage_neuron_ids"])
    traces = np.asarray(data["voltage_mV"])
    if traces.shape != (len(ids), len(time)):
        raise ValueError("Membrane potential rows must match the recorded neuron IDs and times")
    number = min(len(ids), max(0, int(max_neurons)))
    columns = 2 if number > 1 else 1
    rows = max(1, (number + columns - 1) // columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, max(3.0, rows * 1.9 + 1.0)),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    duration = np.asarray(data["counts_25ms"]).shape[1] * float(data["bin_ms"])
    try:
        for index, ax in enumerate(axes.flat):
            if index >= number:
                ax.set_visible(False)
                continue
            neuron = int(ids[index])
            label = "E" if neuron < n_excitatory else "I"
            ax.plot(
                time, traces[index], linewidth=0.65, color="tab:blue" if label == "E" else "tab:red"
            )
            ax.set_title(f"Neuron {neuron} ({label})", fontsize=10)
            ax.set_xlim(0, duration)
            ax.grid(alpha=0.2)
        if number == 0:
            axes[0, 0].set_visible(True)
            axes[0, 0].text(
                0.5,
                0.5,
                "No membrane potentials recorded",
                ha="center",
                va="center",
                transform=axes[0, 0].transAxes,
            )
        fig.supxlabel("Time within trial (ms)")
        fig.supylabel("Membrane potential (mV)")
        fig.suptitle(f"{title}\nMembrane potentials - {number} selected neurons", fontsize=12)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, metadata={"Title": f"{title} - Membrane potentials"})
    finally:
        plt.close(fig)
    return path


def save_liquid_debug(data, path, *, title, n_excitatory=80):
    """代表1試行の全ニューロン発火・集団発火率・選択ニューロン膜電位を保存する。"""
    import matplotlib

    matplotlib.use(RUNTIME_BACKEND_SETTINGS["MATPLOTLIB_BACKEND"])
    import matplotlib.pyplot as plt

    counts = np.asarray(data["counts_25ms"])
    bin_ms = float(data["bin_ms"])
    times = np.asarray(data["spike_times_ms"])
    ids = np.asarray(data["spike_neuron_ids"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout="constrained")
    try:
        for mask, color, label in [
            (ids < n_excitatory, "tab:blue", "E"),
            (ids >= n_excitatory, "tab:red", "I"),
        ]:
            axes[0].scatter(times[mask], ids[mask], s=3, color=color, label=label)
        axes[0].axhline(n_excitatory - 0.5, color="gray", linewidth=0.7)
        axes[0].set(ylabel="Neuron index", ylim=(-1, counts.shape[0]))
        axes[0].legend(loc="upper right")
        centers = (np.arange(counts.shape[1]) + 0.5) * bin_ms
        for block, color, label in [
            (counts[:n_excitatory], "tab:blue", "E"),
            (counts[n_excitatory:], "tab:red", "I"),
        ]:
            if len(block):
                axes[1].plot(
                    centers, block.mean(axis=0) / (bin_ms / 1000), color=color, label=label
                )
        axes[1].set(ylabel="Mean firing rate (Hz)")
        axes[1].legend(loc="upper right")
        for neuron, trace in zip(data["voltage_neuron_ids"], data["voltage_mV"]):
            axes[2].plot(data["voltage_times_ms"], trace, linewidth=0.7, label=f"N{neuron}")
        axes[2].set(
            xlabel="Time within trial (ms)",
            ylabel="Membrane potential (mV)",
            xlim=(0, counts.shape[1] * bin_ms),
        )
        axes[2].legend(ncol=5, fontsize=7, loc="upper right")
        fig.suptitle(title)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
    return path


def save_liquid_positions(excitatory, inhibitory, path, *, title):
    """Liquidの実座標を、興奮性は青・抑制性は赤の3次元散布図でPDFへ保存する。"""
    import matplotlib.pyplot as plt

    # 座標はネットワーク内の無次元値。物理的な距離への換算は行わない。
    groups = [
        ("Excitatory", excitatory, "tab:blue", "^"),
        ("Inhibitory", inhibitory, "tab:red", "o"),
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7), layout="constrained")
    try:
        ax = fig.add_subplot(111, projection="3d")
        for label, positions, color, marker in groups:
            positions = np.asarray(positions, dtype=float)
            if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
                raise ValueError("Neuron positions must be a finite N x 3 array")
            ax.scatter(
                *positions.T,
                color=color,
                marker=marker,
                s=40,
                alpha=0.85,
                label=f"{label} (n={len(positions)})",
            )
        ax.set(
            xlabel="X (dimensionless)",
            ylabel="Y (dimensionless)",
            zlabel="Z (dimensionless)",
            title=title,
            xlim=(0, 1),
            ylim=(0, 1),
            zlim=(0, 1),
        )
        ax.set_box_aspect((1, 1, 1))
        ax.legend(loc="upper left")
        fig.savefig(path)
    finally:
        plt.close(fig)
    return path


def save_weight_histogram_panels(groups, bin_edges, y_limits, path, *, title):
    """結合群別の重み分布を1枚にまとめ、指定された共通軸で保存する。"""
    import matplotlib.pyplot as plt

    columns = min(2, len(groups))
    rows = (len(groups) + columns - 1) // columns
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        rows, columns, figsize=(7 * columns, 3.3 * rows), squeeze=False, layout="constrained"
    )
    try:
        for ax, (name, values) in zip(axes.flat, groups.items()):
            ax.hist(values, bins=bin_edges[name], color="steelblue", edgecolor="white")
            ax.set(
                title=f"{name} | n={len(values):,}",
                xlabel="Synaptic weight",
                ylabel="Number of connections",
                xlim=(bin_edges[name][0], bin_edges[name][-1]),
                ylim=(0, y_limits[name]),
            )
            if len(values):
                ax.axvline(np.mean(values), color="darkorange", linestyle="--", label="Mean")
                ax.legend()
        for ax in list(axes.flat)[len(groups) :]:
            ax.set_visible(False)
        fig.suptitle(title)
        fig.savefig(path)
    finally:
        plt.close(fig)
    return path
