"""保存された内部状態をヒートマップや平均推移として可視化する処理。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .plotting import try_import_pyplot
from .run_paths import jsonable, safe_stem
from .separation_metrics import discover_internal_state_files


def _load_internal_state_npz(fp: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(fp) as data:
        x_state = np.asarray(data["x_state"], dtype=np.float64)
        t_ms = np.asarray(data["t_ms"], dtype=np.float64)
    if x_state.ndim != 2:
        raise ValueError(f"x_state must be 2D, got shape={x_state.shape} in {fp}")
    if t_ms.ndim != 1:
        raise ValueError(f"t_ms must be 1D, got shape={t_ms.shape} in {fp}")
    if x_state.shape[1] != t_ms.shape[0]:
        n_time = min(x_state.shape[1], t_ms.shape[0])
        x_state = x_state[:, :n_time]
        t_ms = t_ms[:n_time]
    return x_state, t_ms


def _select_neurons(
    x_state: np.ndarray,
    *,
    max_neurons: int,
    sort_by_activity: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    n_neurons = int(x_state.shape[0])
    n_select = min(max(1, int(max_neurons)), n_neurons)
    activity = np.sum(np.abs(x_state), axis=1)
    if sort_by_activity:
        order = np.argsort(activity)[::-1]
    else:
        order = np.arange(n_neurons)
    selected = order[:n_select]
    return x_state[selected, :], selected


def save_internal_state_heatmap(
    out_fp: Path,
    x_state: np.ndarray,
    t_ms: np.ndarray,
    *,
    title: str,
    max_neurons: int = 200,
    sort_by_activity: bool = True,
    cmap: str | None = None,
    save_selected_csv: bool = True,
) -> Path:
    plt = try_import_pyplot()
    if plt is None:
        raise RuntimeError("matplotlib is required to save internal-state plots.")

    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    view, selected = _select_neurons(
        x_state,
        max_neurons=max_neurons,
        sort_by_activity=sort_by_activity,
    )
    if t_ms.size > 1:
        duration = float(t_ms[-1] + np.median(np.diff(t_ms)))
    elif t_ms.size == 1:
        duration = float(t_ms[-1] + 1.0)
    else:
        duration = float(view.shape[1])

    fig, ax = plt.subplots(figsize=(11, 6))
    finite = view[np.isfinite(view)]
    if finite.size:
        vmin_data = float(np.nanmin(finite))
        vmax_data = float(np.nanmax(finite))
    else:
        vmin_data = 0.0
        vmax_data = 1.0
    if vmin_data < 0.0:
        vabs = float(np.nanpercentile(np.abs(finite), 99.5)) if finite.size else 1.0
        vabs = vabs if vabs > 0 else 1.0
        vmin = -vabs
        vmax = vabs
        color_map = cmap or "coolwarm"
    else:
        vmax = float(np.nanpercentile(view, 99.5)) if np.any(np.isfinite(view)) else 1.0
        vmax = vmax if vmax > 0 else 1.0
        vmin = 0.0
        color_map = cmap or "magma"
    image = ax.imshow(
        view,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[0.0, duration, 0, view.shape[0]],
        cmap=color_map,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Selected neuron rank")
    ax.set_title(
        f"{title}\n"
        f"showing {view.shape[0]}/{x_state.shape[0]} neurons"
        + (" sorted by activity" if sort_by_activity else "")
    )
    fig.colorbar(image, ax=ax, label="internal state x_i(t)")

    if not np.any(view):
        ax.text(
            0.5,
            0.5,
            "all selected internal states are 0",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
        )

    fig.tight_layout()
    fig.savefig(out_fp, dpi=150)
    plt.close(fig)

    if save_selected_csv:
        selected_fp = out_fp.with_name(out_fp.stem + "_selected_neurons.csv")
        with selected_fp.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["rank", "neuron_index", "activity_sum_abs"])
            activity = np.sum(np.abs(x_state), axis=1)
            for rank, neuron_index in enumerate(selected):
                writer.writerow([rank, int(neuron_index), float(activity[neuron_index])])

    return out_fp


def save_internal_state_mean_trace(
    out_fp: Path,
    x_state: np.ndarray,
    t_ms: np.ndarray,
    *,
    title: str,
) -> Path:
    plt = try_import_pyplot()
    if plt is None:
        raise RuntimeError("matplotlib is required to save internal-state plots.")

    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    mean_trace = np.mean(x_state, axis=0)
    max_trace = np.max(x_state, axis=0)
    active_fraction = np.mean(x_state > 0, axis=0)

    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(t_ms, mean_trace, linewidth=1.4, label="mean x_i(t)")
    ax1.plot(t_ms, max_trace, linewidth=1.0, alpha=0.75, label="max x_i(t)")
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("Internal state")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(t_ms, active_fraction, color="black", linewidth=1.0, alpha=0.55, label="active fraction")
    ax2.set_ylabel("Fraction of neurons with x_i(t) > 0")
    ax2.set_ylim(0.0, 1.0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_fp, dpi=150)
    plt.close(fig)
    return out_fp


def save_internal_state_summary_csv(
    out_fp: Path,
    x_state: np.ndarray,
    t_ms: np.ndarray,
) -> Path:
    # x_state を ニューロン x 時間 のヒートマップとして保存する。
    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    activity = np.sum(np.abs(x_state), axis=1)
    rows = {
        "n_neurons": [int(x_state.shape[0])],
        "n_time": [int(x_state.shape[1])],
        "duration_ms": [float(t_ms[-1]) if t_ms.size else 0.0],
        "state_min": [float(np.nanmin(x_state)) if x_state.size else 0.0],
        "state_max": [float(np.nanmax(x_state)) if x_state.size else 0.0],
        "state_mean": [float(np.nanmean(x_state)) if x_state.size else 0.0],
        "state_std": [float(np.nanstd(x_state)) if x_state.size else 0.0],
        "active_neurons": [int(np.count_nonzero(activity > 0))],
        "active_fraction": [float(np.mean(activity > 0)) if activity.size else 0.0],
        "total_abs_activity": [float(np.sum(activity))],
    }
    with out_fp.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows))
        writer.writeheader()
        writer.writerow({key: value[0] for key, value in rows.items()})
    return out_fp


def save_internal_state_overview(
    npz_fp: Path,
    out_dir: Path,
    *,
    title_prefix: str = "",
    max_neurons: int = 200,
    sort_by_activity: bool = True,
) -> dict:
    npz_fp = Path(npz_fp)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_state, t_ms = _load_internal_state_npz(npz_fp)
    tag = safe_stem(npz_fp.stem.replace("_liquid_internal_state_all", ""))
    title = f"{title_prefix} {tag}".strip()

    heatmap_fp = save_internal_state_heatmap(
        out_dir / f"{tag}_internal_state_heatmap.png",
        x_state,
        t_ms,
        title=title,
        max_neurons=max_neurons,
        sort_by_activity=sort_by_activity,
        save_selected_csv=False,
    )
    return {
        "source_file": str(npz_fp),
        "out_dir": str(out_dir),
        "n_neurons": int(x_state.shape[0]),
        "n_time": int(x_state.shape[1]),
        "heatmap_png": str(heatmap_fp),
    }


def save_internal_state_overviews(
    internal_state_dir: Path,
    out_dir: Path,
    *,
    materials: Sequence[str] | None = None,
    first_sample_only: bool = True,
    max_samples_per_class: int | None = None,
    max_neurons: int = 200,
    sort_by_activity: bool = True,
) -> list[dict]:
    # internal_states 以下の各素材ファイルを読み、確認用 heatmap をまとめて保存する。
    internal_state_dir = Path(internal_state_dir)
    out_dir = Path(out_dir)
    material_to_files = discover_internal_state_files(internal_state_dir)
    material_names = list(materials) if materials is not None else sorted(material_to_files)

    results = []
    for material in material_names:
        files = list(material_to_files.get(str(material), []))
        if first_sample_only:
            files = files[:1]
        elif max_samples_per_class is not None:
            files = files[: int(max_samples_per_class)]
        for fp in files:
            material_out_dir = out_dir / safe_stem(material)
            results.append(
                save_internal_state_overview(
                    fp,
                    material_out_dir,
                    title_prefix=str(material),
                    max_neurons=max_neurons,
                    sort_by_activity=sort_by_activity,
                )
            )

    manifest_fp = out_dir / "internal_state_visualization_manifest.json"
    manifest_fp.parent.mkdir(parents=True, exist_ok=True)
    manifest_fp.write_text(
        json.dumps(jsonable({"results": results}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return results
