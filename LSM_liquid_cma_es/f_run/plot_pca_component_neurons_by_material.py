"""Plot material responses for neurons that contribute strongly to each PCA component."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d_tools.plotting import try_import_pyplot
from d_tools.run_paths import jsonable
from d_tools.separation_metrics import discover_internal_state_files
from f_run.run_best_params_accuracy import find_default_internal_state_dir


DEFAULT_TOP_NEURONS = 20
DEFAULT_MAX_COMPONENTS = 20


def find_latest_pca_model() -> Path:
    root = PROJECT_ROOT / "g_tactile_results" / "liquid_run"
    candidates = []
    if root.exists():
        for run_dir in root.iterdir():
            if not run_dir.is_dir() or run_dir.name == "_runtime_cache":
                continue
            for model_dir in run_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                candidate = model_dir / "pca" / "pca_model.npz"
                if candidate.is_file():
                    candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError("No pca_model.npz was found under g_tactile_results/liquid_run.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def infer_internal_state_dir(pca_model_fp: Path) -> Path:
    candidate = Path(pca_model_fp).parent.parent / "internal_states"
    if candidate.is_dir():
        return candidate
    return find_default_internal_state_dir()


def _window_mask(
    t_ms: np.ndarray,
    *,
    window_start_ms: float | None = None,
    window_end_ms: float | None = None,
) -> np.ndarray:
    mask = np.ones(t_ms.shape, dtype=bool)
    if window_start_ms is not None:
        mask &= t_ms >= float(window_start_ms)
    if window_end_ms is not None:
        mask &= t_ms <= float(window_end_ms)
    if not np.any(mask):
        raise ValueError("selected time window is empty")
    return mask


def load_component_neuron_rankings(
    pca_model_fp: Path,
    *,
    top_neurons: int,
    max_components: int | None = None,
    component_start: int = 1,
) -> list[dict]:
    with np.load(pca_model_fp, allow_pickle=True) as data:
        components = np.asarray(data["components"], dtype=np.float64)
        n_neurons = int(np.asarray(data["state_neurons"]).item())
        n_time = int(np.asarray(data["state_time_steps"]).item())
        ratios = np.asarray(data["explained_variance_ratio"], dtype=np.float64)

    if components.ndim != 2 or components.shape[1] != n_neurons * n_time:
        raise ValueError(
            f"pca_model components shape {components.shape} does not match "
            f"state_neurons={n_neurons}, state_time_steps={n_time}"
        )

    start_index = max(0, int(component_start) - 1)
    n_components = components.shape[0]
    if max_components is not None:
        n_components = min(n_components, start_index + int(max_components))
    top_neurons = min(max(1, int(top_neurons)), n_neurons)

    rankings = []
    for component_index in range(start_index, n_components):
        component_map = components[component_index].reshape(n_neurons, n_time)
        contribution = np.sum(component_map * component_map, axis=1)
        total = float(np.sum(contribution))
        contribution_ratio = contribution / total if total > 0 else np.zeros_like(contribution)
        signed_mean_loading = np.mean(component_map, axis=1)
        order = np.argsort(contribution_ratio)[::-1][:top_neurons]
        rankings.append(
            {
                "component_index": int(component_index),
                "pc": int(component_index + 1),
                "explained_variance_ratio": (
                    float(ratios[component_index]) if component_index < ratios.shape[0] else None
                ),
                "neuron_indices": order.astype(np.int32),
                "contribution_ratio": contribution_ratio[order],
                "signed_mean_loading": signed_mean_loading[order],
            }
        )
    return rankings


def load_material_mean_state(
    files: list[Path],
    neuron_indices: np.ndarray,
    *,
    max_samples: int | None = None,
    window_start_ms: float | None = None,
    window_end_ms: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, int]:
    selected_files = files[: int(max_samples)] if max_samples is not None else files
    if not selected_files:
        raise ValueError("no files selected")

    states = []
    spike_times_by_rank = []
    spike_ranks = []
    has_exact_spikes = False
    t_ref = None
    for fp in selected_files:
        with np.load(fp, allow_pickle=True) as data:
            x_state = np.asarray(data["x_state"], dtype=np.float64)
            t_ms = np.asarray(data["t_ms"], dtype=np.float64)
            spike_t_ms = (
                np.asarray(data["spike_t_ms"], dtype=np.float64)
                if "spike_t_ms" in data.files
                else np.asarray([], dtype=np.float64)
            )
            spike_i = (
                np.asarray(data["spike_i"], dtype=np.int32)
                if "spike_i" in data.files
                else np.asarray([], dtype=np.int32)
            )
        mask = _window_mask(
            t_ms,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
        )
        valid_neurons = neuron_indices[neuron_indices < x_state.shape[0]]
        if valid_neurons.shape[0] != neuron_indices.shape[0]:
            raise ValueError(
                f"{fp} has {x_state.shape[0]} neurons, but selected neurons include "
                f"index {int(np.max(neuron_indices))}"
            )
        x_selected = x_state[valid_neurons, :][:, mask]
        t_selected = t_ms[mask]
        if t_ref is None:
            t_ref = t_selected
        n_time = min(x_selected.shape[1], t_ref.shape[0])
        states.append(x_selected[:, :n_time])
        t_ref = t_ref[:n_time]

        if spike_t_ms.size and spike_i.size:
            spike_mask = np.ones(spike_t_ms.shape, dtype=bool)
            if window_start_ms is not None:
                spike_mask &= spike_t_ms >= float(window_start_ms)
            if window_end_ms is not None:
                spike_mask &= spike_t_ms <= float(window_end_ms)
            rank_by_neuron = {int(neuron): rank for rank, neuron in enumerate(neuron_indices)}
            selected_spike_mask = spike_mask & np.asarray(
                [int(index) in rank_by_neuron for index in spike_i],
                dtype=bool,
            )
            if np.any(selected_spike_mask):
                selected_times = spike_t_ms[selected_spike_mask]
                selected_ranks = np.asarray(
                    [rank_by_neuron[int(index)] for index in spike_i[selected_spike_mask]],
                    dtype=np.float64,
                )
                spike_times_by_rank.append(selected_times)
                spike_ranks.append(selected_ranks)
                has_exact_spikes = True

    min_time = min(item.shape[1] for item in states)
    stacked = np.stack([item[:, :min_time] for item in states], axis=0)
    mean_state = np.mean(stacked, axis=0)
    active_fraction = np.mean(stacked > 0.0, axis=0)
    if spike_times_by_rank:
        spike_times_out = np.concatenate(spike_times_by_rank).astype(np.float64, copy=False)
        spike_ranks_out = np.concatenate(spike_ranks).astype(np.float64, copy=False)
    else:
        spike_times_out = np.asarray([], dtype=np.float64)
        spike_ranks_out = np.asarray([], dtype=np.float64)
    return (
        mean_state,
        active_fraction,
        spike_times_out,
        spike_ranks_out,
        np.asarray(t_ref[:min_time], dtype=np.float64),
        bool(has_exact_spikes),
        len(selected_files),
    )


def save_component_material_raster(
    out_fp: Path,
    t_ms: np.ndarray,
    *,
    material: str,
    pc: int,
    explained_variance_ratio: float | None,
    neuron_indices: np.ndarray,
    contribution_ratio: np.ndarray,
    sample_count: int,
    spike_t_ms: np.ndarray,
    spike_rank: np.ndarray,
) -> Path:
    plt = try_import_pyplot()
    if plt is None:
        raise RuntimeError("matplotlib is required to save plots")

    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    duration = float(t_ms[-1]) if t_ms.size else float(np.max(spike_t_ms, initial=0.0))
    title = f"{material} | PC{pc} top-neuron spike-time raster | n={sample_count}"
    if explained_variance_ratio is not None:
        title += f" | explained={explained_variance_ratio * 100:.1f}%"

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.28 * len(neuron_indices))))
    ax.scatter(
        spike_t_ms,
        spike_rank,
        s=7,
        marker="|",
        linewidths=0.8,
        color="black",
        alpha=0.28,
    )
    ax.text(
        0.99,
        0.02,
        f"spikes={spike_t_ms.size}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="0.25",
        fontsize=9,
    )
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Top neuron rank")
    ax.set_title(title)
    ax.set_xlim(float(t_ms[0]) if t_ms.size else 0.0, duration)
    ax.set_ylim(len(neuron_indices) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(neuron_indices)))
    ax.set_yticklabels(
        [
            f"{rank + 1}: n{int(neuron)} ({float(ratio) * 100:.1f}%)"
            for rank, (neuron, ratio) in enumerate(zip(neuron_indices, contribution_ratio))
        ],
        fontsize=8,
    )
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=150)
    plt.close(fig)
    return out_fp


def plot_pca_component_neurons_by_material(
    *,
    internal_state_dir: Path,
    pca_model_fp: Path,
    out_dir: Path,
    top_neurons: int = DEFAULT_TOP_NEURONS,
    max_components: int | None = DEFAULT_MAX_COMPONENTS,
    component_start: int = 1,
    max_samples_per_class: int | None = None,
    window_start_ms: float | None = None,
    window_end_ms: float | None = None,
) -> dict:
    internal_state_dir = Path(internal_state_dir)
    pca_model_fp = Path(pca_model_fp)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    material_to_files = discover_internal_state_files(internal_state_dir)
    rankings = load_component_neuron_rankings(
        pca_model_fp,
        top_neurons=top_neurons,
        max_components=max_components,
        component_start=component_start,
    )

    rows = []
    output_files = []
    for ranking in rankings:
        pc = int(ranking["pc"])
        pc_dir = out_dir / f"PC{pc:02d}"
        neuron_indices = np.asarray(ranking["neuron_indices"], dtype=np.int32)
        contribution_ratio = np.asarray(ranking["contribution_ratio"], dtype=np.float64)

        for material, files in sorted(material_to_files.items()):
            (
                mean_state,
                active_fraction,
                spike_t_ms,
                spike_rank,
                t_ms,
                exact_spike_times,
                sample_count,
            ) = load_material_mean_state(
                list(files),
                neuron_indices,
                max_samples=max_samples_per_class,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
            )
            if not exact_spike_times or spike_t_ms.size == 0:
                continue
            raster_fp = save_component_material_raster(
                pc_dir / f"{material}_PC{pc:02d}_top{len(neuron_indices)}_neurons_raster.png",
                t_ms,
                material=material,
                pc=pc,
                explained_variance_ratio=ranking["explained_variance_ratio"],
                neuron_indices=neuron_indices,
                contribution_ratio=contribution_ratio,
                sample_count=sample_count,
                spike_t_ms=spike_t_ms,
                spike_rank=spike_rank,
            )
            output_files.append(raster_fp)
            rows.append(
                {
                    "component": f"PC{pc}",
                    "material": material,
                    "samples": int(sample_count),
                    "top_neurons": " ".join(str(int(i)) for i in neuron_indices),
                    "top_neuron_contribution_ratios": " ".join(
                        f"{float(v):.12g}" for v in contribution_ratio
                    ),
                    "mean_state_peak": float(np.max(mean_state)) if mean_state.size else 0.0,
                    "mean_state_mean": float(np.mean(mean_state)) if mean_state.size else 0.0,
                    "active_fraction_peak": (
                        float(np.max(active_fraction)) if active_fraction.size else 0.0
                    ),
                    "exact_spike_times": bool(exact_spike_times),
                    "spike_count": int(spike_t_ms.size),
                    "raster_png": str(raster_fp),
                }
            )

    csv_fp = out_dir / "pca_component_neurons_by_material_summary.csv"
    if not rows:
        raise RuntimeError(
            "No spike-time raster plots were created because the selected internal-state "
            "files do not contain spike_t_ms/spike_i. Re-run liquid generation after "
            "the spike-save patch, then run this plotting script again."
        )

    with csv_fp.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "internal_state_dir": str(internal_state_dir),
        "pca_model": str(pca_model_fp),
        "out_dir": str(out_dir),
        "top_neurons": int(top_neurons),
        "max_components": max_components,
        "component_start": int(component_start),
        "max_samples_per_class": max_samples_per_class,
        "window_start_ms": window_start_ms,
        "window_end_ms": window_end_ms,
        "summary_csv": str(csv_fp),
        "n_plots": int(len(output_files)),
    }
    summary_fp = out_dir / "pca_component_neurons_by_material_summary.json"
    summary_fp.write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_json"] = str(summary_fp)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot each material using neurons with high contribution to each PCA component."
    )
    parser.add_argument("--internal-state-dir", type=Path, default=None)
    parser.add_argument("--pca-dir", type=Path, default=None)
    parser.add_argument("--pca-model", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--top-neurons", type=int, default=DEFAULT_TOP_NEURONS)
    parser.add_argument("--max-components", type=int, default=DEFAULT_MAX_COMPONENTS)
    parser.add_argument("--component-start", type=int, default=1)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--window-start-ms", type=float, default=None)
    parser.add_argument("--window-end-ms", type=float, default=None)
    args = parser.parse_args()

    pca_model_fp = (
        Path(args.pca_model)
        if args.pca_model is not None
        else (
            (Path(args.pca_dir) / "pca_model.npz")
            if args.pca_dir is not None
            else find_latest_pca_model()
        )
    )
    internal_state_dir = (
        Path(args.internal_state_dir)
        if args.internal_state_dir is not None
        else infer_internal_state_dir(pca_model_fp)
    )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else pca_model_fp.parent / "pca_component_neurons_by_material"
    )

    summary = plot_pca_component_neurons_by_material(
        internal_state_dir=internal_state_dir,
        pca_model_fp=pca_model_fp,
        out_dir=out_dir,
        top_neurons=int(args.top_neurons),
        max_components=int(args.max_components),
        component_start=int(args.component_start),
        max_samples_per_class=args.max_samples_per_class,
        window_start_ms=args.window_start_ms,
        window_end_ms=args.window_end_ms,
    )
    print(
        "[pca-component-neurons] "
        f"saved {summary['n_plots']} plots to {summary['out_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
