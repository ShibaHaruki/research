# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "liquid_sensor_pca_results"

SENSOR_ORDER = ["sensor1", "sensor2", "sensor3", "all"]
SENSOR_LABELS = {
    "sensor1": "sensor 1",
    "sensor2": "sensor 2",
    "sensor3": "sensor 3",
    "all": "all sensors",
}
MATERIALS = [
    "Al_board", "buta_omote", "buta_ura", "cork",
    "denim", "rubber_board", "washi", "wood_board",
]
WINDOWS_MS = [5, 10, 20, 25, 50, 100, 250, 500]


def bin_spikes(spikes: np.ndarray, window_ms: int) -> np.ndarray:
    n_sample, n_neuron, total_ms = spikes.shape
    if total_ms % window_ms != 0:
        raise ValueError(f"T={total_ms} is not divisible by window_ms={window_ms}")
    n_window = total_ms // window_ms
    return spikes.reshape(n_sample, n_neuron, n_window, window_ms).sum(axis=-1)


def cosine_against_reference(reference: np.ndarray, targets: np.ndarray) -> np.ndarray:
    reference = reference.astype(np.float64, copy=False)
    targets = targets.astype(np.float64, copy=False)
    numerator = np.einsum("nw,snw->sn", reference, targets)
    reference_norm = np.linalg.norm(reference, axis=1)
    target_norm = np.linalg.norm(targets, axis=2)
    denominator = target_norm * reference_norm[None, :]
    similarity = np.full_like(numerator, np.nan, dtype=np.float64)
    valid = denominator > 0
    similarity[valid] = numerator[valid] / denominator[valid]
    return similarity


def analyze_one_window(sout_rec: np.ndarray,
                       sensor_mode: str,
                       window_ms: int,
                       reference_material: int,
                       reference_sample: int):
    reference_binned = bin_spikes(
        np.asarray(sout_rec[reference_material, reference_sample], dtype=np.float32)[None, ...],
        window_ms,
    )[0]

    neuron_rows = []
    summary_rows = []
    group_values = {}

    for material_idx, material in enumerate(MATERIALS):
        binned = bin_spikes(
            np.asarray(sout_rec[material_idx], dtype=np.float32),
            window_ms,
        )
        if material_idx == reference_material:
            target_indices = np.arange(binned.shape[0])
            target_indices = target_indices[target_indices != reference_sample]
            targets = binned[target_indices]
            group_name = f"same_{MATERIALS[reference_material]}"
            pair_type = "same_material"
        else:
            targets = binned
            group_name = material
            pair_type = "different_material"

        similarity = cosine_against_reference(reference_binned, targets)
        neuron_mean = np.nanmean(similarity, axis=0)
        neuron_std = np.nanstd(similarity, axis=0)
        group_values[group_name] = neuron_mean

        valid_neurons = np.isfinite(neuron_mean)
        summary_rows.append({
            "sensor_mode": sensor_mode,
            "window_ms": window_ms,
            "reference_material": MATERIALS[reference_material],
            "reference_sample_index": reference_sample,
            "comparison_group": group_name,
            "pair_type": pair_type,
            "cosine_mean_across_neurons": float(np.nanmean(neuron_mean)),
            "cosine_std_across_neurons": float(np.nanstd(neuron_mean)),
            "n_valid_neurons": int(np.sum(valid_neurons)),
            "n_target_trials": int(targets.shape[0]),
        })

        for neuron_idx in range(len(neuron_mean)):
            neuron_rows.append({
                "sensor_mode": sensor_mode,
                "window_ms": window_ms,
                "reference_material": MATERIALS[reference_material],
                "reference_sample_index": reference_sample,
                "comparison_group": group_name,
                "pair_type": pair_type,
                "neuron_index": neuron_idx,
                "cosine_mean": neuron_mean[neuron_idx],
                "cosine_std": neuron_std[neuron_idx],
                "n_target_trials": int(targets.shape[0]),
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(neuron_rows), group_values


def plot_neuron_heatmap(group_values: dict,
                        out_dir: Path,
                        sensor_mode: str,
                        window_ms: int,
                        reference_material: int):
    reference_name = MATERIALS[reference_material]
    group_order = [f"same_{reference_name}"] + [
        material for idx, material in enumerate(MATERIALS) if idx != reference_material
    ]
    matrix = np.column_stack([group_values[group] for group in group_order])

    fig, ax = plt.subplots(figsize=(9.0, 10.0))
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    ax.set_xticks(np.arange(len(group_order)))
    ax.set_xticklabels(group_order, rotation=40, ha="right")
    ax.set_xlabel("Compared material")
    ax.set_ylabel("Liquid neuron index")
    ax.set_title(
        f"Cosine similarity to first {reference_name} trial | "
        f"{SENSOR_LABELS[sensor_mode]} | window={window_ms} ms"
    )
    fig.colorbar(image, ax=ax, label="Mean cosine similarity")
    fig.tight_layout()
    out_path = out_dir / f"reference_{reference_name}_neuron_cosine_heatmap_{window_ms}ms.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_group_summary(summary_df: pd.DataFrame,
                       out_dir: Path,
                       sensor_mode: str,
                       window_ms: int,
                       reference_material: int):
    reference_name = MATERIALS[reference_material]
    group_order = [f"same_{reference_name}"] + [
        material for idx, material in enumerate(MATERIALS) if idx != reference_material
    ]
    plot_df = summary_df.set_index("comparison_group").reindex(group_order).reset_index()

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(plot_df))
    colors = ["tab:blue"] + ["tab:orange"] * (len(plot_df) - 1)
    ax.bar(
        x,
        plot_df["cosine_mean_across_neurons"],
        yerr=plot_df["cosine_std_across_neurons"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, rotation=40, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title(
        f"Similarity to first {reference_name} trial | "
        f"{SENSOR_LABELS[sensor_mode]} | window={window_ms} ms"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = out_dir / f"reference_{reference_name}_cosine_summary_{window_ms}ms.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--sensor-mode", choices=SENSOR_ORDER + ["each"], default="each")
    parser.add_argument("--windows-ms", default="5,10,20,25,50,100,250,500")
    parser.add_argument(
        "--reference-sample",
        type=int,
        default=0,
        help="Zero-based sample index used as the reference for each material.",
    )
    parser.add_argument(
        "--reference-material",
        choices=MATERIALS + ["each"],
        default="each",
        help="Reference material. Use 'each' to analyze all eight materials.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sensor_modes = SENSOR_ORDER if args.sensor_mode == "each" else [args.sensor_mode]
    windows_ms = [int(value) for value in args.windows_ms.split(",") if value.strip()]
    reference_materials = (
        list(range(len(MATERIALS)))
        if args.reference_material == "each"
        else [MATERIALS.index(args.reference_material)]
    )

    for sensor_mode in sensor_modes:
        rec_path = args.result_dir / sensor_mode / "liquid_sout_rec_rep1.npy"
        if not rec_path.exists():
            raise FileNotFoundError(rec_path)
        sout_rec = np.load(rec_path, mmap_mode="r")
        if not 0 <= args.reference_sample < sout_rec.shape[1]:
            raise IndexError(
                f"reference sample {args.reference_sample} is outside 0..{sout_rec.shape[1] - 1}"
            )

        print(f"[loaded] {rec_path} shape={sout_rec.shape}")

        for reference_material in reference_materials:
            reference_name = MATERIALS[reference_material]
            out_dir = (
                args.result_dir
                / sensor_mode
                / "reference_material_neuron_cosine"
                / reference_name
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            all_summary = []
            all_neurons = []

            for window_ms in windows_ms:
                print(
                    f"[{sensor_mode}] reference={reference_name} "
                    f"window={window_ms} ms"
                )
                summary_df, neuron_df, group_values = analyze_one_window(
                    sout_rec=sout_rec,
                    sensor_mode=sensor_mode,
                    window_ms=window_ms,
                    reference_material=reference_material,
                    reference_sample=args.reference_sample,
                )
                all_summary.append(summary_df)
                all_neurons.append(neuron_df)
                plot_neuron_heatmap(
                    group_values,
                    out_dir,
                    sensor_mode,
                    window_ms,
                    reference_material,
                )
                plot_group_summary(
                    summary_df,
                    out_dir,
                    sensor_mode,
                    window_ms,
                    reference_material,
                )

            summary_all = pd.concat(all_summary, ignore_index=True)
            neurons_all = pd.concat(all_neurons, ignore_index=True)
            summary_path = out_dir / f"reference_{reference_name}_cosine_summary.csv"
            neuron_path = out_dir / f"reference_{reference_name}_cosine_by_neuron.csv"
            summary_all.to_csv(summary_path, index=False)
            neurons_all.to_csv(neuron_path, index=False)
            print(f"[saved] {summary_path}")
            print(f"[saved] {neuron_path}")


if __name__ == "__main__":
    main()
