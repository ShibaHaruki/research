# -*- coding: utf-8 -*-
"""Create reviewer-ready PCA scatter/loading figures for STDP, t-STDP, and SRDP."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pca import (
    COLORS,
    DISPLAY_NAMES,
    MARKERS,
    SAMPLE_FOR_CLS,
    T_n,
    build_feature_matrix,
    compute_pca_2d,
    find_npy_path,
)


METHODS = (
    ("STDP_1", "STDP"),
    ("T_STDP_1", "t-STDP"),
    ("SRDP_1", "SRDP"),
)
# Matplotlib fills a multi-column legend by columns. This input order displays
# rows as: Al, rubber, wood, outer / back, cork, denim, Japanese paper.
LEGEND_ORDER = (0, 2, 5, 3, 7, 4, 1, 6)
OUT_DIR = Path(__file__).resolve().parent / "pca_2d_results" / "integrated_pca_loadings"


def orient_loading(loading):
    """Resolve the arbitrary PCA sign for stable display."""
    flat = loading.ravel()
    return -loading if flat[np.argmax(np.abs(flat))] < 0 else loading


def calculate_result(rule_name, rep):
    path = find_npy_path(rule_name, rep)
    spikes = np.load(path)
    n_classes, n_samples, n_neurons, duration_ms = spikes.shape
    if SAMPLE_FOR_CLS > n_samples:
        raise ValueError(f"Not enough samples in {path}")

    features = build_feature_matrix(spikes[:, :SAMPLE_FOR_CLS], T_n=T_n)
    scores, explained, eigenvectors = compute_pca_2d(features)
    n_time_bins = duration_ms // T_n
    loadings = [
        orient_loading(eigenvectors[:, component].real.reshape(n_neurons, n_time_bins))
        for component in range(2)
    ]
    return {
        "scores": scores,
        "explained": explained,
        "loadings": loadings,
        "n_classes": n_classes,
        "n_neurons": n_neurons,
        "duration_ms": duration_ms,
    }


def plot_integrated(results, rep, output_path):
    fig = plt.figure(figsize=(14.5, 9.2))
    grid = fig.add_gridspec(
        4,
        3,
        height_ratios=(2.7, 0.30, 1.0, 1.0),
        hspace=0.38,
        wspace=0.25,
    )

    all_loadings = [
        loading
        for result in results
        for loading in result["loadings"]
    ]
    color_limit = max(float(np.max(np.abs(loading))) for loading in all_loadings)
    heatmap_axes = []
    legend_handles = None
    legend_labels = None
    last_image = None

    for column, ((_, method_name), result) in enumerate(zip(METHODS, results)):
        scatter_ax = fig.add_subplot(grid[0, column])
        for class_index in range(result["n_classes"]):
            start = class_index * SAMPLE_FOR_CLS
            end = start + SAMPLE_FOR_CLS
            scatter_ax.scatter(
                result["scores"][start:end, 0],
                result["scores"][start:end, 1],
                color=COLORS[class_index % len(COLORS)],
                marker=MARKERS[class_index % len(MARKERS)],
                s=17,
                linewidths=0.7,
                alpha=0.8,
                label=DISPLAY_NAMES[class_index].replace("_", " "),
            )
        scatter_ax.set_xlabel(f"PC1 ({result['explained'][0] * 100:.1f}%)", fontsize=14)
        scatter_ax.set_ylabel(f"PC2 ({result['explained'][1] * 100:.1f}%)", fontsize=14)
        x_values = result["scores"][:, 0]
        y_values = result["scores"][:, 1]
        axis_span = max(float(np.ptp(x_values)), float(np.ptp(y_values)), 1.0) * 1.08
        x_center = (float(np.min(x_values)) + float(np.max(x_values))) / 2
        y_center = (float(np.min(y_values)) + float(np.max(y_values))) / 2
        x_limits = (x_center - axis_span / 2, x_center + axis_span / 2)
        y_limits = (y_center - axis_span / 2, y_center + axis_span / 2)
        scatter_ax.set_xlim(x_limits)
        scatter_ax.set_ylim(y_limits)
        tick_step = 10.0
        scatter_ax.set_xticks(
            np.arange(np.ceil(x_limits[0] / tick_step) * tick_step, x_limits[1], tick_step)
        )
        scatter_ax.set_yticks(
            np.arange(np.ceil(y_limits[0] / tick_step) * tick_step, y_limits[1], tick_step)
        )
        scatter_ax.set_box_aspect(1)
        scatter_ax.set_aspect("equal", adjustable="box")
        scatter_ax.text(
            -0.12,
            1.04,
            chr(ord("A") + column),
            transform=scatter_ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="bottom",
        )
        scatter_ax.text(
            0.5,
            1.04,
            method_name,
            transform=scatter_ax.transAxes,
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
        scatter_ax.tick_params(labelsize=11)
        if legend_handles is None:
            legend_handles, legend_labels = scatter_ax.get_legend_handles_labels()

        for row, component in ((2, 0), (3, 1)):
            heat_ax = fig.add_subplot(grid[row, column])
            heatmap_axes.append(heat_ax)
            last_image = heat_ax.imshow(
                result["loadings"][component],
                aspect="auto",
                cmap="coolwarm",
                vmin=-color_limit,
                vmax=color_limit,
                extent=(0, result["duration_ms"], result["n_neurons"], 0),
                interpolation="nearest",
            )
            heat_ax.set_xlabel("Time (ms)", fontsize=12)
            if column == 0:
                heat_ax.set_ylabel("readout neuron", fontsize=12)
            heat_ax.set_xticks(np.arange(0, result["duration_ms"] + 1, 100))
            heat_ax.set_yticks(np.arange(0, result["n_neurons"] + 1, 10))
            heat_ax.tick_params(labelsize=10)
            heat_ax.text(
                0.02,
                0.92,
                f"PC{component + 1} ({result['explained'][component] * 100:.1f}%)",
                transform=heat_ax.transAxes,
                fontsize=11,
                fontweight="bold",
                va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
            )

    legend_handles = [legend_handles[index] for index in LEGEND_ORDER]
    legend_labels = [legend_labels[index] for index in LEGEND_ORDER]
    legend_ax = fig.add_subplot(grid[1, :])
    legend_ax.axis("off")
    legend_ax.legend(
        legend_handles,
        legend_labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.15),
        ncol=4,
        frameon=False,
        fontsize=16,
        markerscale=2.0,
    )

    fig.subplots_adjust(left=0.07, right=0.90, top=0.95, bottom=0.07)
    fig.canvas.draw()
    heatmap_positions = [axis.get_position() for axis in heatmap_axes]
    colorbar_bottom = min(position.y0 for position in heatmap_positions)
    colorbar_top = max(position.y1 for position in heatmap_positions)
    colorbar_left = max(position.x1 for position in heatmap_positions) + 0.025
    colorbar_ax = fig.add_axes(
        (colorbar_left, colorbar_bottom, 0.012, colorbar_top - colorbar_bottom)
    )
    colorbar = fig.colorbar(last_image, cax=colorbar_ax, orientation="vertical")
    colorbar.set_label("PCA loading", fontsize=12)
    colorbar.ax.tick_params(labelsize=10)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rep in range(1, 11):
        print(f"processing rep{rep}")
        results = [calculate_result(rule_name, rep) for rule_name, _ in METHODS]
        output_path = OUT_DIR / f"pca_scatter_pc1_pc2_loadings_rep{rep}.png"
        pdf_path = plot_integrated(results, rep, output_path)
        print("saved:", output_path)
        print("saved:", pdf_path)


if __name__ == "__main__":
    main()
