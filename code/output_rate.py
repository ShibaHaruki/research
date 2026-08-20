# -*- coding: utf-8 -*-
import re
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# 入力設定
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

RULES = [
    "off_1",
    "SRDP_1",
    "STDP_1",
    "T_STDP_1",
]

REP = 3
CYCLE_IDX = 1            # 0始まり
TOTAL_MS = 500.0
WINDOW_MS = 25.0

# 元データの並びに対応する内部名（保存ファイル名用）
MATERIAL_KEYS = [
    "Al_board",
    "buta_omote",
    "buta_ura",
    "cork",
    "denim",
    "rubber_board",
    "washi",
    "wood_board"
]

# グラフ表示名（PCA図の表記に合わせる）
MATERIAL_LABELS = [
    "aluminum bd.",
    "outer_pigskin",
    "back_pigskin",
    "cork",
    "denim",
    "rubber bd.",
    "japanese paper",
    "wood bd."
]

OUTPUT_SUFFIX = "sout_rec"

RESULT_ROOT = SCRIPT_DIR / "rate_analysis_results"

LINESTYLES = [
    "-", "--", "-.", ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 1)),
    (0, (5, 2, 1, 2)),
]

MARKERS = [
    "o", "s", "^", "D",
    "v", "x", "*", "+"
]

COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]


# =========================
# 共通関数
# =========================
def clean_rule_name(name: str) -> str:
    return name.replace("_1", "")


def find_existing_file(base_prefix: str, rep: int, suffix_candidates):
    for suffix in suffix_candidates:
        path = SCRIPT_DIR / f"{base_prefix}_{suffix}_rep{rep}.npy"
        if path.exists():
            return path, suffix
    return None, None


def parse_prefix_rep(npy_path: Path):
    m = re.match(r"(.+)_([A-Za-z0-9]+_rec)_rep(\d+)\.npy$", npy_path.name)
    if m is None:
        m = re.match(r"(.+)_(.+_rec)_rep(\d+)\.npy$", npy_path.name)
    if m is None:
        raise ValueError(f"ファイル名が想定形式ではありません: {npy_path.name}")
    prefix = m.group(1)
    rec_name = m.group(2)
    rep = int(m.group(3))
    return prefix, rec_name, rep


def save_figure_pdf_only(base_path: Path):
    plt.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")


def aggregate_one_cycle_to_windows(spike_one_cycle, total_ms=500.0, window_ms=25.0):
    """
    spike_one_cycle: (n_mat, n_neuron, n_bins)
    """
    n_mat, n_neuron, n_bins = spike_one_cycle.shape

    ms_per_bin = total_ms / n_bins
    bins_per_win = int(round(window_ms / ms_per_bin))

    if bins_per_win <= 0:
        raise ValueError("bins_per_win <= 0 です")

    if not np.isclose(bins_per_win * ms_per_bin, window_ms, atol=1e-9):
        raise ValueError(
            f"window_ms={window_ms} ms を bin幅 {ms_per_bin:.6f} ms で割り切れません"
        )

    n_win = n_bins // bins_per_win
    used_bins = n_win * bins_per_win

    spike_trim = spike_one_cycle[..., :used_bins]
    spike_win = spike_trim.reshape(n_mat, n_neuron, n_win, bins_per_win).sum(axis=-1)

    x_ms = (np.arange(n_win) + 0.5) * window_ms
    return spike_win, x_ms, ms_per_bin


def compute_one_cycle_rates(spike_win, window_ms=25.0):
    """
    spike_win: (n_mat, n_neuron, n_win)
    """
    window_sec = window_ms / 1000.0
    rate_per_neuron = spike_win.astype(np.float64) / window_sec
    pop_rate = rate_per_neuron.mean(axis=1)
    return pop_rate, rate_per_neuron


def plot_all_materials(pop_rate, x_ms, mat_labels, title_name, layer_name, out_base_path):
    plt.figure(figsize=(8.5, 4.8))

    for i, label in enumerate(mat_labels):
        plt.plot(
            x_ms,
            pop_rate[i],
            color=COLORS[i % len(COLORS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            marker=MARKERS[i % len(MARKERS)],
            markevery=max(1, len(x_ms) // 8),
            linewidth=1.8,
            markersize=5,
            label=label
        )

    plt.xlabel("Time (ms)")
    plt.ylabel(f"Mean firing rate of {layer_name} layer (Hz)")
    plt.title(title_name)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    save_figure_pdf_only(out_base_path)
    plt.close()


def plot_each_material(pop_rate, x_ms, mat_labels, mat_keys, title_name, prefix_clean, cycle_idx, layer_name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (label, key) in enumerate(zip(mat_labels, mat_keys)):
        plt.figure(figsize=(6.2, 4.0))
        plt.plot(
            x_ms,
            pop_rate[i],
            color=COLORS[i % len(COLORS)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            marker=MARKERS[i % len(MARKERS)],
            markevery=max(1, len(x_ms) // 8),
            linewidth=1.8,
            markersize=5
        )

        plt.xlabel("Time (ms)")
        plt.ylabel(f"Mean firing rate of {layer_name} layer (Hz)")
        plt.title(f"{title_name} | {label}")
        plt.tight_layout()

        save_figure_pdf_only(
            out_dir / f"{prefix_clean}_{layer_name}_{key}_cycle{cycle_idx}_pop_rate"
        )
        plt.close()


def plot_heatmap(pop_rate, x_ms, mat_labels, title_name, out_base_path):
    plt.figure(figsize=(8.5, 4.8))
    plt.imshow(pop_rate, aspect="auto", interpolation="nearest", cmap="viridis")

    plt.yticks(np.arange(len(mat_labels)), mat_labels)

    step = max(1, len(x_ms) // 10)
    xt = np.arange(0, len(x_ms), step)
    plt.xticks(xt, [f"{int(x_ms[k])}" for k in xt])

    plt.xlabel("Time window center (ms)")
    plt.ylabel("Material")
    plt.title(title_name)
    plt.colorbar(label="Mean firing rate (Hz)")
    plt.tight_layout()

    save_figure_pdf_only(out_base_path)
    plt.close()


def analyze_spike_rec_file(npy_path: Path, layer_name: str, out_base_dir: Path):
    if not npy_path.exists():
        raise FileNotFoundError(f"ファイルがありません: {npy_path}")

    prefix, rec_name, rep = parse_prefix_rep(npy_path)
    prefix_clean = clean_rule_name(prefix)

    spike_rec = np.load(npy_path)
    if spike_rec.ndim != 4:
        raise ValueError(f"shape が4次元ではありません: {spike_rec.shape}")

    n_mat, n_sample, n_neuron, n_bins = spike_rec.shape
    print(f"[{layer_name}] loaded: {npy_path.name} shape={spike_rec.shape}")

    if not (0 <= CYCLE_IDX < n_sample):
        raise IndexError(f"CYCLE_IDX={CYCLE_IDX} は範囲外です。0 <= idx < {n_sample}")

    mat_keys = MATERIAL_KEYS[:n_mat]
    mat_labels = MATERIAL_LABELS[:n_mat]

    spike_one_cycle = spike_rec[:, CYCLE_IDX, :, :]

    spike_win, x_ms, ms_per_bin = aggregate_one_cycle_to_windows(
        spike_one_cycle,
        total_ms=TOTAL_MS,
        window_ms=WINDOW_MS
    )
    print(f"[{layer_name}] ms_per_bin={ms_per_bin:.4f}, n_win={len(x_ms)}")

    pop_rate, rate_per_neuron = compute_one_cycle_rates(
        spike_win,
        window_ms=WINDOW_MS
    )

    out_base_dir.mkdir(parents=True, exist_ok=True)

    plot_all_materials(
        pop_rate=pop_rate,
        x_ms=x_ms,
        mat_labels=mat_labels,
        title_name=prefix_clean,
        layer_name=layer_name,
        out_base_path=out_base_dir / f"{prefix_clean}_{layer_name}_cycle{CYCLE_IDX}_pop_rate_25ms_rep{rep}"
    )

    plot_each_material(
        pop_rate=pop_rate,
        x_ms=x_ms,
        mat_labels=mat_labels,
        mat_keys=mat_keys,
        title_name=prefix_clean,
        prefix_clean=prefix_clean,
        cycle_idx=CYCLE_IDX,
        layer_name=layer_name,
        out_dir=out_base_dir / f"{prefix_clean}_{layer_name}_cycle{CYCLE_IDX}_pop_rate_each_25ms_rep{rep}"
    )

    plot_heatmap(
        pop_rate=pop_rate,
        x_ms=x_ms,
        mat_labels=mat_labels,
        title_name=prefix_clean,
        out_base_path=out_base_dir / f"{prefix_clean}_{layer_name}_cycle{CYCLE_IDX}_pop_rate_heatmap_25ms_rep{rep}"
    )

    return pop_rate, rate_per_neuron


def analyze_one_rule(rule: str):
    print("=" * 80)
    print(f"processing rule: {rule}")

    rule_clean = clean_rule_name(rule)
    rule_dir = RESULT_ROOT / rule_clean
    rule_dir.mkdir(parents=True, exist_ok=True)

    out_path, _ = find_existing_file(rule, REP, [OUTPUT_SUFFIX])
    if out_path is None:
        print(f"[skip] output file not found for {rule}")
    else:
        analyze_spike_rec_file(
            npy_path=out_path,
            layer_name="output",
            out_base_dir=rule_dir / "output_layer"
        )


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    print("save root:", RESULT_ROOT)

    for rule in RULES:
        analyze_one_rule(rule)

    print("=" * 80)
    print("done")


if __name__ == "__main__":
    main()