# -*- coding: utf-8 -*-
import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# =========================
# 設定
# =========================
DIR_NAME = [
    "Al_board", "buta_omote", "buta_ura", "cork",
    "denim", "rubber_board", "washi", "wood_board"
]

DISPLAY_NAMES = [
    "aluminum bd.",
    "outer_pigskin",
    "back_pigskin",
    "cork",
    "denim",
    "rubber bd.",
    "japanese paper",
    "wood bd."
]

RULES = [
    "off_1",
    "SRDP_1",
    "STDP_1",
    "T_STDP_1",
]

REP_START = 1
REP_END = 10
T_n = 25
SAMPLE_FOR_CLS = 100
TARGET_DATASETS = sys.argv[1:]

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

OUT_DIR = SCRIPT_DIR / "pca_2d_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARKERS = ["o", "s", "^", "D", "v", "x", "*", "+"]
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


def find_npy_path(rule_name: str, rep: int) -> Path:
    candidates = [SCRIPT_DIR / f"{rule_name}_sout_rec_rep{rep}.npy"]
    folder_name = "T_STDP" if rule_name.startswith("T_STDP_") else rule_name.split("_", 1)[0]
    candidates.append(
        SCRIPT_DIR / folder_name / f"{rule_name}_sout_rec_rep{rep}.npy"
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def save_figure_png(base_path: Path):
    plt.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")


def build_feature_matrix(s_for_cls, T_n):
    n_sozai, sample_for_cls, N_out, T = s_for_cls.shape
    n_interval = T // T_n

    Y = np.zeros((N_out * n_interval, n_sozai * sample_for_cls), dtype=np.float64)

    for i in range(n_sozai):
        for j in range(sample_for_cls):
            col_idx = i * sample_for_cls + j
            vec = np.zeros(N_out * n_interval, dtype=np.float64)

            for k in range(N_out):
                for l in range(n_interval):
                    vec[k * n_interval + l] = np.sum(
                        s_for_cls[i, j, k, l * T_n:(l + 1) * T_n]
                    )

            Y[:, col_idx] = vec

    return Y


def compute_pca_2d(Y):
    scaler = StandardScaler()
    Y_std = scaler.fit_transform(Y.T).T

    cov_matrix = np.cov(Y_std)
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)

    order = np.argsort(eig_values.real)[::-1]
    eig_values = eig_values.real[order]
    eig_vectors = eig_vectors[:, order]

    eig_vectors_2d = eig_vectors[:, :2].real
    Y_pca2 = np.dot(Y_std.T, eig_vectors_2d)
    explained = eig_values / np.sum(eig_values)

    return Y_pca2, explained, eig_vectors


def plot_component_heatmaps(eig_vectors, explained, n_neurons, n_time_bins, out_dir):
    n_components = min(20, eig_vectors.shape[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []

    for component_index in range(n_components):
        loading = eig_vectors[:, component_index].real.reshape(n_neurons, n_time_bins)
        fig, ax = plt.subplots(figsize=(8.0, 5.5))
        image = ax.imshow(loading, aspect="auto", cmap="coolwarm")
        ax.set_xlabel("Time bin")
        ax.set_ylabel("Output neuron")
        ax.set_title(
            f"PC{component_index + 1} loading heatmap | "
            f"explained={explained[component_index] * 100:.1f}%"
        )
        fig.colorbar(image, ax=ax, label="PCA loading")
        fig.tight_layout()
        path = out_dir / f"pca_component_pc{component_index + 1:02d}_heatmap.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        images.append((loading, component_index))

    n_cols = 4
    n_rows = int(np.ceil(n_components / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.0, 3.0 * n_rows), squeeze=False)
    for ax, (loading, component_index) in zip(axes.flat, images):
        image = ax.imshow(loading, aspect="auto", cmap="coolwarm")
        ax.set_title(
            f"PC{component_index + 1} ({explained[component_index] * 100:.1f}%)",
            fontsize=9,
        )
        ax.set_xlabel("Time bin", fontsize=8)
        ax.set_ylabel("Neuron", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[n_components:]:
        ax.axis("off")
    fig.suptitle("PCA component loadings: neuron x time bin", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_components_pc1_pc20_heatmap_grid.png", dpi=180)
    plt.close(fig)


def plot_pca_2d(Y_pca2, explained, display_names, sample_for_cls, title_name, out_base_path):
    plt.figure(figsize=(8.5, 6.5))

    for i in range(len(display_names)):
        start = i * sample_for_cls
        end = start + sample_for_cls
        color = COLORS[i % len(COLORS)]

        plt.scatter(
            Y_pca2[start:end, 0],
            Y_pca2[start:end, 1],
            c=color,
            marker=MARKERS[i % len(MARKERS)],
            s=35,
            linewidths=1.0,
            label=display_names[i],
            alpha=0.8
        )

    plt.xlabel(f"PC1 ({explained[0] * 100:.2f}%)", fontsize=14)
    plt.ylabel(f"PC2 ({explained[1] * 100:.2f}%)", fontsize=14)
    plt.legend(fontsize=11, ncol=2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    save_figure_png(out_base_path)
    plt.close()


def plot_explained_variance(explained, title_name, out_base_path):
    n_plot = min(20, len(explained))
    components = np.arange(1, n_plot + 1)
    explained_plot = explained[:n_plot]
    cumulative = np.cumsum(explained_plot)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.bar(
        components,
        explained_plot,
        color="tab:blue",
        alpha=0.85,
        label="explained",
    )
    ax.plot(
        components,
        cumulative,
        color="black",
        marker="o",
        linewidth=1.8,
        label="cumulative",
    )
    for component, value in zip(components, explained_plot):
        ax.text(
            component,
            value + 0.005,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(f"PCA explained variance | {title_name}")
    ax.set_xticks(components)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_one_rule(rule_name: str, rep: int):
    npy_path = find_npy_path(rule_name, rep)

    if not npy_path.exists():
        print(f"[skip] file not found: {npy_path}")
        return

    sout_org = np.load(npy_path)
    print("loaded:", npy_path)
    print("shape :", sout_org.shape)

    if sout_org.ndim != 4:
        raise ValueError(f"shape が4次元ではありません: {sout_org.shape}")

    n_sozai, n_sample, N_out, T = sout_org.shape

    if SAMPLE_FOR_CLS > n_sample:
        raise ValueError(
            f"SAMPLE_FOR_CLS={SAMPLE_FOR_CLS} が n_sample={n_sample} を超えています"
        )

    if len(DIR_NAME) != len(DISPLAY_NAMES):
        raise ValueError("DIR_NAME と DISPLAY_NAMES の要素数が一致していません")

    if n_sozai > len(DISPLAY_NAMES):
        raise ValueError(
            f"データ中の素材数 n_sozai={n_sozai} が DISPLAY_NAMES の数 {len(DISPLAY_NAMES)} を超えています"
        )

    s_for_cls = sout_org[:, 0:SAMPLE_FOR_CLS, :, :]

    Y = build_feature_matrix(s_for_cls, T_n=T_n)
    Y_pca2, explained, eig_vectors = compute_pca_2d(Y)
    print(
        f"explained variance: PC1={explained[0] * 100:.2f}%, "
        f"PC2={explained[1] * 100:.2f}%, "
        f"PC1+PC2={np.sum(explained[:2]) * 100:.2f}%"
    )

    rule_clean = clean_rule_name(rule_name)
    out_base_path = OUT_DIR / f"{rule_clean}_rep{rep}_pca_2d"

    plot_pca_2d(
        Y_pca2=Y_pca2,
        explained=explained,
        display_names=DISPLAY_NAMES[:n_sozai],
        sample_for_cls=SAMPLE_FOR_CLS,
        title_name=rule_clean,
        out_base_path=out_base_path
    )
    plot_explained_variance(
        explained=explained,
        title_name=rule_clean,
        out_base_path=OUT_DIR / f"{rule_clean}_rep{rep}_pca_explained_variance",
    )
    plot_component_heatmaps(
        eig_vectors=eig_vectors,
        explained=explained,
        n_neurons=N_out,
        n_time_bins=T // T_n,
        out_dir=OUT_DIR / f"{rule_clean}_rep{rep}_pca_component_heatmaps",
    )

    print("saved:", out_base_path.with_suffix(".png"))
    print(
        "saved:",
        (OUT_DIR / f"{rule_clean}_rep{rep}_pca_explained_variance").with_suffix(".png"),
    )
    print(
        "saved:",
        OUT_DIR / f"{rule_clean}_rep{rep}_pca_component_heatmaps",
    )
    print("-" * 80)


def main():
    print("input dir:", SCRIPT_DIR)
    print("save dir :", OUT_DIR)
    print("=" * 80)

    rules = TARGET_DATASETS if TARGET_DATASETS else RULES

    for rule_name in rules:
        for rep in range(REP_START, REP_END + 1):
            print(f"processing: {rule_name} rep{rep}")
            run_one_rule(rule_name, rep)

    print("done")


if __name__ == "__main__":
    main()
