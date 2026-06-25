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

REP = 3
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
    return SCRIPT_DIR / f"{rule_name}_sout_rec_rep{rep}.npy"


def save_figure_pdf(base_path: Path):
    plt.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")


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
    eig_vectors = eig_vectors[:, order]

    eig_vectors_2d = eig_vectors[:, :2].real
    Y_pca2 = np.dot(Y_std.T, eig_vectors_2d)

    return Y_pca2


def plot_pca_2d(Y_pca2, display_names, sample_for_cls, title_name, out_base_path):
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

    plt.xlabel("PC1", fontsize=14)
    plt.ylabel("PC2", fontsize=14)
    plt.title(title_name, fontsize=18)
    plt.legend(fontsize=11, ncol=2)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    save_figure_pdf(out_base_path)
    plt.close()


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
    Y_pca2 = compute_pca_2d(Y)

    rule_clean = clean_rule_name(rule_name)
    out_base_path = OUT_DIR / f"{rule_clean}_rep{rep}_pca_2d"

    plot_pca_2d(
        Y_pca2=Y_pca2,
        display_names=DISPLAY_NAMES[:n_sozai],
        sample_for_cls=SAMPLE_FOR_CLS,
        title_name=rule_clean,
        out_base_path=out_base_path
    )

    print("saved:", out_base_path.with_suffix(".pdf"))
    print("-" * 80)


def main():
    print("input dir:", SCRIPT_DIR)
    print("save dir :", OUT_DIR)
    print("=" * 80)

    rules = TARGET_DATASETS if TARGET_DATASETS else RULES
    rep = 1 if TARGET_DATASETS else REP

    for rule_name in rules:
        print(f"processing: {rule_name}")
        run_one_rule(rule_name, rep)

    print("done")


if __name__ == "__main__":
    main()
