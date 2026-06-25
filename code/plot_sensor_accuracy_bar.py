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
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]


def load_accuracy_results(result_dir: Path, accuracy_t_n: int, n_sample: int) -> pd.DataFrame:
    summary_path = result_dir / f"liquid_accuracy_summary_Tn{accuracy_t_n}_n{n_sample}.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
    else:
        rows = []
        for sensor_mode in SENSOR_ORDER:
            path = result_dir / sensor_mode / f"liquid_accuracy_Tn{accuracy_t_n}_n{n_sample}.csv"
            if path.exists():
                rows.append(pd.read_csv(path))
        if not rows:
            raise FileNotFoundError(
                f"No accuracy CSV found under {result_dir} for Tn={accuracy_t_n}, n={n_sample}"
            )
        df = pd.concat(rows, ignore_index=True)

    df = df[df["sensor_mode"].isin(SENSOR_ORDER)].copy()
    df["sensor_mode"] = pd.Categorical(df["sensor_mode"], categories=SENSOR_ORDER, ordered=True)
    return df.sort_values("sensor_mode")


def plot_accuracy_bar(df: pd.DataFrame, result_dir: Path, accuracy_t_n: int, n_sample: int):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(df))
    ax.bar(
        x,
        df["acc_mean"],
        yerr=df["acc_std"],
        color=COLORS[:len(df)],
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([SENSOR_LABELS[str(mode)] for mode in df["sensor_mode"]])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Liquid-layer classification accuracy | T_n={accuracy_t_n} | n={n_sample}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_path = result_dir / f"liquid_accuracy_summary_bar_Tn{accuracy_t_n}_n{n_sample}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--accuracy-t-n", type=int, default=500)
    parser.add_argument("--n-sample", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_accuracy_results(args.result_dir, args.accuracy_t_n, args.n_sample)
    print(df[["sensor_mode", "acc_mean", "acc_std"]].to_string(index=False))
    plot_accuracy_bar(df, args.result_dir, args.accuracy_t_n, args.n_sample)


if __name__ == "__main__":
    main()
