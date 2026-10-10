"""Visualize held-out predictions with PCA fitted only on training features."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .pca import fit_pca


def save_heldout_plots(train_features, test_features, materials, train_files,
                       test_files, confusion, out_dir: Path, t_n_ms: float) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_x = train_features.reshape(-1, train_features.shape[-1])
    test_x = test_features.reshape(-1, test_features.shape[-1])
    pca = fit_pca(train_x, n_components=3, standardize=True)
    test_scores = ((test_x - pca["mean"]) / pca["scale"]) @ pca["components"].T
    ratios = pca["explained_variance_ratio"]
    np.savez_compressed(out_dir / "pca_model_train_only.npz",
                        mean=pca["mean"], scale=pca["scale"],
                        components=pca["components"], explained_variance_ratio=ratios)
    score_rows = []
    for dataset, scores, files in (("train", pca["scores"], train_files),
                                   ("test", test_scores, test_files)):
        index = 0
        for material, class_files in zip(materials, files):
            for fp in class_files:
                score_rows.append({"dataset": dataset, "material": material, "file": fp,
                                   **{f"PC{i+1}": float(v) for i, v in enumerate(scores[index])}})
                index += 1
    pd.DataFrame(score_rows).to_csv(out_dir / "pca_scores.csv", index=False)
    pd.DataFrame({"component": np.arange(1, len(ratios)+1),
                  "explained_variance_ratio": ratios,
                  "cumulative_ratio": np.cumsum(ratios)}).to_csv(
                      out_dir / "pca_explained_variance.csv", index=False)
    colors = plt.get_cmap("tab10")(np.arange(len(materials)))
    labels_train = np.repeat(np.arange(len(materials)), train_features.shape[1])
    labels_test = np.repeat(np.arange(len(materials)), test_features.shape[1])
    def pc_label(i):
        return f"PC{i+1} ({ratios[i]*100:.1f}% of training variance)"

    if len(ratios) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
        for ax, scores, labels, title in (
            (axes[0], pca["scores"], labels_train, "Search data (training)"),
            (axes[1], test_scores, labels_test, "Unused trials (test)"),
        ):
            for i, material in enumerate(materials):
                points = scores[labels == i]
                ax.scatter(points[:, 0], points[:, 1], color=colors[i], s=25,
                           alpha=0.7, label=material)
            ax.set(title=title, xlabel=pc_label(0), ylabel=pc_label(1))
            ax.grid(alpha=0.2)
        axes[1].legend(fontsize=9)
        fig.suptitle("PCA fitted on search data; same axes for unused trials")
        fig.tight_layout()
        fig.savefig(out_dir / "pca_train_test_2d.png", dpi=160)
        plt.close(fig)
    if len(ratios) >= 3:
        fig = plt.figure(figsize=(15, 7))
        axes = [fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")]
        combined = np.vstack([pca["scores"], test_scores])
        for ax, scores, labels, title in (
            (axes[0], pca["scores"], labels_train, "Search data (training)"),
            (axes[1], test_scores, labels_test, "Unused trials (test)"),
        ):
            for i, material in enumerate(materials):
                points = scores[labels == i]
                ax.scatter(*points[:, :3].T, color=colors[i], s=20, alpha=0.7, label=material)
            for i, setter in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
                lo, hi = combined[:, i].min(), combined[:, i].max()
                pad = max((hi-lo)*0.05, 0.1)
                setter(lo-pad, hi+pad)
            ax.set(title=title, xlabel="PC1", ylabel="PC2", zlabel="PC3")
        axes[1].legend(fontsize=8)
        fig.suptitle("PCA fitted on search data (3D)")
        fig.tight_layout()
        fig.savefig(out_dir / "pca_train_test_3d.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(1, len(ratios)+1), ratios, label="Explained")
    ax.plot(np.arange(1, len(ratios)+1), np.cumsum(ratios), "o-", label="Cumulative")
    ax.set(xlabel="Principal component", ylabel="Training variance ratio", ylim=(0, 1.05))
    ax.set_xticks(np.arange(1, len(ratios)+1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pca_explained_variance.png", dpi=160)
    plt.close(fig)

    normalized = confusion / confusion.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Fraction of true class")
    ax.set_xticks(range(len(materials)), materials, rotation=45, ha="right")
    ax.set_yticks(range(len(materials)), materials)
    for i in range(len(materials)):
        for j in range(len(materials)):
            ax.text(j, i, f"{int(confusion[i,j])}\n{normalized[i,j]:.0%}", ha="center",
                    va="center", color="white" if normalized[i,j] > 0.5 else "black")
    ax.set(xlabel="Predicted material", ylabel="True material", title="Unused-trial confusion matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion8.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    accuracies = np.diag(normalized)
    ax.bar(materials, accuracies)
    for i, value in enumerate(accuracies):
        ax.text(i, value+0.02, f"{value:.1%}", ha="center")
    ax.set(ylim=(0, 1.12), ylabel="Accuracy", title="Unused-trial accuracy by material")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_material.png", dpi=160)
    plt.close(fig)
    summary = {"fit_dataset": "search data only", "feature_type": "classification rate features",
               "feature_window_ms": t_n_ms, "standardize": True,
               "explained_variance_ratio": ratios.tolist(),
               "train_samples": len(train_x), "test_samples": len(test_x),
               "note": "PCA is for visualization; the readout uses the full features."}
    (out_dir / "pca_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
