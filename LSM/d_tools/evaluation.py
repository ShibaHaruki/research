"""分類指標の集計と、保存済み出力発火記録の交差検証・結果保存。"""

from __future__ import annotations
from d_tools.recording_io import read_record
from a_experiment.experiment_sets import TEXTURE_GROUPS
from d_tools.progress import progress_items

import numpy as np
from sklearn.metrics import confusion_matrix


def confusion_percentages(counts) -> np.ndarray:
    """Normalize each true-class row to percentages; empty rows remain zero."""
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1] or counts.size == 0:
        raise ValueError("Expected a nonempty square confusion matrix")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("Confusion counts must be finite and nonnegative")
    totals = counts.sum(1, keepdims=True)
    return (
        np.divide(counts, totals, out=np.zeros(counts.shape, dtype=float), where=totals != 0) * 100
    )


def evaluate_classification(true_labels, predicted_labels, *, labels=None) -> dict:
    """Return overall/per-class accuracy (0..1), counts and row percentages."""
    true = np.asarray(true_labels)
    predicted = np.asarray(predicted_labels)
    if true.ndim != 1 or predicted.shape != true.shape or not len(true):
        raise ValueError("Expected equal-length, nonempty 1D true and predicted labels")
    classes = np.unique(np.concatenate([true, predicted])) if labels is None else np.asarray(labels)
    if classes.ndim != 1 or not len(classes) or len(np.unique(classes)) != len(classes):
        raise ValueError("labels must be a nonempty sequence of unique classes")
    if not np.isin(true, classes).all() or not np.isin(predicted, classes).all():
        raise ValueError("labels must include every true and predicted class")
    cm = confusion_matrix(true, predicted, labels=classes)
    percentages = confusion_percentages(cm)
    return {
        "accuracy": float(np.mean(predicted == true)),
        "labels": classes,
        "confusion_matrix": cm,
        "confusion_percent": percentages,
        "per_class_accuracy": np.diag(percentages) / 100,
        "class_support": cm.sum(1),
    }


def extract_features(sout_rec, T_n_ms, *, bin_width_ms):
    """素材×試行×出力ニューロン×時間ビンの発火回数を、指定時間幅の発火率へ変換する。"""
    values = np.asarray(sout_rec)
    if values.ndim != 4 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(
            "Expected finite, nonnegative material x trial x output x bin spike counts"
        )
    ratio = float(T_n_ms) / float(bin_width_ms)
    step = round(ratio)
    if step < 1 or not np.isclose(step, ratio) or values.shape[-1] % step:
        raise ValueError(
            "Aggregation window must be a multiple of recording bin and divide the trial duration"
        )
    grouped = values.reshape(*values.shape[:-1], values.shape[-1] // step, step).sum(axis=-1)
    return (grouped / (float(T_n_ms) / 1000.0)).reshape(values.shape[0], values.shape[1], -1)


def texture_confusion(counts, materials):
    """8素材予測を従来の触感3種類に集約する。3種類で分類器を再学習する処理ではない。"""

    lookup = {material: i for i, group in enumerate(TEXTURE_GROUPS.values()) for material in group}
    unknown = set(materials) - lookup.keys()
    if unknown:
        raise ValueError(f"No texture group defined for materials: {sorted(unknown)}")
    groups = [lookup[material] for material in materials]
    result = np.zeros((3, 3), dtype=np.int64)
    for i, actual in enumerate(groups):
        for j, predicted in enumerate(groups):
            result[actual, predicted] += counts[i, j]
    return result


def classify_dataset(
    dataset_source,
    *,
    rep_start=1,
    rep_end=None,
    n_folds=10,
    T_n_list=None,
    base_seed=1,
    thread_limit=4,
):
    """保存済みLiquidまたは出力記録を標準化なしで分類し、正答率・混同行列を保存する。

    出力層の可塑性学習には使わなかった記録内で、分類器の層別交差検証を行う。
    ネットワークの実行や重みの更新は行わない。
    """
    # 基本の分類指標だけを使う場合は、記録読込・保存用の依存を読み込まない。
    import re
    from pathlib import Path
    from sklearn.model_selection import StratifiedKFold
    from threadpoolctl import threadpool_limits
    from d_tools.classification import classify
    from d_tools.recording_io import decode

    folder = Path(
        dataset_source["dataset_dir"] if isinstance(dataset_source, dict) else dataset_source
    )
    metadata = decode(read_record(folder / "test_config_snapshot.json"))
    materials = list(
        metadata["materials"] if "materials" in metadata else metadata["test"]["TEST_MAT"]
    )
    bin_width = float(metadata["bin_width_ms"])
    # Liquid単独の記録では出力層用のダミーファイルを作らず、Liquid自身を読む。
    population = metadata.get("classification_population", "readout")
    prefix = "sliq" if population == "liquid" else "sout"
    # どの学習則・層でも1 ms記録から評価時だけ集約する。旧集約済み記録も読める。
    fine_folder = folder / "rec" / ("Liquid" if population == "liquid" else "readout")
    fine_files = sorted(fine_folder.glob(f"{prefix}_rec_rep*.npy"))
    if fine_files:
        files = fine_files
        bin_width = float(metadata["recording_bin_width_ms"])
    else:
        files = sorted(folder.glob(f"{prefix}_rec_rep*.npy"))
    selected = []
    for file in files:
        match = re.fullmatch(rf"{prefix}_rec_rep(\d+)\.npy", file.name)
        if match and int(match[1]) >= rep_start and (rep_end is None or int(match[1]) <= rep_end):
            selected.append((int(match[1]), file))
    if not selected:
        raise FileNotFoundError(f"No test recordings selected in {folder}")
    from a_experiment.paths import evaluation_directory

    output = evaluation_directory(folder, cfg=metadata)
    summaries = []
    all_details = []
    texture_summaries = []
    texture_details = []
    matrices = []
    texture_matrices = []

    # 計算を始める前に素材と3種類の対応を検証する。
    texture_confusion(np.zeros((len(materials), len(materials)), dtype=int), materials)
    with threadpool_limits(limits=thread_limit):
        for rep, file in selected:
            counts = np.load(file)
            if counts.ndim != 4 or counts.shape[0] != len(materials):
                raise ValueError("Material count or record shape mismatch")
            if "evaluation_neuron_ids" in metadata:
                # 記録はE/I全体を保持し、分類器へ渡すニューロンだけを選ぶ。
                ids = np.asarray(metadata["evaluation_neuron_ids"])
                if (
                    ids.ndim != 1
                    or not len(ids)
                    or not np.issubdtype(ids.dtype, np.integer)
                    or len(np.unique(ids)) != len(ids)
                    or np.any(ids < 0)
                    or np.any(ids >= counts.shape[2])
                ):
                    raise ValueError("Invalid evaluation neuron IDs")
                counts = counts[:, :, ids, :]
            n_samples = counts.shape[1]
            if (
                n_folds < 2
                or n_samples < n_folds
                or n_samples - (n_samples + n_folds - 1) // n_folds < 2
            ):
                raise ValueError(
                    "Insufficient trials: each fold needs at least 2 classifier-training trials per class"
                )
            labels = np.repeat(np.arange(len(materials)), n_samples)
            for window in T_n_list or [25]:
                features = extract_features(counts, window, bin_width_ms=bin_width).reshape(
                    len(labels), -1
                )
                cm = np.zeros((len(materials), len(materials)), np.int64)
                fold_rows = []
                texture_fold_rows = []
                cv = StratifiedKFold(
                    n_splits=n_folds, shuffle=True, random_state=base_seed + rep - 1
                )
                for fold, (train, test) in progress_items(
                    enumerate(cv.split(features, labels), 1),
                    total=n_folds,
                    description=f"Evaluation | {window:g} ms",
                    rep=rep,
                    unit="fold",
                    label=lambda item: f"fold {item[0]}",
                ):
                    active = np.any(features[train] != 0, axis=0)
                    if not active.any():
                        raise ValueError(
                            "No output spikes in classifier training trials; adjust output gain"
                        )
                    predicted = classify(
                        features[train][:, active], labels[train], features[test][:, active]
                    )
                    metrics = evaluate_classification(
                        labels[test], predicted, labels=np.arange(len(materials))
                    )
                    cm += metrics["confusion_matrix"]
                    row = {
                        "rep": rep,
                        "T_n_ms": window,
                        "fold": fold,
                        "accuracy": metrics["accuracy"],
                        "n_train": len(train),
                        "n_test": len(test),
                    }
                    fold_rows.append(row)
                    all_details.append(row)
                    group_cm = texture_confusion(metrics["confusion_matrix"], materials)
                    group_row = {**row, "accuracy": float(np.trace(group_cm) / group_cm.sum())}
                    texture_fold_rows.append(group_row)
                    texture_details.append(group_row)
                # 各テスト試行は、分類CVの評価側へ1回ずつ現れる。
                assert int(cm.sum()) == len(labels)
                summary = {
                    "rep": rep,
                    "T_n_ms": window,
                    "folds": n_folds,
                    "accuracy": float(np.trace(cm) / cm.sum()),
                    "mean_fold_accuracy": float(np.mean([r["accuracy"] for r in fold_rows])),
                    "std_fold_accuracy": float(np.std([r["accuracy"] for r in fold_rows], ddof=1)),
                }
                summaries.append(summary)
                cm3 = texture_confusion(cm, materials)
                texture_summary = {
                    **summary,
                    "accuracy": float(np.trace(cm3) / cm3.sum()),
                    "mean_fold_accuracy": float(
                        np.mean([r["accuracy"] for r in texture_fold_rows])
                    ),
                    "std_fold_accuracy": float(
                        np.std([r["accuracy"] for r in texture_fold_rows], ddof=1)
                    ),
                }
                texture_summaries.append(texture_summary)
                matrices.append({"rep": rep, "T_n_ms": window, "counts": cm.tolist()})
                texture_matrices.append({"rep": rep, "T_n_ms": window, "counts": cm3.tolist()})
                print(
                    f'[classification] rep{rep}, {window:g} ms: {summary["accuracy"]:.2%}',
                    flush=True,
                )
    from d_tools.evaluation_excel import save_classification_workbook

    save_classification_workbook(
        output,
        materials=materials,
        summaries=summaries,
        details=all_details,
        matrices=matrices,
        texture_summaries=texture_summaries,
        texture_details=texture_details,
        texture_matrices=texture_matrices,
        source=folder,
        folds=n_folds,
        base_seed=base_seed,
    )
    return summaries
