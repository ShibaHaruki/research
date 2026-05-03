"""run_test の sout_rec を読み、10-fold 分類評価と Excel 結果を保存する入口。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from c_configs.FIXED import cfg_run
from d_tools.compat import first_value, test_bin_steps


RUN_CFG = getattr(cfg_run, "CFG_RUN", {})
matplotlib.use(RUN_CFG.get("MATPLOTLIB_BACKEND", "Agg"))
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / RUN_CFG["RESULTS_DIR"]
TEST_RESULT_DIR = RESULTS_PATH / RUN_CFG.get("TEST_RESULT_DIR", "test_run")

CURRENT_SOUT_RE = re.compile(r"^sout_rec_rep(?P<rep>\d+)\.npy$")
LEGACY_SOUT_RE = re.compile(r"^(?P<prefix>.+)_sout_rec_rep(?P<rep>\d+)\.npy$")

DEFAULT_MATERIALS = [
    "Al_board",
    "buta_omote",
    "buta_ura",
    "cork",
    "denim",
    "rubber_board",
    "washi",
    "wood_board",
]

GROUP_LABELS_3CLS = ["smooth", "rough", "pigskin"]
GROUP_MEMBERS_3CLS = {
    "smooth": {"Al_board", "rubber_board", "wood_board"},
    "rough": {"cork", "denim", "washi"},
    "pigskin": {"buta_omote", "buta_ura"},
}

RIDGE_LAMBDA = 1e-6
RIDGE_LAMBDA_RETRY = 1e-3
NAN_FILL_VALUE = 0.0


def _dataset_id(dataset_dir: Path) -> str:
    dataset_dir = Path(dataset_dir).resolve()
    try:
        return dataset_dir.relative_to(TEST_RESULT_DIR.resolve()).as_posix()
    except ValueError:
        return dataset_dir.name


def _entry_id(dataset_dir: Path, legacy_prefix: str | None = None) -> str:
    base_id = _dataset_id(dataset_dir)
    return base_id if not legacy_prefix else f"{base_id}/{legacy_prefix}"


def discover_dataset_entries(root: Path) -> list[dict]:
    # 指定フォルダ以下から sout_rec_rep*.npy を探し、評価対象データセットとしてまとめる。
    root = Path(root)
    current_map: dict[Path, dict[int, Path]] = {}
    legacy_map: dict[tuple[Path, str], dict[int, Path]] = {}
    for fp in root.rglob("*sout_rec_rep*.npy"):
        current_match = CURRENT_SOUT_RE.match(fp.name)
        if current_match:
            rep = int(current_match.group("rep"))
            current_map.setdefault(fp.parent.resolve(), {})[rep] = fp.resolve()
            continue

        legacy_match = LEGACY_SOUT_RE.match(fp.name)
        if legacy_match:
            rep = int(legacy_match.group("rep"))
            prefix = str(legacy_match.group("prefix"))
            legacy_map.setdefault((fp.parent.resolve(), prefix), {})[rep] = fp.resolve()

    entries = []
    for dataset_dir, rep_files in current_map.items():
        entries.append(
            {
                "dataset_dir": dataset_dir,
                "dataset_id": _entry_id(dataset_dir),
                "rep_files": rep_files,
                "legacy_prefix": None,
                "output_dir": dataset_dir / "results_10fold",
            }
        )

    for (dataset_dir, prefix), rep_files in legacy_map.items():
        entries.append(
            {
                "dataset_dir": dataset_dir,
                "dataset_id": _entry_id(dataset_dir, prefix),
                "rep_files": rep_files,
                "legacy_prefix": prefix,
                "output_dir": dataset_dir / "results_10fold" / prefix,
            }
        )

    return sorted(entries, key=lambda item: item["dataset_id"])


def _looks_like_dataset_dir(path: Path) -> bool:
    return any(
        CURRENT_SOUT_RE.match(fp.name) or LEGACY_SOUT_RE.match(fp.name)
        for fp in path.glob("*sout_rec_rep*.npy")
    )


def _load_metadata(dataset_dir: Path) -> dict:
    dataset_dir = Path(dataset_dir)
    for name in ("test_config_snapshot.json", "config_snapshot.json"):
        snapshot_fp = dataset_dir / name
        if snapshot_fp.exists():
            return json.loads(snapshot_fp.read_text(encoding="utf-8"))
    return {}


def _write_used_parameters_text(out_dir: Path, metadata: dict, evaluation_params: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_metadata": metadata,
        "evaluation": evaluation_params,
    }
    out_fp = out_dir / "used_parameters.txt"
    out_fp.write_text(
        "Used Parameters\n"
        "===============\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return out_fp


def _materials_from_metadata(metadata: dict) -> list[str]:
    test_mats = metadata.get("test", {}).get("TEST_MAT")
    if test_mats:
        return list(test_mats)
    train_mats = metadata.get("training", {}).get("TRAINING_MAT")
    if train_mats:
        return list(train_mats)
    return list(DEFAULT_MATERIALS)


def _source_rule_from_metadata(metadata: dict) -> str:
    for section in ("source_models", "models"):
        value = metadata.get(section, {}).get("LEARNING_RULE_MODEL")
        if value:
            return str(value)
    return "unknown"


def _bin_width_ms_from_records(dataset_dir: Path, rep: int) -> float | None:
    trials_fp = Path(dataset_dir) / f"test_trials_rep{rep}.csv"
    if not trials_fp.exists():
        return None

    df = pd.read_csv(trials_fp, nrows=1)
    if df.empty:
        return None

    duration_ms = float(df.iloc[0].get("duration_ms", np.nan))
    n_bins = float(df.iloc[0].get("n_bins", np.nan))
    if not np.isfinite(duration_ms) or not np.isfinite(n_bins) or n_bins <= 0:
        return None
    return duration_ms / n_bins


def _bin_width_ms_from_metadata(metadata: dict) -> float | None:
    common = metadata.get("common", {})
    test = metadata.get("test", {})
    dt_s = first_value(common.get("dt_ms"))
    bin_steps = test_bin_steps(test, None)
    if dt_s is None or bin_steps is None:
        return None
    return float(dt_s) * 1000.0 * float(bin_steps)


def infer_bin_width_ms(dataset_dir: Path, rep: int, *, n_bins: int | None = None) -> float:
    metadata = _load_metadata(dataset_dir)
    value = _bin_width_ms_from_records(dataset_dir, rep)
    if value is not None:
        return float(value)
    value = _bin_width_ms_from_metadata(metadata)
    if value is not None:
        return float(value)
    if n_bins and n_bins > 0:
        legacy_duration_ms = float(RUN_CFG.get("LEGACY_TEST_DURATION_MS", 500.0))
        return legacy_duration_ms / float(n_bins)
    raise ValueError(f"Could not infer bin width for {dataset_dir} rep{rep}.")


def extract_features(sout_rec: np.ndarray, T_n_ms: int, *, bin_width_ms: float) -> np.ndarray:
    # sout_rec を T_n_ms ごとの発火率特徴に変換する。
    # 形は 素材 x サンプル x (出力ニューロン×時間区間) になる。
    n_materials, n_samples, n_out, n_bins = sout_rec.shape
    ratio = float(T_n_ms) / float(bin_width_ms)
    bins_per_window = int(round(ratio))

    if bins_per_window <= 0 or not np.isclose(ratio, bins_per_window, atol=1e-6):
        raise ValueError(
            f"T_n={T_n_ms} ms is not compatible with saved bin width {bin_width_ms:.6g} ms."
        )
    if n_bins % bins_per_window != 0:
        raise ValueError(
            f"Saved bin count {n_bins} is not divisible by {bins_per_window} "
            f"(T_n={T_n_ms} ms, bin_width_ms={bin_width_ms:.6g})."
        )

    n_interval = n_bins // bins_per_window
    reshaped = sout_rec.reshape(
        n_materials,
        n_samples,
        n_out,
        n_interval,
        bins_per_window,
    )
    spike_sum = reshaped.sum(axis=-1)
    rate = spike_sum / (float(T_n_ms) / 1000.0)
    features = rate.reshape(n_materials, n_samples, n_out * n_interval).astype(
        np.float64,
        copy=False,
    )

    if not np.isfinite(features).all():
        features = np.nan_to_num(
            features,
            nan=NAN_FILL_VALUE,
            posinf=NAN_FILL_VALUE,
            neginf=NAN_FILL_VALUE,
        )
    return features


def extract_rate_series(
    sout_rec: np.ndarray,
    T_n_ms: int,
    *,
    bin_width_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_materials, n_samples, n_out, n_bins = sout_rec.shape
    ratio = float(T_n_ms) / float(bin_width_ms)
    bins_per_window = int(round(ratio))

    if bins_per_window <= 0 or not np.isclose(ratio, bins_per_window, atol=1e-6):
        raise ValueError(
            f"T_n={T_n_ms} ms is not compatible with saved bin width {bin_width_ms:.6g} ms."
        )
    if n_bins % bins_per_window != 0:
        raise ValueError(
            f"Saved bin count {n_bins} is not divisible by {bins_per_window} "
            f"(T_n={T_n_ms} ms, bin_width_ms={bin_width_ms:.6g})."
        )

    n_interval = n_bins // bins_per_window
    reshaped = sout_rec.reshape(
        n_materials,
        n_samples,
        n_out,
        n_interval,
        bins_per_window,
    )
    spike_sum = reshaped.sum(axis=-1)
    rate_hz = spike_sum / (float(T_n_ms) / 1000.0)
    mean_rate_hz = np.mean(rate_hz, axis=(1, 2))
    time_ms = np.arange(n_interval, dtype=np.float64) * float(T_n_ms)
    return time_ms, mean_rate_hz


def save_spike_rate_trends(
    out_dir: Path,
    *,
    dataset_id: str,
    rule_name: str,
    rep: int,
    T_n_ms: int,
    materials: list[str],
    time_ms: np.ndarray,
    mean_rate_hz: np.ndarray,
) -> tuple[Path, Path, Path]:
    # 分類精度だけでなく、素材ごとの平均スパイク率の時間推移も保存する。
    trend_dir = Path(out_dir) / "spike_rate_trends"
    trend_dir.mkdir(parents=True, exist_ok=True)

    stem = f"rep{rep:02d}_Tn_{T_n_ms}ms"
    csv_fp = trend_dir / f"{stem}_mean_rate.csv"
    line_fp = trend_dir / f"{stem}_mean_rate.png"
    heatmap_fp = trend_dir / f"{stem}_mean_rate_heatmap.png"

    df = pd.DataFrame({"time_ms": time_ms})
    for material_index, material in enumerate(materials):
        df[material] = mean_rate_hz[material_index]
    df.to_csv(csv_fp, index=False)

    plt.figure(figsize=(10, 5))
    for material_index, material in enumerate(materials):
        plt.plot(time_ms, mean_rate_hz[material_index], linewidth=1.8, label=material)
    plt.xlabel("Time [ms]")
    plt.ylabel("Mean spike rate [Hz]")
    plt.title(f"{dataset_id} | {rule_name} | rep{rep:02d} | T_n={T_n_ms} ms")
    if len(materials) <= 8:
        plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(line_fp, dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(materials))))
    im = ax.imshow(mean_rate_hz, aspect="auto", origin="lower")
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Material")
    ax.set_title(f"{dataset_id} | {rule_name} | rep{rep:02d} | T_n={T_n_ms} ms")
    ax.set_yticks(np.arange(len(materials)))
    ax.set_yticklabels(materials)
    if len(time_ms) <= 25:
        ax.set_xticks(np.arange(len(time_ms)))
        ax.set_xticklabels([f"{int(t)}" for t in time_ms], rotation=45, ha="right")
        ax.set_xlabel("Time [ms]")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean spike rate [Hz]")
    fig.tight_layout()
    fig.savefig(heatmap_fp, dpi=150)
    plt.close(fig)

    return csv_fp, line_fp, heatmap_fp


def safe_pinv_cov(cov: np.ndarray, ridge: float) -> np.ndarray:
    dim = cov.shape[0]
    cov2 = cov + ridge * np.eye(dim, dtype=cov.dtype)
    if not np.isfinite(cov2).all():
        cov2 = np.nan_to_num(cov2, nan=0.0, posinf=0.0, neginf=0.0)
    return np.linalg.pinv(cov2)


def group_indices_3cls(materials: list[str]) -> list[int] | None:
    group_index_by_name = {
        material: group_index
        for group_index, group_label in enumerate(GROUP_LABELS_3CLS)
        for material in GROUP_MEMBERS_3CLS[group_label]
    }
    indices = []
    for material in materials:
        if material not in group_index_by_name:
            return None
        indices.append(group_index_by_name[material])
    return indices


def aggregate_confusion_3cls(conf_8cls: np.ndarray, materials: list[str]) -> np.ndarray | None:
    material_groups = group_indices_3cls(materials)
    if material_groups is None:
        return None

    conf_3cls = np.zeros((len(GROUP_LABELS_3CLS), len(GROUP_LABELS_3CLS)), dtype=float)
    for true_index, true_group in enumerate(material_groups):
        for pred_index, pred_group in enumerate(material_groups):
            conf_3cls[true_group, pred_group] += conf_8cls[true_index, pred_index]
    return conf_3cls


def eval_10fold(features: np.ndarray, rng: np.random.Generator, n_folds: int, materials: list[str]):
    # Mahalanobis 距離で 10-fold 分類を行い、8素材分類と3グループ分類の結果を返す。
    n_materials, n_samples, dim = features.shape
    if n_samples < n_folds:
        raise ValueError(
            f"Need at least {n_folds} samples per class, but n_sample={n_samples}."
        )

    all_indices = np.arange(n_samples)
    rng.shuffle(all_indices)
    fold_indices = np.array_split(all_indices, n_folds)

    conf_8_total = np.zeros((n_materials, n_materials), dtype=float)
    conf_3_total = np.zeros((len(GROUP_LABELS_3CLS), len(GROUP_LABELS_3CLS)), dtype=float)
    has_group_map = group_indices_3cls(materials) is not None
    acc_list_8 = []
    acc_list_3 = []

    for fold in range(n_folds):
        test_idx = np.asarray(fold_indices[fold], dtype=int)
        train_idx = np.setdiff1d(all_indices, test_idx)

        class_mean = np.zeros((n_materials, dim), dtype=np.float64)
        class_cov_inv: list[np.ndarray] = []

        for class_index in range(n_materials):
            train_data = features[class_index, train_idx, :]
            if not np.isfinite(train_data).all():
                train_data = np.nan_to_num(
                    train_data,
                    nan=NAN_FILL_VALUE,
                    posinf=NAN_FILL_VALUE,
                    neginf=NAN_FILL_VALUE,
                )

            class_mean[class_index, :] = np.mean(train_data, axis=0)
            cov = np.cov(train_data, rowvar=False)
            if np.ndim(cov) == 0:
                cov = np.array([[float(cov)]], dtype=np.float64)
            if not np.isfinite(cov).all():
                cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

            try:
                class_cov_inv.append(safe_pinv_cov(cov, RIDGE_LAMBDA))
            except np.linalg.LinAlgError:
                class_cov_inv.append(safe_pinv_cov(cov, RIDGE_LAMBDA_RETRY))

        conf_8_fold = np.zeros((n_materials, n_materials), dtype=float)
        for true_class in range(n_materials):
            for sample_index in test_idx:
                x = features[true_class, sample_index, :]
                if not np.isfinite(x).all():
                    x = np.nan_to_num(
                        x,
                        nan=NAN_FILL_VALUE,
                        posinf=NAN_FILL_VALUE,
                        neginf=NAN_FILL_VALUE,
                    )

                distances = np.zeros(n_materials, dtype=np.float64)
                for class_index in range(n_materials):
                    diff = x - class_mean[class_index, :]
                    val = (diff.reshape(1, -1) @ class_cov_inv[class_index] @ diff.reshape(-1, 1)).item()
                    distances[class_index] = np.sqrt(max(val, 0.0))

                pred_class = int(np.argmin(distances))
                conf_8_fold[true_class, pred_class] += 1.0

        total_samples_fold = float(np.sum(conf_8_fold))
        acc_list_8.append(float(np.trace(conf_8_fold) / total_samples_fold))
        conf_8_total += conf_8_fold

        if has_group_map:
            conf_3_fold = aggregate_confusion_3cls(conf_8_fold, materials)
            if conf_3_fold is not None:
                correct_3_fold = float(np.trace(conf_3_fold))
                acc_list_3.append(correct_3_fold / total_samples_fold)
                conf_3_total += conf_3_fold

    total_samples = int(np.sum(conf_8_total))
    acc_8_overall = float(np.trace(conf_8_total) / total_samples)
    acc_8_mean = float(np.mean(acc_list_8))

    if has_group_map and total_samples > 0 and acc_list_3:
        acc_3_overall = float(np.trace(conf_3_total) / total_samples)
        acc_3_mean = float(np.mean(acc_list_3))
    else:
        conf_3_total = None
        acc_3_overall = np.nan
        acc_3_mean = np.nan

    return (
        conf_8_total,
        conf_3_total,
        acc_8_overall,
        acc_8_mean,
        acc_3_overall,
        acc_3_mean,
        total_samples,
    )


def save_excel(
    out_dir: Path,
    *,
    dataset_id: str,
    rule_name: str,
    rep: int,
    T_n_ms: int,
    materials: list[str],
    conf_8cls: np.ndarray,
    conf_3cls: np.ndarray | None,
    acc_8_overall: float,
    acc_8_mean: float,
    acc_3_overall: float,
    acc_3_mean: float,
    n_sample_per_class: int,
    total_samples: int,
    bin_width_ms: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    excel_fp = out_dir / f"rep{rep:02d}_Tn_{T_n_ms}ms_10fold_conf_matrices.xlsx"

    with pd.ExcelWriter(excel_fp, engine="openpyxl") as writer:
        pd.DataFrame(conf_8cls, index=materials, columns=materials).to_excel(
            writer,
            sheet_name="conf_8cls",
        )
        if conf_3cls is not None:
            pd.DataFrame(conf_3cls, index=GROUP_LABELS_3CLS, columns=GROUP_LABELS_3CLS).to_excel(
                writer,
                sheet_name="conf_3cls",
            )

        pd.DataFrame(
            {
                "dataset_id": [dataset_id],
                "rule_name": [rule_name],
                "rep": [rep],
                "T_n_ms": [T_n_ms],
                "saved_bin_width_ms": [bin_width_ms],
                "accuracy8_overall": [acc_8_overall],
                "accuracy8_mean": [acc_8_mean],
                "accuracy3_overall": [acc_3_overall],
                "accuracy3_mean": [acc_3_mean],
                "n_sample_per_class": [n_sample_per_class],
                "total_samples": [total_samples],
            }
        ).to_excel(writer, sheet_name="accuracy", index=False)

    return excel_fp


def _coerce_dataset_entry(dataset_source: Path | dict) -> dict:
    if isinstance(dataset_source, dict):
        return dataset_source

    dataset_dir = Path(dataset_source).resolve()
    entries = discover_dataset_entries(dataset_dir)
    direct_entries = [entry for entry in entries if entry["dataset_dir"] == dataset_dir]
    if not direct_entries:
        raise FileNotFoundError(f"No sout_rec data found in {dataset_dir}")
    if len(direct_entries) == 1:
        return direct_entries[0]

    current_entries = [entry for entry in direct_entries if entry["legacy_prefix"] is None]
    if len(current_entries) == 1:
        return current_entries[0]

    raise ValueError(
        f"Multiple legacy dataset groups found in {dataset_dir}. "
        "Use auto-discovery from the parent directory or point to one current-format dataset dir."
    )


def classify_dataset(
    dataset_source: Path | dict,
    *,
    rep_start: int,
    rep_end: int,
    n_folds: int,
    T_n_list: list[int],
    base_seed: int,
) -> list[dict]:
    # 1データセット内の rep と T_n を順番に評価し、Excel・CSV・スパイク率グラフを保存する。
    entry = _coerce_dataset_entry(dataset_source)
    dataset_dir = Path(entry["dataset_dir"]).resolve()
    rep_files = dict(entry["rep_files"])
    if not rep_files:
        raise FileNotFoundError(f"No sout_rec data found in {dataset_dir}")

    dataset_id = str(entry["dataset_id"])
    metadata = _load_metadata(dataset_dir)
    materials = _materials_from_metadata(metadata)
    rule_name = (
        str(entry["legacy_prefix"])
        if entry.get("legacy_prefix")
        else _source_rule_from_metadata(metadata)
    )
    out_dir = Path(entry["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_params = {
        "dataset_id": dataset_id,
        "dataset_dir": str(dataset_dir),
        "rule_name": rule_name,
        "rep_start": rep_start,
        "rep_end": rep_end,
        "n_folds": n_folds,
        "T_n_list": T_n_list,
        "base_seed": base_seed,
    }
    _write_used_parameters_text(out_dir, metadata, eval_params)

    summary_rows = []
    for rep in range(rep_start, rep_end + 1):
        fp = rep_files.get(rep)
        if fp is None:
            print(f"[SKIP] {dataset_id} rep{rep:02d}: file not found")
            continue

        sout_rec = np.load(fp)
        if sout_rec.ndim != 4:
            print(f"[SKIP] {dataset_id} rep{rep:02d}: invalid shape {tuple(sout_rec.shape)}")
            continue

        n_materials, n_sample, n_out, n_bins = sout_rec.shape
        if n_sample < n_folds:
            raise ValueError(
                f"10-fold needs n_sample >= {n_folds}, but got {n_sample} in {fp.name}"
            )

        if len(materials) != n_materials:
            raise ValueError(
                f"Material count mismatch in {dataset_id}: metadata has {len(materials)}, "
                f"sout_rec has {n_materials}."
            )

        bin_width_ms = infer_bin_width_ms(dataset_dir, rep, n_bins=n_bins)
        rng = np.random.default_rng(base_seed + rep)

        print("=" * 72)
        print(
            f"[DATASET {dataset_id}] [RULE {rule_name}] [REP {rep:02d}] "
            f"file={fp.name} shape={tuple(sout_rec.shape)} bin_width_ms={bin_width_ms:.6g}"
        )
        print("=" * 72)

        for T_n_ms in T_n_list:
            features = extract_features(sout_rec, T_n_ms, bin_width_ms=bin_width_ms)
            time_ms, mean_rate_hz = extract_rate_series(
                sout_rec,
                T_n_ms,
                bin_width_ms=bin_width_ms,
            )
            n_interval = features.shape[-1] // n_out if n_out > 0 else 0
            print(
                f"[REP {rep:02d}] T_n={T_n_ms} ms "
                f"n_interval={n_interval} dim={features.shape[-1]}"
            )

            (
                conf_8cls,
                conf_3cls,
                acc_8_overall,
                acc_8_mean,
                acc_3_overall,
                acc_3_mean,
                total_samples,
            ) = eval_10fold(features, rng, n_folds, materials)

            print("Accuracy 8cls total:", acc_8_overall, "fold-mean:", acc_8_mean)
            print("Accuracy 3cls total:", acc_3_overall, "fold-mean:", acc_3_mean)

            excel_fp = save_excel(
                out_dir,
                dataset_id=dataset_id,
                rule_name=rule_name,
                rep=rep,
                T_n_ms=T_n_ms,
                materials=materials,
                conf_8cls=conf_8cls,
                conf_3cls=conf_3cls,
                acc_8_overall=acc_8_overall,
                acc_8_mean=acc_8_mean,
                acc_3_overall=acc_3_overall,
                acc_3_mean=acc_3_mean,
                n_sample_per_class=n_sample,
                total_samples=total_samples,
                bin_width_ms=bin_width_ms,
            )
            rate_csv_fp, rate_plot_fp, rate_heatmap_fp = save_spike_rate_trends(
                out_dir,
                dataset_id=dataset_id,
                rule_name=rule_name,
                rep=rep,
                T_n_ms=T_n_ms,
                materials=materials,
                time_ms=time_ms,
                mean_rate_hz=mean_rate_hz,
            )
            _write_used_parameters_text(out_dir / "spike_rate_trends", metadata, eval_params)
            print("Saved Excel:", excel_fp)
            print("Saved spike-rate trend:", rate_plot_fp)

            summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_dir": str(dataset_dir),
                    "rule_name": rule_name,
                    "rep": rep,
                    "T_n_ms": T_n_ms,
                    "saved_bin_width_ms": bin_width_ms,
                    "acc8_overall": acc_8_overall,
                    "acc8_mean": acc_8_mean,
                    "acc3_overall": acc_3_overall,
                    "acc3_mean": acc_3_mean,
                    "n_sample_per_class": n_sample,
                    "n_out": n_out,
                    "saved_n_bins": n_bins,
                    "total_samples": total_samples,
                    "file": fp.name,
                    "excel_file": excel_fp.name,
                    "rate_csv_file": rate_csv_fp.name,
                    "rate_plot_file": rate_plot_fp.name,
                    "rate_heatmap_file": rate_heatmap_fp.name,
                }
            )

    if summary_rows:
        dataset_summary = pd.DataFrame(summary_rows)
        summary_xlsx = out_dir / "summary.xlsx"
        summary_csv = out_dir / "summary.csv"
        dataset_summary.to_csv(summary_csv, index=False)
        with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
            dataset_summary.to_excel(writer, sheet_name="summary", index=False)
        print(f"[saved] {summary_xlsx}")

    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify saved test sout_rec outputs with 10-fold Mahalanobis evaluation."
    )
    parser.add_argument(
        "--test-dir",
        action="append",
        default=None,
        help=(
            "Test result directory containing sout_rec_rep*.npy. "
            "Can be given multiple times. If omitted, search all under test_run."
        ),
    )
    parser.add_argument("--rep-start", type=int, default=1)
    parser.add_argument("--rep-end", type=int, default=10)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument(
        "--tn-ms",
        type=int,
        nargs="+",
        default=[25],
        help="Feature aggregation window(s) in ms. Default: 25",
    )
    parser.add_argument(
        "--dataset-filter",
        action="append",
        default=None,
        help="Keep only datasets whose relative path contains this text. Can be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # --test-dir がなければ test_run 全体からデータセットを自動検出する。
    if args.test_dir:
        candidate_dirs = [Path(path).resolve() for path in args.test_dir]
    else:
        candidate_dirs = [TEST_RESULT_DIR.resolve()]

    dataset_entries: list[dict] = []
    for candidate in candidate_dirs:
        if _looks_like_dataset_dir(candidate):
            dataset_entries.extend(discover_dataset_entries(candidate))
        else:
            dataset_entries.extend(discover_dataset_entries(candidate))

    if args.dataset_filter:
        keep_texts = [text for text in args.dataset_filter if text]
        dataset_entries = [
            entry
            for entry in dataset_entries
            if any(text in str(entry["dataset_id"]) for text in keep_texts)
        ]

    selected_entries = sorted(dataset_entries, key=lambda item: item["dataset_id"])
    print("[TEST_RESULT_DIR]", TEST_RESULT_DIR.resolve())
    print("[DISCOVERED DATASETS]", len(selected_entries))
    for entry in selected_entries:
        reps = sorted(entry["rep_files"])
        print(f"  - {entry['dataset_id']} reps={reps}")

    if not selected_entries:
        print("No datasets found.")
        return 0

    all_rows = []
    for entry in selected_entries:
        all_rows.extend(
            classify_dataset(
                entry,
                rep_start=args.rep_start,
                rep_end=args.rep_end,
                n_folds=args.folds,
                T_n_list=list(args.tn_ms),
                base_seed=args.base_seed,
            )
        )

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        summary_csv = TEST_RESULT_DIR / "results_10fold_summary.csv"
        summary_xlsx = TEST_RESULT_DIR / "results_10fold_summary.xlsx"
        df_all.to_csv(summary_csv, index=False)
        with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="summary", index=False)
        print(f"[saved] {summary_xlsx}")

    print("All finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
