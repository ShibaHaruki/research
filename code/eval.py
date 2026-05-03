# -*- coding: utf-8 -*-
import sys
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================
# 0) ユーザー設定（ここだけ触ればOK）
# =========================================================

# 例：学習則ホワイトリスト（完全一致 or prefix一致）
RULE_WHITELIST = [
    "STDP_1",
    "SRDP_1",
    "off_1",
    "T_STDP_1",
]

# rep 範囲
REP_START = 1
REP_END   = 10

# 10-fold
N_FOLDS = 10

# Tn
Tn_list = [25]
if len(sys.argv) >= 2:
    Tn_list = [int(sys.argv[1])]

BASE_SEED = 1

DIR_NAME = [
    "Al_board", "buta_omote", "buta_ura", "cork",
    "denim", "rubber_board", "washi", "wood_board"
]

# ---- 数値安定化パラメータ ----
RIDGE_LAMBDA = 1e-6
RIDGE_LAMBDA_RETRY = 1e-3
NAN_FILL_VALUE = 0.0

# =========================================================
# 1) パス設定
# =========================================================
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

DATA_DIR = SCRIPT_DIR
print("[DATA_DIR]", DATA_DIR.resolve())

OUT_ROOT = DATA_DIR / "results_10fold"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

REX = re.compile(r"^(?P<rule>.+)_sout_rec_rep(?P<rep>\d+)\.npy$")

def discover_sout_files(data_dir: Path):
    files = [Path(p) for p in glob.glob(str(data_dir / "*_sout_rec_rep*.npy"))]
    print(f"[FOUND sout_rec FILES] {len(files)}")

    mapping = {}  # dataset_name(rule文字列) -> {rep:int -> Path}

    for f in files:
        m = REX.match(f.name)
        if not m:
            continue

        rule = m.group("rule")
        rep = int(m.group("rep"))

        # rep範囲フィルタ
        if not (REP_START <= rep <= REP_END):
            continue

        # シャッフル系は除外
        if "_shuf_" in rule:
            continue

        mapping.setdefault(rule, {})
        mapping[rule][rep] = f

    print(f"[DISCOVERED DATASETS] {len(mapping)}")
    for name in sorted(mapping.keys())[:30]:
        reps = sorted(mapping[name].keys())
        print(f"  - {name} reps={reps}")

    return mapping

def should_run(dataset_name: str) -> bool:
    
    if not RULE_WHITELIST:
        return True

    rule = dataset_name
    rule_head = rule.split("_")[0]

    for w in RULE_WHITELIST:
        if rule == w or rule_head == w:
            return True
        if rule.startswith(w + "_"):
            return True

    return False

# =========================================================
# 3) 特徴抽出
# =========================================================
def extract_features(sout_rec: np.ndarray, T_n: int) -> np.ndarray:
    n_sozai, n_sample, N_out, T = sout_rec.shape

    if T % T_n != 0:
        raise ValueError(f"T={T} is not divisible by T_n={T_n}")

    n_interval = T // T_n
    X = sout_rec.reshape(n_sozai, n_sample, N_out, n_interval, T_n)
    spike_sum = X.sum(axis=-1)
    rate = spike_sum / (T_n / 1000.0)

    features = rate.reshape(
        n_sozai, n_sample, N_out * n_interval
    ).astype(np.float64, copy=False)

    if not np.isfinite(features).all():
        features = np.nan_to_num(
            features,
            nan=NAN_FILL_VALUE,
            posinf=NAN_FILL_VALUE,
            neginf=NAN_FILL_VALUE
        )

    return features

# =========================================================
# 4) 10-fold CV（Mahalanobis）※安定化入り
# =========================================================
def safe_pinv_cov(cov: np.ndarray, ridge: float) -> np.ndarray:
    d = cov.shape[0]
    cov2 = cov + ridge * np.eye(d, dtype=cov.dtype)

    if not np.isfinite(cov2).all():
        cov2 = np.nan_to_num(cov2, nan=0.0, posinf=0.0, neginf=0.0)

    return np.linalg.pinv(cov2)

def eval_10fold(features: np.ndarray, rng: np.random.Generator, n_folds: int):
    n_sozai, n_sample, dim = features.shape

    if n_sample < n_folds:
        raise ValueError(f"Need at least {n_folds} samples per class, but n_sample={n_sample}")

    all_indices = np.arange(n_sample)
    rng.shuffle(all_indices)
    fold_indices = np.array_split(all_indices, n_folds)

    conf_8_total = np.zeros((n_sozai, n_sozai))
    conf_3_total = np.zeros((3, 3))
    acc_list8 = []
    acc_list3 = []

    for fold in range(n_folds):
        test_idx = np.array(fold_indices[fold])
        train_idx = np.setdiff1d(all_indices, test_idx)

        vec_ave = np.zeros((n_sozai, dim))
        cov_inv = np.zeros((n_sozai, dim, dim))

        for c in range(n_sozai):
            train_data = features[c, train_idx, :]

            if not np.isfinite(train_data).all():
                train_data = np.nan_to_num(
                    train_data,
                    nan=NAN_FILL_VALUE,
                    posinf=NAN_FILL_VALUE,
                    neginf=NAN_FILL_VALUE
                )

            vec_ave[c, :] = np.mean(train_data, axis=0)

            cov = np.cov(train_data.T)
            if np.ndim(cov) == 0:
                cov = np.array([[float(cov)]])

            if not np.isfinite(cov).all():
                cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

            try:
                cov_inv[c, :, :] = safe_pinv_cov(cov, RIDGE_LAMBDA)
            except np.linalg.LinAlgError:
                cov_inv[c, :, :] = safe_pinv_cov(cov, RIDGE_LAMBDA_RETRY)

        conf_8_fold = np.zeros((n_sozai, n_sozai))

        for true_c in range(n_sozai):
            for idx in test_idx:
                x = features[true_c, idx, :]

                if not np.isfinite(x).all():
                    x = np.nan_to_num(
                        x,
                        nan=NAN_FILL_VALUE,
                        posinf=NAN_FILL_VALUE,
                        neginf=NAN_FILL_VALUE
                    )

                diff_sozai = np.zeros(n_sozai)
                for m in range(n_sozai):
                    diff = x - vec_ave[m, :]
                    val = (diff.reshape(1, -1) @ cov_inv[m, :, :] @ diff.reshape(-1, 1)).item()
                    diff_sozai[m] = np.sqrt(max(val, 0.0))

                pred_c = int(np.argmin(diff_sozai))
                conf_8_fold[true_c, pred_c] += 1

        total_samples_fold = np.sum(conf_8_fold)
        correct_fold_8 = np.trace(conf_8_fold)
        acc_list8.append(correct_fold_8 / total_samples_fold)

        # 8 -> 3 統合
        mtrx1_f = np.zeros((8, 3))
        mtrx1_f[:, 0] = conf_8_fold[:, 0] + conf_8_fold[:, 5] + conf_8_fold[:, 7]
        mtrx1_f[:, 1] = conf_8_fold[:, 3] + conf_8_fold[:, 4] + conf_8_fold[:, 6]
        mtrx1_f[:, 2] = conf_8_fold[:, 1] + conf_8_fold[:, 2]

        mtrx2_f = np.zeros((3, 3))
        mtrx2_f[0, :] = mtrx1_f[0, :] + mtrx1_f[5, :] + mtrx1_f[7, :]
        mtrx2_f[1, :] = mtrx1_f[3, :] + mtrx1_f[4, :] + mtrx1_f[6, :]
        mtrx2_f[2, :] = mtrx1_f[1, :] + mtrx1_f[2, :]

        correct_fold_3 = mtrx2_f[0, 0] + mtrx2_f[1, 1] + mtrx2_f[2, 2]
        acc_list3.append(correct_fold_3 / total_samples_fold)

        conf_8_total += conf_8_fold
        conf_3_total += mtrx2_f

    total_samples = np.sum(conf_8_total)
    correct_8 = np.trace(conf_8_total)
    accuracy8_overall = correct_8 / total_samples
    accuracy8_mean = np.mean(acc_list8)

    correct_3_total = conf_3_total[0, 0] + conf_3_total[1, 1] + conf_3_total[2, 2]
    accuracy3_overall = correct_3_total / total_samples
    accuracy3_mean = np.mean(acc_list3)

    return (
        conf_8_total, conf_3_total,
        accuracy8_overall, accuracy8_mean,
        accuracy3_overall, accuracy3_mean,
        int(total_samples)
    )

# =========================================================
# 5) Excel保存
# =========================================================
def save_excel(out_dir: Path, dataset_name: str, rep: int, T_n: int,
               conf8: np.ndarray, conf3: np.ndarray,
               acc8_overall: float, acc8_mean: float,
               acc3_overall: float, acc3_mean: float,
               n_sample: int, total_samples: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    excel_filename = out_dir / f"{dataset_name}_rep{rep:02d}_Tn_{T_n}_10fold_conf_matrices.xlsx"

    with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
        pd.DataFrame(conf8, index=DIR_NAME, columns=DIR_NAME).to_excel(writer, sheet_name="conf_8cls")
        pd.DataFrame(conf3).to_excel(writer, sheet_name="conf_3cls", index=False)

        pd.DataFrame({
            "dataset": [dataset_name],
            "rep": [rep],
            "T_n": [T_n],
            "accuracy8_overall": [acc8_overall],
            "accuracy8_mean": [acc8_mean],
            "accuracy3_overall": [acc3_overall],
            "accuracy3_mean": [acc3_mean],
            "n_sample_per_class": [n_sample],
            "total_samples": [total_samples],
        }).to_excel(writer, sheet_name="accuracy", index=False)

    print("Saved Excel:", excel_filename)

# =========================================================
# 6) メイン
# =========================================================
def main():
    mapping = discover_sout_files(DATA_DIR)
    mapping = {name: rep2path for name, rep2path in mapping.items() if should_run(name)}

    selected = sorted(mapping.keys())
    print("[SELECTED DATASETS]", selected)

    if len(selected) == 0:
        print("No datasets matched RULE_WHITELIST. Check names and filenames.")
        return

    summary_rows = []

    for dataset_name in selected:
        rep2path = mapping[dataset_name]
        dataset_out_dir = OUT_ROOT / dataset_name
        dataset_out_dir.mkdir(parents=True, exist_ok=True)

        for rep in range(REP_START, REP_END + 1):
            if rep not in rep2path:
                print(f"[SKIP] {dataset_name} rep{rep:02d}: file not found")
                continue

            infname = rep2path[rep]
            sout_rec = np.load(infname)

            if sout_rec.ndim != 4:
                print(f"[SKIP] {dataset_name} rep{rep:02d}: invalid shape {sout_rec.shape}")
                continue

            n_sozai, n_sample, N_out, T = sout_rec.shape
            if n_sample < N_FOLDS:
                raise ValueError(
                    f"10-fold needs n_sample>= {N_FOLDS}, but got {n_sample} in {infname.name}"
                )

            rng = np.random.default_rng(BASE_SEED + rep)

            print("=" * 60)
            print(f"[DATASET {dataset_name}] [REP {rep:02d}] file={infname.name} shape={sout_rec.shape}")
            print("=" * 60)

            for T_n in Tn_list:
                features = extract_features(sout_rec, T_n)
                n_interval = (T // T_n) if (T % T_n == 0) else None
                print(f"[REP {rep:02d}] T_n={T_n} n_interval={n_interval} dim={features.shape[-1]}")

                (
                    conf8, conf3,
                    acc8_overall, acc8_mean,
                    acc3_overall, acc3_mean,
                    total_samples
                ) = eval_10fold(features, rng, N_FOLDS)

                print("Accuracy 8cls total:", acc8_overall, "fold-mean:", acc8_mean)
                print("Accuracy 3cls total:", acc3_overall, "fold-mean:", acc3_mean)

                save_excel(
                    dataset_out_dir, dataset_name, rep, T_n,
                    conf8, conf3,
                    acc8_overall, acc8_mean,
                    acc3_overall, acc3_mean,
                    n_sample, total_samples
                )

                summary_rows.append({
                    "dataset": dataset_name,
                    "rep": rep,
                    "T_n": T_n,
                    "acc8_overall": acc8_overall,
                    "acc8_mean": acc8_mean,
                    "acc3_overall": acc3_overall,
                    "acc3_mean": acc3_mean,
                    "n_sample_per_class": n_sample,
                    "total_samples": total_samples,
                    "file": infname.name,
                })

    if len(summary_rows) > 0:
        df_summary = pd.DataFrame(summary_rows)
        summary_xlsx = OUT_ROOT / "summary_all_datasets.xlsx"
        with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
            df_summary.to_excel(writer, sheet_name="summary", index=False)
        print("\nSaved summary:", summary_xlsx)

    print("\nAll finished.")

if __name__ == "__main__":
    main()


