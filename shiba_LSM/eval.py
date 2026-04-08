import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance
import pandas as pd  # Excel保存用

rng = np.random.default_rng(1)

# 素材名（行ラベル用）
dir_name = ["Al_board", "buta_omote", "buta_ura",
            "cork", "denim", "rubber_board", "washi", "wood_board"]
n_sozai = len(dir_name)

# ==== T_n を一発で複数回す（それ以外は変更しない）====
Tn_list = [25, 50, 100, 125, 250, 500]

# もし従来通り「1つだけ」実行したい場合は引数を渡す
# 例: python eval_10fold.py 25
if len(sys.argv) >= 2:
    Tn_list = [int(sys.argv[1])]

# ==== LSM 出力データ読み込み ====
# ファイル名は実際のものに合わせて書き換えてください
infname = 'T_STDP_4_sout_rec.npy'  # 例: 'STDP_off_sout_rec.npy'
sout_rec_org = np.load(infname)   # shape: (n_sozai, n_sample_all, N_out, T)
print("T_STDP_2_sout_rec.npy shape:", sout_rec_org.shape)

n_sozai_all = sout_rec_org.shape[0]   # 8
n_sample_all = sout_rec_org.shape[1]  # 例: 100
N_out = sout_rec_org.shape[2]         # 出力ニューロン数
T = sout_rec_org.shape[3]             # 時間ビン数

# ここでは「全サンプル」を使用
sout_rec = sout_rec_org

n_sozai = sout_rec.shape[0]          # 8
n_sample = sout_rec.shape[1]         # 例: 100

# =========================================================
# 2. 10-fold 交差検証の分割は1回だけ作る（全T_n共通）
# =========================================================
n_folds = 10
if n_sample < n_folds:
    raise ValueError(f"10-fold CV には各クラス少なくとも {n_folds} サンプル必要です (今は {n_sample})")

all_indices = np.arange(n_sample)
rng.shuffle(all_indices)
fold_indices = np.array_split(all_indices, n_folds)

# =========================================================
# T_n を順番に回して全部出す
# =========================================================
for T_n in Tn_list:

    n_interval = int(T / T_n)            # 区間数
    dim = N_out * n_interval             # 特徴次元数

    print(f"n_sozai={n_sozai}, n_sample={n_sample}, N_out={N_out}, "
          f"T={T}, n_interval={n_interval}, dim={dim}")

    # =========================================================
    # 1. すべてのサンプルに対して特徴ベクトルを作成
    #    features[i, j, :] : 素材 i, サンプル j の (N_out*n_interval,) ベクトル
    # =========================================================
    features = np.zeros((n_sozai, n_sample, dim))

    for i in range(n_sozai):
        for j in range(n_sample):
            for k in range(N_out):
                for l in range(n_interval):
                    # 区間 l の発火数を合計し [spikes / s] に換算
                    spike_sum = np.sum(sout_rec[i, j, k, l*T_n:(l+1)*T_n])
                    features[i, j, k*n_interval + l] = spike_sum / (T_n / 1000.0)

    print("Feature extraction done:", features.shape)  # 例: (8, 100, dim)

    # =========================================================
    # 2. 10-fold 交差検証
    #    100×8 データのうち，90×8 で学習，10×8 でテストを 10 回繰り返す
    # =========================================================
    conf_mtrx_8_total = np.zeros((n_sozai, n_sozai))
    conf_mtrx_3_total = np.zeros((3, 3))
    acc_list8 = []
    acc_list3 = []

    for fold in range(n_folds):
        test_idx = np.array(fold_indices[fold])
        train_idx = np.setdiff1d(all_indices, test_idx)

        # ---- 学習データからクラスごとの平均・共分散逆行列を計算 ----
        vec_ave = np.zeros((n_sozai, dim))
        cov_inv = np.zeros((n_sozai, dim, dim))

        for c in range(n_sozai):
            train_data = features[c, train_idx, :]   # shape: (90, dim) の想定
            vec_ave[c, :] = np.mean(train_data, axis=0)
            cov = np.cov(train_data.T)
            cov_inv[c, :, :] = np.linalg.pinv(cov)

        # ---- この fold の混同行列（8クラス） ----
        conf_mtrx_8_fold = np.zeros((n_sozai, n_sozai))

        for true_c in range(n_sozai):
            for idx in test_idx:
                x = features[true_c, idx, :]  # (dim,)

                diff_sozai = np.zeros(n_sozai)
                for m in range(n_sozai):
                    diff = x - vec_ave[m, :]
                    diff_sozai[m] = np.sqrt(
                        (diff.reshape(1, -1) @ cov_inv[m, :, :] @ diff.reshape(-1, 1))
                    )

                pred_c = int(np.argmin(diff_sozai))
                conf_mtrx_8_fold[true_c, pred_c] += 1

        # ---- この fold の正答率（8クラス） ----
        total_samples_fold = np.sum(conf_mtrx_8_fold)
        correct_fold_8 = np.trace(conf_mtrx_8_fold)
        acc_fold_8 = correct_fold_8 / total_samples_fold
        acc_list8.append(acc_fold_8)

        # ---- この fold で 8クラス→3クラスにまとめて精度算出 ----
        mtrx1_f = np.zeros((8, 3))
        mtrx1_f[:, 0] = conf_mtrx_8_fold[:, 0] + conf_mtrx_8_fold[:, 5] + conf_mtrx_8_fold[:, 7]
        mtrx1_f[:, 1] = conf_mtrx_8_fold[:, 3] + conf_mtrx_8_fold[:, 4] + conf_mtrx_8_fold[:, 6]
        mtrx1_f[:, 2] = conf_mtrx_8_fold[:, 1] + conf_mtrx_8_fold[:, 2]

        mtrx2_f = np.zeros((3, 3))
        mtrx2_f[0, :] = mtrx1_f[0, :] + mtrx1_f[5, :] + mtrx1_f[7, :]
        mtrx2_f[1, :] = mtrx1_f[3, :] + mtrx1_f[4, :] + mtrx1_f[6, :]
        mtrx2_f[2, :] = mtrx1_f[1, :] + mtrx1_f[2, :]

        correct_fold_3 = mtrx2_f[0, 0] + mtrx2_f[1, 1] + mtrx2_f[2, 2]
        acc_fold_3 = correct_fold_3 / total_samples_fold
        acc_list3.append(acc_fold_3)

        # ---- 全体の混同行列に加算 ----
        conf_mtrx_8_total += conf_mtrx_8_fold
        conf_mtrx_3_total += mtrx2_f

        print(f"Fold {fold+1}/{n_folds} done. "
              f"acc8={acc_fold_8:.4f}, acc3={acc_fold_3:.4f}")

    # =========================================================
    # 3. 正答率計算（全 fold 統合後）
    # =========================================================
    total_samples = np.sum(conf_mtrx_8_total)
    correct_8 = np.trace(conf_mtrx_8_total)
    accuracy8_overall = correct_8 / total_samples
    accuracy8_mean = np.mean(acc_list8)

    correct_3_total = (conf_mtrx_3_total[0, 0]
                       + conf_mtrx_3_total[1, 1]
                       + conf_mtrx_3_total[2, 2])
    accuracy3_overall = correct_3_total / total_samples
    accuracy3_mean = np.mean(acc_list3)

    print("8-class confusion matrix (total):")
    print(conf_mtrx_8_total)
    print("Accuracy of 8 classes (total):", accuracy8_overall)
    print("Accuracy of 8 classes (fold-mean):", accuracy8_mean)

    print("3-class confusion matrix (total):")
    print(conf_mtrx_3_total)
    print("Accuracy of 3 classes (total):", accuracy3_overall)
    print("Accuracy of 3 classes (fold-mean):", accuracy3_mean)

    # =========================================================
    # 4. Excel に結果を書き出し（保存方法は元のまま）
    # =========================================================
    excel_filename = f"T_STDP_4_Tn_{T_n}_10fold_conf_matrices.xlsx"
    with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
        df_conf8 = pd.DataFrame(conf_mtrx_8_total,
                                index=dir_name,
                                columns=dir_name)
        df_conf8.to_excel(writer, sheet_name="conf_8cls")

        df_conf3 = pd.DataFrame(conf_mtrx_3_total)
        df_conf3.to_excel(writer, sheet_name="conf_3cls", index=False)

        df_acc = pd.DataFrame({
            "accuracy8_overall": [accuracy8_overall],
            "accuracy8_mean": [accuracy8_mean],
            "accuracy3_overall": [accuracy3_overall],
            "accuracy3_mean": [accuracy3_mean],
            "n_sample_per_class": [n_sample],
            "total_samples": [int(total_samples)],
        })
        df_acc.to_excel(writer, sheet_name="accuracy", index=False)

    print("Saved Excel:", excel_filename)

