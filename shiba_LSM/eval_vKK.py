import sys
import numpy as np
#import math
#import matplotlib.pyplot as plt
#from scipy.spatial import distance
#from scipy.interpolate import inter1d

rng = np.random.default_rng(1)

dir_name = ["Al_board", "buta_omote", "buta_ura", "cork", "denim", "rubber_board", "washi", "wood_board"]
n_sozai = len(dir_name)

# コマンドライン引数から T_n を取得
T_n = int(sys.argv[1])
loop_size = 10

infname = 'T_STDP_1_sout_rec.npy'
sout_rec_org = np.load(infname)

n_sozai = sout_rec_org.shape[0]
n_sample = sout_rec_org.shape[1]
N_out   = sout_rec_org.shape[2]
T       = sout_rec_org.shape[3]

# T_n が割り切れるかチェック（任意）
if T % T_n != 0:
    raise ValueError(f"T={T} が T_n={T_n} で割り切れません。")

n_interval = int(T / T_n)

sample_for_cls = 80
sample_for_tst = n_sample - sample_for_cls
accuracy8 = np.zeros(loop_size)
accuracy3 = np.zeros(loop_size)

for loop_i in range(loop_size):
    # サンプル方向にシャッフル
    sout_rec = rng.permutation(sout_rec_org, axis=1)
    s_for_cls = sout_rec[:, 0:sample_for_cls, :, :]
    s_for_tst = sout_rec[:, sample_for_cls:n_sample, :, :]

    sout_rec_fr  = np.zeros([n_sozai, sample_for_cls, N_out * n_interval])
    sout_rec_ave = np.zeros([n_sozai, N_out * n_interval])
    sout_rec_cov = np.zeros([n_sozai, N_out * n_interval, N_out * n_interval])
    cov_mtrx     = np.zeros([n_sozai, N_out * n_interval, N_out * n_interval])

    # ---------- 学習データから特徴量・平均・共分散 ----------
    for i in range(n_sozai):
        for j in range(sample_for_cls):
            for k in range(N_out):
                for l in range(n_interval):
                    sout_rec_fr[i, j, k * n_interval + l] = (
                        np.sum(s_for_cls[i, j, k, l*T_n:(l+1)*T_n]) / (T_n / 1000.0)
                    )
        # ★ 修正：平均は「学習に使ったサンプル数」で割る
        sout_rec_ave[i, :] = np.sum(sout_rec_fr[i, :, :], axis=0) / sample_for_cls
        # sout_rec_ave[i, :] = sout_rec_fr[i, :, :].mean(axis=0) でもOK

        sout_rec_cov[i, :, :] = np.cov(sout_rec_fr[i, :, :].T)
        # （必要なら正則化：eps * I を足すのもアリ）
        # eps = 1e-3
        # sout_rec_cov[i, :, :] += eps * np.eye(N_out * n_interval)

    for i in range(n_sozai):
        cov_mtrx[i, :, :] = np.linalg.pinv(sout_rec_cov[i, :, :])

    feature_vec = np.zeros([N_out * n_interval])
    diff_sozai  = np.zeros(n_sozai)
    seigo       = np.zeros([n_sozai * sample_for_tst, 2])

    # ---------- テストデータでマハラノビス距離を計算 ----------
    idx = 0
    for i in range(n_sozai):
        for j in range(sample_for_tst):
            diff_sozai[:]  = 0.0
            feature_vec[:] = 0.0

            # 発火率ベクトル作成
            for k in range(N_out):
                for l in range(n_interval):
                    feature_vec[k * n_interval + l] = (
                        np.sum(s_for_tst[i, j, k, l*T_n:(l+1)*T_n]) / (T_n / 1000.0)
                    )

            # ★ 修正：マハラノビス距離の計算（DeprecationWarning 回避）
            for m in range(n_sozai):
                data_minus_ave = feature_vec - sout_rec_ave[m, :]
                quad = data_minus_ave @ cov_mtrx[m, :, :] @ data_minus_ave
                diff_sozai[m] = float(np.sqrt(quad))

            seigo[idx, 0] = int(i)                     # 真のクラス
            seigo[idx, 1] = int(np.argmin(diff_sozai)) # 予測クラス
            idx += 1

    # ---------- 8クラス混同行列 & 精度 ----------
    y_true = seigo[:, 0].astype(int)
    y_pred = seigo[:, 1].astype(int)

    conf_mtrx = np.zeros((n_sozai, n_sozai), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_mtrx[t, p] += 1

    accuracy8[loop_i] = np.mean(y_true == y_pred)
    print(accuracy8[loop_i])
    print(conf_mtrx)

    # ---------- 3クラスへ統合 ----------

    # 元のコードのマージ規則：
    #  列(・行)0,5,7 → グループ0
    #  列(・行)3,4,6 → グループ1
    #  列(・行)1,2   → グループ2
    group0 = [0, 5, 7]
    group1 = [3, 4, 6]
    group2 = [1, 2]

    group_map = np.full(n_sozai, -1, dtype=int)
    group_map[group0] = 0
    group_map[group1] = 1
    group_map[group2] = 2

    y_true3 = group_map[y_true]
    y_pred3 = group_map[y_pred]

    # 3クラスの混同行列
    conf3 = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true3, y_pred3):
        conf3[t, p] += 1

    accuracy3[loop_i] = np.mean(y_true3 == y_pred3)
    print(accuracy3[loop_i])
    print(conf3)

print('Accuracy of 8 classes')
print(np.mean(accuracy8))
print('Accuracy of 3 classes')
print(np.mean(accuracy3))
