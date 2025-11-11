#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# sinewave_generation.py: 本書の図4.6に対応するサンプルコード
#################################################################

import numpy as np
import matplotlib.pyplot as plt
from model import ESN, Tikhonov


np.random.seed(seed=0)

# 与えられた周波数の正弦波を生成
def generate_sinwave(freqs):
    phase_diff = 2.0*np.pi*freqs  # 角周波数の変化量
    phase = np.cumsum(2.0*np.pi*freqs)  # 位相
    sin_wave = np.sin(phase)  # 振幅1の正弦波

    return sin_wave


# 周波数が時間変化する正弦波の生成
def freq_generator(T, n_changepoint, seed=0):
    '''
    :param T: データ長
    :param n_changepoint: 周波数変化点の数
    :return: 周波数、正弦波
    '''
    changepoints = np.linspace(0, T-1, n_changepoint, endpoint=True)
    changepoints = changepoints.astype(np.int32)
    const_intervals = list(zip(changepoints, np.roll(changepoints, -1)))[:-1]

    freqs = np.zeros(T)
    np.random.seed(seed=seed)
    for (t0, t1) in const_intervals:
        freqs[t0:t1] = np.random.rand() / 10
    y = generate_sinwave(freqs)

    return freqs, y


if __name__ == "__main__":

    # データ長
    T_trans = 500  # 過渡期
    T_train = 5000  # 訓練データ
    T_test = 500  # 検証データ

    # データ生成
    T = T_trans + T_train
    u, d = freq_generator(T + T_test, 30, seed=0)

    # 訓練・検証用情報
    train_U = u[:T].reshape(-1, 1)
    train_D = d[:T].reshape(-1, 1)

    test_U = u[T:].reshape(-1, 1)
    test_D = d[T:].reshape(-1, 1)

    # ESNモデル
    N_x = 200  # リザバーのノード数
    model = ESN(train_U.shape[1], train_D.shape[1], N_x, density=0.05,
                input_scale=1, rho=0.7, fb_scale=1, fb_seed=1,
                leaking_rate=0.95) 
    
    # 学習
    train_Y = model.train(train_U, train_D, 
                          Tikhonov(N_x, train_D.shape[1], 1e-5), 
                          trans_len = T_trans)

    # モデル出力
    test_Y = model.predict(test_U)

    # 評価（検証誤差, RMSE, NRMSE）
    RMSE = np.sqrt(((test_D - test_Y) ** 2).mean())
    NRMSE = RMSE/np.sqrt(np.var(test_D))
    print('RMSE =', RMSE)
    print('NRMSE =', NRMSE)

    # グラフ表示用データ
    T_disp = (-500, 500)
    t_axis = np.arange(T_disp[0], T_disp[1])  # 時間軸
    disp_U = np.concatenate((train_U[T_disp[0]:], test_U[:T_disp[1]]))
    disp_D = np.concatenate((train_D[T_disp[0]:], test_D[:T_disp[1]]))
    disp_Y = np.concatenate((train_Y[T_disp[0]:], test_Y[:T_disp[1]]))

    # グラフ表示
    plt.rcParams["font.size"] = 12
    fig = plt.figure(figsize=(7, 5))
    plt.subplots_adjust(hspace=0.3)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.text(-0.15, 1, '(a)', transform=ax1.transAxes)
    ax1.text(0.2, 1.05, "Training", transform=ax1.transAxes)
    ax1.text(0.7, 1.05, "Testing", transform=ax1.transAxes)
    plt.plot(t_axis, disp_U[:,0], color='k')
    plt.ylabel("Input")
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.text(-0.15, 1, '(b)', transform=ax2.transAxes)
    plt.plot(t_axis, disp_D[:,0], color='k', label='Target')
    plt.plot(t_axis, disp_Y[:,0], color='gray', linestyle='--', label='Model')
    plt.xlabel("n")
    plt.ylabel("Output")
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    plt.show()
    
