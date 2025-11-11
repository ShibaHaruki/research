#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# delay_task.py: 本書の図3.11に対応するサンプルコード
#################################################################

import numpy as np
import matplotlib.pyplot as plt
from model import ESN, Tikhonov


np.random.seed(seed=0)

if __name__ == '__main__':

    # 時系列入力データ生成
    T = 500  # 長さ
    u = np.random.rand(T,1)-0.5  # 区間[-0.5, 0.5]の乱数系列

    # 時系列出力データ生成
    delay = [4, 8, 12]  # 遅延長
    d = np.empty((T, len(delay)))
    for k in range(len(delay)):
        for t in range(T):
            d[t, k] = u[t-delay[k]]  # 遅延系列

    # 学習用情報
    T_trans = 200  # 過渡期の長さ
    train_U = u[T_trans:T].reshape(-1, 1)
    train_D = d[T_trans:T, :].reshape(-1, len(delay))

    # ESNモデル
    N_x = 20  # リザバーのノード数
    model = ESN(train_U.shape[1], train_D.shape[1], N_x, density=0.05, 
                input_scale=1.0, rho=0.8)

    # 学習（線形回帰）
    model.train(train_U, train_D, Tikhonov(N_x, train_D.shape[1], 0.0))

    # モデル出力
    train_Y = model.predict(train_U)

    # グラフ表示用データ
    T_disp = (0, T-T_trans)
    time_axis = np.arange(T_trans, T)  # 時間軸
    disp_U = train_U[T_disp[0]:T_disp[1]]  # 入力
    disp_D = train_D[T_disp[0]:T_disp[1], :]  # 目標出力
    disp_Y = train_Y[T_disp[0]:T_disp[1], :]  # モデル出力

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 9))
    plt.subplots_adjust(hspace=0.3)

    ax1 = fig.add_subplot(4, 1, 1)
    ax1.text(-0.15, 1, '(a)', transform=ax1.transAxes)
    plt.plot(time_axis, disp_U[:, 0], color='gray', linestyle=':')
    plt.ylim([-0.6, 0.6])
    plt.ylabel('Input')

    ax2 = fig.add_subplot(4, 1, 2)
    ax2.text(-0.15, 1, '(b)', transform=ax2.transAxes)
    plt.plot(time_axis, disp_D[:, 0], color='gray', linestyle=':', label='Target')
    plt.plot(time_axis, disp_Y[:, 0], color='k', label='Model')
    plt.ylim([-0.6, 0.6])
    plt.ylabel('Output (k=4)')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')

    ax3 = fig.add_subplot(4, 1, 3)
    ax3.text(-0.15, 1, '(c)', transform=ax3.transAxes)
    plt.plot(time_axis, disp_D[:, 1], color='gray', linestyle=':', label='Target')
    plt.plot(time_axis, disp_Y[:, 1], color='k', label='Model')
    plt.ylim([-0.6, 0.6])
    plt.ylabel('Output (k=8)')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')

    ax4 = fig.add_subplot(4, 1, 4)
    ax4.text(-0.15, 1, '(d)', transform=ax4.transAxes)
    plt.plot(time_axis, disp_D[:, 2], color='gray', linestyle=':', label='Target')
    plt.plot(time_axis, disp_Y[:, 2], color='k', label='Model')
    plt.ylim([-0.6, 0.6])
    plt.ylabel('Output (k=12)')
    plt.xlabel('n')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')

    plt.show()

