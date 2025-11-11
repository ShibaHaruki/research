#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# edge_of_chaos.py: 本書の図3.10に対応するサンプルコード
#################################################################

import numpy as np
import matplotlib.pyplot as plt
from model import Input, Reservoir
from tqdm.notebook import tqdm


if __name__ == '__main__':

    # 時系列入力データ生成
    T = 2000  # 長さ
    period = 50  # 周期
    time = np.arange(T)
    u = np.sin(2*np.pi*time/period)  # 正弦波信号
    
    # スペクトル半径rhoの値を変えながらループ
    p_all = np.empty((0, 101))
    rho_list = np.arange(0.0,2.0,0.02)
    for rho in tqdm(rho_list):

        # 入力層とリザバーを生成
        N_x = 100  # リザバーのノード数
        input = Input(N_u=1, N_x=N_x, input_scale=1.0, seed=0)
        reservoir = Reservoir(N_x=N_x, density=0.05, rho = rho,
                              activation_func=np.tanh, leaking_rate=1.0, seed=0)

        # リザバー状態の時間発展
        U = u[:T].reshape(-1, 1)
        x_all = np.empty((0, 100))
        for t in range(T):
            x_in = input(U[t])
            x = reservoir(x_in)
            x_all = np.vstack((x_all, x))

        # 1周期おきの状態
        T_trans = 1000  # 過渡期
        p = np.hstack((rho*np.ones((int((T-T_trans)/period), 1)), 
                       x_all[T_trans:T:period, 0:100]))
        p_all = np.vstack((p_all, p))

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize = (7, 7))
    plt.subplots_adjust(hspace=0.3)

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.text(-0.15, 1, '(a)', transform=ax1.transAxes)
    plt.scatter(p_all[:,0], p_all[:,1], color='k', marker='o', s=5)
    plt.ylabel('p_1')
    plt.grid(True)

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.text(-0.15, 1, '(b)', transform=ax2.transAxes)
    plt.scatter(p_all[:,0], p_all[:,2], color='k', marker='o', s=5)
    plt.ylabel('p_2')
    plt.grid(True)

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.text(-0.15, 1, '(c)', transform=ax3.transAxes)
    plt.scatter(p_all[:,0], p_all[:,3], color='k', marker='o', s=5)
    plt.xlabel(r'$\rho$')
    plt.ylabel('p_3')
    plt.grid(True)

    plt.show()
