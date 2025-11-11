#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# nonlinearity.py: 本書の図3.8, 図3.9に対応するサンプルコード
# （図3.8ではrho=0.5，図3.9ではrho=1.5と設定してください．）
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model import Input, Reservoir


if __name__ == '__main__':
    
    # 時系列入力データ生成
    T = 500  # 長さ
    period = 50  # 周期
    time = np.arange(T)  # 時間
    u = np.sin(2*np.pi*time/period)  # 正弦波

    # 入力層とリザバーの生成
    N_x = 100  # リザバーの大きさ
    input = Input(N_u=1, N_x=N_x, input_scale=1.0, seed=0)

    # リザバー構築（図3.8: rho=0.5, 図3.9: rho=1.5）
    reservoir = Reservoir(N_x=N_x, density=0.05, rho=0.5,
                          activation_func = np.tanh, leaking_rate=1.0, seed=0)

    # リザバー状態の時間発展
    U = u[:T].reshape(-1, 1) 
    x_all = np.empty((0, N_x))
    for n in range(T):
        x_in = input(U[n])
        x = reservoir(x_in)
        x_all = np.vstack((x_all, x))

    # グラフ表示
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 3, wspace=0.4, hspace=0.4)

    # (n, x_1)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(time, x_all[:,0], color='k', linewidth=2)
    ax1.set_xlabel('n')
    ax1.set_ylabel('x_1')
    ax1.grid(True)
    ax1.text(-0.3, 1, '(a)', transform=ax1.transAxes)

    # (n, x_2)
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(time, x_all[:,1], color='k', linewidth=2)
    ax2.set_xlabel('n')
    ax2.set_ylabel('x_2')
    ax2.grid(True)
    ax2.text(-0.3, 1, '(b)', transform=ax2.transAxes)

    # (n, x_3)
    ax3 = plt.subplot(gs[0, 2])
    ax3.plot(time, x_all[:,2], color='k', linewidth=2)
    ax3.set_xlabel('n')
    ax3.set_ylabel('x_3')
    ax3.grid(True)
    ax3.text(-0.3, 1, '(c)', transform=ax3.transAxes)

    # (u, x_1)
    ax4 = plt.subplot(gs[1, 0])
    ax4.plot(u, x_all[:,0], color='k', linewidth=2)
    ax4.set_xlabel('u')
    ax4.set_ylabel('x_1')
    ax4.grid(True)
    ax4.text(-0.3, 1, '(d)', transform=ax4.transAxes)

    # (u, x_2)
    ax5 = plt.subplot(gs[1, 1])
    ax5.plot(u, x_all[:,1], color='k', linewidth=2)
    ax5.set_xlabel('u')
    ax5.set_ylabel('x_2')
    ax5.grid(True)
    ax5.text(-0.3, 1, '(e)', transform=ax5.transAxes)

    # (u, x_3)
    ax6 = plt.subplot(gs[1, 2])
    ax6.plot(u, x_all[:,2], color='k', linewidth=2)
    ax6.set_xlabel('u')
    ax6.set_ylabel('x_3')
    ax6.grid(True)
    ax6.text(-0.3, 1, '(f)', transform=ax6.transAxes)

    plt.show()
