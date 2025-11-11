#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# tempral_parity.py: 本書の図4.2に対応するサンプルコード
#################################################################

import numpy as np
import matplotlib.pyplot as plt
from model import ESN, Tikhonov


np.random.seed(seed=0)

if __name__ == '__main__':

    # 時系列入力データ
    T = 50  # 時間長
    u = np.random.randint(0, 2, T)  # 2値系列
    
    # 時系列出力データ
    tau = 4  # delay
    k = 3 # 3-bit
    d = np.zeros(T)
    for n in range(tau+k-1, T):
        tmp = u[n-tau]+u[n-tau-1]+u[n-tau-2]  # 3-bit PARITY関数
        if tmp % 2 == 0:
            d[n] = 0  # 1の数が偶数なら0
        else:
            d[n] = 1  # 1の数が奇数なら1

    # 実数に変換
    u = u.astype(np.float)
    d = d.astype(np.float)

    # 学習用情報
    train_U = u[tau+k-1:T].reshape(-1, 1)
    train_D = d[tau+k-1:T].reshape(-1, 1)
    
    # ESNモデル
    N_x = 40  # リザバーのノード数
    model = ESN(train_U.shape[1], train_D.shape[1], N_x, density=0.1, 
                input_scale=1.0, rho=0.9)

    # 学習（線形回帰）
    model.train(train_U, train_D, Tikhonov(N_x, train_D.shape[1], 0.0))

    # モデル出力
    train_Y = model.predict(train_U)
    
    # 評価（ビット誤り率, BER）
    train_Y_binary = np.zeros(T-tau-k+1)
    for n in range(T-tau-k+1):
        if train_Y[n, 0] <= 0.5:
            train_Y_binary[n] = 0
        else:
            train_Y_binary[n] = 1
        
    BER = np.linalg.norm(train_Y_binary[0:T-tau-k+1]-d[tau+k-1:T],1)/(T-tau-k+1)
    print('BER =', BER)

    # グラフ表示用データ
    T_disp = (0, T-tau-k+1)
    t_axis = np.arange(tau+k-1, T)  # 時間軸
    disp_U = train_U[T_disp[0]:T_disp[1]]  # 入力
    disp_D = train_D[T_disp[0]:T_disp[1],:]  # 目標出力
    disp_Y = train_Y[T_disp[0]:T_disp[1],:]  # モデル出力

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 5))
    plt.subplots_adjust(hspace=0.3)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.text(-0.15, 1, '(a)', transform=ax1.transAxes)
    ax1.set_yticks([0.0, 1.0])
    plt.plot(t_axis, disp_U[:, 0], color='k', marker='o')
    plt.xlim([0, T])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('Input')
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.text(-0.15, 1, '(b)', transform=ax2.transAxes)
    ax2.set_yticks([0, 0.5, 1])
    plt.plot(t_axis, disp_D[:, 0], color='k', marker='o', label='Target')
    plt.plot(t_axis, disp_Y[:, 0], color='gray', marker='s', 
             linestyle = '--', label='Model')
    plt.plot([0, T], [0.5, 0.5], color='k', linestyle = ':')
    plt.xlim([0, T])
    plt.ylim([-0.3,1.3])
    plt.xlabel('n')
    plt.ylabel('Output')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    
    plt.show()
