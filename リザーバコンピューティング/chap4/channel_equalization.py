#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# channel_equalization.py: 本書の図4.4に対応するサンプルコード
#################################################################

import numpy as np
import matplotlib.pyplot as plt
from model import ESN, Tikhonov


np.random.seed(seed=0)

if __name__ == '__main__':

    # 時系列出力データ（送信信号：4値シンボル列）
    T = 50  # 長さ
    symbol = [-3, -1, 1, 3]
    s = np.random.choice(symbol, T)
    s = s.astype(np.float)
    d = np.zeros(T)
    for n in range(2, T): 
        d[n] = s[n-2]

    # 時系列入力データ（受信信号）
    tau1 = 7
    tau2 = 2
    q = np.zeros(T)
    u = np.zeros(T)
    for n in range(tau1, T-tau2):
        q[n] = 0.08*d[n+2] - 0.12*d[n+1] + d[n] + 0.18*d[n-1] - 0.1*d[n-2] \
               + 0.09*d[n-3] - 0.05*d[n-4] + 0.04*d[n-5] + 0.03*d[n-6] \
               + 0.01*d[n-7]
        u[n] = q[n] + 0.036*(q[n]**2) - 0.011*(q[n]**3)

    # ノイズ付加
    SNRdB = 24
    SNR = 10**(SNRdB/10)
    nu = np.random.normal(0, np.sqrt(1/SNR), T)
    u = u + nu

    # 学習用情報
    train_U = u[tau1:T-tau2].reshape(-1, 1)
    train_D = d[tau1:T-tau2].reshape(-1, 1)

    # ESNモデル
    N_x = 40  # リザバーのノード数
    model = ESN(train_U.shape[1], train_D.shape[1], N_x, density=0.1, 
                input_scale=1.0, rho=0.9)

    # 学習(線形回帰)
    model.train(train_U, train_D, Tikhonov(N_x, train_D.shape[1], 0.0))
    
    # モデル出力
    train_Y = model.predict(train_U)

    # 評価（シンボル誤り率, SER）
    dif = 0
    train_Y_symbol = np.zeros(T-tau1-tau2-2)
    for n in range(T-tau1-tau2-2):
        if train_Y[n, 0] > 2:
            train_Y_symbol[n] = 3
        elif train_Y[n, 0] > 0:
            train_Y_symbol[n] = 1
        elif train_Y[n, 0] > -2:
            train_Y_symbol[n] = -1
        else:
            train_Y_symbol[n] = -3
        if int(train_Y_symbol[n]) != int(d[n+tau1]): 
            dif = dif + 1

    SER = dif/(T-tau1-tau2-2)
    print('SER =', SER)

    # グラフ表示データ
    T_disp = (0, T-tau1-tau2)
    t_axis = np.arange(tau1, T-tau2)  # 時間軸
    disp_U = train_U[T_disp[0]:T_disp[1]]  # 入力
    disp_D = train_D[T_disp[0]:T_disp[1], :]  # 目標出力
    disp_Y = train_Y[T_disp[0]:T_disp[1], :]  # モデル出力

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 5))
    plt.subplots_adjust(hspace=0.3)
    
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.text(-0.15, 1, '(a)', transform=ax1.transAxes)
    plt.plot(t_axis, disp_U[:, 0], color='k', marker='o')
    plt.xlim([0, T])
    plt.ylim([-4, 4])
    plt.ylabel('Input')
    
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.text(-0.15, 1, '(b)', transform=ax2.transAxes)
    ax2.set_yticks([-3, -1, 1, 3])
    plt.plot(t_axis, disp_D[:, 0], color='k', marker='o', label='Target')
    shift = 0.3  # プロットの重複を避けるためモデル出力を-0.3シフトする
    plt.plot(t_axis, disp_Y[:, 0]-shift, color='gray', marker='s', 
             linestyle = '--', label='Model')
    plt.plot([0, T], [-3, -3], linestyle = ':', color='k')
    plt.plot([0, T], [-1, -1], linestyle = ':', color='k')
    plt.plot([0, T], [1, 1], linestyle = ':', color='k')
    plt.plot([0, T], [3, 3], linestyle = ':', color='k')
    plt.xlim([0, T])
    plt.ylim([-4,4])
    plt.xlabel('n')
    plt.ylabel('Output')
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    
    plt.show()
