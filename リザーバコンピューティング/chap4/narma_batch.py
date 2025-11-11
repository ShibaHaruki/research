#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# narma_batch.py: 本書の図4.7に対応するサンプルコード
#################################################################

import numpy as np
import matplotlib.pyplot as plt
from model import ESN, Tikhonov


np.random.seed(seed=0)

# NARMAモデル
class NARMA:
    # パラメータの設定
    def __init__(self, m, a1, a2, a3, a4):
        self.m = m
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

    def generate_data(self, T, y_init, seed=0):
        n = self.m
        y = y_init
        np.random.seed(seed=seed)
        u = np.random.uniform(0, 0.5, T)

        # 時系列生成
        while n < T:
            y_n = self.a1*y[n-1] + self.a2*y[n-1]*(np.sum(y[n-self.m:n-1])) \
                + self.a3*u[n-self.m]*u[n] + self.a4
            y.append(y_n)
            n += 1

        return u, np.array(y)


if __name__ == '__main__':

    # データ長
    T = 900  # 訓練用
    T_test = 100  # 検証用

    order = 10  # NARMAモデルの次数
    dynamics = NARMA(order, a1=0.3, a2=0.05, a3=1.5, a4=0.1)
    y_init = [0] * order
    u, d = dynamics.generate_data(T + T_test, y_init)

    # 学習・テスト用情報
    train_U = u[:T].reshape(-1, 1)
    train_D = d[:T].reshape(-1, 1)
    test_U = u[T:].reshape(-1, 1)
    test_D = d[T:].reshape(-1, 1)

    # ESNモデル
    N_x = 50  # リザバーのノード数
    model = ESN(train_U.shape[1], train_D.shape[1], N_x, 
                density=0.15, input_scale=0.1, rho=0.9,
                fb_scale=0.1, fb_seed=0)

    # 学習（リッジ回帰）
    train_Y = model.train(train_U, train_D, 
                          Tikhonov(N_x, train_D.shape[1], 1e-4)) 

    # モデル出力
    test_Y = model.predict(test_U)

    # 評価（テスト誤差RMSE, NRMSE）
    RMSE = np.sqrt(((test_D - test_Y) ** 2).mean())
    NRMSE = RMSE/np.sqrt(np.var(test_D))
    print('RMSE =', RMSE)
    print('NRMSE =', NRMSE)

    # グラフ表示用データ
    T_disp = (-100, 100)
    t_axis = np.arange(T_disp[0], T_disp[1])
    disp_U = np.concatenate((train_U[T_disp[0]:], test_U[:T_disp[1]]))
    disp_D = np.concatenate((train_D[T_disp[0]:], test_D[:T_disp[1]]))
    disp_Y = np.concatenate((train_Y[T_disp[0]:], test_Y[:T_disp[1]]))

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 5))
    plt.subplots_adjust(hspace=0.3)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.text(-0.15, 1, '(a)', transform=ax1.transAxes)
    ax1.text(0.2, 1.05, 'Training', transform=ax1.transAxes)
    ax1.text(0.7, 1.05, 'Testing', transform=ax1.transAxes)
    plt.plot(t_axis, disp_U[:,0], color='k')
    plt.ylabel('Input')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.text(-0.15, 1, '(b)', transform=ax2.transAxes)
    plt.plot(t_axis, disp_D[:,0], color='k', label='Target')
    plt.plot(t_axis, disp_Y[:,0], color='gray', linestyle='--', label='Model')
    plt.xlabel('n')
    plt.ylabel('Output')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    plt.show()
