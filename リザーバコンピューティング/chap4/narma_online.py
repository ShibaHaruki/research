#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# narma_online.py: 本書の図4.8に対応するサンプルコード
#################################################################

import numpy as np
import matplotlib.pyplot as plt
from model import ESN, RLS


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
        np.random.seed(seed)
        u = np.random.uniform(0, 0.5, T)

        # 時系列生成
        while n < T:
            y_n = self.a1*y[n-1] + self.a2*y[n-1]*(np.sum(y[n-self.m:n-1])) \
                 + self.a3*u[n-self.m]*u[n] + self.a4
            y.append(y_n)
            n += 1

        return u, np.array(y)


if __name__ == '__main__':

    # NARMAタスクの入出力データ
    T1 = 200  # データ長
    order1 = 10  # 次数
    dynamics = NARMA(order1, a1=0.3, a2=0.05, a3=1.5, a4=0.1)
    y_init = [0] * order1
    u1, d1 = dynamics.generate_data(T1, y_init)

    T2 = 200
    order2 = 10
    dynamics = NARMA(order2, a1=0.3, a2=0.03, a3=1.0, a4=0.1)
    y_init = d1[T1-order2:T1].tolist()
    u2, d2 = dynamics.generate_data(T2, y_init)

    T3 = 200
    order3 = 10
    dynamics = NARMA(order3, a1=0.3, a2=0.01, a3=0.1, a4=0.1)
    y_init = d2[T2-order3:T2].tolist()
    u3, d3 = dynamics.generate_data(T3, y_init)

    u = np.concatenate((u1, u2, u3))
    d = np.concatenate((d1, d2, d3))
    
    # 訓練用情報
    train_U = u.reshape(-1, 1)
    train_D = d.reshape(-1, 1)

    # ESNモデル
    N_x = 200  # リザバーのノード数
    model = ESN(train_U.shape[1], train_D.shape[1], N_x, density=0.05, 
                input_scale=0.1, rho=0.8, fb_scale=0.1, fb_seed=0,
                noise_level=0.01)
    
    # オンライン学習と予測
    train_Y, Wout_size = model.adapt(train_U, train_D, 
                                     RLS(N_x, train_D.shape[1], delta=1e-4, 
                                         lam=0.995, update=5))

    # 評価（誤差RMSE, NRMSE）
    RMSE = np.sqrt(((train_D - train_Y) ** 2).mean())
    NRMSE = RMSE/np.sqrt(np.var(train_D))
    print('RMSE =', RMSE)
    print('NRMSE =', NRMSE)

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 7))
    plt.subplots_adjust(hspace=0.3)

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.text(-0.15, 1, '(a)', transform=ax1.transAxes)
    plt.plot(train_D, label='Target', color='k')
    plt.plot(train_Y, label='Model', color='gray', linestyle='--')
    plt.axvline(x=T1, ymin=0, color='k', linestyle=':')
    plt.axvline(x=T1+T2, ymin=0, color='k', linestyle=':')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.text(-0.15, 1, '(b)', transform=ax2.transAxes)
    plt.plot(np.abs(train_D - train_Y), color='k')
    plt.yscale('log')
    plt.ylabel('Error')
    plt.axvline(x=T1, ymin=0, color='k', linestyle=':')
    plt.axvline(x=T1+T2, ymin=0, color='k', linestyle=':')

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.text(-0.15, 1, '(c)', transform=ax3.transAxes)
    plt.plot(Wout_size, color='k')
    plt.yscale('log')
    plt.ylabel('Av. abs. weights')
    plt.xlabel('n')
    plt.axvline(x=T1, ymin=0, color='k', linestyle=':')
    plt.axvline(x=T1+T2, ymin=0, color='k', linestyle=':')

    plt.show()
