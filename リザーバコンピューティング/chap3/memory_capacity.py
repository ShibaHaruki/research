#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# memory_capacity.py: 本書の図3.12に対応するサンプルコード
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
    delay = np.arange(20)  # 遅延長
    d = np.empty((T, len(delay)))
    for k in range(len(delay)):
        for t in range(T):
            d[t, k] = u[t-delay[k]]  # 遅延系列

    # 学習用情報
    T_trans = 200  # 過渡期
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

    # 忘却曲線
    DC = np.zeros((len(delay), 1))  # 決定係数
    MC = 0.0  # 記憶容量
    for k in range(len(delay)):
        corr = np.corrcoef(np.vstack((train_D.T[k, k:], train_Y.T[k, k:])))
        DC[k] = corr[0, 1] ** 2
        MC += DC[k]
    
    # グラフ表示
    plt.rcParams['font.size'] = 12
    plt.plot(delay, DC, color='k', marker='o')
    plt.ylim([0, 1.1])
    plt.xticks([0, 5, 10, 15, 20])
    plt.title('MC ~ %3.2lf' % MC, x=0.8, y=0.9)
    plt.xlabel('Delay k')
    plt.ylabel('Determination coefficient')

    plt.show()
