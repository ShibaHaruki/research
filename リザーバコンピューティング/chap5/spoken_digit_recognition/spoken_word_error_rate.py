# !/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# spoken_word_error.py: 本書の図5.3に対応するサンプルコード
#################################################################

#################################################################
# ■ 音声データの準備
#
# https://github.com/dsiufl/Reservoir-Computing
# より，"Lyon_decimation_128"というフォルダをダウンロードして
# 同じフォルダに置く．
#
# これは，TI 46-Word Corpus
# https://catalog.ldc.upenn.edu/LDC93S9
# に含まれる音声データの一部をLyon's auditory modelで前処理した後の
# コクリアグラムのデータセットである（計500ファイル）．
# （注：オリジナルデータの使用にはLinguistic Data Consortiumのライセンスが必要）
#
# ファイル名は"s1_u1_d0.mat"のようになっており，
# sはspeaker (s1,s2,s5,s6,s7)
# uはutterance (u1, u2, ..., u9, u10)
# dはdigit (d0, d1, ..., d9)
# をそれぞれ表す．
# 
# 各.matファイルは構造体'spec'に行列データをもつ．
# 行列サイズは，チャネル数（n_channel=77）× 時間長（n_tau=データ依存）．
# 
# L115のtrain_listに含まれるutteranceを訓練用，それ以外を検証用とする．
#
#################################################################

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.io import loadmat
from model import ESN
from tqdm.notebook import tqdm


np.random.seed(seed=0)

# 音声信号を前処理したデータ(コクリアグラム)の読み込み
def read_speech_data(dir_name, utterance_train_list):
    ''' 
    :入力：データファイル(.mat)の入っているディレクトリ名(dir_name)
    :出力：入力データ(input_data)と教師データ(teacher_data)
    '''
    # .matファイルのみを取得
    data_files = glob.glob(os.path.join(dir_name, '*.mat'))

    # データに関する情報
    n_channel = 77  # チャネル数
    n_label = 10  # ラベル数(digitの数)

    # 初期化
    train_input = np.empty((0, n_channel))  # 教師入力
    train_output = np.empty((0, n_label))  # 教師出力
    train_length = np.empty(0, np.int)  # データ長
    train_label = np.empty(0, np.int)  # 正解ラベル
    test_input = np.empty((0, n_channel))  # 教師入力
    test_output = np.empty((0, n_label))  # 教師出力
    test_length = np.empty(0, np.int)  # データ長
    test_label = np.empty(0, np.int)  # 正解ラベル
    
    # データ読み込み
    if len(data_files) > 0:
        print("%d files in %s を読み込んでいます..." \
              % (len(data_files), dir_name))
        for each_file in data_files:
            data = loadmat(each_file)
            utterance = int(each_file[-8])  # 各speakerの発話番号
            digit = int(each_file[-5]) # 発話された数字
            if utterance in utterance_train_list:  # 訓練用
                # 入力データ（構造体'spec'に格納されている）
                train_input = np.vstack((train_input, data['spec'].T))
                # 出力データ（n_tau x 10，値はすべて'-1'）
                tmp = -np.ones([data['spec'].shape[1], 10])
                tmp[:, digit] = 1  # digitの列のみ1
                train_output = np.vstack((train_output, tmp))
                # データ長
                train_length = np.hstack((train_length, data['spec'].shape[1]))
                # 正解ラベル
                train_label = np.hstack((train_label, digit))
            else:  # 検証用
                # 入力データ（構造体'spec'に格納されている）
                test_input = np.vstack((test_input, data['spec'].T))
                # 出力データ（n_tau x 10，値はすべて'-1'）
                tmp = -np.ones([data['spec'].shape[1], 10])
                tmp[:, digit] = 1  # digitの列のみ1
                test_output = np.vstack((test_output, tmp))
                # データ長
                test_length = np.hstack((test_length, data['spec'].shape[1]))
                # 正解ラベル
                test_label = np.hstack((test_label, digit))
    else:
        print("ディレクトリ %s にファイルが見つかりません．" % (indir))
        return
    return train_input, train_output, train_length, train_label, \
           test_input, test_output, test_length, test_label


if __name__ == "__main__":

    # 訓練データ，検証データの取得
    train_list = [1, 2, 3, 4, 5]  # u1-u5が訓練用，残りが検証用
    train_input, train_output, train_length, train_label, test_input, \
    test_output, test_length, test_label = \
    read_speech_data(dir_name='./Lyon_decimation_128',
                     utterance_train_list=train_list)
    print("データ読み込み完了．訓練と検証を行っています...")
    
    N_x_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    train_WER = np.empty(0)
    test_WER = np.empty(0)
    bar = tqdm(total = np.sum(N_x_list))
    for N_x in N_x_list:
        print("リザバーの大きさ: %d" % N_x)
        bar.update(N_x)
        
        # ESNモデル
        model = ESN(train_input.shape[1], train_output.shape[1], N_x,
                    density=0.05, input_scale=1.0e+4, rho=0.9, fb_scale=0.0)

        ########## 訓練データに対して
        # リザバー状態行列
        stateCollectMat = np.empty((0, N_x))
        for i in range(len(train_input)):
            u_in = model.Input(train_input[i])
            r_out = model.Reservoir(u_in)
            stateCollectMat = np.vstack((stateCollectMat, r_out))
    
        # 教師出力データ行列
        teachCollectMat = train_output
    
        # 学習（疑似逆行列）
        Wout = np.dot(teachCollectMat.T, np.linalg.pinv(stateCollectMat.T))

        # ラベル出力
        Y_pred = np.dot(Wout, stateCollectMat.T)
        pred_train = np.empty(0, np.int)
        start = 0
        for i in range(len(train_length)):
            tmp = Y_pred[:,start:start+train_length[i]]  # 1つのデータに対する出力
            max_index = np.argmax(tmp, axis=0)  # 最大出力を与える出力ノード番号
            histogram = np.bincount(max_index)  # 出力ノード番号のヒストグラム
            pred_train = np.hstack((pred_train, np.argmax(histogram)))  # 最頻値
            start = start + train_length[i]
    
        # 訓練誤差(Word Error Rate, WER)
        count = 0
        for i in range(len(train_length)):
            if pred_train[i] != train_label[i]:
                count = count + 1 
        print("訓練誤差： WER = %5.4lf" % (count/len(train_length)))
        train_WER = np.hstack((train_WER, count/len(train_length)))

        ########## 検証データに対して
        # リザバー状態行列
        stateCollectMat = np.empty((0, N_x))
        for i in range(len(test_input)):
            u_in = model.Input(test_input[i])
            r_out = model.Reservoir(u_in)
            stateCollectMat = np.vstack((stateCollectMat, r_out))

        # ラベル出力
        Y_pred = np.dot(Wout, stateCollectMat.T)
        pred_test = np.empty(0, np.int)
        start = 0
        for i in range(len(test_length)):
            tmp = Y_pred[:,start:start+test_length[i]]  # 1つのデータに対する出力
            max_index = np.argmax(tmp, axis=0)  # 最大出力を与える出力ノード番号
            histogram = np.bincount(max_index)  # 出力ノード番号のヒストグラム
            pred_test = np.hstack((pred_test, np.argmax(histogram)))  # 最頻値
            start = start + test_length[i]
        
        # 検証誤差(WER)
        count = 0
        for i in range(len(test_length)):
            if pred_test[i] != test_label[i]:
                count = count + 1 
        print("検証誤差： WER = %5.4lf" % (count/len(test_length)))
        test_WER = np.hstack((test_WER, count/len(test_length)))

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 5))
    
    plt.plot(N_x_list, train_WER, marker='o', fillstyle='none',
             markersize=8, color='k', label='Training')         
    plt.plot(N_x_list, test_WER, marker='s', 
             markersize=8, color='k', label='Testing')
    plt.xticks(N_x_list)
    plt.xlabel("Size of reservoir")
    plt.ylabel("WER")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')

    plt.show()
