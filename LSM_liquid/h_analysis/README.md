# Jupyterでの解析

[`incremental_pca_accuracy.ipynb`](incremental_pca_accuracy.ipynb)で、使用する主成分数を1〜40へ増やしたときの分類正答率を調べます。

## 使い方

1. Jupyter Notebook対応のエディタでノートブックを開く。
2. カーネルを`Python (LSM_liquid)`、または`../.mlflow-venv/Scripts/python.exe`にする。
3. 上からすべてのセルを実行する。

保存済みの発火応答を読むため、gain探索やリキッド層の再計算は行いません。
現在の入力gain179の結果、興奮性80個、25 msビン、8素材×324試行が初期値です。
添付の過去条件の正答率を転記するのではなく、選んだデータから計算します。

学習・評価は従来と同じ10分割（各素材226学習・98評価）です。
標準化せず、PCAは各分割の学習試行だけで求めます。
分類の計算は`d_tools/component_evaluation.py`、PCAと分類器は既存の共通モジュールを使います。

## 出力

`results/gain179_incremental_pca_accuracy/`に保存します。

- `incremental_pca_accuracy_01_40.png`：正答率のグラフ
- `incremental_pca_accuracy_01_40.svg`：拡大しても文字が鮮明な図
- `accuracy_by_pc_count.csv`：主成分数ごとの平均正答率・標準偏差
- `accuracy_by_split.csv`：分割ごとの全特徴・1〜40成分の正答率
- `analysis_metadata.json`：データと解析コードのハッシュ・設定・計算時間

ノートブック内にも図と表を表示します。横軸は`Number of PCs`、縦軸は正答率（%）です。
青い誤差棒は10分割間の標本標準偏差で、信頼区間ではありません。
黒破線は全1600特徴を使った平均正答率、灰色点線は12.5%の一様ランダム分類です。

同じ入力・条件・解析コードでは保存済みCSVを再利用します。
再計算したい場合は、ノートブック内の`RECOMPUTE = True`へ変更してください。
