# Tools

再利用する処理本体を置く場所です。実行入口ではなく、`f_run/` から呼ばれます。

## VAE / CMA-ES

```text
internal_state_vae.py  内部状態の読み込み、1D-CNN VAE学習、潜在表現、PCA可視化、Silhouette/DR計算
cma_es_search.py       CMA-ES本体、候補パラメータ生成、cfg反映、VAE潜在空間での評価
```

## LSM共通ツール

```text
internal_state.py      内部状態保存
separation_metrics.py  DR/Sb/Sw/線形分離などの指標
pca.py                 PCA処理
run_paths.py           結果保存パス生成
visualization.py       可視化
```
