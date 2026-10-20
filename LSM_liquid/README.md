# LSM_liquid

入力gain・EE・EI・IEの順次調整と、8素材×324試行の状態記録に必要なコードだけを残しています。
実験の入口は[`f_run/calibrate_liquid.py`](f_run/calibrate_liquid.py)です。

```powershell
# 条件・手順の確認
.\.mlflow-venv\Scripts\python.exe .\workflow.py plan

# ①〜⑩の調整・状態記録
.\.mlflow-venv\Scripts\python.exe .\workflow.py run

# 同じ条件で途中再開
.\.mlflow-venv\Scripts\python.exe .\workflow.py run --resume
```

gain調整後は全素材の状態記録と各素材1試行のdebugを保存します。分類・PCAは自動実行せず、[解析ノートブックの案内](h_analysis/README.md)から実行します。

|場所|役割|
|---|---|
|[`f_run/calibrate_liquid.py`](f_run/calibrate_liquid.py)|①〜⑩の設定・探索・状態保存|
|[`f_run/run_common.py`](f_run/run_common.py)|入力データ読み込み・フィルタ・設定の組み立て|
|[`b_network/build_network.py`](b_network/build_network.py)|ニューロン・結合とネットワーク全体の構築|
|[`b_network/monitor.py`](b_network/monitor.py)|発火・膜電位モニタの作成、E/I記録の統合、記録対象の選択|
|`a_tactile/`|入力フィルタとデータの基本条件|
|`b_network/`|ニューロン・シナプス・結合構築|
|`c_configs/FIXED/`|共通設定、固定した結合と学習・評価分割|
|`d_tools/`|発火活動集計・PCA・分類・評価・描画・重み保存|
|`g_tactile_results/`|保存済みの実験結果|
|[`h_analysis/incremental_pca_accuracy.ipynb`](h_analysis/incremental_pca_accuracy.ipynb)|JupyterでPC数1〜40と分類正答率を計算・描画|

旧調整・検証・PCA解析・ニューロン分類の独立スクリプト、旧実験実行入口、未使用の設定候補とノートブックは削除しています。
分類・混同行列の作成は解析側で行います。既存の実験結果、入力データ、Python環境は保持しています。

`fast_liquid_calibration.pyx`と対応する`.pyd`、そのビルド用スクリプトは統合版で必要なため残しています。

## 状態の保存形式

既定の保存先は `g_tactile_results/sequential_calibration/all_materials/<素材>/features.npz` です。

- `states_hz`: 324試行×20時間ビン×100ニューロンの発火率（Hz）。1ビン25 ms、1試行500 ms。E80個にI20個が続きます。
- `counts_25ms`: 既存の解析用の発火回数（324試行×100ニューロン×20時間ビン）。
- `trial_ids`、`neuron_ids`、`neuron_types`、`bin_ms`、`time_bin_edges_ms`: 各軸の対応。

試行ごとに状態をリセットします。全試行の膜電位・シナプス変数を保存する形式ではありません。
各素材の `debug_sid203/` に、代表試行の発火ラスタ・E/I平均発火率・選択10ニューロンの膜電位をまとめた `debug.png` と記録データ `debug.npz` を保存します。
