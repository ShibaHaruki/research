# 実行入口

処理は現在の `b_network`・`c_configs/FIXED`・`d_tools` を使用します。旧コードの退避先をimportすることはありません。

|ファイル|役割|
|---|---|
|`run_calibration.py`|入力・EE・EI・IE・出力gainの調整、全素材の状態保存・debug|
|`run_liquid.py`|現在の設定でリキッド単体を実行し、状態と代表試行debugを保存|
|`run_training.py`|E→出力結合の学習と初期・最終重み保存|
|`run_test.py`|学習時の設定・重みで、重みを固定してテスト|
|`run_train_test.py`|学習→テスト|
|`run_train_test_eval.py`|学習→テスト→出力を使った素材分類評価|
|`run_test_classification.py`|保存済みテスト出力から分類評価だけを実行|
|`run_common.py`|データ読込・フィルタ、設定管理、試行分割、ネットワーク準備、試行実行、debug呼び出し|

詳しいコマンドは[プロジェクトの案内](../README.md)を参照してください。


各素材1試行のdebugは `d_tools/debug.py` に共通化しています。
- 校正・リキッド記録：従来どおり各素材の代表試行を確認します。リキッド記録では最初に記録する試行を使います。
- 学習・テスト：終了後の固定重みを別ネットワークへ読み込み、各素材を1回ずつ確認します。学習・テスト本体の状態には加えません。
- 保存済み記録を分類する処理はネットワークを動かさず、テスト時のdebugを利用します。

学習・テストのdebugは `c_configs/FIXED/cfg_run.py` の `DEBUG_EACH_MATERIAL`（標準True）、`DEBUG_TRIAL_ID`（標準203）で指定します。
保存先は学習フォルダの `debug/training_rep1/`、テストフォルダの `debug/test_rep1/` 以下です。
各素材・各層について `debug.png`（発火・集団発火率・膜電位）と `debug.npz` を保存します。


出力層のgain校正も `run_calibration.py` の通常の `run` に含まれます。
リキッドのgainを固定し、全8素材×324試行で、リキッドE80個と出力層の平均発火率を合わせます。
各試行の500 ms内の発火回数をニューロン数と試行時間で割ってHzを計算し、2592試行を等しく重み付けします。
差を「リキッドEの発火率 − 出力層の発火率」とし、正なら出力gainを増やし、負なら減らします。
調整幅を半分ずつ狭め、0.0001刻みの評価済み候補から、全試行の平均差の絶対値が最も小さいgainを採用します。
同じgainを全試行に使うため、試行ごとに完全一致することを要求する条件ではありません。
E80個→出力40個は全結合（100%、各出力80入力、総結合数3,200本）、I→出力なし、学習無効、試行ごとに状態を初期化します。

```powershell
.\.mlflow-venv\Scripts\python.exe .\f_run\run_calibration.py output --liquid-dir .\g_tactile_results\calibration --out-dir .\g_tactile_results\sequential_output_rate_match_new
```

`--out-dir`には未作成のフォルダを指定します。
結果の `output_calibration/all_materials/<素材>/liquid/features.npz` と `output/features.npz` に全324試行を保存します。
各ファイルの `states_hz` は「試行×時間ビン×ニューロン」、`counts_25ms` は「試行×ニューロン×時間ビン」です。
同じ素材フォルダの `debug_sid203/` にリキッド・出力それぞれのdebug画像と元データを保存します。
`trial_rate_comparison.csv` は2592試行ごとの差、`material_rate_comparison.csv` は素材ごとの平均です。
`output_summary.json` と `search_history.csv` に採用値・探索履歴を保存します。
`selected_parameters.json` の出力gainと `config_snapshot.json` が校正用設定です。通常のFIXED設定は自動変更しません。

校正用の数値計算は `run_calibration.py` 内のNumPy処理で行います。別の高速モジュールやビルド操作は不要です。

## SVDに基づく120出力の実験

`run_svd_output120.py` は保存済み校正結果を使う独立した実験です。通常のFIXEDの出力数40を変更せず、PC1〜PC13の60パターンを正負2枝ずつの120出力へ写します。

```powershell
python f_run/run_svd_output120.py prepare
python f_run/run_svd_output120.py run --resume
```

E80個→出力120個を全結合とし、結合gainはmax(a,0)/max(-a,0)、共通倍率は1です。ゼロgainの接続はシミュレーション上の入力を作りません。`--gain-scale` を変える場合は別の `--out-dir` を指定してください。PCA/SVDの学習データ、結合gain、全状態、各素材sid203のdebug、分類結果を `g_tactile_results/svd_signed_output120` へ保存します。図・表の表示は `h_analysis/svd_signed_output120.ipynb` を使います。既存の校正データは上書きしません。

### SVD120出力の平均発火率合わせ

`python f_run/run_svd_output120_rate_match.py` で、SVD由来の相対gainを保った共通倍率を調整します。目標は学習1,808試行におけるリキッドE80個の平均発火率で、出力側は無活動枝も含む全120個の平均です。評価784試行は倍率の選択に使いません。倍率1の結果を残し、調整後の全素材状態・debug・分類結果は `g_tactile_results/svd_signed_output120_rate_matched` に保存します。許容差の既定値は1 Hz、倍率の刻みは0.0001です。
