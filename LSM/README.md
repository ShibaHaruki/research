# LSM

入力gain・EE・EI・IEの順次調整と、8素材×324試行の状態記録に必要なコードだけを残しています。
リキッド校正の入口は[`f_run/run_calibration.py`](f_run/run_calibration.py)です。出力層の学習・テストの入口は下の案内を参照してください。

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
|[`f_run/run_calibration.py`](f_run/run_calibration.py)|①〜⑩の設定・探索・状態保存|
|[`f_run/run_common.py`](f_run/run_common.py)|入力データ読み込み・フィルタ・設定の組み立て|
|[`b_network/build_network.py`](b_network/build_network.py)|ニューロン・結合とネットワーク全体の構築|
|[`b_network/monitor.py`](b_network/monitor.py)|発火・膜電位モニタの作成、E/I記録の統合、記録対象の選択|
|`a_tactile/`|入力フィルタとデータの基本条件|
|`b_network/`|ニューロン・シナプス・結合構築|
|`c_configs/FIXED/`|共通設定、固定した結合と学習・評価分割|
|`d_tools/`|発火活動集計・PCA・分類・評価・描画・重み保存|
|`g_tactile_results/`|保存済みの実験結果|
|[`h_analysis/calibration_liquid.ipynb`](h_analysis/calibration_liquid.ipynb)|リキッド層のPCA・分類精度・応答パターンの解析|
|[`h_analysis/calibration_output.ipynb`](h_analysis/calibration_output.ipynb)|リキッド層と出力層のPCA・分類精度を比較|

旧調整・検証・PCA解析・ニューロン分類の独立スクリプト、旧実験実行入口、未使用の設定候補とノートブックは削除しています。
分類・混同行列の作成は解析側で行います。既存の実験結果、入力データ、Python環境は保持しています。

校正用の数値計算は `f_run/run_calibration.py` 内のNumPy処理にまとめています。Cythonのビルドは不要です。

## 状態の保存形式

既定の保存先は `g_tactile_results/calibration/all_materials/<素材>/features.npz` です。

- `states_hz`: 324試行×20時間ビン×100ニューロンの発火率（Hz）。1ビン25 ms、1試行500 ms。E80個にI20個が続きます。
- `counts_25ms`: 既存の解析用の発火回数（324試行×100ニューロン×20時間ビン）。
- `trial_ids`、`neuron_ids`、`neuron_types`、`bin_ms`、`time_bin_edges_ms`: 各軸の対応。

試行ごとに状態をリセットします。全試行の膜電位・シナプス変数を保存する形式ではありません。
各素材の `debug_sid203/` に、代表試行の発火ラスタ・E/I平均発火率・選択10ニューロンの膜電位をまとめた `debug.png` と記録データ `debug.npz` を保存します。

## LSMへの統合（2026-10-20）

現行のコード・tool・設定・解析・Python環境はLSM_liquidの構成を引き継いでいます。
旧LSMのコードは `_before_liquid_merge_20261020/` に退避しました。旧LSMの実験結果は `g_tactile_results/` に残し、現在の `calibration/` を追加しています。
旧版の実行入口は退避先に保存しています。現在は `workflow.py`、`f_run/` の各入口と、現行構成に合わせた `run_experiment.py` を使用します。
移動後はこのLSMフォルダを開き、Pythonに `.mlflow-venv/Scripts/python.exe`、Jupyterに `Python (LSM)` を選んでください。

## 出力層

`b_network/build_network.py`で、リキッド層の後ろに固定重みの出力層を構築できます。設定は `c_configs/FIXED/cfg_network.py` にあります。

- 出力：40個の全興奮性LIFニューロン（`N_out`）。
- リキッドE→出力E：結合確率0.11、gain0.2963（gainは確率10%での校正値のため、発火率を合わせる場合は再校正）。リキッドI→出力E：確率・gainとも0で、結合は作りません。
- 出力層への直接センサー入力、出力層内の再帰、リキッドへの戻り結合はありません。
- 出力gainは全8素材×324試行でリキッドEと出力の平均発火率を合わせた採用値です。学習の有無は選択した学習則で決まります。

各素材のリキッド・出力層のdebugは、校正・学習・テストの実行時に共通の `d_tools/debug.py` で作成します。

## 出力層の学習・テスト・分類評価

旧LSMの5つの入口を現行構成へ組み込みました。実行の共通処理は `f_run/run_common.py`、重みの保存・復元は `d_tools/weight_export.py`、分類評価は `d_tools/evaluation.py` にまとめています。

### 一括実行

```powershell
.\.mlflow-venv\Scripts\python.exe .\f_run\run_train_test_eval.py --rule SRDP --out-dir .\g_tactile_results\readout_SRDP_01
```

同じ処理を短い入口から実行できます。

```powershell
.\.mlflow-venv\Scripts\python.exe .\workflow.py train-test-eval --rule SRDP --out-dir .\g_tactile_results\readout_SRDP_01
```

- 学習則は `--rule off/STDP/T_STDP/SRDP/BCPNN` で選びます。未指定時は `cfg_models.py` の設定（初期状態はoff）に従います。
- 可塑性を適用するのはリキッドE→出力Eだけです。I→出力は接続せず、入力・リキッド内の結合は固定します。
- SRDPの `A_minus` は式中で引く正の係数、STDPの `A_minus` は加える負の係数です。T-STDPの正の抑圧項は減算します。パラメータ値は最適化したものではありません。
- 試行数とエポックは `a_tactile/experiment_sets.py`、学習則パラメータは `c_configs/FIXED/cfg_learning_rule_models.py`、出力層の数・結合は `cfg_network.py` で設定します。
- 既定では全8素材、素材ごとに学習100試行・テスト100試行、1エポック、分類は10-fold、25 msビンです。
- 同一repの学習とテストの試行IDは重複させません。全素材で同じID分割を使い、学習時の素材・試行順はエポックごとにシャッフルします。
- 膜電位・シナプス状態は既定で次の試行へ引き継ぎ、素材切替時の無入力100 msは入れません。`--state-policy reset_each_trial` で試行ごとの初期化へ変更できます。学習からテストへは重みだけを渡し、動的状態は初期化します。
- `NUM_REPEAT` に従って繰り返します。`--rep 1` なら1回目だけを実行します。一括実行では `--rep-end` で範囲終端も指定できます。
- 分類は出力スパイクの時間ビン特徴を使う、標準化なしの正則化マハラノビス分類です。出力層自体の学習は教師なしで、分類器をテスト記録内の層別交差検証で学習・評価します。

### 個別実行

```powershell
# 学習だけ
.\.mlflow-venv\Scripts\python.exe .\f_run\run_training.py --rule SRDP --out-dir .\g_tactile_results\readout_SRDP_01

# 保存した重みでテストだけ
.\.mlflow-venv\Scripts\python.exe .\f_run\run_test.py --train-dir .\g_tactile_results\readout_SRDP_01

# 学習とテストだけ
.\.mlflow-venv\Scripts\python.exe .\f_run\run_train_test.py --rule SRDP --out-dir .\g_tactile_results\readout_SRDP_02

# 保存したテスト出力を分類
.\.mlflow-venv\Scripts\python.exe .\f_run\run_test_classification.py --test-dir .\g_tactile_results\readout_SRDP_01\test --folds 10 --tn-ms 25
```

### 保存内容

指定した実験フォルダに、設定スナップショット、学習・テストID分割、学習履歴、初期・最終の結合重みを保存します。`test/sout_rec_rep1.npy` は「素材×テスト試行×出力ニューロン×時間ビン」の発火回数です。
`test/results_10fold/` に分類正答率CSV・混同行列CSV/PNG・評価条件を保存します。既に学習済み・テスト済みの結果がある場合は上書きせず停止するため、別の `--out-dir` を使用してください。
解析ノートブックの表示専用設定はそのままです。

## 校正済み設定でリキッド単体を実行する

`f_run/run_liquid.py` はgain探索を行わず、FIXEDの現在の設定でリキッド単体を実行します。出力層・可塑性学習・自動PCAは無効です。

```powershell
# 8素材それぞれ全324試行を記録
.\.mlflow-venv\Scripts\python.exe .\f_run\run_liquid.py --samples 324

# workflowからも同じ処理を呼べる
.\.mlflow-venv\Scripts\python.exe .\workflow.py liquid --samples 324

# 各素材の特定1試行だけ確認する例
.\.mlflow-venv\Scripts\python.exe .\workflow.py liquid --materials aluminum_board Japanese_paper --trial-ids 203 --out-dir .\g_tactile_results\liquid_run\debug_203
```

引数なしの場合、`a_tactile/experiment_sets.py` の `LIQUID_SETS`（現在は8素材・各100試行）を使用します。素材間で同じ試行ID順を使い、順序は `sample_seq_liquid.npy` に保存します。

既定の保存先は `g_tactile_results/liquid_run/recording/` です。`all_materials/<素材>/features.npz` の `states_hz` は「試行×20時間ビン×100ニューロン」の発火率、`counts_25ms` は「試行×100ニューロン×20時間ビン」の発火回数です（現在の100ニューロン・25 ms・500 ms設定の場合）。入力信号自体はコピーしません。

各素材の最初の1試行について、`debug_sid<試行番号>/layer_1/` に発火ラスタ・平均発火率・選択ニューロン膜電位のPNGとデータを保存します。設定スナップショットと結合重みも保存します。

状態は `cfg_run.py` の `LIQUID_STATE_POLICY`（初期値carry）に従って試行間で引き継ぎます。初期化して記録する場合は `--state-policy reset_each_trial` を指定してください。素材切替の100 ms無入力期間は入れません。既存フォルダへは上書きせず停止します。別条件では新しい `--out-dir` を指定してください。

`run_calibration.py` はgain探索とfresh試行での記録、`run_liquid.py` は設定済みのネットワークの記録に使い分けます。

## EXPERIMENTSの実行

`c_configs/EXPERIMENTS/` に、実験ごとの設定上書きをまとめています。`run_experiment.py` は実験名または実験定義.pyのパスを読み、現在の入口を順次実行します。

```powershell
.\.mlflow-venv\Scripts\python.exe .\run_experiment.py exp_001_base --dry-run
.\.mlflow-venv\Scripts\python.exe .\workflow.py experiment exp_002_gain_sweep_example --dry-run
```

定義例と実行方法は [c_configs/EXPERIMENTS/README.md](c_configs/EXPERIMENTS/README.md) を参照してください。


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
E→出力の結合確率は11%（出力40個すべてが1本以上の入力を持つ確率は約99.64%）、I→出力なし、学習無効、試行ごとに状態を初期化します。

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
