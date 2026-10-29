# LSM

3センサーの触覚入力をRI・SIフィルタに通し、LiquidとReadoutの状態記録・学習・素材分類を行います。LSMフォルダを作業ディレクトリにし、NumPy・pandas・Brian2・scikit-learn・Matplotlibを利用できるPythonを選んでください。

## 設定の場所

|場所|内容|
|---|---|
|`a_experiment/experiment_sets.py`|素材・試行数・seed・状態引き継ぎ・debug・学習条件・評価条件・実行環境|
|`a_experiment/paths.py`|入力データと実験結果の場所、結果フォルダの階層|
|`a_experiment/input_filters.py`|入力フィルタとセンサー・フィルタgain|
|`a_experiment/input_processing.py`|CSV読込、センサー・フィルタ経路、1試行の入力電流作成|
|`c_configs/FIXED/`|現在使うニューロン・シナプス・結合・学習則の設定|
|`c_configs/SPACE/`|探索用の候補定義。STDP sweepのseed数もここで指定。epoch数はexperiment_sets.pyのNUM_TRAINING_EPOCHを共用|
|`b_network/`|ネットワーク・モニター・モデル・初期重み|
|`d_tools/`|共通の記録・集計・分類・PCA・描画|
|`h_analysis/`|保存済み状態を読むJupyterノート|

## 現在の既定条件

- Liquidは100個（E80・I20）。入力は3センサー×RI/SIの6経路。
- Readoutは1群80個。STDPの時定数5〜250 msを5 ms刻みで比較する50群を定義しています。
- Liquid E→Readoutの結合確率は0.11、gainは0.2963。I→Readoutの結合は作りません。
- 入力・Liquid共通条件は`c_configs/FIXED/cfg_network.py`、STDP用は`c_configs/FIXED/cfg_network_stdp.py`、off用は`c_configs/FIXED/cfg_network_off.py`です。`--rule off`では出力80個・1群の固定重み構成を選びます。
- 共通の時間刻みは0.1 ms。共通設定`dt_s`は秒単位（0.0001）です。旧設定の`dt_ms`は読み込み時に移行します。
- `SLICE_START=3000`、`SLICE_END=8000`から試行時間500 msを求めます。
- 分類・debug用の時間幅は`TEST_SETS["base"]["BIN_STEPS"]`で共通指定し、現在は25 msです。保存する発火数は1 msビンに統一し、分類時に必要な時間幅へ集約します。
- すべての学習則・off・Liquid単独の既定は、試行ごとに膜電位・シナプス状態・学習トレースを初期化します。学習済みの結合重みは次の試行へ引き継ぎます。校正も固定条件のfresh試行で行います。

## 実行入口

|ファイル|役割|
|---|---|
|`f_run/run_calibration.py`|gain校正・検証・状態保存・固定スパイクの相互相関|
|`f_run/run_liquid.py`|出力層を作らずLiquidのdebug・発火記録・分類評価|
|`f_run/run_train.py`|Readout結合の学習|
|`f_run/run_test.py`|学習済み重み、またはoffの固定重みでテスト|
|`f_run/run_train_test.py`|学習→テスト|
|`f_run/run_train_test_eval.py`|学習→テスト→分類評価|
|`f_run/run_test_eval.py`|保存済みテスト記録の分類評価|


```powershell
# 校正計画を表示するだけ（探索は開始しない）
python f_run/run_calibration.py plan

# offと同じ100試行でLiquidだけを記録・分類する（既定はE80個を分類）
python f_run/run_liquid.py --rep 1 --workers 1

# E/Iを合わせた全100ニューロンで分類する場合（別の保存先で比較）
python f_run/run_liquid.py --population all --out-dir g_tactile_results/liquid_run_all

# 全324試行の記録だけを行う場合
python f_run/run_liquid.py --samples 324 --record-only --out-dir g_tactile_results/liquid_recording_324

# 50時定数×15seedの実行計画を表示するだけ
python f_run/run_train_test_eval.py --sweep --plan

# 保存済みのテスト記録を分類する
python f_run/run_test_eval.py --test-dir <testフォルダ>

# off用の出力80個を固定重みでテストし、分類評価する（既定は試行初期化、保存先off）
python f_run/run_train_test_eval.py --rule off


# 相互相関の計算方法を自己テストする
python f_run/run_calibration.py correlation --self-test
```

`--sweep`では、各seed内でLiquidを共有し、時定数ごとの80ニューロンを同時に計算します。seed間は別プロセスです。`--sweep`なしでは、FIXEDに定義した出力群全体を1つの実験として保存します。学習は8素材を1回ずつ含むランダム順の組で進め、1組ごとに全出力結合を保存します。

Liquid単独の実行本体は`f_run/liquid.py`です。既定ではoffと同じ保存済みのテスト試行ID・素材順を使用し、入力と再帰結合・Liquidのseedも共通です。全100ニューロンを記録し、分類対象だけをE80個（`--population E`）または全100個（`--population all`）に選びます。Liquidのseedは固定なので、rep番号だけを増やしてもLiquidの結合や重みは変わりません。`--resume`では完成済みの記録を再利用して分類します。未完了の記録はテストの最初から再計算します。

```text
g_tactile_results/liquid_run/
├─ network_parameters.json
└─ rep1/test/
   ├─ test_config_snapshot.json
   ├─ test_trial_ids_rep1.npy
   ├─ debug/liquid/
   │  ├─ 素材名/                       # raster・membrane_potential・input_currentのPDF
   │  └─ weight/                       # 入力・再帰結合の重み分布PDF
   ├─ eval/classification.xlsx          # offと同じ3シート形式
   ├─ weight/                          # 入力・再帰結合のみ。出力層への結合なし
   └─ rec/
      ├─ recording.npz
      └─ Liquid/sliq_rec_rep1.npy        # 全100ニューロン、1 msビン
```

Liquid単独・off・各学習則とも、25 ms等へ集約した別ファイルは保存せず、分類時に`rec`内の1 ms記録を指定時間幅へ集約します。

## 処理の置き場所

`f_run/run_*.py`から実験を開始します。学習本体は`f_run/training.py`、テスト本体は`testing.py`、学習とテストの連携は`train_test.py`、評価手順は`evaluation.py`です。`common.py`は設定組み立てとネットワーク・試行実行、`repetitions.py`はrepのプロセス並列、`sweep.py`は時定数比較の管理を担当します。

入力処理は`a_experiment/input_processing.py`で共通化し、通常の学習・テストはキャリブレーション用ファイルを読み込みません。保存形式の変換は`d_tools/recording_io.py`、パラメータ記録は`parameter_recording.py`にまとめ、toolから実行ファイルへの依存をなくしています。

## コードの書式と単位

- Pythonの書式はBlack・1行100文字を基準にし、設定を`pyproject.toml`に置いています。
- 複数の文をセミコロンで連結せず、処理ごとに行を分けます。
- 秒は`_s`、ミリ秒は`_ms`、発火率は`_hz`を目安に、時間・発火率の単位を名前で示します。共通時間刻みは`dt_s`です。
- ニューロンや学習則の標準的なパラメータ名（`tau_plus`など）は維持し、設定・計算箇所で単位を明記します。
- 長い実行処理は、準備・計算・保存・検証を名前付き関数に分けています。

## 保存形式

```text
g_tactile_results/readout_run/
└─ p_E_readout-0.11__g_e_out-0.2963/
   ├─ network_parameters.json
   └─ STDP/
      └─ tau_plus-5.0__tau_minus-5.0/
         ├─ learning_rule_parameters.json
         └─ rep1/
            ├─ training/
            │  ├─ debug/
            │  └─ weight/
            │     ├─ intial/
            │     ├─ final/
            │     └─ Progression_of_weights/
            └─ test/
               ├─ test_config_snapshot.json
               ├─ debug/
               ├─ eval/classification.xlsx
               ├─ weight/
               └─ rec/
                  ├─ Liquid/
                  └─ readout/
```

- JSONはパラメータ記録だけに使います。進捗・集計・接続情報はNPZ、発火数や重みはNPY/NPZです。旧JSON結果は読み込みに対応します。
- 学習済み重みのテストに必要な設定は`training/weight/intial/weights.npz`内に保存し、`training/config_snapshot.json`は新規作成しません。過去のJSON形式も読み込みに対応します。
- 学習中の代表試行では、指定本数の重みと選択ニューロンの膜電位をシミュレーションと同じ刻みで記録します。
- 8素材を1回ずつ提示した区切りで全出力結合を保存し、前回との差の二乗和をCSV・PDFへ記録します。
- 分類結果は`classification.xlsx`1冊に保存します。参照形式と同じ`conf_8cls`（8素材の個数の混同行列）、`conf_3cls`（触感3種類の個数の混同行列）、`accuracy`（正答率・試行数）の3シート構成です。3種類の順は0＝つるつる、1＝ざらざら、2＝中間です。正答率は参照と同じ0〜1の数値で保存します。複数の時間窓を指定した場合は、同じシートの下に条件順で表を追加します。評価条件と分割別結果は、再開時の照合用にブック内の定義名へ保存します。書式は`d_tools/classification_template.xlsx`で管理します。Excel出力はCodex文書ランタイムのNode.jsとartifact-toolを使用し、必要なら`LSM_DOCUMENT_RUNTIME`でdependenciesフォルダを指定します。
- debugには素材ごとのラスター・膜電位・入力電流、学習時の重み変化を保存します。
- offでは`off/rep番号/test/`へ保存し、trainingフォルダは作りません。
- off・各学習則で共通の`tqdm`進捗表示を使い、debugは素材数、学習は提示試行数、テストは素材ごとの試行数、分類評価は分割数を表示します。共有Liquid実行では子プロセスの進捗を実行元のターミナルへ転送します。表示処理は`d_tools/progress.py`で管理します。

## 解析ノート

- `h_analysis/calibration_liquid.ipynb`: LiquidのPCA・分類・応答パターン・ニューロン削減・時間幅別評価。
- `h_analysis/calibration_output.ipynb`: 校正済みLiquid EとReadoutの発火率・PCA・分類・ニューロン削減の比較。

ノートは保存済み校正結果を使います。通常のReadoutテスト記録の分類は`run_test_eval.py`で実行します。入力データと既存の実験結果は保持します。

状態初期化の既定値は`COMMON_SETS["base"]`の`LIQUID_STATE_POLICY`・`READOUT_STATE_POLICY`です。過去の保存済み設定は変更しません。異なる状態条件の結果を同じ保存先へ混在させず、既存結果がある場合は別の保存先を指定してください。offは状態条件によらず`off/rep番号/test/`へ保存します。
