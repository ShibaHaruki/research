# 実行入口

入力作成は`a_experiment/input_processing.py`へ、記録形式は`d_tools/recording_io.py`へまとめています。ネットワークやモデルの処理は`b_network`、`c_configs/FIXED`、`a_experiment`、`d_tools`を使います。

|ファイル|役割|
|---|---|
|`run_calibration.py`|gain校正・検証・状態記録・相互相関解析|
|`run_liquid.py`|現在のLiquidだけを動かして状態と代表試行debugを保存|
|`run_train.py`|Liquid E→Readoutの学習、初期・最終重みと学習推移の保存|
|`run_test.py`|保存済み学習重み、またはoffの固定重みでテスト|
|`run_train_test.py`|学習→テスト|
|`run_train_test_eval.py`|学習→テスト→素材分類|
|`run_test_eval.py`|保存済みテスト記録の分類だけを実行|
|`common.py`|独立した設定の組み立て、試行分割、ネットワーク・試行実行|
|`training.py`|通常学習と共有Liquidの学習処理|
|`testing.py`|固定重みテスト、発火記録、保存済み重みの読み込み|
|`train_test.py`|学習からテストへの連携と共有Liquidのテスト|
|`evaluation.py`|評価条件の検証と、学習・テスト・分類の連携|
|`repetitions.py`|repを単位とした独立プロセスの並列実行|
|`sweep.py`|時定数候補の準備・seedごとの子プロセス管理・再開|

## 設定・記録

- 新規実験では各repの結果フォルダに`split_rep*.npz`を作りません。分割は共通の`sample_seq/sample_seq_rep*.npz`から読みます。過去の実験に残る旧分割ファイルは、再現用に読み込みのみ対応します。

- 試行番号は`g_tactile_results/sample_seq/sample_seq_liquid.npy`から読み込みます。rep1なら分割を`sample_seq_rep1.npz`へ保存し、rep2以降も同じ形式でrep番号ごとに保存します。以後は各ファイルの`train_ids`・`test_ids`を読み込みます。`unused_ids`には残り124試行を保存します。元の順序ファイルは上書きせず、存在しない初回だけseedから作成します。明示的に別の`DATA_SPLIT_REP`を指定した場合は、同フォルダ内の`seed_<seed値>/`へ分けます。

- 通常実行のrepは出力のseed番号です。Liquidのseedは固定し、出力のseedは`BASE_SEED + rep`にします。同じrepなら学習則・時定数によらず同じサイズ・確率・gainの出力結合と初期重みが一致します。学習・テスト分割は既定で共通（`DATA_SPLIT_REP=1`）です。繰り返し数は`COMMON_SETS["base"]["NUM_REPEAT"]`で指定します。
- `FIXED/cfg_network.py`は入力・Liquid共通条件、`FIXED/cfg_network_stdp.py`はSTDP時定数50候補の構成、`FIXED/cfg_network_off.py`は固定重みの出力80個・1群の構成です。通常実行で`--rule off`を指定するとoff用を選びます。入力・Liquid・結合確率・gainはSTDP用と共通です。
- 共通条件・debugは`a_experiment/experiment_sets.py`の`COMMON_SETS["base"]`、重みの詳細記録本数は`TRAINING_SETS["base"]`、評価条件は`EVAL_SET["base"]`にあります。
- 学習中のdebugは各素材が最初に提示された実際の試行を記録します。重み変化も学習中の値です。
- 固定重みテストは、最初に別ネットワークで`DEBUG_TRIAL_ID`の試行を各素材1回ずつdebugします。その後、時計・状態を初期化したネットワークへ同じ重みを読み直し、全試行のテスト→分類評価を行います。rep並列時も各repがこの順で進みます。
- 試行時間は切り出し範囲から、発火数ビン幅は共通の`BIN_STEPS`から計算します。
- JSONはパラメータだけに使い、summary・recording・topology・進捗はNPZへ保存します。旧結果のJSONも読み込めます。
- `test/eval/`には`classification.xlsx`を1冊だけ保存します。8素材と触感3種類は同じ予測を集約した結果で、3種類の分類器を別学習する処理ではありません。

## 校正・相互相関

```powershell
python f_run/run_calibration.py plan
python f_run/run_calibration.py run
python f_run/run_calibration.py run --resume
python f_run/run_calibration.py verify --out-dir <校正フォルダ>
python f_run/run_calibration.py output --liquid-dir <Liquid校正フォルダ> --out-dir <新しい保存先>
python f_run/run_calibration.py correlation --self-test
python f_run/run_calibration.py correlation
```

引数なしで`run_calibration.py`を実行すると相互相関解析を行います。gain探索を始める場合は`run`を指定します。

## 共有Liquidの時定数比較

通常実行でも`--workers`でrepごとの並列数を指定できます。例えば`python f_run/run_train_test_eval.py --rule off --rep 1 --rep-end 15 --workers 5`はrep1〜15を最大5プロセスで実行し、各repでテスト・分類を完結します。rep範囲を省略すると`NUM_REPEAT`に従います。

```powershell
# 実行計画だけを確認する
python f_run/run_train_test_eval.py --sweep --plan

# 実験を実行する場合
python f_run/run_train_test_eval.py --sweep

# 同じ条件で再開する場合
python f_run/run_train_test_eval.py --sweep --resume
```

時定数候補・各出力群の学習則・パラメータは`FIXED/cfg_network_stdp.py`、seed数・epochは`SPACE/cfg_learning_rule_models.py`を使います。各seedは同じ学習・テスト分割を使い、時定数群は同じ初期結合・初期重みから始めます。固定データを共有するため、候補間でLiquid設定や試行時間が異なる場合は実行を拒否します。

保存先の詳細と通常の入口は[プロジェクトの案内](../README.md)を参照してください。
