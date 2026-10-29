# 統合版の共通処理

|ファイル|役割|
|---|---|
|`pca.py`|状態配列のPCA、ローディングのヒートマップ・時間波形・横並び図の作成（標準化なし）|
|`activity.py`|発火率、試行ごとの無活動数・割合、全試行での無活動数・割合を集計|
|`classification.py`|学習特徴とラベルから正則化マハラノビス分類器を作り、評価特徴のラベルと距離を算出|
|`evaluation.py`|正答率・クラス別正答率・混同行列の集計、保存済み出力発火記録の時間窓集約・交差検証・結果保存|
|`evaluation_excel.py`|8素材・3種類分類の全結果をclassification.xlsxにまとめて保存し、再開用の評価条件も読み出す|
|`plotting.py`|混同行列を英語表記・0〜100%の共通色スケールで保存|
|`weight_export.py`|重み行列の出力、結合構造付き重みの保存・復元、出力重みの集計、debugのLiquid/Readout別重み分布PDFの保存|
|`training_recording.py`|実学習中の代表試行debug、指定本数の結合の時系列、全素材を1回提示するごとの結合保存と重み変化二乗和の推移|
|`debug.py`|共通試行記録、入力フィルタ後の信号と実入力結合に基づくLiquid外部入力の抽出、debug保存|
|`parameter_recording.py`|設定の独立したスナップショット、共通設定JSONと学習則パラメータJSONの保存・同条件の照合|
|`recording_io.py`|タプルキー・NumPy値・パスの変換、パラメータJSON・集計NPZの保存、旧JSON読込|
|`__init__.py`|Pythonパッケージ定義|

キャリブレーションの順序・合格基準・設定反映は`f_run/run_calibration.py`にまとめています。設定反映は同ファイルの`config()`が担当し、再利用する集計・分類・評価・描画はこのフォルダに分けています。
入力の経路・CSV取得・フィルタ適用は`a_experiment/input_processing.py`で共通化しています。ネットワーク構築時は`f_run/common.py`で現在の`(センサー番号, フィルタ名)`→`layers`形式を直接読みます。旧形式の変換処理は使用しません。

## 集計・分類・評価・描画

```python
from d_tools.activity import summarize_activity
from d_tools.classification import classify
from d_tools.evaluation import evaluate_classification
from d_tools.plotting import save_confusion_matrix

# counts.shape = (試行数, ニューロン数, 時間ビン数)
activity = summarize_activity(counts, bin_ms=25, neuron_ids=[0, 1, 2])
# activity['trial_neuron_rate_hz']: 試行×対象ニューロンの発火率（Hz）
# activity['silent_neuron_percent_per_trial']: 各試行で無活動だった割合（%）
# activity['never_active_neuron_ids']: 全試行で一度も発火しなかったニューロン

# train_features/test_features は「試行×特徴」。PCAスコアも渡せる。
predicted, distances = classify(train_features, train_labels, test_features,
                                return_distances=True)
metrics = evaluate_classification(test_labels, predicted, labels=class_labels)
save_confusion_matrix(metrics['confusion_matrix'], class_labels,
                      'confusion.png', title='Classification results')
```

分類器を別の評価データへ使い回す場合は`fit_classifier()`でモデルを作り、`predict_classifier()`へ渡します。
距離は正則化したマハラノビス距離の二乗で、列はモデルの`classes`順です。
`accuracy`と`per_class_accuracy`は0〜1、`confusion_percent`は0〜100です。
発火率の計算には状態値ではなく、各時間ビン内のスパイク数を渡してください。

## 状態配列からPCA

```python
from d_tools.pca import fit_transform_pca, transform_pca

# states_train.shape = (学習試行数, ニューロン数, 時間ビン数)
scores_train, model = fit_transform_pca(states_train, n_components=40)
scores_test = transform_pca(states_test, model)
```

発火率・発火数・膜電位・シナプス状態などの有限の実数値を入力できます。
ニューロン数と時間ビン数は自動で読み取り、各ニューロン・各時間ビンを1特徴として並べます。
出力は「試行数×主成分数」です。時間方向の平均は取りません。

一度のPCAでは、各試行および学習・評価間でニューロンと時間ビンの数・並び・意味をそろえてください。
形が違えば、特徴数の積が同じでもエラーにします。
主成分数を省略すると、40・特徴数・学習試行数−1の最小値を使います。
分類評価では学習試行だけで軸を求め、評価試行には同じモデルを適用します。

## PCAのローディング図

```python
from d_tools.pca import fit_transform_pca, save_loading_plots

# 3次元の状態からPCAとローディング図をまとめて作る。
scores, model = fit_transform_pca(states_train, n_components=40,
                                 output_dir='pca_results', bin_ms=25)

# すでに求めたモデルから図だけ作ることもできる。
paths = save_loading_plots(model, 'pca_results', bin_ms=25)
```

- ローディングは「主成分係数×√主成分の分散」。スコア計算に使う係数そのものとは分けて保存します。
- ヒートマップは「ニューロン×時間ビン」で、全PCで共通の対称な色範囲を使います。
- 時間波形は、各時間ビンのローディングをニューロン方向に符号付き平均したものです。帯は付けません。
- 時間座標は各ビンの中心です。bin_ms=25なら12.5、37.5、…msになります。
- 数値は`PC01_PC40_loading_data.npz`にも保存します。PC数が少ないときはファイル名もその数に合わせます。
- 図のPC寄与率は、保持した40成分内での比率ではなく、元の学習状態の総分散に対する比率です。

2次元入力のモデルでは、`feature_shape=(ニューロン数, 時間ビン数)`を`save_loading_plots()`に指定してください。
特徴を間引いた場合は、元の特徴順のbool配列を`feature_mask`に渡すと、除外した位置の係数を0で補います。

PCAの解析と図の表示は`h_analysis`のノートブックから実行します。通常のテスト・分類実行では自動PCAを行いません。
評価データをPCAの学習へ混ぜず、10分割の異なる主成分軸の平均も取りません。

- `evaluation.py`：基本の分類指標に加え、保存済み出力スパイクの時間窓集約、層別交差検証、分類結果保存を担当します。分類器は `classification.py`、描画は `plotting.py` を使います。

- `debug.py`：リキッド・出力層に共通する1試行の実行、発火ビン集計、膜電位抽出、debug画像・元データの保存。学習・テストの各素材1試行確認にも使用する。
  出力層の実験では`debug/liquid/素材名/`と`debug/readout/素材名/`に分け、
  全ニューロンのラスタを`raster.pdf`、最大10ニューロンの膜電位を`membrane_potential.pdf`へ保存する。


学習・テストに固有の設定管理・試行分割・ネットワーク準備・状態引き継ぎは `f_run/common.py` が担当します。各素材のdebug実行・集計・保存は `debug.py` の共通処理を利用します。

## 記録ファイルの形式

パラメータ（共通・学習則・実行設定のsnapshot）だけをJSONで保存します。進捗・集計・接続情報などは`recording_io.py`を使ってNPZへ保存します。NPZ内の階層データは文字列として保持し、pickleは使用しません。旧JSONは読み込み専用で対応します。Excelへ渡す中間データは標準入力を使用し、一時JSONファイルも作成しません。
