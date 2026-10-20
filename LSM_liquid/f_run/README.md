# 統合した実行コード

実験の入口は[`calibrate_liquid.py`](calibrate_liquid.py)だけです。
設定・保存形式・コマンドは[プロジェクトの案内](../README.md)を参照してください。

|ファイル|役割|
|---|---|
|`calibrate_liquid.py`|入力・EE・EI・IE調整、8素材全試行の状態保存、各素材1試行のdebug|
|`run_common.py`|入力データ・フィルタ処理と設定の組み立て|
|`fast_liquid_calibration.pyx`|検証済みの高速シミュレーション|
|`fast_liquid_calibration.cp313-win_amd64.pyd`|現在のPython環境用の高速モジュール|
|`build_fast_liquid_calibration.py`|高速モジュールの再ビルド用|
|`__init__.py`|Pythonパッケージ定義|

高速モジュールを再ビルドする場合は、`f_run`を作業フォルダとしてビルド用スクリプトを実行します。
