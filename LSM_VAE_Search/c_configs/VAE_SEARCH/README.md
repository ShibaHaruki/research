# configs

VAE / CMA-ES探索で使う設定ファイルを置く場所です。

- `parameter_values_config.py`: 固定VAE Encoderの共通潜在空間を作るために、手入力したリキッド層パラメータ群を定義します。
- `cma_es_internal_state.py`: CMA-ESの探索範囲、目的関数、発火率・同期ペナルティなどを定義します。

数値を変えて探索条件を調整したいときは、基本的にこのフォルダだけを触ります。

