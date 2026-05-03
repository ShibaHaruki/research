# LSM_VAE_Search

このフォルダは、LSM本体から切り離した VAE / CMA-ES パラメータ探索用の独立プロジェクトです。
フォルダ名はLSM本体の構造に合わせ、既存の役割フォルダだけを使います。
触覚データは従来通り、親フォルダの `tactile_data` を参照します。

## フォルダ構成

- `a_tactile/`: 触覚入力と入力フィルタ。
- `b_network/`: LSMネットワーク構築、モデル定義、重み初期化。
- `c_configs/`: 固定設定、探索空間、VAE/Search用設定。
- `c_configs/VAE_SEARCH/`: 固定VAE Encoder学習とCMA-ES探索の設定。
- `d_tools/`: VAE、CMA-ES、PCA、分離指標、保存・描画などの再利用処理。
- `f_run/`: 実行するPythonファイル。
- `g_tactile_results/`: VAE学習、CMA-ES探索、内部状態などの保存先。

## 主に編集する場所

パラメータ群:

```text
c_configs/VAE_SEARCH/parameter_values_config.py
```

CMA-ES探索範囲・目的関数・VAE事前学習フィルタ:

```text
c_configs/VAE_SEARCH/cma_es_internal_state.py
```

VAEやCMA-ESの処理本体:

```text
d_tools/internal_state_vae.py
d_tools/cma_es_search.py
```

実行ファイル:

```text
f_run/run_fixed_vae_encoder_pretrain.py
f_run/run_cma_es_search.py
```

## 実行例

```powershell
cd "C:\Users\haru4\OneDrive - 学校法人立命館\ドキュメント\研究\研究コード\LSM_VAE_Search"
python .\f_run\run_fixed_vae_encoder_pretrain.py --parameter-values-config parameter_values_config.py --samples-per-material 100
python .\f_run\run_cma_es_search.py --fixed-vae-encoder-dir "固定Encoderのfixed_encoder_vaeフォルダ"
```
