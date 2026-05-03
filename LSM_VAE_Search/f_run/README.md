# Run Scripts

このフォルダは実行入口を置く場所です。
LSM本体と同じ考え方で、直接動かすPythonファイルはここに集約します。

## VAE / CMA-ES

```text
run_fixed_vae_encoder_pretrain.py  複数パラメータ候補で内部状態を作り、固定VAE Encoderを学習
run_cma_es_search.py               固定VAE Encoderを使ってCMA-ES探索
plot_cma_es_progress.py            CMA-ES履歴の可視化
run_internal_state_vae.py          単一liquid結果に対するVAE学習・可視化
```

## LSM互換実行

```text
run_training.py                    training互換用
run_liquid.py                      liquid内部状態生成用
run_test.py                        保存重みを使ったtest用
run_test_classification.py         test結果の分類評価用
run_separation_metrics.py          内部状態からDR/Sb/Swなどを計算
plot_internal_states.py            内部状態ヒートマップの再描画
```

処理本体は `d_tools/`、設定は `c_configs/` に置きます。
