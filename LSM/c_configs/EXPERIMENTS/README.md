# 実験定義（EXPERIMENTS）

各Pythonファイルで `EXPERIMENT` 辞書を定義します。通常のFIXED設定のコピーに、その実験だけの上書きを適用します。FIXEDファイル自体は書き換えません。

- `exp_001_base.py`：現在のFIXEDをそのまま使うベースライン。
- `exp_002_gain_sweep_example.py`：E→出力のgainを0.5・1.0・2.0で比較する例。I→出力は確率・gainとも0のままです。

```powershell
# 実行せず、展開される条件・保存先を確認する
.\.mlflow-venv\Scripts\python.exe .\run_experiment.py exp_002_gain_sweep_example --dry-run

# 1条件だけ実行する
.\.mlflow-venv\Scripts\python.exe .\run_experiment.py exp_002_gain_sweep_example --trial liq_out_gain_0p5

# 学習→テスト→分類評価を行う
.\.mlflow-venv\Scripts\python.exe .\run_experiment.py exp_001_base --target train_test_eval
```

これらの例は、学習則もFIXEDの選択に従います。初期設定のoffなら可塑性学習は行いません。SRDPなどを使う場合は `models.LEARNING_RULE_MODEL` を実験定義で上書きしてください。

```python
EXPERIMENT = {
    "name": "my_srdp_experiment",
    "target": "train_test_eval",
    "base_overrides": {
        "models.LEARNING_RULE_MODEL": "SRDP",
    },
    "trials": [
        {
            "id": "tau_pre_10",
            "overrides": {"learning_rule_models.SRDP.tau_pre": 10.0},
        },
        {
            "id": "tau_pre_20",
            "overrides": {"learning_rule_models.SRDP.tau_pre": 20.0},
        },
    ],
}
```

文字列キーはドット区切りのパスです。リスト添字やIN_ROUTEのタプルキーを含む場合はタプルのパスを使います。

```python
("network", "gain_liq_to_out_pairs", 0, "EE"): 0.6
("network", "IN_ROUTE", (0, "RI"), "layers", 0, "scale", "E"): 179.0
```

`trials`の代わりに `grid`（パス→候補値リスト）を置くと直積に展開します。大きな候補集合は実行前に必ずdry-runで件数を確認してください。

実行先は `training`・`liquid`・`test`・`train_test`・`train_test_eval`・`classify` です。`--target`は実験定義のtargetより優先されます。`liquid`では出力なし、`test`と`classify`では同じ実験・trialの保存済み学習結果を使います。

既定の保存先は `g_tactile_results/experiments/<実験名>/<trial ID>/` です。学習結果は各trialの `training/`、テストは `training/test/`、リキッド単体記録は `liquid/` へ保存します。条件定義JSONと実験全体の `experiment_manifest.csv` も作成します。別の保存先は `--out-dir` で指定します。

現在は条件を順次実行します。CMA-ESは追加していません。
