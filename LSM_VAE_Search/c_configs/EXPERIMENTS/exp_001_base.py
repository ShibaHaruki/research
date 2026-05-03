"""固定設定をそのまま使うベースライン実験の定義。"""

EXPERIMENT = {
    "name": "exp_001_base",
    "description": "Baseline run with the FIXED config values.",
    "target": "training",
    "trials": [
        {
            "id": "base",
            "memo": "No overrides. This should reproduce the current FIXED setup.",
            "overrides": {},
        },
    ],
}
