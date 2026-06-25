"""リキッド内部のゲインを振って比較する例。"""

EXPERIMENT = {
    "name": "exp_002_liquid_gain_sweep_example",
    "target": "liquid",
    "description": "Example sweep for liquid recurrent gain.",
    "trials": [
        {
            "id": "liq_gain_0p5",
            "memo": "All liquid recurrent E/I gains are 0.5.",
            "overrides": {
                "network.liq_intra_gain_pairs": {
                    "EE": 0.5,
                    "EI": 0.5,
                    "IE": 0.5,
                    "II": 0.5,
                },
            },
        },
        {
            "id": "liq_gain_1p0",
            "memo": "All liquid recurrent E/I gains are 1.0.",
            "overrides": {
                "network.liq_intra_gain_pairs": {
                    "EE": 1.0,
                    "EI": 1.0,
                    "IE": 1.0,
                    "II": 1.0,
                },
            },
        },
        {
            "id": "liq_gain_2p0",
            "memo": "All liquid recurrent E/I gains are 2.0.",
            "overrides": {
                "network.liq_intra_gain_pairs": {
                    "EE": 2.0,
                    "EI": 2.0,
                    "IE": 2.0,
                    "II": 2.0,
                },
            },
        },
    ],
}
