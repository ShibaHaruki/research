"""リキッドから出力層へのゲインを振って比較する例。"""

PAIR_05 = {"EE": 0.5, "EI": 0.5, "IE": 0.5, "II": 0.5}
PAIR_10 = {"EE": 1.0, "EI": 1.0, "IE": 1.0, "II": 1.0}
PAIR_20 = {"EE": 2.0, "EI": 2.0, "IE": 2.0, "II": 2.0}


EXPERIMENT = {
    "name": "exp_002_gain_sweep_example",
    "description": "Example sweep for liquid-to-output gain.",
    "target": "training",
    "trials": [
        {
            "id": "liq_out_gain_0p5",
            "memo": "All liquid-to-output E/I gains are 0.5.",
            "overrides": {
                "network.gain_liq_to_out_pairs": [
                    PAIR_05
                ],
            },
        },
        {
            "id": "liq_out_gain_1p0",
            "memo": "All liquid-to-output E/I gains are 1.0.",
            "overrides": {
                "network.gain_liq_to_out_pairs": [
                    PAIR_10
                ],
            },
        },
        {
            "id": "liq_out_gain_2p0",
            "memo": "All liquid-to-output E/I gains are 2.0.",
            "overrides": {
                "network.gain_liq_to_out_pairs": [
                    PAIR_20
                ],
            },
        },
    ],
}
