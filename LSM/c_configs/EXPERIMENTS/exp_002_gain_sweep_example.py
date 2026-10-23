"""リキッドから出力層へのゲインを振って比較する例。"""

PAIR_05 = {"EE": 0.5, "EI": 0.0, "IE": 0.0, "II": 0.0}
PAIR_10 = {"EE": 1.0, "EI": 0.0, "IE": 0.0, "II": 0.0}
PAIR_20 = {"EE": 2.0, "EI": 0.0, "IE": 0.0, "II": 0.0}


EXPERIMENT = {
    "name": "exp_002_gain_sweep_example",
    "description": "Example sweep for liquid-to-output gain.",
    "target": "training",
    "trials": [
        {
            "id": "liq_out_gain_0p5",
            "memo": "Liquid E-to-output gain is 0.5; I-to-output remains disconnected.",
            "overrides": {
                "network.gain_liq_to_out_pairs": [
                    PAIR_05
                ],
            },
        },
        {
            "id": "liq_out_gain_1p0",
            "memo": "Liquid E-to-output gain is 1.0; I-to-output remains disconnected.",
            "overrides": {
                "network.gain_liq_to_out_pairs": [
                    PAIR_10
                ],
            },
        },
        {
            "id": "liq_out_gain_2p0",
            "memo": "Liquid E-to-output gain is 2.0; I-to-output remains disconnected.",
            "overrides": {
                "network.gain_liq_to_out_pairs": [
                    PAIR_20
                ],
            },
        },
    ],
}
