"""Liquid-only run for eval.py-style random-100-neuron classification."""

EXPERIMENT = {
    "name": "exp_classify_random100_quick",
    "target": "liquid",
    "description": "Generate 1 ms internal states for 10-fold random-100-neuron classification.",
    "trials": [
        {
            "id": "nliq1000_samples100",
            "memo": "N_liq=1000, 100 samples per material, 1 ms bins.",
            "overrides": {
                "network.N_liq": [1000],
                "liquid.NUM_LIQUID_SAMPLE": [100],
                "run.INTERNAL_STATE_BIN_MS": 1.0,
                "run.INTERNAL_STATE_PCA_ENABLE": False,
            },
        },
    ],
}
