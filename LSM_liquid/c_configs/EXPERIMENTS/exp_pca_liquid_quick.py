"""Small liquid-only run for internal-state PCA."""

EXPERIMENT = {
    "name": "exp_pca_liquid_quick",
    "target": "liquid",
    "description": "Quick liquid internal-state PCA run with a smaller reservoir.",
    "trials": [
        {
            "id": "nliq100_samples3",
            "memo": "N_liq=100 and 3 samples per material for a quick PCA check.",
            "overrides": {
                "network.N_liq": [100],
                "liquid.NUM_LIQUID_SAMPLE": [3],
                "run.INTERNAL_STATE_PCA_COMPONENTS": 3,
                "run.INTERNAL_STATE_PCA_FEATURE_MODE": "flatten",
                "run.INTERNAL_STATE_PCA_STANDARDIZE": True,
                "run.INTERNAL_STATE_BIN_MS": 1.0,
            },
        },
    ],
}
