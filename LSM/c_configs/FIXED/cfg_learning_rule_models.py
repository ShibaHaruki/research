# FIXED/cfg_learning_rule_models.py

LEARNING_RULE_PRESETS = {
    "off_default": {
        "learning_rule": "off",
    },

    "STDP_default": {
        "learning_rule": "STDP",
        "tau_plus": 11.7,
        "tau_minus": 14.0,
        "A_plus": 0.0007,
        "A_minus": -0.0006,
        "wmin": 0.0,
        "wmax": 1.0,
    },

    "T_STDP_default": {
        "learning_rule": "T_STDP",
        "tau_plus1": 11.7,
        "tau_plus2": 15.0,
        "tau_minus1": 14.0,
        "tau_minus2": 15.0,
        "A2_plus": 0.0007,
        "A3_plus": 0.00000005,
        "A2_minus": 0.0006,
        "A3_minus": 0.00000005,
        "wmin": 0.0,
        "wmax": 1.0,
    },

    "SRDP_default": {
        "learning_rule": "SRDP",
        "tau_plus": 11.7,
        "tau_minus": 14.0,
        "tau_pre": 15.0,
        "tau_post": 15.0,
        "A_plus": 0.0007,
        "A_minus": -0.0006,
        "A_pre": 0.00005,
        "A_post": 0.00005,
        "wmin": 0.0,
        "wmax": 1.0,
    },
}