"""LSM の層数、ニューロン数、接続確率、スケールなどのネットワーク固定設定"""

# FIXED/cfg_network.py
#
# Processing order:
# 1. neuron groups
# 2. input -> liquid
# 3. liquid -> liquid
#
# Pair labels:
# EE: excitatory -> excitatory
# EI: excitatory -> inhibitory
# IE: inhibitory -> excitatory
# II: inhibitory -> inhibitory

CFG_NETWORK = {
    # 1) neuron groups
    # Layer counts are inferred from len(N_liq).
    "N_liq": [1000],
    "r_inh_liq": 0.5,

    # 2) input -> liquid
    # Use IN_ROUTE when you want per-filter input settings.
    # Key: (channel_index, filter_name)
    # p/scale can be scalars or post-type dicts: {"E": ..., "I": ...}.
    "IN_ROUTE": {
        (0, "merkel"): {
            "layers": {
                0: {
                    "p": {"E": 0.2, "I": 0.2},
                    "scale": {"E": 1.7, "I": 1.7},
                },
            },
        },
        (0, "meissner"): {
            "layers": {
                0: {
                    "p": {"E": 0.2, "I": 0.2},
                    "scale": {"E": 1.7, "I": 1.7},
                },
            },
        },
        (0, "RI"): {
            "layers": {
                0: {
                    "p": {"E": 0.00, "I": 0.00},
                    "scale": {"E": 0.1, "I": 0.1},
                },
            },
        },
        (0, "SI"): {
            "layers": {
                0: {
                    "p": {"E": 0.0, "I": 0.0},
                    "scale": {"E": 0.1, "I": 0.1},
                },
            },
        },
        (0, "USI"): {
            "layers": {
                0: {
                    "p": {"E": 0.0, "I": 0.0},
                    "scale": {"E": 0.1, "I": 0.1},
                },
            },
        },
        (1, "merkel"): {
            "layers": {
                0: {
                    "p": {"E": 0.2, "I": 0.2},
                    "scale": {"E": 1.7, "I": 1.7},
                },
            },
        },
        (1, "meissner"): {
            "layers": {
                0: {
                    "p": {"E": 0.2, "I": 0.2},
                    "scale": {"E": 1.7, "I": 1.7},
                },
            },
        },
        (1, "RI"): {
            "layers": {
                0: {
                    "p": {"E": 0.00, "I": 0.00},
                    "scale": {"E": 0.0, "I": 0.0},
                },
            },
        },
        (1, "SI"): {
            "layers": {
                0: {
                    "p": {"E": 0.0, "I": 0.0},
                    "scale": {"E": 0.1, "I": 0.1},
                },
            },
        },
        (1, "USI"): {
            "layers": {
                0: {
                    "p": {"E": 0.0, "I": 0.0},
                    "scale": {"E": 0.1, "I": 0.1},
                },
            },
        },
        (2, "merkel"): {
            "layers": {
                0: {
                    "p": {"E": 0.2, "I": 0.2},
                    "scale": {"E": 0.7, "I": 0.7},
                },
            },
        },
        (2, "meissner"): {
            "layers": {
                0: {
                    "p": {"E": 0.2, "I": 0.2},
                    "scale": {"E": 1.7, "I": 1.7},
                },
            },
        },
        (2, "RI"): {
            "layers": {
                0: {
                    "p": {"E": 0.00, "I": 0.00},
                    "scale": {"E": 0.1, "I": 0.1},
                },
            },
        },
        (2, "SI"): {
            "layers": {
                0: {
                    "p": {"E": 0.0, "I": 0.0},
                    "scale": {"E": 0.1, "I": 0.1},
                },
            },
        },
        (2, "USI"): {
            "layers": {
                0: {
                    "p": {"E": 0.0, "I": 0.0},
                    "scale": {"E": 0.1, "I": 0.1},
                },
            },
        },
    },

    # Optional background Poisson spike input.
    # When enabled, this group keeps firing during the whole Brian2 simulation.
    "poisson_input": {
        "enabled": False,
        # Use N_ratio to set the number of Poisson neurons as a fraction of
        # each liquid layer size. If N_ratio is None, fixed N is used.
        "N_ratio": None,
        "N": 100,
        "rate_hz": 5.0,
        "current": "exc",
        "p": {"E": 0.1, "I": 0.1},
        "scale": {"E": 0.2, "I": 0.2},
    },

    # 3) liquid -> liquid
    "liq_intra_connection": "random",
    "p_liq_intra_pairs": {"EE": 0.5, "EI": 0.5, "IE": 0.5, "II": 0.5},
    "liq_intra_gain_pairs": {"EE": 0.25, "EI": 0.25, "IE": 0.25, "II": 0.25},

}
