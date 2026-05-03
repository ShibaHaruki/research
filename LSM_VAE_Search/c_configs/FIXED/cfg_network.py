"""LSM の層数、ニューロン数、接続確率、スケールなどのネットワーク固定設定。"""

# FIXED/cfg_network.py
#
# Processing order:
# 1. neuron groups
# 2. input -> liquid
# 3. liquid -> liquid
# 4. liquid -> output
#
# Pair labels:
# EE: excitatory -> excitatory
# EI: excitatory -> inhibitory
# IE: inhibitory -> excitatory
# II: inhibitory -> inhibitory

CFG_NETWORK = {
    # 1) neuron groups
    # Layer counts are inferred from len(N_liq) and len(N_out).
    "N_liq": [1000],
    "r_inh_liq": 0.2,

    "N_out": [40],
    "r_inh_out": 0,

    # 2) input -> liquid
    # Use IN_ROUTE when you want per-filter input settings.
    # Key: (channel_index, filter_name)
    # p/scale can be scalars or post-type dicts: {"E": ..., "I": ...}.
    "IN_ROUTE": {
        (0, "merkel"): {
            "layers": {
                0: {
                    "p": {"E": 0.2, "I": 0.2},
                    "scale": {"E": 0.1, "I": 0.1},
                },
            },
        },
        (0, "meissner"): {
            "layers": {
                0: {
                    "p": {"E": 0.2, "I": 0.2},
                    "scale": {"E": 0.1, "I": 0.1},
                },
            },
        },
    },

    # 3) liquid -> liquid
    "liq_intra_connection": "random",
    "p_liq_intra_pairs": {"EE": 0.2, "EI": 0.2, "IE": 0.2, "II": 0.00},
    "liq_intra_gain_pairs": {"EE": 0.5, "EI": 0.5, "IE": 0.5, "II": 0.0},

    # 4) liquid -> output
    # Shape: list by output layer -> dict by liquid layer -> pair dict.
    "p_liq_to_out_pairs": [
        {"EE": 1.0, "EI": 0.0, "IE": 0.0, "II": 0.0}
    ],
    "gain_liq_to_out_pairs": [
        {"EE": 0.8, "EI": 0.5, "IE": 0.4, "II": 0.5}
    ],
}
