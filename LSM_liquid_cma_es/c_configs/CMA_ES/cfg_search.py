"""Search space and objective defaults for the liquid CMA-ES search."""

PARAMS = [
    # Separate E/I Brian2 groups require at least one neuron in each group.
    {"name": "n_liq", "kind": "int_log10", "initial": 1000, "low": 5, "high": 200},

    {"name": "r_inh_liq", "kind": "logit", "initial": 0.20, "low": 0.2, "high": 0.50},

    # {"name": "merkel_p_E", "kind": "logit", "initial": 0.05, "low": 0.00, "high": 0.30},
    # {"name": "merkel_p_I", "kind": "logit", "initial": 0.05, "low": 0.00, "high": 0.30},
    # {"name": "meissner_p_E", "kind": "logit", "initial": 0.05, "low": 0.00, "high": 0.30},
    # {"name": "meissner_p_I", "kind": "logit", "initial": 0.05, "low": 0.00, "high": 0.30},
    # {"name": "merkel_gain_E", "kind": "log10", "initial": 0.10, "low": 0.01, "high": 1.50},
    # {"name": "merkel_gain_I", "kind": "log10", "initial": 0.10, "low": 0.01, "high": 1.50},
    # {"name": "meissner_gain_E", "kind": "log10", "initial": 0.10, "low": 0.01, "high": 1.50},
    # {"name": "meissner_gain_I", "kind": "log10", "initial": 0.10, "low": 0.01, "high": 1.50},

    {"name": "RI_p_E", "kind": "linear", "initial": 0.05, "low": 0.01, "high": 0.30},
    {"name": "RI_p_I", "kind": "linear", "initial": 0.05, "low": 0.01, "high": 0.30},
    {"name": "RI_gain_E", "kind": "linear", "initial": 0.10, "low": 1, "high": 10.0},
    {"name": "RI_gain_I", "kind": "linear", "initial": 0.10, "low": 1, "high": 10.0},

    {"name": "SI_p_E", "kind": "linear", "initial": 0.05, "low": 0.1, "high": 0.30},
    {"name": "SI_p_I", "kind": "linear", "initial": 0.05, "low": 0.1, "high": 0.30},
    {"name": "SI_gain_E", "kind": "linear", "initial": 0.10, "low": 1, "high": 10.0},
    {"name": "SI_gain_I", "kind": "linear", "initial": 0.10, "low": 1, "high": 10.0},

    # {"name": "USI_p_E", "kind": "linear", "initial": 0.05, "low": 0.00, "high": 0.00},
    # {"name": "USI_p_I", "kind": "linear", "initial": 0.05, "low": 0.00, "high": 0.00},
    # {"name": "USI_gain_E", "kind": "linear", "initial": 0.10, "low": 0.00, "high": 0.00},
    # {"name": "USI_gain_I", "kind": "linear", "initial": 0.10, "low": 0.00, "high": 0.00},

    {"name": "RI_opt_gain", "kind": "log10", "initial": 1.0, "low": 10, "high": 100.0},
    {"name": "SI_opt_gain", "kind": "log10", "initial": 1.0, "low": 10, "high": 100.0},
    # {"name": "USI_opt_gain", "kind": "log10", "initial": 1.0, "low": 0, "high": 0.0},
    # {"name": "merkel_opt_gain", "kind": "log10", "initial": 1.0, "low": 0.00, "high": 0.0},
    # {"name": "meissner_opt_gain", "kind": "log10", "initial": 1.0, "low": 0.00, "high": 0.0},

    {"name": "rec_p_ee", "kind": "logit", "initial": 0.05, "low": 0.20, "high": 0.60},
    {"name": "rec_p_ei", "kind": "logit", "initial": 0.05, "low": 0.10, "high": 0.60},
    {"name": "rec_p_ie", "kind": "logit", "initial": 0.05, "low": 0.01, "high": 0.60},
    #{"name": "rec_p_ii", "kind": "logit", "initial": 0.00, "low": 0.0, "high": 0.0},

    {"name": "rec_gain_ee", "kind": "log10", "initial": 0.10, "low": 1.0, "high": 5.00},
    {"name": "rec_gain_ei", "kind": "log10", "initial": 0.10, "low": 1.0, "high": 5.00},
    {"name": "rec_gain_ie", "kind": "log10", "initial": 0.10, "low": 1.0, "high": 5.00},
    #{"name": "rec_gain_ii", "kind": "log10", "initial": 0.10, "low": 1.0, "high": 5.00},

    {"name": "lif_tau_exc", "kind": "log10", "initial": 10.0, "low": 1.0, "high": 40.0},
    {"name": "lif_tau_inh", "kind": "log10", "initial": 10.0, "low": 1.0, "high": 40.0},
    {"name": "lif_ref_exc", "kind": "log10", "initial": 2.0, "low": 0.5, "high": 10.0},
    {"name": "lif_ref_inh", "kind": "log10", "initial": 2.0, "low": 0.5, "high": 10.0},

    {"name": "lif_bias", "kind": "linear", "initial": -65.0, "low": -70.0, "high": -30.0},

    {"name": "syn_tau_r", "kind": "log10", "initial": 2.0, "low": 0.5, "high": 10.0},
    {"name": "syn_tau_d", "kind": "log10", "initial": 30.0, "low": 5.0, "high": 100.0},
]


OBJECTIVE_DEFAULTS = {
    "metric": "accuracy8_overall",
    "α": 100.0,
    "β": 1.0,
    "γ": 1.0,
    "δ": 20.0,
    "ε": 1.0,
    "κ": 2000.0,
}
# 目的関数 =
# − α × 精度
# + β × 精度の分散
# + γ × スパイク数 / κ
# + δ × 無活動ニューロン割合
# − ε × Fisher比 

# CMA-ES本体のパラメータ
CMA_ES_DEFAULTS = {
    "generations": 10,                #世代数
    "population_size": 10,            #個体数
    "sigma0": 0.50,                   #ステップサイズ
    "randomize_initial_center": True, #初期中心をランダム化
}

# 評価条件
ACCURACY_DEFAULTS ={
    "test_size": 0.20, #testの割合
    "fold": 10,       #hold回数
    "t_n_ms": 25.0,    #分割幅(ms)

}

SEARCH_OTHER_DEFAULTS = {
    # 実行・並列計算
    "brian_codegen_target": "numpy", # or cyhton

    "search_name": "liquid_search001",

    "share_filter_input_params_across_sensors": True,  
    "search_input_filters": ["RI", "SI"], #["RI", "SI", "USI", "merkel", "meissner"]

    "seed": 0,
    "n_starts": 1,
    "start_spread": None,
    "jobs": None,
    "start_jobs": 1,
    
    "internal_state_bin_ms": 1.0,
    "samples_per_class": 50,
    "neurons": 0,

    # 評価条件
     # 0 means all available neurons
    # At least one evaluation is required. Use neurons=0 to evaluate all
    # available neurons; repeats controls repeated random neuron selections.
    "repeats": 1,
}

# Existing code imports SEARCH_DEFAULTS. Keep this flat compatibility view
# while maintaining a clear separation above.
SEARCH_DEFAULTS = {**CMA_ES_DEFAULTS, **SEARCH_OTHER_DEFAULTS,**ACCURACY_DEFAULTS}
