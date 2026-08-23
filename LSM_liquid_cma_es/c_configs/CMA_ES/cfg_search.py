"""LSMのパラメータをCMA-ESで最適化するための、探索範囲・評価方法・実行条件をまとめた設定ファイル"""

PARAMS = [

    #{"name": "n_liq", "step": 10, "initial": 1000, "low": 100, "high": 100},

    #{"name": "r_inh_liq", "step": 0.1, "initial": 0.20, "low": 0.1, "high": 0.7},

    {"name": "RI_p_E", "step": 0.01, "initial": 0.05, "low": 0.01, "high": 0.60},
    {"name": "RI_p_I", "step": 0.01, "initial": 0.05, "low": 0.01, "high": 0.60},
    {"name": "RI_gain_E", "step": 0.1, "initial": 0.10, "low": 1, "high": 100.0},
    {"name": "RI_gain_I", "step": 0.1, "initial": 0.10, "low": 1, "high": 100.0},

    {"name": "SI_p_E", "step": 0.01, "initial": 0.05, "low": 0.01, "high": 0.60},
    {"name": "SI_p_I", "step": 0.01, "initial": 0.05, "low": 0.01, "high": 0.60},
    {"name": "SI_gain_E", "step": 0.1, "initial": 0.10, "low": 1, "high": 100.0},
    {"name": "SI_gain_I", "step": 0.1, "initial": 0.10, "low": 1, "high": 100.0},

    {"name": "RI_opt_gain", "step": 1, "initial": 1.0, "low": 10, "high": 200.0},
    {"name": "SI_opt_gain", "step": 1, "initial": 1.0, "low": 10, "high": 200.0},

    {"name": "rec_p_ee", "step": 0.01, "initial": 0.05, "low": 0.01, "high": 0.60},
    {"name": "rec_p_ei", "step": 0.01, "initial": 0.05, "low": 0.01, "high": 0.60},
    {"name": "rec_p_ie", "step": 0.01, "initial": 0.05, "low": 0.01, "high": 0.60},
    # {"name": "rec_p_ii", "step": 0.01, "initial": 0.00, "low": 0.0, "high": 0.0},

    {"name": "rec_gain_ee", "step": 0.1, "initial": 0.10, "low": 1.0, "high": 100.00},
    {"name": "rec_gain_ei", "step": 0.1, "initial": 0.10, "low": 1.0, "high": 100.00},
    {"name": "rec_gain_ie", "step": 0.1, "initial": 0.10, "low": 1.0, "high": 100.00},
    # {"name": "rec_gain_ii", "step": 0.1, "initial": 0.10, "low": 1.0, "high": 20.00},

    #{"name": "lif_tau_exc", "step": 0.1, "initial": 10.0, "low": 1.0, "high": 20.0},
    #{"name": "lif_tau_inh", "step": 0.1, "initial": 10.0, "low": 1.0, "high": 20.0},
    # {"name": "lif_ref_exc", "step": "linear", "initial": 2.0, "low": 0.5, "high": 10.0},
    # {"name": "lif_ref_inh", "step": "linear", "initial": 2.0, "low": 0.5, "high": 10.0},

    {"name": "lif_bias", "step": 1, "initial": -65.0, "low": -70.0, "high": 0.0},

    {"name": "syn_tau_r", "step": 1, "initial": 2.0, "low": 1, "high": 100.0},
    {"name": "syn_tau_d", "step": 1, "initial": 30.0, "low": 5.0, "high": 200.0},
]


OBJECTIVE_DEFAULTS = {
    "metric": "accuracy8_overall",
    "α": 3.0,
    "β": 1.0,
    "γ": 4.0,
    "δ": 2.0,
    "κ": 2000.0,
}

# 目的関数 =
# − α × 精度
# + β × 精度の分散
# + γ × E/Iニューロン数の割合で分配した総スパイク上限ペナルティ
# + δ × 無活動ニューロン割合

# CMA-ES本体のパラメータ
CMA_ES_DEFAULTS = {
    "generations": 40,                #世代数 20
    "population_size": 20,            #個体数 (推奨 λ=4+[3ln(パラメータ数)]) 
    "sigma0": 1.0,                   #ステップサイズ
    "randomize_initial_center": False, #初期中心はPARAMSのinitial値を使用
}

# 評価条件
ACCURACY_DEFAULTS ={
    "test_size": 0.30, #testの割合
    "t_n_ms": 25.0,    #分割幅(ms)

    "neurons": "all", # 評価に使うニューロン数。"all"なら全ニューロン。
    "neuron_selection_repeats": 1, #ニューロン集合を何回選び直して評価するか
    "hold": 10, #反復ホールドアウト回数
}

SEARCH_OTHER_DEFAULTS = {
    
    "brian_codegen_target": "numpy", # numpy or cython
    "seed": 0,
    
    "jobs": 10, #1世代あたりの並列計算する数。Noneなら自動でCPUコア数に合わせる

    "n_starts": 1,# 異なる初期中心での探索回数 
    "start_jobs": 1, #異なる初期中心で並列計算する数。

    "search_name": "liquid_search007",

    "samples_per_class": 50,
    "internal_state_bin_ms": 1.0,

    # Silent-neuron metric: use all trials when the value is 0.
    "silent_min_spikes_per_neuron": 1,
    "silent_min_trial_fraction": 0.125,
    "silent_trials_per_material": 0,  # 0 means all available trials
    "spike_limit": 10000.0,

    "share_filter_input_params_across_sensors": True,  
    "search_input_filters": ["RI", "SI"], #["RI", "SI", "USI", "merkel", "meissner"]

}

SEARCH_DEFAULTS = {**CMA_ES_DEFAULTS, **SEARCH_OTHER_DEFAULTS,**ACCURACY_DEFAULTS}
