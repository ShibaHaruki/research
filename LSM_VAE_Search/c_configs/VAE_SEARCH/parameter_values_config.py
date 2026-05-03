# -*- coding: utf-8 -*-
"""固定VAE Encoderの事前学習に使うパラメータ群。

ここでは「パラメータごとの min / max / step」で管理します。
各パラメータについて、最小値から最大値まで step 間隔で候補値を作ります。

PARAMETER_RANGE_MODE:
    "zip"  : 同じ位置の値同士をまとめる。1値だけのパラメータは全候補で使い回します。
             候補数が増えにくいので、VAE事前学習ではまずこちらがおすすめです。
    "grid" : 全組み合わせを作る。パラメータが多いと爆発的に増えるので注意してください。

0.0 の値も意味があるため、削除せず明示的に残します。
"""

PARAMETER_RANGE_MODE = "grid"
MAX_GENERATED_PARAMETER_SETS = 100000
RANDOM_SAMPLE_COUNT = 200
RANDOM_SEED = 0
INCLUDE_DEFAULT_PARAMETER_SET = True


VAE_PRETRAIN = {
    "name": "fixed_vae_encoder_pretrain",

    # Number of internal-state samples per material for fixed VAE Encoder pretraining.
    "samples_per_material": 10,

    # Number of parallel liquid worker processes used during VAE pretraining.
    # 1 means sequential execution; 2 or more runs parameter candidates in separate processes.
    "parallel_liquid_workers": 50,

    # Brian2 codegen target during VAE search. "auto" uses cython only when C++ Build Tools are available.
    # Without Microsoft Visual C++ Build Tools, it automatically falls back to "numpy".
    "brian_codegen_target": "auto",

    # Print VAE training loss every N epochs. Use 1 to show every epoch, 0 to disable.
    "vae_progress_interval": 1,

    # Filter settings for internal states used in VAE training.
    # Zero-spike samples use patience so one bad sample does not discard all candidates too aggressively.

    # Cleanup temporary liquid_run folders created only for VAE pretraining.
    "vae_pretrain_cleanup": {
        "enabled": True,
        "remove_excluded_immediately": True,
        "remove_used_after_training": True,
        "remove_liquid_run_after_training": True,
    },

    "vae_pretrain_filter": {
        "enabled": True,
        # If True, samples with no liquid spikes are not saved for VAE training.
        "skip_zero_spike_samples": True,
        # If True, parameter sets with repeated zero-spike samples are excluded from VAE training.
        "exclude_zero_spike_samples": True,
        # Exclude the whole parameter set when zero-spike samples reach this count.
        "zero_spike_patience": 3,
        "min_sample_mean_rate_hz": 0.0,
        "min_mean_rate_hz": 0.0,
        "min_file_mean_rate_hz": 0.0,
        "max_mean_rate_hz": 1000.0,
        "max_file_mean_rate_hz": 1000.0,
        "max_population_peak_rate_hz": 1000.0,
    },
}

PARAMETER_SET_IDS = [
    "range_min",
    "range_max",
]

PARAMETER_SET_MEMOS = [
    "各rangeの小さい側を使う候補",
    "各rangeの大きい側を使う候補",
]

# [min, max, step] または {"min": ..., "max": ..., "step": ...} で書けます。
# min == max の場合、その値だけを全候補で使います。
PARAMETER_RANGES = {
    # 入力 merkel -> liquid L1
    "merkel_p_E":       [0.2, 0.20, 0.0],
    "merkel_p_I":       [0.2, 0.20, 0.0],
    "merkel_scale_E":   [0.2, 0.2, 0.0],
    "merkel_scale_I":   [0.2, 0.2, 0.0],

    # 入力 meissner -> liquid L1
    "meissner_p_E":     [0.2, 0.2, 0.0],
    "meissner_p_I":     [0.2, 0.2, 0.0],
    "meissner_scale_E": [0.25, 0.25, 0.0],
    "meissner_scale_I": [0.25, 0.25, 0.0],

    # liquid 層内結合確率。II は元LSMで 0.00 と定義されているため明示的に残します。
    "p_liq_EE":         [0.1, 0.50, 0.1],
    "p_liq_EI":         [0.1, 0.50, 0.1],
    "p_liq_IE":         [0.1, 0.50, 0.1],
    "p_liq_II":         [0.0, 0.00, 0.0],

    # liquid 層内結合ゲイン。II は元LSMで 0.0 と定義されているため明示的に残します。
    "gain_liq_EE":      [0.25, 0.25, 0.25],
    "gain_liq_EI":      [0.25, 0.25, 0.25],
    "gain_liq_IE":      [0.25, 0.25, 0.25],
    "gain_liq_II":      [0.0, 0.0, 0.0],

    # LIF ニューロンモデル。元LSMの FIXED/cfg_neuron_models.py の値を明示的に含めます。
    "lif_tau_exc":      [10.0, 10.0, 1.0],
    "lif_tau_inh":      [10.0, 10.0, 1.0],
    "lif_ref_exc":      [2.0, 2.0, 0.0],
    "lif_ref_inh":      [2.0, 2.0, 0.0],
    "lif_bias":         [-65.0, -65.0, 0.0],
    "lif_v_thr":        [-40.0, -40.0, 5.0],
    "lif_v_reset":      [-65.0, -65.0, 5.0],

    # double_exp シナプスモデル。元LSMの FIXED/cfg_synapse_models.py の値を明示的に含めます。
    "syn_tau_r":        [1.0, 1.0, 1.0],
    "syn_tau_d":        [30.0, 30.0, 5.0],
}

