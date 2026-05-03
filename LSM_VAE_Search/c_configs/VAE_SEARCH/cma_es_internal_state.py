"""内部状態VAEの評価指標を使うCMA-ES探索設定。"""

PAIR_PROB_BOUNDS = [0.0, 0.20]
PAIR_GAIN_BOUNDS = [0.0, 2.00]
INPUT_PROB_BOUNDS = [0.01, 0.60]
INPUT_SCALE_BOUNDS = [0.001, 0.080]
NETWORK_N_LIQ_BOUNDS = [100, 2000]
NETWORK_N_OUT_BOUNDS = [1, 200]
INHIBITORY_RATIO_BOUNDS = [0.0, 0.9]
LIQ_TO_OUT_PROB_BOUNDS = [0.0, 1.0]
LIQ_TO_OUT_GAIN_BOUNDS = [0.0, 2.0]
NEURON_TAU_BOUNDS = [1.0, 30.0]
NEURON_REF_BOUNDS = [0.0, 10.0]
BIAS_BOUNDS = [-80.0, -40.0]
V_THR_BOUNDS = [-60.0, -30.0]
V_RESET_BOUNDS = [-80.0, -40.0]
SYNAPSE_TAU_R_BOUNDS = [0.5, 10.0]
SYNAPSE_TAU_D_BOUNDS = [5.0, 80.0]

CMA_ES = {
    "name": "cma_es_vae16_J",
    "description": "CMA-ES search that maximizes J = Silhouette + DR - firing-rate penalty - synchrony penalty.",
    "target": "liquid",

    "metric": {
        "name": "VAE_J",
        "direction": "maximize",
        # True 縺ｫ縺吶ｋ縺ｨ縲∝呵｣懊＃縺ｨ縺ｫ蛻･VAE繧剃ｽ懊ｉ縺壹∬､・焚蛟呵｣懊ｒ縺ｾ縺ｨ繧√◆蜈ｱ騾壽ｽ懷惠遨ｺ髢薙〒隧穂ｾ｡縺吶ｋ縲・
        "common_latent_space": True,
        # generation: 蜷後§荳紋ｻ｣縺ｮ蛟呵｣懊□縺代〒蜈ｱ騾啖AE
        # cumulative: 縺昴ｌ縺ｾ縺ｧ縺ｫ隧穂ｾ｡縺励◆蜈ｨ蛟呵｣懊ｒ縺ｾ縺ｨ繧√※蜈ｱ騾啖AE
        "common_vae_scope": "cumulative",
        # J = Silhouette + DR - 逋ｺ轣ｫ邇・・繝翫Ν繝・ぅ - 蜷梧悄繝壹リ繝ｫ繝・ぅ縲・
        "objective": {
            "silhouette_weight": 1.0,
            "DR_weight": 1.0,
        },
        "vae": {
            "latent_dim": 16,
            "window_ms": 10.0,
            "step_ms": 10.0,
            "hidden_channels": 64,
            "beta": 1e-3,
            "epochs": 50,
            "batch_size": 32,
            "lr": 1e-3,
            "seed": 0,
            "device": "auto",
            "standardize": True,
            "max_samples_per_class": 100,
            "progress_interval": 1,
            # 縺薙％縺ｫ莠句燕蟄ｦ鄙偵＠縺欸AE繝輔か繝ｫ繝縲√∪縺溘・ common_vae_model.pt / vae_model.pt 繧呈欠螳壹☆繧九→縲・
            # CMA-ES荳ｭ縺ｯVAE繧貞・蟄ｦ鄙偵○縺壹∝崋螳哘ncoder縺ｨ縺励※菴ｿ縺・・
            "fixed_encoder_dir": "",
        },
        "vae_pretrain_filter": {
            # 固定VAE Encoderの学習前に、発火しすぎたパラメータ候補を自動除外します。
            # run_liquidで保存された spike_bin_mean は spikes/ms なので、内部でHzへ変換して判定します。
            "enabled": True,
            # 1サンプルでも完全に無発火なら、その候補の全素材をVAE学習から除外します。
            "exclude_zero_spike_samples": True,
            "min_sample_mean_rate_hz": 0.0,
            "min_mean_rate_hz": 0.0,
            "min_file_mean_rate_hz": 0.0,
            "max_mean_rate_hz": 120.0,
            "max_file_mean_rate_hz": 180.0,
            "max_population_peak_rate_hz": 300.0,
        },
        "vae_pretrain_cleanup": {
            # VAE学習後は候補ごとの内部状態が重くなるため、internal_statesだけを削除します。
            # 除外候補はVAEに使わないので、その場で削除します。
            "enabled": True,
            "remove_excluded_immediately": True,
            "remove_used_after_training": True,
        },
        "penalties": {
            # spike_bin_mean 縺ｯ spikes/ms 縺ｧ菫晏ｭ倥＆繧後ｋ縺溘ａ縲・000蛟阪＠縺ｦ Hz 縺ｫ逶ｴ縺吶・
            "rate_scale_hz": 1000.0,
            "target_rate_min_hz": 1.0,
            "target_rate_max_hz": 80.0,
            "rate_ref_hz": 80.0,
            "rate_weight": 1.0,
            # 蜷梧悄蠎ｦ縺ｯ std_t(population_rate) / mean_t(population_rate) 縺ｧ隕九ｋ縲・
            "sync_max": 1.0,
            "sync_ref": 1.0,
            "sync_weight": 1.0,
        },
    },

    "cma": {
        "generations": 10,
        "population_size": None,
        "sigma0": 0.25,
        "seed": 0,
        "jobs": 1,
    },

    # 謗｢邏｢荳ｭ縺ｯ蛟呵｣懊＃縺ｨ縺ｫ run_liquid 繧貞屓縺吶◆繧√√Λ繧､繝冶｡ｨ遉ｺ繧ПCA縺ｯ豁｢繧√※譎る俣繧堤ｯ邏・☆繧九・
    # 蜀・Κ迥ｶ諷倶ｿ晏ｭ倥・逶ｮ逧・未謨ｰ縺ｫ蠢・ｦ√↑縺ｮ縺ｧ譛牙柑縺ｫ縺吶ｋ縲・
    "base_overrides": {
        "run.LIVE_PLOT_ENABLE": False,
        "run.LIVE_RASTER_ENABLE": False,
        "run.INTERNAL_STATE_ENABLE": True,
        "run.INTERNAL_STATE_PCA_ENABLE": False,
        "liquid.NUM_LIQUID_SAMPLE": 100,
    },

    # CMA-ES 縺ｯ [0, 1] 遨ｺ髢薙〒蛟呵｣懊ｒ蜃ｺ縺励√％縺薙〒謖・ｮ壹＠縺・bounds 縺ｸ謌ｻ縺励※ cfg 縺ｫ蜿肴丐縺吶ｋ縲・
    "parameters": [
        {
            "name": "N_liq_L1",
            "path": ("network", "N_liq", 0),
            "bounds": NETWORK_N_LIQ_BOUNDS,
            "kind": "int",
        },
        {
            "name": "r_inh_liq",
            "path": "network.r_inh_liq",
            "bounds": INHIBITORY_RATIO_BOUNDS,
        },
        {
            "name": "N_out_O1",
            "path": ("network", "N_out", 0),
            "bounds": NETWORK_N_OUT_BOUNDS,
            "kind": "int",
        },
        {
            "name": "r_inh_out",
            "path": "network.r_inh_out",
            "bounds": INHIBITORY_RATIO_BOUNDS,
        },
        {
            "name": "p_liq_to_out_EE",
            "path": ("network", "p_liq_to_out_pairs", 0, "EE"),
            "bounds": LIQ_TO_OUT_PROB_BOUNDS,
        },
        {
            "name": "p_liq_to_out_EI",
            "path": ("network", "p_liq_to_out_pairs", 0, "EI"),
            "bounds": LIQ_TO_OUT_PROB_BOUNDS,
        },
        {
            "name": "p_liq_to_out_IE",
            "path": ("network", "p_liq_to_out_pairs", 0, "IE"),
            "bounds": LIQ_TO_OUT_PROB_BOUNDS,
        },
        {
            "name": "p_liq_to_out_II",
            "path": ("network", "p_liq_to_out_pairs", 0, "II"),
            "bounds": LIQ_TO_OUT_PROB_BOUNDS,
        },
        {
            "name": "gain_liq_to_out_EE",
            "path": ("network", "gain_liq_to_out_pairs", 0, "EE"),
            "bounds": LIQ_TO_OUT_GAIN_BOUNDS,
        },
        {
            "name": "gain_liq_to_out_EI",
            "path": ("network", "gain_liq_to_out_pairs", 0, "EI"),
            "bounds": LIQ_TO_OUT_GAIN_BOUNDS,
        },
        {
            "name": "gain_liq_to_out_IE",
            "path": ("network", "gain_liq_to_out_pairs", 0, "IE"),
            "bounds": LIQ_TO_OUT_GAIN_BOUNDS,
        },
        {
            "name": "gain_liq_to_out_II",
            "path": ("network", "gain_liq_to_out_pairs", 0, "II"),
            "bounds": LIQ_TO_OUT_GAIN_BOUNDS,
        },
        {
            "name": "merkel_p_E",
            "path": ("network", "IN_ROUTE", (0, "merkel"), "layers", 0, "p", "E"),
            "bounds": INPUT_PROB_BOUNDS,
        },
        {
            "name": "merkel_p_I",
            "path": ("network", "IN_ROUTE", (0, "merkel"), "layers", 0, "p", "I"),
            "bounds": INPUT_PROB_BOUNDS,
        },
        {
            "name": "merkel_scale_E",
            "path": ("network", "IN_ROUTE", (0, "merkel"), "layers", 0, "scale", "E"),
            "bounds": INPUT_SCALE_BOUNDS,
        },
        {
            "name": "merkel_scale_I",
            "path": ("network", "IN_ROUTE", (0, "merkel"), "layers", 0, "scale", "I"),
            "bounds": INPUT_SCALE_BOUNDS,
        },
        {
            "name": "meissner_p_E",
            "path": ("network", "IN_ROUTE", (0, "meissner"), "layers", 0, "p", "E"),
            "bounds": INPUT_PROB_BOUNDS,
        },
        {
            "name": "meissner_p_I",
            "path": ("network", "IN_ROUTE", (0, "meissner"), "layers", 0, "p", "I"),
            "bounds": INPUT_PROB_BOUNDS,
        },
        {
            "name": "meissner_scale_E",
            "path": ("network", "IN_ROUTE", (0, "meissner"), "layers", 0, "scale", "E"),
            "bounds": INPUT_SCALE_BOUNDS,
        },
        {
            "name": "meissner_scale_I",
            "path": ("network", "IN_ROUTE", (0, "meissner"), "layers", 0, "scale", "I"),
            "bounds": INPUT_SCALE_BOUNDS,
        },
        {
            "name": "p_liq_EE",
            "path": "network.p_liq_intra_pairs.EE",
            "bounds": PAIR_PROB_BOUNDS,
        },
        {
            "name": "p_liq_EI",
            "path": "network.p_liq_intra_pairs.EI",
            "bounds": PAIR_PROB_BOUNDS,
        },
        {
            "name": "p_liq_IE",
            "path": "network.p_liq_intra_pairs.IE",
            "bounds": PAIR_PROB_BOUNDS,
        },
        {
            "name": "p_liq_II",
            "path": "network.p_liq_intra_pairs.II",
            "bounds": PAIR_PROB_BOUNDS,
        },
        {
            "name": "gain_liq_EE",
            "path": "network.liq_intra_gain_pairs.EE",
            "bounds": PAIR_GAIN_BOUNDS,
        },
        {
            "name": "gain_liq_EI",
            "path": "network.liq_intra_gain_pairs.EI",
            "bounds": PAIR_GAIN_BOUNDS,
        },
        {
            "name": "gain_liq_IE",
            "path": "network.liq_intra_gain_pairs.IE",
            "bounds": PAIR_GAIN_BOUNDS,
        },
        {
            "name": "gain_liq_II",
            "path": "network.liq_intra_gain_pairs.II",
            "bounds": PAIR_GAIN_BOUNDS,
        },
        {
            "name": "lif_tau_exc",
            "path": "neuron_models.LIF.tau_exc",
            "bounds": NEURON_TAU_BOUNDS,
        },
        {
            "name": "lif_tau_inh",
            "path": "neuron_models.LIF.tau_inh",
            "bounds": NEURON_TAU_BOUNDS,
        },
        {
            "name": "lif_ref_exc",
            "path": "neuron_models.LIF.ref_exc",
            "bounds": NEURON_REF_BOUNDS,
        },
        {
            "name": "lif_ref_inh",
            "path": "neuron_models.LIF.ref_inh",
            "bounds": NEURON_REF_BOUNDS,
        },
        {
            "name": "lif_bias",
            "path": "neuron_models.LIF.bias",
            "bounds": BIAS_BOUNDS,
        },
        {
            "name": "lif_v_thr",
            "path": "neuron_models.LIF.v_thr",
            "bounds": V_THR_BOUNDS,
        },
        {
            "name": "lif_v_reset",
            "path": "neuron_models.LIF.v_reset",
            "bounds": V_RESET_BOUNDS,
        },
        {
            "name": "syn_tau_r",
            "path": "synapse_models.double_exp.tau_r",
            "bounds": SYNAPSE_TAU_R_BOUNDS,
        },
        {
            "name": "syn_tau_d",
            "path": "synapse_models.double_exp.tau_d",
            "bounds": SYNAPSE_TAU_D_BOUNDS,
        },
    ],
}





