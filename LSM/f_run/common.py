"""実験設定の組み立て・復元・検証と、ネットワークの準備・試行実行。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from brian2 import prefs, start_scope, defaultclock, ms, seed, BrianLogger
from b_network.build_network import make_liquid_network
from d_tools.parameter_recording import SNAPSHOT_SECTIONS
from a_experiment.input_processing import input_rows, input_layout
from d_tools.debug import execute_trial, run_material_debug
from d_tools.weight_export import load_weights

from a_experiment.experiment_sets import (
    RUNTIME_BACKEND_SETTINGS,
    EVAL_SET,
    COMMON_SETS,
    LIQUID_SETS,
    TEST_SETS,
    TRAINING_SETS,
    output_learning_rules,
    output_learning_parameters,
    learning_rule_name,
    trial_duration_ms,
    recording_bin_ms,
)
from c_configs.FIXED import (
    cfg_models,
    cfg_learning_rule_models,
    cfg_network_stdp,
    cfg_network_off,
    cfg_neuron_models,
    cfg_synapse_models,
)
from c_configs.FIXED.cfg_filter import (
    FILTER_FUNCS,
    FILTER_GAIN,
    INPUT_FILTER_MAP,
    OPT_FILTER_GAIN,
    SENSOR_GAIN,
)


def _module_config(module, attr_name: str) -> dict:
    """設定モジュールの辞書を、元の定義を変更しない独立コピーとして取り出す。"""
    value = getattr(module, attr_name, {})
    return deepcopy(value) if isinstance(value, dict) else {}


def _fill_missing_model_params(net_cfg: dict, model_params: dict) -> None:
    """ネットワークで未指定の項目だけを、選択したモデルの既定値で補う。"""
    for key, value in model_params.items():
        net_cfg.setdefault(key, deepcopy(value))


def _apply_shared_input_routes(net_cfg: dict) -> None:
    """共有入力設定を有効にした場合だけ、各センサーの経路へ反映する。"""
    shared_cfg = net_cfg.pop("SHARED_IN_ROUTE", {})
    if not shared_cfg.get("enabled", False):
        return
    shared_filters = shared_cfg.get("filters", {})
    for (_, filter_name), route_cfg in net_cfg.get("IN_ROUTE", {}).items():
        shared = shared_filters.get(str(filter_name))
        if not shared:
            continue
        for layer_cfg in route_cfg.get("layers", {}).values():
            for key in ("p", "scale"):
                if key in shared:
                    layer_cfg[key] = deepcopy(shared[key])


def build_cfg(*, network_preset=None):
    """共通条件と固定設定をまとめる。offを指定すると出力80個の1群を使う。"""
    if network_preset not in (None, "STDP", "off"):
        raise ValueError("network_preset must be STDP or off")
    # 未指定時は従来のSTDP構成を使い、offは専用の1群構成から組み立てる。
    network_module = cfg_network_off if network_preset == "off" else cfg_network_stdp
    cfg = {
        "common": COMMON_SETS["base"],
        "training": TRAINING_SETS["base"],
        "liquid": LIQUID_SETS["base"],
        "test": TEST_SETS["base"],
        "eval": deepcopy(EVAL_SET["base"]),
        "network": _module_config(network_module, "CFG_NETWORK"),
        "run": {},  # 保存先など、実行時に追加する値だけを保持する。
        "models": _module_config(cfg_models, "CFG_MODELS"),
        "learning_rule_models": _module_config(
            cfg_learning_rule_models, "CFG_LEARNING_RULE_MODELS"
        ),
        "neuron_models": _module_config(cfg_neuron_models, "CFG_NEURON_MODELS"),
        "synapse_models": _module_config(cfg_synapse_models, "CFG_SYNAPSE_MODELS"),
        "filter_funcs": FILTER_FUNCS,
        "input_filter_map": INPUT_FILTER_MAP,
        "sensor_gain": SENSOR_GAIN,
        "filter_gain": FILTER_GAIN,
        "opt_filter_gain": OPT_FILTER_GAIN,
    }
    cfg["models"]["LEARNING_RULE_MODEL"] = learning_rule_name(cfg)
    # ネストしたリスト・辞書も含め、設定定義とは独立した実験用コピーを返す。
    return deepcopy(cfg)


def build_network_cfg(cfg: dict) -> dict:
    """実験用の設定を、Brian2のネットワーク構築で使う形式と単位へ変換する。"""
    common = cfg["common"]
    model_cfg = cfg["models"]
    network_cfg = cfg["network"]
    input_filter_map = cfg["input_filter_map"]
    channels, filters = input_layout(input_filter_map)

    neuron_model = model_cfg["NEURON_MODEL"]
    synapse_model = model_cfg["SYNAPSE_MODEL"]

    dt_s = float(common["dt_s"])
    net_cfg = deepcopy(network_cfg)
    _apply_shared_input_routes(net_cfg)
    net_cfg.update(
        {
            "dt_ms": dt_s * 1000.0,
            "BASE_SEED": int(common["BASE_SEED"]),
            "neuron_model": neuron_model,
            "learning_rule": learning_rule_name(cfg),
            "learning_rule_liq_to_out": output_learning_rules(cfg),
            "OUTPUT_LEARNING_PARAMETERS": output_learning_parameters(cfg),
            "learning_rule_models": deepcopy(cfg.get("learning_rule_models", {})),
            "synapse_model": synapse_model,
            "USE_INPUT_CHANNELS": channels,
            "USE_INPUT_FILTERS": filters,
            "INPUT_ROWS": input_rows(input_filter_map),
            "NUM_CHANNEL": len(channels),
        }
    )

    _fill_missing_model_params(net_cfg, cfg["neuron_models"].get(neuron_model, {}))
    _fill_missing_model_params(net_cfg, cfg["synapse_models"].get(synapse_model, {}))

    # Use only explicitly configured (sensor, filter) routes in input-row order.
    # Legacy route strings, filter aliases and IN_ROUTE_LAYERS are not supported.
    routes = net_cfg.get("IN_ROUTE", {})
    selected_routes = {}
    for key in net_cfg["INPUT_ROWS"]:
        if key not in routes:
            raise KeyError(f"IN_ROUTE must explicitly define {key}")
        route = routes[key]
        if (
            not isinstance(route, dict)
            or not isinstance(route.get("layers"), dict)
            or not route["layers"]
        ):
            raise ValueError(f"IN_ROUTE[{key}] must contain a nonempty 'layers' dictionary")
        selected_routes[key] = deepcopy(route)
    net_cfg["IN_ROUTE"] = selected_routes
    return net_cfg


def first(value):
    """単一値と候補リストのどちらからも、現在使用する先頭値を取り出す。"""
    return value[0] if isinstance(value, (list, tuple)) else value


def cfg_from_snapshot(data):
    """旧形式の設定も現行のcommon・eval配置に読み替える。"""
    cfg = deepcopy(build_cfg())
    for key in SNAPSHOT_SECTIONS:
        cfg[key] = deepcopy(data.get(key, cfg[key]) if key == "eval" else data[key])
    # 旧名dt_msの値は秒単位だったため、数値を変えずdt_sへ読み替える。
    from a_experiment.experiment_sets import simulation_dt_s

    cfg["common"]["dt_s"] = simulation_dt_s(cfg["common"])
    cfg["common"].pop("dt_ms", None)
    # 過去のsnapshotでrunにあった共通項目を、新しいcommon配置に読み替える。
    for key in (
        "LIQUID_STATE_POLICY",
        "DEBUG_EACH_MATERIAL",
        "DEBUG_TRIAL_ID",
        "READOUT_STATE_POLICY",
        "DEBUG_VOLTAGE_NEURONS",
        "LIQUID_DEBUG_ENABLE",
    ):
        legacy = cfg["run"].pop(key, COMMON_SETS["base"][key])
        cfg["common"].setdefault(key, legacy)
    # 旧snapshotの独立した膜電位記録間隔は、現在は使用しない。
    cfg["common"].pop("DEBUG_VOLTAGE_DT_MS", None)
    cfg["run"].pop("DEBUG_VOLTAGE_DT_MS", None)
    cfg["run"].pop("INTERNAL_STATE_DT_MS", None)
    cfg["run"].pop("INTERNAL_STATE_BIN_MS", None)
    cfg["run"].pop("WEIGHT_TRACE_DT_MS", None)
    legacy_trace_count = cfg["run"].pop(
        "WEIGHT_TRACE_SYNAPSES", TRAINING_SETS["base"]["WEIGHT_TRACE_SYNAPSES"]
    )
    cfg["training"].setdefault("WEIGHT_TRACE_SYNAPSES", legacy_trace_count)
    for key, default in EVAL_SET["base"].items():
        legacy = cfg["run"].pop(key, deepcopy(default))
        if "eval" not in data:
            cfg["eval"][key] = legacy
    return cfg


def load_cfg(folder):
    """保存済みパラメータを読み込み、旧形式も現在の設定配置へ変換する。"""
    from d_tools.parameter_recording import read_training_configuration

    data = read_training_configuration(folder)
    if data is None:
        raise FileNotFoundError(f"No saved configuration in {folder}")
    return cfg_from_snapshot(data)


def validate_cfg(cfg):
    """試行数・素材・記録幅・状態の引き継ぎ条件を検証して実行条件を返す。"""
    cfg["models"]["LEARNING_RULE_MODEL"] = learning_rule_name(cfg)
    output_learning_parameters(cfg)
    n_train = int(first(cfg["training"]["NUM_TRAINING_SAMPLE"]))
    n_test = int(first(cfg["test"]["NUM_TEST_SAMPLE"]))
    epochs = int(first(cfg["training"]["NUM_TRAINING_EPOCH"]))
    if min(n_train, n_test, epochs) < 1 or n_train + n_test > 324:
        raise ValueError("Require positive train/test samples and epochs; train+test <= 324")
    materials = list(cfg["training"]["TRAINING_MAT"])
    if (
        not materials
        or len(set(materials)) != len(materials)
        or materials != list(cfg["test"]["TEST_MAT"])
    ):
        raise ValueError("Training/test materials must be the same unique, nonempty list")
    bin_ms = recording_bin_ms(cfg)
    duration = trial_duration_ms(cfg["common"])
    if bin_ms <= 0 or duration <= 0 or not np.isclose(duration / bin_ms, round(duration / bin_ms)):
        raise ValueError("Slice duration must be positive and divisible by the recording bin width")
    if cfg["common"].get("READOUT_STATE_POLICY", "reset_each_trial") not in ("carry", "reset_each_trial"):
        raise ValueError("READOUT_STATE_POLICY must be carry or reset_each_trial")
    return materials, n_train, n_test, epochs, bin_ms, duration


def cfg_for_repetition(cfg, rep):
    """Liquidとデータを共通にし、学習則に依存せずrep番号から出力seedを決める。"""
    if isinstance(rep, bool) or int(rep) != rep or rep < 1:
        raise ValueError("rep must be a positive integer")
    # 呼出し元の設定を変更せず、repごとの完全な設定を保存・再利用できるようにする。
    result = deepcopy(cfg)
    output_seed = int(result["common"]["BASE_SEED"]) + int(rep)
    result["network"]["OUTPUT_SEED"] = output_seed
    # 出力群数や学習則が違っても、同じサイズの各群は同じ結合・初期重みになる。
    result["network"]["OUTPUT_SEEDS"] = [output_seed] * len(output_learning_rules(result))
    result["run"]["READOUT_SEED_INDEX"] = int(rep)
    # rep間では出力の乱数だけを比較する。明示的なデータ分割指定は維持する。
    result["run"].setdefault("DATA_SPLIT_REP", 1)
    return result


def sample_split(cfg, rep):
    """sample_seq内の保存済み分割から、学習用とテスト用の試行番号を読む。"""
    from a_experiment.paths import SAMPLE_SEQUENCE_ROOT
    from a_experiment.sample_sequences import load_sample_split

    if int(rep) < 1:
        raise ValueError("rep must be >=1")
    _, n_train, n_test, _, _, _ = validate_cfg(cfg)
    data_rep = int(cfg["run"].get("DATA_SPLIT_REP", 1))
    if data_rep < 1:
        raise ValueError("DATA_SPLIT_REP must be positive")
    data_seed = int(cfg["common"]["BASE_SEED"]) + data_rep - 1
    # 通常は全rep・学習則で既存ファイルを共有する。明示した別分割だけseed別に保存する。
    directory = (
        SAMPLE_SEQUENCE_ROOT if data_rep == 1 else SAMPLE_SEQUENCE_ROOT / f"seed_{data_seed}"
    )
    return load_sample_split(
        data_seed,
        n_train,
        n_test,
        rep=rep,
        n_samples=int(cfg["common"]["NUM_SAMPLE"]),
        directory=directory,
    )


def configure_brian_runtime():
    """共通の実行方式とログ設定をBrian2へ反映する。"""
    prefs.codegen.target = RUNTIME_BACKEND_SETTINGS["BRIAN_CODEGEN_TARGET"]
    if prefs.codegen.target == "cython":
        from a_experiment.paths import CYTHON_CACHE_ROOT

        # 同じ計算コードの再コンパイルを避け、rep間でも生成済みコードを再利用する。
        CYTHON_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        prefs.codegen.runtime.cython.cache_dir = str(CYTHON_CACHE_ROOT)
    for name in RUNTIME_BACKEND_SETTINGS["BRIAN_SUPPRESS_LOG_NAMES"]:
        BrianLogger.suppress_name(name)


def build_runtime(cfg, *, learning):
    """選択した学習則でネットワークを作り、試行開始時の状態を保存する。"""
    start_scope()
    configure_brian_runtime()
    defaultclock.dt = float(cfg["common"]["dt_s"]) * 1000 * ms
    base_seed = int(cfg["common"]["BASE_SEED"])
    seed(base_seed)
    np.random.seed(base_seed)
    effective = deepcopy(cfg)
    effective["network"]["output_enabled"] = True
    if not learning:
        effective["models"]["LEARNING_RULE_MODEL"] = "off"
        effective["network"]["learning_rule_liq_to_out"] = ["off"] * len(output_learning_rules(cfg))
    net_cfg = build_network_cfg(effective)
    objects = make_liquid_network(
        net_cfg,
        len(net_cfg["INPUT_ROWS"]),
        np.random.default_rng(base_seed),
        cfg["run"],
        common_cfg=cfg["common"],
    )
    if not objects["G_out"]:
        raise ValueError("At least one output layer is required")
    # 学習ではトレース・重みだけを使い、全試行の発火イベント蓄積を避ける。
    for monitor in objects["M_out"]:
        monitor.active = not learning
    objects["net"].store("trial_initial")
    return objects


def run_trial(
    objects,
    cfg,
    material,
    sid,
    *,
    record,
    record_liquid=False,
    record_bin_ms=None,
    training_debug=False,
    before_run=None,
):
    """1試行の入力を提示し、必要な場合だけ発火数やdebug用データを返す。"""
    from a_experiment.input_processing import filtered_trial

    _, _, _, _, bin_ms, duration = validate_cfg(cfg)
    if cfg["common"].get("READOUT_STATE_POLICY", "reset_each_trial") == "reset_each_trial":
        # 全層の動的状態を戻す一方、前の試行までに学習した結合重みは保持する。
        learned = [np.asarray(syn.w[:]).copy() for syn in objects["S_out"]]
        objects["net"].restore("trial_initial")
        for syn, values in zip(objects["S_out"], learned):
            syn.w = values
    if before_run is not None:
        before_run()
    result = execute_trial(
        objects,
        filtered_trial(material, int(sid), cfg),
        dt_ms=float(cfg["common"]["dt_s"]) * 1000,
        duration_ms=duration,
        bin_ms=bin_ms if record_bin_ms is None else record_bin_ms,
        record_liquid=record and record_liquid,
        record_output=record,
        voltage=training_debug,
    )
    if training_debug:
        return result
    if not record:
        return None
    if record_liquid:
        return {
            name: np.concatenate([data["counts"] for data in result[key]], axis=0)
            for name, key in [("liquid", "liquid"), ("readout", "output")]
        }
    return np.concatenate([data["counts"] for data in result["output"]], axis=0)


def debug_readout_materials(cfg, weights_dir, out_dir):
    """学習・テストの設定を共通debugへ渡し、固定重みで各素材を1回確認する。"""
    from a_experiment.input_processing import filtered_trial

    if not cfg["common"].get("DEBUG_EACH_MATERIAL", True):
        return []
    materials, _, _, _, bin_ms, duration = validate_cfg(cfg)
    sid = int(cfg["common"].get("DEBUG_TRIAL_ID", 203))
    if not 1 <= sid <= 324:
        raise ValueError("DEBUG_TRIAL_ID must be 1..324")

    def make_network():
        objects = build_runtime(cfg, learning=False)
        load_weights(weights_dir, objects)
        return objects

    return run_material_debug(
        materials,
        trial_id=sid,
        make_network=make_network,
        load_current=lambda material, trial_id: filtered_trial(material, trial_id, cfg),
        dt_ms=float(cfg["common"]["dt_s"]) * 1000,
        duration_ms=duration,
        bin_ms=bin_ms,
        out_dir=out_dir,
    )


def add_training_arguments(parser):
    """学習・テストの入口で共通するコマンド引数を登録する。"""
    parser.add_argument("--rule", choices=["off", "STDP", "T_STDP", "SRDP", "BCPNN"], default=None)
    parser.add_argument("--materials", nargs="+", default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rep", type=int, default=None)
    parser.add_argument("--rep-end", type=int, default=None, help="実行する最後のrep番号")
    parser.add_argument("--state-policy", choices=["carry", "reset_each_trial"], default=None)
    parser.add_argument("--out-dir", type=Path, default=None)


def cfg_from_args(args, *, base_cfg=None):
    """コマンド引数で指定された条件だけを、独立した実験設定に反映する。"""
    # 通常の--rule offは専用構成を使用する。明示されたbase_cfgの構造は維持する。
    network_preset = "off" if args.rule == "off" else None
    cfg = deepcopy(build_cfg(network_preset=network_preset) if base_cfg is None else base_cfg)
    if args.rule is not None:
        old_rules = output_learning_rules(cfg)
        old_parameters = cfg["network"].get("OUTPUT_LEARNING_PARAMETERS")
        cfg["models"]["LEARNING_RULE_MODEL"] = args.rule
        cfg["network"]["learning_rule_liq_to_out"] = [args.rule] * len(old_rules)
        if old_parameters is not None:
            cfg["network"]["OUTPUT_LEARNING_PARAMETERS"] = [
                deepcopy(params) if rule == args.rule else {}
                for rule, params in zip(old_rules, old_parameters)
            ]
    if args.materials is not None:
        if not set(args.materials) <= set(TRAINING_SETS["base"]["TRAINING_MAT"]):
            raise ValueError("Unknown material")
        cfg["training"]["TRAINING_MAT"] = args.materials
        cfg["test"]["TEST_MAT"] = args.materials
    for value, section, key in [
        (args.train_samples, "training", "NUM_TRAINING_SAMPLE"),
        (args.test_samples, "test", "NUM_TEST_SAMPLE"),
        (args.epochs, "training", "NUM_TRAINING_EPOCH"),
    ]:
        if value is not None:
            cfg[section][key] = [value]
    if args.state_policy is not None:
        cfg["common"]["READOUT_STATE_POLICY"] = args.state_policy
        # run_trialはネットワーク全体をrestoreするため、Liquid側の記録条件も揃える。
        cfg["common"]["LIQUID_STATE_POLICY"] = args.state_policy
    if args.out_dir is not None:
        cfg["run"]["READOUT_TRAIN_DIR"] = str(args.out_dir.resolve())
    validate_cfg(cfg)
    return cfg


def repetition_numbers(cfg, rep=None, rep_end=None):
    """未指定ならCOMMONのNUM_REPEATに従い、指定時は対象repだけを実行する。"""
    total = int(cfg["common"].get("NUM_REPEAT", 1))
    start = 1 if rep is None else int(rep)
    end = (total if rep is None else start) if rep_end is None else int(rep_end)
    if total < 1 or start < 1 or end < start:
        raise ValueError("Invalid NUM_REPEAT or repetition range")
    return list(range(start, end + 1))
