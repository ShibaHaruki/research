"""実験で共通して使う素材名、乱数、サンプル数などの基本設定。"""

# 実行時キャッシュのサブフォルダと、環境変数の対応設定。
RUNTIME_CACHE_SETTINGS = {
    "CACHE_DIR": "_runtime_cache",
    "CACHE_SUBDIRS": ("home", "matplotlib", "tmp"),
    "ENV_CACHE_MAP": {
        "USERPROFILE": "home",
        "HOME": "home",
        "MPLCONFIGDIR": "matplotlib",
        "TEMP": "tmp",
        "TMP": "tmp",
    },
}

# 描画とBrian2の実行方式・ログの共通設定。
RUNTIME_BACKEND_SETTINGS = {
    "MATPLOTLIB_BACKEND": "Agg",
    "BRIAN_CODEGEN_TARGET": "numpy",
    "BRIAN_SUPPRESS_LOG_NAMES": ["resolution_conflict"],
}


def simulation_dt_s(common):
    """秒単位の時間刻みを返す。旧snapshotのdt_msも、従来どおり秒として読む。"""
    from math import isfinite

    # 新規設定はdt_sに統一し、dt_msは過去に保存した設定の読み込みだけに使う。
    value = common["dt_s"] if "dt_s" in common else common["dt_ms"]
    dt_s = float(value)
    if not isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive (seconds)")
    if "dt_s" in common and "dt_ms" in common and dt_s != float(common["dt_ms"]):
        raise ValueError("dt_s conflicts with the legacy dt_ms value (seconds)")
    return dt_s


def trial_duration_ms(common):
    """切り出し点数×秒単位の時間刻みから、ミリ秒単位の試行時間を求める。"""
    from math import isfinite
    from numbers import Integral

    start, end = common["SLICE_START"], common["SLICE_END"]
    dt_s = simulation_dt_s(common)
    if (
        not isinstance(start, Integral)
        or not isinstance(end, Integral)
        or isinstance(start, bool)
        or isinstance(end, bool)
        or start < 0
        or end <= start
        or not isfinite(dt_s)
        or dt_s <= 0
    ):
        raise ValueError("Require integer 0 <= SLICE_START < SLICE_END and positive finite dt_s")
    return (end - start) * dt_s * 1000.0


def recording_bin_ms(cfg):
    """Liquid・Readout共通の発火数ビン幅を、整数ステップ数からmsへ変換する。"""
    from math import isfinite

    values = cfg["test"]["BIN_STEPS"]  # 候補リストの先頭を現在の記録幅として使う。
    if isinstance(values, (list, tuple)):
        if not values:
            raise ValueError("BIN_STEPS must not be empty")
        steps = values[0]
    else:
        steps = values
    dt_s = simulation_dt_s(cfg["common"])
    if (
        isinstance(steps, bool)
        or not isfinite(float(steps))
        or float(steps) <= 0
        or not float(steps).is_integer()
        or not isfinite(dt_s)
        or dt_s <= 0
    ):
        raise ValueError(
            "BIN_STEPS must be a positive integer and dt_s must be positive and finite"
        )
    return int(steps) * dt_s * 1000.0


def output_learning_rules(cfg):
    """出力層順の学習則。旧設定ではmodelsの共通値を使う。"""
    network = cfg.get("network", {})
    sizes = network.get("N_out", [80])
    count = len(sizes) if isinstance(sizes, (list, tuple)) else 1
    rules = network.get("learning_rule_liq_to_out")
    if rules is None:
        rules = [cfg.get("models", {}).get("LEARNING_RULE_MODEL", "off")] * count
    if not isinstance(rules, (list, tuple)) or len(rules) != count:
        raise ValueError("learning_rule_liq_to_out needs one learning-rule name per output layer")
    supported = {"off", "STDP", "T_STDP", "SRDP", "BCPNN"}
    if any(rule not in supported for rule in rules):
        raise ValueError(f"Unknown output learning rules: {rules}")
    return list(rules)


def output_learning_parameters(cfg):
    """各出力層のモデルに対応するFIXED値に、層別指定があれば上書きする。"""
    from copy import deepcopy

    rules = output_learning_rules(cfg)
    overrides = cfg.get("network", {}).get("OUTPUT_LEARNING_PARAMETERS")
    if overrides is not None and (
        not isinstance(overrides, (list, tuple)) or len(overrides) != len(rules)
    ):
        raise ValueError("OUTPUT_LEARNING_PARAMETERS needs one dictionary per output layer")
    result = []
    for index, rule in enumerate(rules):
        params = deepcopy(cfg.get("learning_rule_models", {}).get(rule, {}))
        if overrides is not None:
            if not isinstance(overrides[index], dict):
                raise ValueError("Per-output learning parameters must be dictionaries")
            params.update(deepcopy(overrides[index]))
        result.append(params)
    return result


def learning_rule_name(cfg):
    """保存先・学習有無の判定用。異なるモデルの出力層を含む場合はmixed。"""
    rules = output_learning_rules(cfg)
    return "off" if not rules else (rules[0] if len(set(rules)) == 1 else "mixed")


# 従来のcode/off/eval.pyと同じ8素材→3種類の対応。
TEXTURE_GROUPS = {
    "つるつる": ["aluminum_board", "rubber_board", "wood_board"],
    "ざらざら": ["cork", "denim", "Japanese_paper"],
    "中間": ["outer_pigskin", "back_pigskin"],
}

# 共通の設定
COMMON_SETS = {
    "base": {
        "VELOCITY": [20, 30, 40, 50, 60, 70, 80, 90, 100],  # センサーのなぞる速度
        "BASE_SEED": 12345,
        "dt_s": 0.1 * 1e-3,  # シミュレーションの時間刻み（秒）。0.0001 s = 0.1 ms。
        "NUM_REPEAT": 1,
        "NUM_SENSOR": 3,
        "SLICE_START": 3000,
        "SLICE_END": 8000,
        "NUM_SAMPLE": 324,  # 各素材の総サンプル数
        "LIQUID_STATE_POLICY": "reset_each_trial",  # 試行ごとに膜電位・シナプス状態を初期化する。
        "READOUT_STATE_POLICY": "reset_each_trial",  # 学習済み重みを保持し、膜電位・学習トレースなどを初期化する。
        "LIQUID_DEBUG_ENABLE": True,  # Liquid単独実行で各素材の最初の試行をdebug記録する。
        "DEBUG_EACH_MATERIAL": True,  # 各素材1試行を確認し、学習時は結合重みの変化も記録する。
        "DEBUG_TRIAL_ID": 203,
        "DEBUG_VOLTAGE_NEURONS": 10,  # Liquid・Readoutそれぞれで膜電位を表示するニューロン数。
    },
}

# training に関する設定
TRAINING_SETS = {
    "base": {
        "TRAINING_MAT": [
            "aluminum_board",
            "outer_pigskin",
            "back_pigskin",
            "cork",
            "denim",
            "rubber_board",
            "Japanese_paper",
            "wood_board",
        ],
        "NUM_TRAINING_EPOCH": [10],  # 学習の繰り返し回数。
        "NUM_TRAINING_SAMPLE": [100],
        "WEIGHT_TRACE_SYNAPSES": 10,  # 学習中の重み変化を詳しく記録する結合数。
    },
}

# test に関する設定
TEST_SETS = {
    "base": {
        "TEST_MAT": [
            "aluminum_board",
            "outer_pigskin",
            "back_pigskin",
            "cork",
            "denim",
            "rubber_board",
            "Japanese_paper",
            "wood_board",
        ],
        "NUM_TEST_SAMPLE": [100],
        "BIN_STEPS": [250],
    },
}

# 分類評価の設定
EVAL_SET = {
    "base": {
        "EVAL_N_FOLDS": 10,  # 層化交差検証の分割数。
        "EVAL_BASE_SEED": 1,  # 評価用の乱数seed。
        "EVAL_TN_MS": [25],  # 発火数を集約して分類する時間窓（ms）。
    },
}


LIQUID_SETS = {
    "base": {
        "LIQUID_MAT": [
            "aluminum_board",
            "outer_pigskin",
            "back_pigskin",
            "cork",
            "denim",
            "rubber_board",
            "Japanese_paper",
            "wood_board",
        ],
        "NUM_LIQUID_SAMPLE": [100],
    },
}
