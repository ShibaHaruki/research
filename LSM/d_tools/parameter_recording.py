"""共通設定と学習則設定を、それぞれ対応する結果フォルダへ保存する。"""

from d_tools.recording_io import read_record, encode, save_record
from copy import deepcopy
from pathlib import Path

from a_experiment.paths import parameter_snapshot_files
from a_experiment.experiment_sets import (
    learning_rule_name,
    output_learning_rules,
    output_learning_parameters,
)

# パラメータ記録へ含める設定区分。入力フィルタの関数本体は保存しない。
SNAPSHOT_SECTIONS = (
    "common",
    "training",
    "test",
    "eval",
    "liquid",
    "network",
    "run",
    "models",
    "neuron_models",
    "synapse_models",
    "learning_rule_models",
)


def snapshot(cfg):
    """再現に必要な設定だけを、元の実験設定と独立したコピーで取り出す。"""
    return {key: deepcopy(cfg[key]) for key in SNAPSHOT_SECTIONS}


def parameter_records_equal(saved, current):
    """旧名dt_msを秒単位のdt_sへ読み替え、保存済み条件と現在の条件を比較する。"""
    from a_experiment.experiment_sets import simulation_dt_s

    records = [deepcopy(saved), deepcopy(current)]
    for record in records:
        common = record.get("common")
        if common is not None:
            common["dt_s"] = simulation_dt_s(common)
            common.pop("dt_ms", None)
    return encode(records[0]) == encode(records[1])


def read_training_configuration(folder):
    """初期重み内の設定を読み、過去の結果では従来のJSONへフォールバックする。"""
    import json
    import numpy as np
    from d_tools.recording_io import decode
    from a_experiment.paths import training_weight_directory

    folder = Path(folder)
    weights = training_weight_directory(folder, "initial") / "weights.npz"
    if weights.is_file():
        with np.load(weights, allow_pickle=False) as saved:
            if "configuration" in saved:
                return decode(json.loads(str(saved["configuration"].item())))
    legacy = folder / "config_snapshot.json"
    if legacy.is_file():
        return decode(read_record(legacy))
    return None


def has_training_configuration(folder):
    """テストに必要な保存設定が、新旧いずれかの形式で存在するか確認する。"""
    return read_training_configuration(folder) is not None


def save_training_configuration(folder, cfg):
    """設定を既存の初期重みNPZに含め、training直下に設定ファイルを増やさない。"""
    import json
    import tempfile
    import numpy as np
    from a_experiment.paths import training_weight_directory

    weights = training_weight_directory(folder, "initial") / "weights.npz"
    with np.load(weights, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    arrays["configuration"] = np.asarray(json.dumps(encode(snapshot(cfg)), ensure_ascii=False))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=weights.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        temporary.replace(weights)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_parameter_snapshots(cfg, *, rep=1):
    """共通設定と学習則設定を分け、同じ保存先に異なる条件が混在しないよう確認する。"""
    files = parameter_snapshot_files(cfg, rep)
    common = deepcopy(snapshot(cfg))
    common["input_filters"] = {
        key: deepcopy(cfg[key])
        for key in ("input_filter_map", "sensor_gain", "filter_gain", "opt_filter_gain")
        if key in cfg
    }
    common["input_filters"]["functions"] = {
        name: getattr(function, "__name__", str(function))
        for name, function in cfg.get("filter_funcs", {}).items()
    }
    common.pop("learning_rule_models", None)
    common["models"].pop("LEARNING_RULE_MODEL", None)
    # repごと・学習則ごとに異なる値は、初期重みNPZ内の完全な設定が保持する。
    for name in (
        "OUTPUT_SEED",
        "OUTPUT_SEEDS",
        "OUTPUT_LEARNING_PARAMETERS",
        "learning_rule_liq_to_out",
    ):
        common["network"].pop(name, None)
    for name in ("READOUT_TRAIN_DIR", "READOUT_SEED_INDEX", "DATA_SPLIT_REP"):
        common["run"].pop(name, None)
    common["repetition_parameters"] = {
        "liquid_seed": int(cfg["common"]["BASE_SEED"]),
        "sweep_output_seed_rule": "BASE_SEED + rep number",
        "actual_output_seed_and_data_split": "See each rep/training/config_snapshot.json (off: rep/config_snapshot.json)",
    }
    rule = learning_rule_name(cfg)
    rules = output_learning_rules(cfg)
    parameters = output_learning_parameters(cfg)
    learning = {
        "learning_rule": rule,
        "parameters": parameters[0] if len(rules) == 1 else parameters,
        "time_constant_unit": "ms",
        "amplitudes": "Stored values including their signs",
    }
    if len(rules) > 1:
        learning["outputs"] = [
            {"output_index": i, "learning_rule": name, "parameters": params}
            for i, (name, params) in enumerate(zip(rules, parameters))
        ]
    for kind, payload in [("network", common), ("learning", learning)]:
        path = Path(files[kind])
        payload = encode(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not parameter_records_equal(read_record(path), payload):
                raise ValueError(f"Parameters differ from the saved experiment: {path}")
            continue
        # 新規保存を競合させても既存記録は上書きせず、先に保存された内容を比較する。
        try:
            save_record(path, payload, overwrite=False)
        except FileExistsError:
            if not parameter_records_equal(read_record(path), payload):
                raise ValueError(f"Parameters differ from the saved experiment: {path}")
    return files
