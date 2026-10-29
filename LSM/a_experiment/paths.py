"""実験結果の保存先とフォルダ名を定義する。

出力層の基本構造:
g_tactile_results/readout_run/結合確率とgain/学習則名/学習則パラメータ/
"""

from __future__ import annotations

from numbers import Integral, Real
import json
from pathlib import Path
import re
import hashlib
from a_experiment.experiment_sets import (
    learning_rule_name,
    output_learning_rules,
    output_learning_parameters,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "g_tactile_results"
# 学習・テストで共通使用する試行番号の並びと、分割済みの番号を保存する場所。
SAMPLE_SEQUENCE_ROOT = RESULTS_ROOT / "sample_seq"
# Cythonのコンパイル結果を再利用する共通キャッシュ。実験結果とは分けて管理する。
CYTHON_CACHE_ROOT = RESULTS_ROOT / "_runtime_cache" / "cython"
READOUT_FOLDER = "readout_run"
TACTILE_DATA_ROOT = PROJECT_ROOT.parent
TACTILE_DATA_DIR_NAME = "tactile_data"
TACTILE_DATA_DIRECTORY = TACTILE_DATA_ROOT / TACTILE_DATA_DIR_NAME
LIQUID_RESULTS_ROOT = RESULTS_ROOT / "liquid_run"
CALIBRATION_ROOT = RESULTS_ROOT / "calibration"
CORRELATION_ROOT = RESULTS_ROOT / "liquid_readout_crosscorrelation_allpairs"

# フォルダ名に含めない項目。学習そのものの設定値には影響しない。
EXCLUDED_PARAMETERS = {"wmin", "wmax"}
EXCLUDED_PARAMETERS_BY_RULE = {"STDP": {"A_plus", "A_minus"}}


def _component(value) -> str:
    """Windowsのフォルダ名に使用できる表記にする。"""
    result = re.sub(r"[^\w.-]+", "_", str(value)).strip("._")
    if not result:
        raise ValueError("Folder name must contain usable characters")
    return result


def learning_parameter_directory(cfg: dict) -> str:
    """選択した学習則のパラメータから、組み合わせを表すフォルダ名を作る。"""
    rule = learning_rule_name(cfg)
    rules = output_learning_rules(cfg)
    if len(rules) > 1:
        params = output_learning_parameters(cfg)
        digest = hashlib.sha256(
            json.dumps({"rules": rules, "parameters": params}, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        kinds = "-".join(dict.fromkeys(rules))
        return _component(f"outputs-{len(rules)}x{kinds}__{digest}")
    params = output_learning_parameters(cfg)[0] if rules else {}
    excluded = EXCLUDED_PARAMETERS | EXCLUDED_PARAMETERS_BY_RULE.get(rule, set())
    parts = []
    for key, value in params.items():
        if key in excluded:
            continue
        if isinstance(value, Integral):
            rendered = str(int(value))
        elif isinstance(value, Real):
            rendered = repr(float(value))
        else:
            rendered = str(value)
        parts.append(_component(f"{key}-{rendered}"))
    return "__".join(parts) if parts else "no_parameters"


def readout_connection_directory(cfg: dict, *, results_root=None) -> Path:
    """readout_run直下をE→出力の結合確率とgainで分ける。複数層は層順に並べる。"""
    root = RESULTS_ROOT if results_root is None else Path(results_root)
    network = cfg["network"]
    probabilities = network["p_liq_to_out_pairs"]
    gains = network["gain_liq_to_out_pairs"]
    if not probabilities or len(probabilities) != len(gains):
        raise ValueError("Output connection probabilities/gains must have matching nonempty layers")

    def compact(values):
        rendered = [repr(float(value["EE"])) for value in values]
        if len(set(rendered)) == 1:
            return rendered[0] if len(rendered) == 1 else f"{rendered[0]}x{len(rendered)}"
        name = "-".join(rendered)
        return (
            name
            if len(name) <= 75
            else f"{len(rendered)}layers-" + hashlib.sha256(name.encode()).hexdigest()[:12]
        )

    probability_name = compact(probabilities)
    gain_name = compact(gains)
    name = _component(f"p_E_readout-{probability_name}__g_e_out-{gain_name}")
    return root / READOUT_FOLDER / name


def readout_rule_directory(cfg: dict, *, results_root=None) -> Path:
    """結果ルート/readout_run/結合確率とgain/学習則名。"""
    rule = learning_rule_name(cfg)
    return readout_connection_directory(cfg, results_root=results_root) / _component(rule)


def readout_configuration_directory(cfg: dict, *, rule_dir=None, results_root=None) -> Path:
    """学習則名/パラメータ組み合わせ。"""
    root = (
        readout_rule_directory(cfg, results_root=results_root)
        if rule_dir is None
        else Path(rule_dir)
    )
    if learning_rule_name(cfg) == "off":
        return root
    return root / learning_parameter_directory(cfg)


def training_directory(cfg: dict, rep: int = 1) -> Path:
    """学習ありは条件/rep番号/training、offは従来の条件ルートを返す。"""
    override = cfg["run"].get("READOUT_TRAIN_DIR")
    root = Path(override).resolve() if override else readout_configuration_directory(cfg)
    if learning_rule_name(cfg) == "off":
        return root
    if rep < 1:
        raise ValueError("Repetition must be positive")
    if root.name == "training" or root.parent.name == "training":
        return root
    if re.fullmatch(r"rep[1-9]\d*", root.name):
        return root / "training"
    return root / f"rep{rep}" / "training"


def test_directory(train_dir) -> Path:
    """rep番号内でtrainingとtestを兄弟にする。"""
    root = Path(train_dir).resolve()
    if root.name == "training":
        return root.parent / "test"
    if root.parent.name == "training":
        return root.parent.parent / "test" / root.name
    return root / "test"  # 過去の学習結果を指定して読む場合。


def training_weight_directory(train_dir, stage: str) -> Path:
    """ユーザー指定のintial・final・Progression_of_weightsを配置する。"""
    names = {
        "initial": "intial",
        "intial": "intial",
        "final": "final",
        "progression": "Progression_of_weights",
        "Progression_of_weights": "Progression_of_weights",
    }
    if stage not in names:
        raise ValueError("Unknown weight stage")
    return Path(train_dir) / "weight" / names[stage]


def parameter_snapshot_files(cfg: dict, rep: int = 1) -> dict[str, Path]:
    """標準階層は維持し、任意の保存先ではその範囲内にパラメータを保存する。"""
    folder = training_directory(cfg, rep)  # 実際のtraining保存先、またはoff条件の保存先。
    rule = learning_rule_name(cfg)
    if rule == "off":
        # 標準の「結合条件/off」だけ、共通設定を結合条件フォルダへ置く。
        standard = folder.name == "off" and folder.parent.name.startswith("p_E_readout-")
        network_root = folder.parent if standard else folder
        return {
            "network": network_root / "network_parameters.json",
            "learning": folder / "learning_rule_parameters.json",
        }
    # rep/training、training、旧training/repのいずれからも条件ルートを求める。
    if folder.name == "training":
        parameters = (
            folder.parent.parent
            if re.fullmatch(r"rep[1-9]\d*", folder.parent.name)
            else folder.parent
        )
    elif folder.parent.name == "training":
        parameters = folder.parent.parent
    else:
        parameters = folder
    standard = parameters.parent.name == rule and parameters.name == learning_parameter_directory(
        cfg
    )
    network_root = parameters.parent.parent if standard else parameters
    return {
        "network": network_root / "network_parameters.json",
        "learning": parameters / "learning_rule_parameters.json",
    }


def trained_weight_directory(train_dir, rep: int) -> Path:
    """新しい保存先を優先し、過去のweights/final/rep番号も読み込めるようにする。"""
    current = training_weight_directory(train_dir, "final")
    legacy = Path(train_dir) / "weights/final" / f"rep{rep}"
    return legacy if not current.exists() and legacy.exists() else current


def test_artifact_directories(test_dir, *, cfg=None, rep=1) -> dict[str, Path]:
    root = Path(test_dir)
    folders = {name: root / name for name in ("debug", "eval", "weight", "rec")}
    return folders


def off_repetition_directory(cfg: dict, rep: int) -> Path:
    """学習offの固定重み実験は、off直下でrepごとに分ける。"""
    if rep < 1:
        raise ValueError("Repetition must be positive")
    return training_directory(cfg) / f"rep{rep}"


def off_test_directory(cfg: dict, rep: int) -> Path:
    return off_repetition_directory(cfg, rep) / "test"


def liquid_test_directory(cfg: dict, rep: int) -> Path:
    """出力層なしの固定Liquid実験を、liquid_run/rep番号/testへ保存する。"""
    if rep < 1:
        raise ValueError("Repetition must be positive")
    root = Path(cfg["run"].get("LIQUID_OUTPUT_DIR", LIQUID_RESULTS_ROOT))
    return root.resolve() / f"rep{rep}" / "test"


def off_test_artifact_directories(cfg: dict, rep: int) -> dict[str, Path]:
    """off/rep番号/testの下にdebug・eval・weight・recを配置する。"""
    root = off_test_directory(cfg, rep)
    return {name: root / name for name in ("debug", "eval", "weight", "rec")}


def off_recording_files(cfg: dict, rep: int) -> dict[str, Path]:
    """1 ms発火記録の保存先。LiquidにはE/I両方、readoutには出力層全体を記録する。"""
    return test_recording_files(off_test_directory(cfg, rep), rep)


def test_recording_files(test_dir, rep: int) -> dict[str, Path]:
    root = Path(test_dir) / "rec"
    return {
        "liquid": root / "Liquid" / f"sliq_rec_rep{rep}.npy",
        "readout": root / "readout" / f"sout_rec_rep{rep}.npy",
    }


def debug_population_directory(
    debug_dir, population: str, material: str, *, layer_index=0, layer_count=1
) -> Path:
    """debug/liquidまたはreadout/素材名。複数層の場合だけlayer番号を追加する。"""
    names = {"liquid": "liquid", "output": "readout", "readout": "readout"}
    if population not in names or not 0 <= layer_index < layer_count:
        raise ValueError("Invalid debug population or layer index")
    root = Path(debug_dir) / names[population] / _component(material)
    return root if layer_count == 1 else root / f"layer_{layer_index+1}"


def evaluation_directory(test_dir, *, cfg=None) -> Path:
    """学習則にかかわらずtest/evalに分類結果を保存する。"""
    return Path(test_dir) / "eval"


def readout_seed_directory(cfg: dict, seed_index: int, *, rule_dir=None) -> Path:
    """同一パラメータの結果を、出力層の乱数seedごとに分ける。"""
    if seed_index < 1:
        raise ValueError("Seed index must be positive")
    return readout_configuration_directory(cfg, rule_dir=rule_dir) / f"rep{seed_index}"


def shared_readout_seed_directory(cfg: dict, seed_index: int, *, rule_dir=None) -> Path:
    """1つのLiquidから複数出力群を学習する際の共通記録。"""
    if seed_index < 1:
        raise ValueError("Seed index must be positive")
    root = readout_rule_directory(cfg) if rule_dir is None else Path(rule_dir)
    return root / "_shared" / f"seed_{seed_index:03d}"


def experiment_directory(
    cfg: dict, experiment_name: str, trial_id: str, *, target: str, results_root=None
) -> Path:
    """実験定義から実行する各条件の保存先。"""
    root = RESULTS_ROOT if results_root is None else Path(results_root)
    parent = root if target == "liquid" else readout_configuration_directory(cfg, results_root=root)
    return parent / "experiments" / _component(experiment_name) / _component(trial_id)


def experiment_manifest_directory(
    cfg: dict, experiment_name: str, *, target="training", results_root=None
) -> Path:
    """進捗記録もE→出力の結合確率・gainごとに保存する。"""
    root = RESULTS_ROOT if results_root is None else Path(results_root)
    parent = root if target == "liquid" else readout_connection_directory(cfg, results_root=root)
    return parent / "experiments" / _component(experiment_name)
