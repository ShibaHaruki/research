"""実験結果の保存先とフォルダ名を定義する。

出力層の基本構造:
g_tactile_results/readout_run/結合確率とgain/学習則名/学習則パラメータ/
"""
from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "g_tactile_results"
READOUT_FOLDER = "readout_run"

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
    rule = cfg["models"].get("LEARNING_RULE_MODEL", "off")
    params = cfg.get("learning_rule_models", {}).get(rule, {})
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
    probability_name = "-".join(repr(float(layer["EE"])) for layer in probabilities)
    gain_name = "-".join(repr(float(layer["EE"])) for layer in gains)
    name = _component(f"p_E_readout-{probability_name}__g_e_out-{gain_name}")
    return root / READOUT_FOLDER / name


def readout_rule_directory(cfg: dict, *, results_root=None) -> Path:
    """結果ルート/readout_run/結合確率とgain/学習則名。"""
    rule = cfg["models"].get("LEARNING_RULE_MODEL", "off")
    return readout_connection_directory(cfg, results_root=results_root) / _component(rule)


def readout_configuration_directory(cfg: dict, *, rule_dir=None, results_root=None) -> Path:
    """学習則名/パラメータ組み合わせ。"""
    root = readout_rule_directory(cfg, results_root=results_root) if rule_dir is None else Path(rule_dir)
    return root / learning_parameter_directory(cfg)


def training_directory(cfg: dict) -> Path:
    """学習結果の保存先。明示的なREADOUT_TRAIN_DIR指定にも対応する。"""
    override = cfg["run"].get("READOUT_TRAIN_DIR")
    return Path(override).resolve() if override else readout_configuration_directory(cfg)


def test_directory(train_dir) -> Path:
    """対応する学習フォルダ内にテスト結果を置く。"""
    return Path(train_dir).resolve() / "test"


def experiment_directory(cfg: dict, experiment_name: str, trial_id: str, *, target: str,
                         results_root=None) -> Path:
    """実験定義から実行する各条件の保存先。"""
    root = RESULTS_ROOT if results_root is None else Path(results_root)
    parent = root if target == "liquid" else readout_configuration_directory(cfg, results_root=root)
    return parent / "experiments" / _component(experiment_name) / _component(trial_id)


def experiment_manifest_directory(cfg: dict, experiment_name: str, *, target="training",
                                  results_root=None) -> Path:
    """進捗記録もE→出力の結合確率・gainごとに保存する。"""
    root = RESULTS_ROOT if results_root is None else Path(results_root)
    parent = root if target == "liquid" else readout_connection_directory(cfg, results_root=root)
    return parent / "experiments" / _component(experiment_name)
