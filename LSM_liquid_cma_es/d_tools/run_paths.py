"""設定内容から結果保存フォルダ名を作り、使用パラメータを保存する処理。"""

from __future__ import annotations

import json
from hashlib import sha1
from pathlib import Path

import numpy as np


PAIR_KEYS = ("EE", "EI", "IE", "II")


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def safe_stem(value: object) -> str:
    text = str(value)
    stem = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
    return stem or "item"


def value_stem(value) -> str:
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return safe_stem(text.replace(".", "p"))


def model_dir_name(model_cfg: dict, *, include_learning: bool = False) -> str:
    parts = [
        f"neuron_{safe_stem(model_cfg['NEURON_MODEL'])}",
        f"synapse_{safe_stem(model_cfg['SYNAPSE_MODEL'])}",
    ]
    return "__".join(parts)


def _as_list(value) -> list:
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if value is None:
        return []
    return [value]


def _is_pair_config(value) -> bool:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return True
    return isinstance(value, dict) and any(key in value for key in PAIR_KEYS)


def _positive(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) > 0
    if isinstance(value, dict):
        return any(_positive(v) for v in value.values())
    if isinstance(value, (list, tuple, np.ndarray)):
        return any(_positive(v) for v in value)
    return bool(value)


def _layer_value(value, layer_index: int):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        return value[layer_index] if layer_index < len(value) else value[0]
    return value


def _format_layers(indices: list[int], prefix: str) -> str:
    values = sorted(set(int(index) for index in indices))
    if not values:
        return f"{prefix}none"
    if len(values) >= 2 and values == list(range(values[0], values[-1] + 1)):
        return f"{prefix}{values[0]}-{prefix}{values[-1]}"
    return "_".join(f"{prefix}{value}" for value in values)


def _route_layer_indices(route_layers: dict) -> list[int]:
    return sorted(int(layer_index) + 1 for layer_index in route_layers)


def _input_to_liquid_layers(net_cfg: dict) -> list[int]:
    route_layers = net_cfg.get("IN_ROUTE_LAYERS")
    if isinstance(route_layers, dict):
        return _route_layer_indices(route_layers)

    route = net_cfg.get("IN_ROUTE", {})
    layers: set[int] = set()
    for info in route.values():
        if isinstance(info, dict) and isinstance(info.get("layers"), dict):
            layers.update(_route_layer_indices(info["layers"]))
    return sorted(layers)


def _liquid_recurrent_layers(net_cfg: dict) -> list[int]:
    n_liq = len(_as_list(net_cfg.get("N_liq")))
    pair_probs = net_cfg.get("p_liq_intra_pairs", {})
    layers = []
    for layer_index in range(n_liq):
        layer_probs = {
            key: _layer_value(value, layer_index)
            for key, value in pair_probs.items()
        } if isinstance(pair_probs, dict) else pair_probs
        if _positive(layer_probs):
            layers.append(layer_index + 1)
    return layers


def network_topology(net_cfg: dict, *, include_output: bool = True) -> dict:
    in_layers = _input_to_liquid_layers(net_cfg)
    rec_layers = _liquid_recurrent_layers(net_cfg)
    return {
        "input_to_liquid": {
            "target_liquid_layers": in_layers,
            "label": f"in2liq_{_format_layers(in_layers, 'L')}",
        },
        "liquid_recurrent": {
            "layers": rec_layers,
            "connection": net_cfg.get("liq_intra_connection"),
            "label": f"liqRec_{_format_layers(rec_layers, 'L')}",
        },
    }


def experiment_dir_name(cfg: dict) -> str | None:
    experiment = cfg.get("experiment")
    if not isinstance(experiment, dict):
        return None

    experiment_id = experiment.get("id")
    if experiment_id is None:
        return None
    return safe_stem(str(experiment_id))


def network_dir_name(net_cfg: dict, include_output: bool = True) -> str:
    # ネットワーク構造が変われば別フォルダになるよう、主要パラメータから名前とハッシュを作る。
    key_cfg = {
        "N_liq": net_cfg.get("N_liq"),
        "r_inh_liq": net_cfg.get("r_inh_liq"),
        # Input route probabilities/scales are part of the effective network
        # and must distinguish CMA-ES candidates in the output directory.
        "IN_ROUTE": net_cfg.get("IN_ROUTE"),
        "IN_ROUTE_LAYERS": net_cfg.get("IN_ROUTE_LAYERS"),
        "liq_intra_connection": net_cfg.get("liq_intra_connection"),
        "p_liq_intra_pairs": net_cfg.get("p_liq_intra_pairs"),
        "liq_intra_gain_pairs": net_cfg.get("liq_intra_gain_pairs"),
    }
    raw = json.dumps(jsonable(key_cfg), sort_keys=True, ensure_ascii=False)
    digest = sha1(raw.encode("utf-8")).hexdigest()[:10]
    n_liq = "-".join(value_stem(v) for v in net_cfg.get("N_liq", []))
    conn = safe_stem(net_cfg.get("liq_intra_connection", "conn"))
    topology = network_topology(net_cfg, include_output=include_output)
    in_label = topology["input_to_liquid"]["label"]
    rec_label = topology["liquid_recurrent"]["label"]

    return f"Nliq_{n_liq}__{in_label}__{rec_label}__{conn}__{digest}"


def parameter_dir_name(net_cfg: dict, include_output: bool = True) -> str:
    return network_dir_name(net_cfg, include_output=include_output)


def _pick_keys(source: dict, keys: tuple[str, ...]) -> dict:
    return {key: source.get(key) for key in keys if key in source}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_text_snapshot(path: Path, title: str, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = [
        f"{title}\n",
        "=" * len(title) + "\n\n",
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False),
        "\n",
    ]
    path.write_text("".join(text), encoding="utf-8")


def network_params_snapshot(cfg: dict, net_cfg: dict, *, include_output: bool = True) -> dict:
    keys = (
        "N_liq",
        "r_inh_liq",
        "IN_ROUTE_LAYERS",
        "IN_ROUTE",
        "USE_INPUT_FILTERS",
        "NUM_CHANNEL",
        "liq_intra_connection",
        "p_liq_intra_pairs",
        "liq_intra_gain_pairs",
    )
    return {
        "network_dir": network_dir_name(net_cfg, include_output=include_output),
        "topology": network_topology(net_cfg, include_output=include_output),
        "network": _pick_keys(net_cfg, keys),
        "input_filter_map": cfg.get("input_filter_map"),
    }


def model_params_snapshot(cfg: dict, *, include_learning: bool = True) -> dict:
    model_cfg = cfg["models"]
    neuron_name = model_cfg["NEURON_MODEL"]
    synapse_name = model_cfg["SYNAPSE_MODEL"]
    model_keys = ("NEURON_MODEL", "SYNAPSE_MODEL")

    snapshot = {
        "model_dir": model_dir_name(model_cfg, include_learning=include_learning),
        "include_learning": include_learning,
        "models": _pick_keys(
            model_cfg,
            model_keys,
        ),
        "neuron_model_params": cfg.get("neuron_models", {}).get(neuron_name, {}),
        "synapse_model_params": cfg.get("synapse_models", {}).get(synapse_name, {}),
    }

    return snapshot


def save_hierarchy_param_snapshots(
    out_dir: Path,
    cfg: dict,
    net_cfg: dict,
    *,
    include_output: bool = True,
    include_learning: bool | None = None,
) -> None:
    # ネットワーク階層とモデル階層に used parameter の snapshot を残す。
    if include_learning is None:
        include_learning = include_output
    out_dir = Path(out_dir)
    if experiment_dir_name(cfg):
        model_dir = out_dir.parent
        network_dir = model_dir.parent
    else:
        model_dir = out_dir
        network_dir = model_dir.parent

    network_snapshot = network_params_snapshot(cfg, net_cfg, include_output=include_output)
    model_snapshot = model_params_snapshot(cfg, include_learning=include_learning)
    _write_json(network_dir / "network_params.json", network_snapshot)
    _write_text_snapshot(network_dir / "network_params.txt", "Network Parameters", network_snapshot)
    _write_json(model_dir / "model_params.json", model_snapshot)
    _write_text_snapshot(model_dir / "model_params.txt", "Model Parameters", model_snapshot)


def make_run_output_dir(
    root_dir: Path,
    cfg: dict,
    net_cfg: dict,
    *,
    include_output: bool = True,
    include_learning: bool | None = None,
) -> Path:
    # 保存先の標準階層を作る入口。training/test/liquid で同じ命名規則を使う。
    if include_learning is None:
        include_learning = include_output
    out_dir = (
        Path(root_dir)
        / network_dir_name(net_cfg, include_output=include_output)
        / model_dir_name(cfg["models"], include_learning=include_learning)
    )
    exp_dir = experiment_dir_name(cfg)
    include_experiment_dir = bool(
        cfg.get("run", {}).get("INCLUDE_EXPERIMENT_DIR", True)
    )
    if exp_dir and include_experiment_dir:
        out_dir = out_dir / exp_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def config_snapshot_payload(
    cfg: dict,
    net_cfg: dict,
    *,
    include_output: bool = True,
    include_learning: bool | None = None,
) -> dict:
    if include_learning is None:
        include_learning = include_output
    model_keys = ("NEURON_MODEL", "SYNAPSE_MODEL")

    snapshot = {
        "models": _pick_keys(cfg["models"], model_keys),
        "include_learning": include_learning,
        "network_dir": network_dir_name(net_cfg, include_output=include_output),
        "model_dir": model_dir_name(cfg["models"], include_learning=include_learning),
        "network": cfg["network"],
        "net_cfg": net_cfg,
        "common": cfg["common"],
        "training": cfg["training"],
        "liquid": cfg.get("liquid", {}),
        "test": cfg.get("test", {}),
        "run": cfg.get("run", {}),
        "input_filter_map": cfg.get("input_filter_map"),
        "search_params": cfg.get("search_params", {}),
        "network_params": network_params_snapshot(cfg, net_cfg, include_output=include_output),
        "model_params": model_params_snapshot(cfg, include_learning=include_learning),
    }
    if "experiment" in cfg:
        snapshot["experiment"] = cfg["experiment"]
    return snapshot


def save_used_parameters_text(
    out_dir: Path,
    cfg: dict,
    net_cfg: dict,
    *,
    include_output: bool = True,
    include_learning: bool | None = None,
    filename: str = "used_parameters.txt",
    extra: dict | None = None,
) -> Path:
    payload = config_snapshot_payload(
        cfg,
        net_cfg,
        include_output=include_output,
        include_learning=include_learning,
    )
    if extra:
        payload["extra"] = extra

    out_fp = Path(out_dir) / filename
    _write_text_snapshot(out_fp, "Used Parameters", payload)
    return out_fp


def save_config_snapshot(
    out_dir: Path,
    cfg: dict,
    net_cfg: dict,
    *,
    include_output: bool = True,
    include_learning: bool | None = None,
) -> None:
    if include_learning is None:
        include_learning = include_output
    save_hierarchy_param_snapshots(
        out_dir,
        cfg,
        net_cfg,
        include_output=include_output,
        include_learning=include_learning,
    )

    snapshot = config_snapshot_payload(
        cfg,
        net_cfg,
        include_output=include_output,
        include_learning=include_learning,
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "config_snapshot.json").write_text(
        json.dumps(jsonable(snapshot), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_used_parameters_text(
        out_dir,
        cfg,
        net_cfg,
        include_output=include_output,
        include_learning=include_learning,
    )
