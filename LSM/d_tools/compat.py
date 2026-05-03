"""古い設定形式と新しい設定形式をつなぐための互換処理。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def first_value(value, default=None):
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value if value is not None else default


def repeat_count(common_cfg: dict, default: int = 1) -> int:
    return int(first_value(common_cfg.get("NUM_REPEAT"), default))


def training_sample_count(training_cfg: dict, common_cfg: dict) -> int:
    return int(
        first_value(
            training_cfg.get("NUM_TRAINING_SAMPLE", training_cfg.get("NUM_TRIANING_SAMPLE")),
            common_cfg.get("NUM_SAMPLE", 0),
        )
    )


def liquid_sample_count(liquid_cfg: dict, training_cfg: dict, common_cfg: dict) -> int:
    return int(
        first_value(
            liquid_cfg.get("NUM_LIQUID_SAMPLE"),
            training_sample_count(training_cfg, common_cfg),
        )
    )


def test_sample_count(test_cfg: dict, common_cfg: dict) -> int:
    return int(first_value(test_cfg.get("NUM_TEST_SAMPLE"), common_cfg.get("NUM_SAMPLE", 0)))


def test_bin_steps(test_cfg: dict, default: int | None = 10) -> int | None:
    value = first_value(test_cfg.get("BIN_STEPS"), default)
    return None if value is None else int(value)


def normalize_target_name(target: str | None) -> str | None:
    if target is None:
        return None
    aliases = {
        "training_test": "train_test",
        "training_test_eval": "train_test_eval",
    }
    return aliases.get(target, target)


def normalize_input_route_key(key: Any) -> tuple[int, str] | None:
    if isinstance(key, tuple) and len(key) == 2:
        return int(key[0]), str(key[1])
    if isinstance(key, list) and len(key) == 2:
        return int(key[0]), str(key[1])
    if isinstance(key, str):
        for sep in (":", "|", ",", "__"):
            if sep in key:
                left, right = key.split(sep, 1)
                try:
                    return int(left), str(right)
                except ValueError:
                    break
    return None


def canonical_input_route(
    network_cfg: dict,
    *,
    input_filter_map: dict[int, list[str]],
) -> dict[tuple[int, str], dict[str, Any]]:
    channels = sorted(input_filter_map)
    if not channels:
        return {}

    filters = list(input_filter_map[channels[0]])
    route: dict[tuple[int, str], dict[str, Any]] = {}

    route_layers = network_cfg.get("IN_ROUTE_LAYERS")
    if isinstance(route_layers, dict):
        for ch in channels:
            for filter_name in input_filter_map[ch]:
                route[(int(ch), str(filter_name))] = {"layers": deepcopy(route_layers)}

    explicit_route = network_cfg.get("IN_ROUTE", {})
    if isinstance(explicit_route, dict):
        for raw_key, value in explicit_route.items():
            norm_key = normalize_input_route_key(raw_key)
            if norm_key is None:
                continue
            if norm_key in route:
                continue

            if isinstance(value, dict) and "layers" in value:
                route[norm_key] = deepcopy(value)
                continue

            if isinstance(value, dict):
                route[norm_key] = {"layers": deepcopy(value)}
                continue

    if route:
        return route

    if not isinstance(route_layers, dict):
        raise KeyError("CFG_NETWORK must define either IN_ROUTE or IN_ROUTE_LAYERS.")

    for ch in channels:
        for filter_name in filters:
            route[(int(ch), str(filter_name))] = {"layers": deepcopy(route_layers)}
    return route
