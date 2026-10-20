"""各モデル設定に必要なキーを確認し、登録表へ追加する共通処理。"""

# models/model_utils.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def ensure_keys(model: dict[str, Any], required: Iterable[str], model_name: str) -> dict[str, Any]:
    missing = [key for key in required if key not in model]
    if missing:
        raise KeyError(f"{model_name} is missing required keys: {missing}")
    return model


def merge_namespace(*namespaces: dict[str, Any] | None) -> dict[str, Any]:
    merged = {}
    for namespace in namespaces:
        if namespace:
            merged.update(namespace)
    return merged


def register_model(
    registry: dict[str, dict[str, Any]],
    name: str,
    model: dict[str, Any],
    required: Iterable[str],
) -> dict[str, Any]:
    if name in registry:
        raise KeyError(f"Model '{name}' is already registered.")
    model = dict(model)
    model.setdefault("name", name)
    model.setdefault("namespace", {})
    ensure_keys(model, required, name)
    registry[name] = model
    return model


def alias_model(registry: dict[str, dict[str, Any]], alias: str, target: str) -> None:
    if target not in registry:
        raise KeyError(f"Cannot alias unknown model '{target}'.")
    registry[alias] = registry[target]
