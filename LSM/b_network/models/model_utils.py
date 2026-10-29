"""各モデル設定に必要なキーを確認し、登録表へ追加する共通処理。"""

# models/model_utils.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def ensure_keys(model: dict[str, Any], required: Iterable[str], model_name: str) -> dict[str, Any]:
    """モデル定義の必須項目を検査し、不足しているキーを知らせる。"""
    missing = [key for key in required if key not in model]
    if missing:
        raise KeyError(f"{model_name} is missing required keys: {missing}")
    return model


def merge_namespace(*namespaces: dict[str, Any] | None) -> dict[str, Any]:
    """複数のモデルの外部定数を、後の指定を優先して統合する。"""
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
    """必須項目を確認して、名前から取得できるモデル登録表へ追加する。"""
    if name in registry:
        raise KeyError(f"Model '{name}' is already registered.")
    model = dict(model)
    model.setdefault("name", name)
    model.setdefault("namespace", {})
    ensure_keys(model, required, name)
    registry[name] = model
    return model
