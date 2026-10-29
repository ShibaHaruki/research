"""パラメータはJSON、進捗・集計・記録情報はNPZで保存し、旧JSONも読み込む。"""

from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def encode(value):
    """辞書・配列・パスを、pickleを使わず保存できる値へ変換する。"""
    if isinstance(value, dict):
        if any(isinstance(key, tuple) for key in value):
            return {"__tuple_keys__": [[list(key), encode(item)] for key, item in value.items()]}
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def is_parameter_record(path):
    """JSON保存を許可するパラメータ記録のファイル名か確認する。"""
    name = Path(path).name
    return name in {
        "network_parameters.json",
        "learning_rule_parameters.json",
        "selected_parameters.json",
        "config_snapshot.json",
        "test_config_snapshot.json",
        "sweep_config.json",
    } or (name.startswith("experiment_parameters_") and name.endswith(".json"))


def save_record(path, data, *, overwrite=True):
    """NPZ内の文字列に階層構造を保持する。pickleを使わず、一時ファイルから置換する。"""
    path = Path(path)
    if path.suffix == ".json" and not is_parameter_record(path):
        raise ValueError(f"Only parameter records may use JSON: {path}")
    if path.suffix not in (".npz", ".json"):
        raise ValueError("Record extension must be .npz or parameter .json")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(encode(data), ensure_ascii=False, indent=2)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix="_record_", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            if path.suffix == ".json":
                stream.write(text.encode("utf-8"))
            else:
                np.savez_compressed(stream, record=np.asarray(text))
        if overwrite:
            temporary.replace(path)
        else:
            # 完成済みの一時ファイルを新規公開する。既存の記録は置換しない。
            # repが同じパラメータを同時保存しても、Windowsの読込中ファイルと衝突しない。
            if os.name == "nt":
                os.rename(temporary, path)  # Windowsのrenameは既存ファイルを上書きしない。
            else:
                os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _candidates(path):
    """現在のNPZと過去のJSONを、指定された形式を優先する順に並べる。"""
    path = Path(path)
    if path.suffix == ".npz":
        return [path, path.with_suffix(".json")]
    if path.suffix == ".json" and not is_parameter_record(path):
        return [path, path.with_suffix(".npz")]
    return [path]


def record_exists(path):
    """新旧いずれかの形式で記録が存在するか調べる。"""
    return any(candidate.is_file() for candidate in _candidates(path))


def read_record(path):
    """新NPZを優先し、過去のJSON結果も書き換えずに読み込む。"""
    for candidate in _candidates(path):
        if not candidate.is_file():
            continue
        if candidate.suffix == ".npz":
            with np.load(candidate, allow_pickle=False) as archive:
                return json.loads(str(archive["record"].item()))
        return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(path)


def decode(value):
    """保存時に変換したタプルキーを戻し、設定を再び使用できる形にする。"""
    if isinstance(value, dict):
        if "__tuple_keys__" in value:
            return {tuple(key): decode(item) for key, item in value["__tuple_keys__"]}
        # IN_ROUTEの層番号はJSON読込後も文字列で使用できる。
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value
