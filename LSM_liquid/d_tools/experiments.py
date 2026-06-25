"""EXPERIMENT 設定を展開し、設定上書きや実験記録を書き出す処理。"""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

from .run_paths import jsonable, safe_stem


MANIFEST_FIELDS = (
    "experiment_name",
    "trial_index",
    "trial_id",
    "target",
    "status",
    "started_at",
    "finished_at",
    "result_dir",
    "message",
    "overrides_json",
)


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def experiment_run_id(experiment_name: str, trial_id: str) -> str:
    return safe_stem(f"{experiment_name}__{trial_id}")


def _dict_key(container: dict, part: str):
    if part in container:
        return part
    try:
        int_key = int(part)
    except ValueError:
        return part
    return int_key if int_key in container else part


def _path_parts(path: Any) -> list[Any]:
    if isinstance(path, str):
        return path.split(".")
    if isinstance(path, (tuple, list)):
        return list(path)
    raise TypeError(f"override path must be str, tuple, or list; got {type(path).__name__}")


def _label_key(path: Any) -> str:
    parts = _path_parts(path)
    if not parts:
        return "value"
    tail = parts[-1]
    if isinstance(tail, tuple):
        return "__".join(str(item) for item in tail)
    return str(tail)


def set_by_path(config: dict[str, Any], path: Any, value: Any) -> None:
    parts = _path_parts(path)
    if not parts:
        raise ValueError("override path is empty")

    current: Any = config
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            if part in current:
                current = current[part]
            else:
                current = current[_dict_key(current, str(part))]
        else:
            raise TypeError(f"Cannot enter '{part}' in override path '{path}'.")

    last = parts[-1]
    if isinstance(current, list):
        current[int(last)] = value
    elif isinstance(current, dict):
        if last in current:
            current[last] = value
        else:
            current[_dict_key(current, str(last))] = value
    else:
        raise TypeError(f"Cannot set '{path}' on {type(current).__name__}.")


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    # EXPERIMENT/CMA-ES から渡された dot-path の上書きを cfg に反映する。
    cfg = deepcopy(config)
    for path, value in (overrides or {}).items():
        set_by_path(cfg, path, deepcopy(value))
    return cfg


def _normalize_override_entry(entry: Any, index: int) -> dict[str, Any]:
    if isinstance(entry, dict) and "overrides" in entry:
        trial_id = str(entry.get("id", f"trial{index:03d}"))
        return {
            "id": safe_stem(trial_id),
            "memo": entry.get("memo", ""),
            "target": entry.get("target"),
            "overrides": dict(entry.get("overrides", {})),
        }

    if isinstance(entry, dict):
        return {
            "id": f"trial{index:03d}",
            "memo": "",
            "target": None,
            "overrides": dict(entry),
        }

    raise TypeError("Each trial/override entry must be a dict.")


def _grid_trials(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    values = [list(grid[key]) for key in keys]
    trials = []
    for index, combo in enumerate(product(*values), start=1):
        overrides = dict(zip(keys, combo))
        label = "__".join(
            f"{safe_stem(_label_key(key))}_{safe_stem(str(value))[:24]}"
            for key, value in overrides.items()
        )
        trials.append(
            {
                "id": f"grid{index:03d}_{label}"[:120],
                "memo": "",
                "target": None,
                "overrides": overrides,
            }
        )
    return trials


def expand_experiment(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    # trials / grid / overrides を、run_experiment が1件ずつ実行できる trial list に展開する。
    base_overrides = dict(experiment.get("base_overrides", {}))

    if "trials" in experiment:
        trials = [
            _normalize_override_entry(entry, index)
            for index, entry in enumerate(experiment["trials"], start=1)
        ]
    elif "grid" in experiment:
        trials = _grid_trials(experiment["grid"])
    elif isinstance(experiment.get("overrides"), list):
        trials = [
            _normalize_override_entry(entry, index)
            for index, entry in enumerate(experiment["overrides"], start=1)
        ]
    else:
        trials = [
            {
                "id": "base",
                "memo": experiment.get("description", ""),
                "target": None,
                "overrides": dict(experiment.get("overrides", {})),
            }
        ]

    normalized = []
    for index, trial in enumerate(trials, start=1):
        overrides = deepcopy(base_overrides)
        overrides.update(deepcopy(trial.get("overrides", {})))
        normalized.append(
            {
                "index": index,
                "id": safe_stem(str(trial.get("id", f"trial{index:03d}"))),
                "memo": trial.get("memo", ""),
                "target": trial.get("target"),
                "overrides": overrides,
            }
        )
    return normalized


def append_manifest_row(path: Path, row: dict[str, Any]) -> None:
    # 実験全体の進捗を CSV に追記する。後から成功/失敗と保存先を確認するため。
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def write_experiment_trial_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
