"""CMA-ESのパラメータ探索で使う共通処理。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from d_tools.experiments import set_by_path
from d_tools.run_paths import jsonable, safe_stem
from d_tools.separation_metrics import (
    linear_separation_property,
    load_internal_state_dataset,
    pairwise_separation_matrix,
    scatter_metrics,
    temporal_separation_metrics,
)


EPS = 1e-12


def fixed_vae_encoder_source(metric_cfg: dict) -> str:
    # 空文字なら固定Encoderなし。指定された場合は探索中にVAEを再学習しない。
    vae_cfg = dict(metric_cfg.get("vae", {}))
    for key in ("fixed_encoder_dir", "fixed_encoder_model", "fixed_encoder_path"):
        value = vae_cfg.get(key)
        if value:
            return str(value)
    return ""


# =========================
# 設定 path の読み書き
# =========================
# CMA-ES では「どの設定値を動かすか」を path で指定する。
# ここでは "network.p_in" や tuple/list の path を実際の cfg 参照に変換する。

def path_parts(path: Any) -> list[Any]:
    if isinstance(path, str):
        return path.split(".")
    if isinstance(path, (tuple, list)):
        return list(path)
    raise TypeError(f"path must be str, tuple, or list; got {type(path).__name__}")


def path_label(path: Any) -> str:
    parts = path_parts(path)
    return safe_stem("__".join(str(part) for part in parts))


def _dict_lookup(container: dict, part: Any):
    if part in container:
        return container[part]
    if isinstance(part, str):
        try:
            int_part = int(part)
        except ValueError:
            int_part = None
        if int_part is not None and int_part in container:
            return container[int_part]
    raise KeyError(part)


def get_by_path(config: dict, path: Any):
    current: Any = config
    for part in path_parts(path):
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = _dict_lookup(current, part)
        else:
            raise TypeError(f"Cannot enter '{part}' in path '{path}'.")
    return current


@dataclass(frozen=True)
class SearchParameter:
    # 1つの探索パラメータを [0, 1] の CMA-ES 空間と実スケールの間で変換する。
    name: str
    path: Any
    lower: float
    upper: float
    x0: float
    scale: str = "linear"
    kind: str = "float"

    def unit_from_value(self, value: float) -> float:
        value = float(value)
        if self.scale in {"log", "log10"}:
            if self.lower <= 0 or self.upper <= 0:
                raise ValueError(f"log-scaled parameter {self.name} needs positive bounds.")
            lo = math.log(self.lower)
            hi = math.log(self.upper)
            return (math.log(max(value, EPS)) - lo) / max(hi - lo, EPS)
        return (value - self.lower) / max(self.upper - self.lower, EPS)

    def value_from_unit(self, unit_value: float):
        u = float(np.clip(unit_value, 0.0, 1.0))
        if self.scale in {"log", "log10"}:
            lo = math.log(self.lower)
            hi = math.log(self.upper)
            value = math.exp(lo + u * (hi - lo))
        else:
            value = self.lower + u * (self.upper - self.lower)

        if self.kind in {"int", "integer"}:
            return int(round(value))
        return float(value)


def normalize_parameter_specs(specs: Sequence[dict], base_cfg: dict) -> list[SearchParameter]:
    params: list[SearchParameter] = []
    for index, spec in enumerate(specs, start=1):
        bounds = spec.get("bounds")
        if bounds is None or len(bounds) != 2:
            raise ValueError(f"parameter #{index} needs bounds=[lower, upper].")
        path = spec["path"]
        lower = float(bounds[0])
        upper = float(bounds[1])
        if upper <= lower:
            raise ValueError(f"parameter {spec.get('name', index)} has invalid bounds: {bounds}")
        x0 = spec.get("x0")
        if x0 is None:
            x0 = get_by_path(base_cfg, path)
        params.append(
            SearchParameter(
                name=str(spec.get("name") or path_label(path)),
                path=path,
                lower=lower,
                upper=upper,
                x0=float(x0),
                scale=str(spec.get("scale", "linear")),
                kind=str(spec.get("kind", "float")),
            )
        )
    if not params:
        raise ValueError("At least one CMA-ES parameter is required.")
    return params


def initial_unit_vector(params: Sequence[SearchParameter]) -> np.ndarray:
    return np.asarray(
        [np.clip(param.unit_from_value(param.x0), 0.0, 1.0) for param in params],
        dtype=np.float64,
    )


def unit_vector_to_values(params: Sequence[SearchParameter], x_unit: Sequence[float]) -> dict[str, Any]:
    return {
        param.name: param.value_from_unit(float(x_unit[index]))
        for index, param in enumerate(params)
    }


def apply_parameter_values(cfg: dict, params: Sequence[SearchParameter], values: dict[str, Any]) -> dict:
    for param in params:
        set_by_path(cfg, param.path, values[param.name])
    return cfg


def stack_features_for_silhouette(features_by_class: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for class_index, features in enumerate(features_by_class):
        arr = np.asarray(features, dtype=np.float64)
        rows.append(arr)
        labels.extend([class_index] * arr.shape[0])
    if not rows:
        raise ValueError("No features to stack.")
    return np.vstack(rows), np.asarray(labels, dtype=np.int64)


def _silhouette_score(features_by_class: Sequence[np.ndarray]) -> float:
    from internal_state_vae import silhouette_score_safe

    x, labels = stack_features_for_silhouette(features_by_class)
    return silhouette_score_safe(x, labels)


def _positive_part(value: float) -> float:
    return max(0.0, float(value))


def compute_activity_penalties(internal_state_dir, penalty_cfg: dict, vae_cfg: dict) -> dict:
    from internal_state_vae import load_windowed_internal_state_dataset

    # 内部状態から平均発火率と同期度を計算し、極端な活動を目的関数から減点する。
    # これにより「分離は高いが全ニューロンが同時に発火する」ような解を避ける。
    dataset = load_windowed_internal_state_dataset(
        internal_state_dir,
        window_ms=float(vae_cfg.get("window_ms", 10.0)),
        step_ms=float(vae_cfg.get("step_ms", 10.0)),
        max_samples_per_class=vae_cfg.get("max_samples_per_class"),
    )
    x = np.asarray(dataset.x, dtype=np.float64)

    # 現在の spike_bin_mean は spikes/ms として保存されるため、1000倍して Hz に直す。
    rate_scale_hz = float(penalty_cfg.get("rate_scale_hz", 1000.0))
    x_hz = x * rate_scale_hz
    mean_rate_hz = float(np.mean(x_hz))
    min_rate_hz = float(penalty_cfg.get("target_rate_min_hz", 1.0))
    max_rate_hz = float(penalty_cfg.get("target_rate_max_hz", 80.0))
    rate_ref_hz = max(float(penalty_cfg.get("rate_ref_hz", max_rate_hz - min_rate_hz)), EPS)
    rate_low = _positive_part(min_rate_hz - mean_rate_hz) / rate_ref_hz
    rate_high = _positive_part(mean_rate_hz - max_rate_hz) / rate_ref_hz
    rate_penalty_raw = float(rate_low * rate_low + rate_high * rate_high)
    rate_weight = float(penalty_cfg.get("rate_weight", 1.0))
    firing_rate_penalty = rate_weight * rate_penalty_raw

    # 同期度は集団発火率の時間方向の揺らぎで見る。
    # 多数のニューロンが同じ bin で発火すると pop_rate がバースト状になり、この値が大きくなる。
    population_rate_hz = np.mean(x_hz, axis=1)
    pop_mean = np.mean(population_rate_hz, axis=1)
    pop_std = np.std(population_rate_hz, axis=1)
    sync_by_sample = pop_std / np.maximum(np.abs(pop_mean), EPS)
    sync_index = float(np.mean(sync_by_sample))
    sync_max = float(penalty_cfg.get("sync_max", 1.0))
    sync_ref = max(float(penalty_cfg.get("sync_ref", 1.0)), EPS)
    sync_penalty_raw = float((_positive_part(sync_index - sync_max) / sync_ref) ** 2)
    sync_weight = float(penalty_cfg.get("sync_weight", 1.0))
    synchrony_penalty = sync_weight * sync_penalty_raw

    return {
        "firing_rate_penalty": float(firing_rate_penalty),
        "synchrony_penalty": float(synchrony_penalty),
        "penalty_total": float(firing_rate_penalty + synchrony_penalty),
        "mean_rate_hz": mean_rate_hz,
        "target_rate_min_hz": min_rate_hz,
        "target_rate_max_hz": max_rate_hz,
        "rate_penalty_raw": rate_penalty_raw,
        "rate_weight": rate_weight,
        "sync_index": sync_index,
        "sync_max": sync_max,
        "sync_penalty_raw": sync_penalty_raw,
        "sync_weight": sync_weight,
        "input_shape_batch_N_K": list(x.shape),
    }


def compute_vae_composite_objective(internal_state_dir, metric_cfg: dict, *, metric_out_dir=None) -> dict:
    # 1. 内部状態から VAE を学習
    # 2. z=mu に対して Silhouette と DR を計算
    # 3. 発火率・同期ペナルティを引いて J を作る
    metric_out_dir = Path(metric_out_dir) if metric_out_dir is not None else Path("vae_objective")
    vae_cfg = dict(metric_cfg.get("vae", {}))
    vae_cfg.setdefault("latent_dim", 16)
    vae_cfg.setdefault("window_ms", 10.0)
    vae_cfg.setdefault("step_ms", 10.0)

    vae_out_dir = metric_out_dir / "vae_latent16"
    fixed_source = fixed_vae_encoder_source(metric_cfg)
    if fixed_source:
        from internal_state_vae import encode_internal_state_with_fixed_vae

        vae_result = encode_internal_state_with_fixed_vae(
            internal_state_dir,
            fixed_source,
            vae_out_dir,
            window_ms=float(vae_cfg.get("window_ms", 10.0)),
            step_ms=float(vae_cfg.get("step_ms", 10.0)),
            batch_size=int(vae_cfg.get("batch_size", 128)),
            device=str(vae_cfg.get("device", "auto")),
            max_samples_per_class=vae_cfg.get("max_samples_per_class"),
        )
    else:
        from internal_state_vae import train_internal_state_vae

        vae_result = train_internal_state_vae(
            internal_state_dir,
            vae_out_dir,
            window_ms=float(vae_cfg.get("window_ms", 10.0)),
            step_ms=float(vae_cfg.get("step_ms", 10.0)),
            latent_dim=int(vae_cfg.get("latent_dim", 16)),
            hidden_channels=int(vae_cfg.get("hidden_channels", 64)),
            beta=float(vae_cfg.get("beta", 1e-3)),
            epochs=int(vae_cfg.get("epochs", 50)),
            batch_size=int(vae_cfg.get("batch_size", 32)),
            lr=float(vae_cfg.get("lr", 1e-3)),
            seed=int(vae_cfg.get("seed", 0)),
            device=str(vae_cfg.get("device", "auto")),
            standardize=bool(vae_cfg.get("standardize", True)),
            max_samples_per_class=vae_cfg.get("max_samples_per_class"),
        )
    penalties = compute_activity_penalties(
        internal_state_dir,
        dict(metric_cfg.get("penalties", {})),
        vae_cfg,
    )
    objective_cfg = dict(metric_cfg.get("objective", {}))
    silhouette_weight = float(objective_cfg.get("silhouette_weight", 1.0))
    dr_weight = float(objective_cfg.get("DR_weight", objective_cfg.get("dr_weight", 1.0)))
    silhouette = float(vae_result["silhouette"])
    dr = float(vae_result["DR"])
    j_score = (
        silhouette_weight * silhouette
        + dr_weight * dr
        - penalties["firing_rate_penalty"]
        - penalties["synchrony_penalty"]
    )

    details = {
        "J": float(j_score),
        "silhouette": silhouette,
        "DR": dr,
        "silhouette_weight": silhouette_weight,
        "DR_weight": dr_weight,
        **penalties,
        "vae": vae_result,
    }
    metric_out_dir.mkdir(parents=True, exist_ok=True)
    (metric_out_dir / "vae_composite_objective.json").write_text(
        json.dumps(jsonable(details), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "metric": str(metric_cfg.get("name", "VAE_J")),
        "direction": str(metric_cfg.get("direction", "maximize")).lower(),
        "score": float(j_score),
        "details": details,
    }


def metric_uses_common_vae(metric_cfg: dict) -> bool:
    # True の場合、候補ごとにVAEを学習せず、世代内/累積候補をまとめた共通VAEで評価する。
    metric_key = str(metric_cfg.get("name", "DR")).lower()
    if fixed_vae_encoder_source(metric_cfg):
        return False
    vae_metric = metric_key in {
        "j",
        "vae_j",
        "vae_composite_j",
        "vae_objective_j",
        "vae_dr",
        "vae_silhouette",
        "vae_trace_sb",
        "vae_trace_sw",
    }
    return vae_metric and bool(metric_cfg.get("common_latent_space", metric_cfg.get("common_vae", False)))


def compute_common_vae_population_metrics(
    candidate_entries: Sequence[dict],
    metric_cfg: dict,
    *,
    metric_out_dir=None,
) -> dict[str, dict]:
    from internal_state_vae import train_common_internal_state_vae

    # 複数候補の内部状態をまとめて1つのVAEを学習し、同じEncoderで各候補を評価する。
    entries = [
        dict(entry)
        for entry in candidate_entries
        if entry.get("internal_state_dir")
    ]
    if not entries:
        raise ValueError("No candidate internal_state_dir entries for common VAE evaluation.")

    metric_out_dir = Path(metric_out_dir) if metric_out_dir is not None else Path("common_vae_objective")
    vae_cfg = dict(metric_cfg.get("vae", {}))
    vae_cfg.setdefault("latent_dim", 16)
    vae_cfg.setdefault("window_ms", 10.0)
    vae_cfg.setdefault("step_ms", 10.0)

    vae_out_dir = metric_out_dir / "common_vae_latent16"
    vae_result = train_common_internal_state_vae(
        entries,
        vae_out_dir,
        dataset_id=str(metric_cfg.get("dataset_id", "")),
        window_ms=float(vae_cfg.get("window_ms", 10.0)),
        step_ms=float(vae_cfg.get("step_ms", 10.0)),
        latent_dim=int(vae_cfg.get("latent_dim", 16)),
        hidden_channels=int(vae_cfg.get("hidden_channels", 64)),
        beta=float(vae_cfg.get("beta", 1e-3)),
        epochs=int(vae_cfg.get("epochs", 50)),
        batch_size=int(vae_cfg.get("batch_size", 32)),
        lr=float(vae_cfg.get("lr", 1e-3)),
        seed=int(vae_cfg.get("seed", 0)),
        device=str(vae_cfg.get("device", "auto")),
        standardize=bool(vae_cfg.get("standardize", True)),
        max_samples_per_class=vae_cfg.get("max_samples_per_class"),
        progress_interval=int(vae_cfg.get("progress_interval", 1)),
    )

    metric_name = str(metric_cfg.get("name", "VAE_J"))
    metric_key = metric_name.lower()
    direction = str(metric_cfg.get("direction", "maximize")).lower()
    objective_cfg = dict(metric_cfg.get("objective", {}))
    silhouette_weight = float(objective_cfg.get("silhouette_weight", 1.0))
    dr_weight = float(objective_cfg.get("DR_weight", objective_cfg.get("dr_weight", 1.0)))
    penalty_cfg = dict(metric_cfg.get("penalties", {}))
    score_key_map = {
        "vae_dr": "DR",
        "vae_silhouette": "silhouette",
        "vae_trace_sb": "trace_Sb",
        "vae_trace_sw": "trace_Sw",
    }

    common_summary = {
        key: value
        for key, value in vae_result.items()
        if key not in {"per_candidate_metrics"}
    }
    results: dict[str, dict] = {}
    per_candidate = vae_result["per_candidate_metrics"]
    metric_out_dir.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        candidate_key = str(entry["candidate_key"])
        latent = per_candidate[candidate_key]
        penalties = compute_activity_penalties(
            entry["internal_state_dir"],
            penalty_cfg,
            vae_cfg,
        )
        if metric_key in {"j", "vae_j", "vae_composite_j", "vae_objective_j"}:
            score = (
                silhouette_weight * float(latent["silhouette"])
                + dr_weight * float(latent["DR"])
                - penalties["firing_rate_penalty"]
                - penalties["synchrony_penalty"]
            )
        elif metric_key in score_key_map:
            score = float(latent[score_key_map[metric_key]])
        else:
            raise ValueError(f"Common VAE does not support metric '{metric_name}'.")

        details = {
            "common_latent_space": True,
            "candidate_key": candidate_key,
            "common_vae_scope": str(metric_cfg.get("common_vae_scope", "generation")),
            "silhouette": float(latent["silhouette"]),
            "DR": float(latent["DR"]),
            "trace_Sb": float(latent["trace_Sb"]),
            "trace_Sw": float(latent["trace_Sw"]),
            "silhouette_weight": silhouette_weight,
            "DR_weight": dr_weight,
            **penalties,
            "common_vae": common_summary,
            "candidate_latent_metrics": latent,
        }
        if metric_key in {"j", "vae_j", "vae_composite_j", "vae_objective_j"}:
            details["J"] = float(score)

        result = {
            "metric": metric_name,
            "direction": direction,
            "score": float(score),
            "details": details,
        }
        results[candidate_key] = result
        (metric_out_dir / f"{candidate_key}_common_vae_metric.json").write_text(
            json.dumps(jsonable(result), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return results


def compute_internal_state_metric(internal_state_dir, metric_cfg: dict, *, metric_out_dir=None) -> dict:
    # metric.name に応じて、通常の内部状態指標、VAE 指標、複合目的関数 J を切り替える。
    metric_name = str(metric_cfg.get("name", "DR"))
    metric_key = metric_name.lower()
    direction = str(metric_cfg.get("direction", "maximize")).lower()

    if metric_key in {"j", "vae_j", "vae_composite_j", "vae_objective_j"}:
        return compute_vae_composite_objective(
            internal_state_dir,
            metric_cfg,
            metric_out_dir=metric_out_dir,
        )

    if metric_key in {"vae_dr", "vae_silhouette", "vae_trace_sb", "vae_trace_sw"}:
        vae_cfg = dict(metric_cfg.get("vae", {}))
        vae_out_dir = metric_out_dir / "vae_metric" if metric_out_dir is not None else "vae_metric"
        fixed_source = fixed_vae_encoder_source(metric_cfg)
        if fixed_source:
            from internal_state_vae import encode_internal_state_with_fixed_vae

            vae_result = encode_internal_state_with_fixed_vae(
                internal_state_dir,
                fixed_source,
                vae_out_dir,
                window_ms=float(vae_cfg.get("window_ms", 10.0)),
                step_ms=float(vae_cfg.get("step_ms", 10.0)),
                batch_size=int(vae_cfg.get("batch_size", 128)),
                device=str(vae_cfg.get("device", "auto")),
                max_samples_per_class=vae_cfg.get("max_samples_per_class"),
            )
        else:
            from internal_state_vae import train_internal_state_vae

            vae_result = train_internal_state_vae(
                internal_state_dir,
                vae_out_dir,
                window_ms=float(vae_cfg.get("window_ms", 10.0)),
                step_ms=float(vae_cfg.get("step_ms", 10.0)),
                latent_dim=int(vae_cfg.get("latent_dim", 16)),
                hidden_channels=int(vae_cfg.get("hidden_channels", 64)),
                beta=float(vae_cfg.get("beta", 1e-3)),
                epochs=int(vae_cfg.get("epochs", 50)),
                batch_size=int(vae_cfg.get("batch_size", 32)),
                lr=float(vae_cfg.get("lr", 1e-3)),
                seed=int(vae_cfg.get("seed", 0)),
                device=str(vae_cfg.get("device", "auto")),
                standardize=bool(vae_cfg.get("standardize", True)),
                max_samples_per_class=vae_cfg.get("max_samples_per_class"),
            )
        key_map = {
            "vae_dr": "DR",
            "vae_silhouette": "silhouette",
            "vae_trace_sb": "trace_Sb",
            "vae_trace_sw": "trace_Sw",
        }
        score_key = key_map[metric_key]
        return {
            "metric": metric_name,
            "direction": direction,
            "score": float(vae_result[score_key]),
            "details": vae_result,
        }

    load_trajectories = metric_key.startswith("temporal_")
    dataset = load_internal_state_dataset(
        internal_state_dir,
        feature_mode=str(metric_cfg.get("feature_mode", "flatten")),
        max_samples_per_class=metric_cfg.get("max_samples_per_class"),
        window_start_ms=metric_cfg.get("window_start_ms"),
        window_end_ms=metric_cfg.get("window_end_ms"),
        load_trajectories=load_trajectories,
    )
    features_by_class = dataset["features_by_class"]

    scatter = scatter_metrics(features_by_class)
    linear = linear_separation_property(
        features_by_class,
        tol=metric_cfg.get("rank_tol"),
    )
    details = {
        "DR": scatter["DR"],
        "trace_Sb": scatter["trace_Sb"],
        "trace_Sw": scatter["trace_Sw"],
        "SPlin": linear["SPlin"],
        "SPlin_normalized": linear["normalized_rank"],
        "n_classes": scatter["n_classes"],
        "n_samples_total": scatter["n_samples_total"],
        "n_features": scatter["n_features"],
        "class_counts": scatter["class_counts"],
        "materials": dataset["materials"],
    }

    if metric_key == "silhouette":
        details["silhouette"] = _silhouette_score(features_by_class)
    if metric_key in {"sppw_between_mean", "sppw_within_mean", "sppw_gap"}:
        pairwise = pairwise_separation_matrix(
            features_by_class,
            batch_size=int(metric_cfg.get("pairwise_batch_size", 256)),
        )
        details.update(
            {
                "SPpw_between_mean": pairwise["SPpw_between_mean"],
                "SPpw_within_mean": pairwise["SPpw_within_mean"],
                "SPpw_gap": pairwise["SPpw_between_mean"] - pairwise["SPpw_within_mean"],
            }
        )
    if metric_key.startswith("temporal_"):
        temporal = temporal_separation_metrics(
            dataset["trajectories_by_class"],
            t_ms=dataset.get("t_ms"),
            rank_tol=metric_cfg.get("rank_tol"),
        )
        details.update(
            {
                "temporal_DR_mean": float(np.mean(temporal["DR"])),
                "temporal_DR_max": float(np.max(temporal["DR"])),
                "temporal_trace_Sb_mean": float(np.mean(temporal["trace_Sb"])),
                "temporal_trace_Sw_mean": float(np.mean(temporal["trace_Sw"])),
                "temporal_SPlin_mean": float(np.mean(temporal["SPlin"])),
                "temporal_SPlin_normalized_mean": float(np.mean(temporal["SPlin_normalized"])),
            }
        )

    score_lookup = {str(key).lower(): value for key, value in details.items()}
    if metric_key not in score_lookup:
        available = ", ".join(sorted(details))
        raise ValueError(f"Unknown metric name '{metric_name}'. Available metrics: {available}")

    return {
        "metric": metric_name,
        "direction": direction,
        "score": float(score_lookup[metric_key]),
        "details": details,
    }


class BoundedCMAES:
    """[0, 1] に正規化した探索空間で動く、NumPy だけの小さな CMA-ES 実装。"""

    def __init__(
        self,
        x0: Sequence[float],
        *,
        sigma0: float = 0.25,
        population_size: int | None = None,
        seed: int = 0,
    ):
        self.rng = np.random.default_rng(int(seed))
        self.n_dim = int(len(x0))
        if self.n_dim <= 0:
            raise ValueError("x0 must not be empty.")
        self.mean = np.asarray(x0, dtype=np.float64).reshape(self.n_dim)
        self.mean = np.clip(self.mean, 0.0, 1.0)
        self.sigma = float(sigma0)

        self.lambda_ = int(population_size or (4 + int(3 * np.log(self.n_dim))))
        self.lambda_ = max(2, self.lambda_)
        self.mu = self.lambda_ // 2
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mueff = float(1.0 / np.sum(self.weights * self.weights))

        n = float(self.n_dim)
        self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
        self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((n + 2.0) ** 2 + self.mueff),
        )
        self.damps = 1.0 + 2.0 * max(0.0, math.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0) + self.cs

        self.pc = np.zeros(self.n_dim, dtype=np.float64)
        self.ps = np.zeros(self.n_dim, dtype=np.float64)
        self.B = np.eye(self.n_dim, dtype=np.float64)
        self.D = np.ones(self.n_dim, dtype=np.float64)
        self.C = np.eye(self.n_dim, dtype=np.float64)
        self.invsqrtC = np.eye(self.n_dim, dtype=np.float64)
        self.chiN = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
        self.counteval = 0

    def ask(self) -> list[dict]:
        # 現在の平均と共分散から次世代の候補パラメータを生成する。
        candidates = []
        for index in range(self.lambda_):
            z = self.rng.standard_normal(self.n_dim)
            y = self.B @ (self.D * z)
            x_raw = self.mean + self.sigma * y
            x = np.clip(x_raw, 0.0, 1.0)
            candidates.append(
                {
                    "index": index,
                    "x": x,
                    "x_raw": x_raw,
                    "z": z,
                }
            )
        return candidates

    def tell(self, candidates: Sequence[dict], losses: Sequence[float]) -> dict:
        # 評価済み loss を使って、平均・分散・探索幅 sigma を更新する。
        losses_arr = np.asarray(losses, dtype=np.float64)
        order = np.argsort(losses_arr)
        sorted_candidates = [candidates[int(index)] for index in order]
        selected = sorted_candidates[: self.mu]

        xold = self.mean.copy()
        x_selected = np.vstack([candidate["x"] for candidate in selected])
        y_selected = (x_selected - xold[None, :]) / max(self.sigma, EPS)
        self.mean = np.sum(x_selected * self.weights[:, None], axis=0)
        y_w = np.sum(y_selected * self.weights[:, None], axis=0)

        self.counteval += len(candidates)
        self.ps = (1.0 - self.cs) * self.ps + math.sqrt(
            self.cs * (2.0 - self.cs) * self.mueff
        ) * (self.invsqrtC @ y_w)
        ps_norm = float(np.linalg.norm(self.ps))
        hsig_den = math.sqrt(1.0 - (1.0 - self.cs) ** (2.0 * self.counteval / self.lambda_))
        hsig = ps_norm / max(hsig_den, EPS) / self.chiN < (1.4 + 2.0 / (self.n_dim + 1.0))

        self.pc = (1.0 - self.cc) * self.pc + float(hsig) * math.sqrt(
            self.cc * (2.0 - self.cc) * self.mueff
        ) * y_w
        rank_mu = np.zeros_like(self.C)
        for weight, y in zip(self.weights, y_selected):
            rank_mu += float(weight) * np.outer(y, y)

        self.C = (
            (1.0 - self.c1 - self.cmu) * self.C
            + self.c1
            * (
                np.outer(self.pc, self.pc)
                + (1.0 - float(hsig)) * self.cc * (2.0 - self.cc) * self.C
            )
            + self.cmu * rank_mu
        )
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        self.sigma *= math.exp((self.cs / self.damps) * (ps_norm / self.chiN - 1.0))

        eigen_values, eigen_vectors = np.linalg.eigh(self.C)
        eigen_values = np.maximum(eigen_values, EPS)
        self.D = np.sqrt(eigen_values)
        self.B = eigen_vectors
        self.invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T

        best_index = int(order[0])
        return {
            "best_candidate_index": best_index,
            "best_loss": float(losses_arr[best_index]),
            "mean": self.mean.copy(),
            "sigma": float(self.sigma),
            "condition_number": float(np.max(self.D) / max(np.min(self.D), EPS)),
        }
