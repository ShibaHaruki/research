# -*- coding: utf-8 -*-
"""Evaluate one manually fixed parameter set with a pretrained fixed VAE Encoder."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_DIR = PROJECT_ROOT / "d_tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from c_configs.FIXED import cfg_run
from cma_es_search import (
    apply_parameter_values,
    compute_internal_state_metric,
    initial_unit_vector,
    normalize_parameter_specs,
    unit_vector_to_values,
)
from d_tools.experiments import apply_overrides, now_text
from d_tools.internal_state import internal_state_config
from d_tools.plotting import try_import_pyplot
from d_tools.run_paths import jsonable, make_run_output_dir, safe_stem
from d_tools.separation_metrics import pairwise_separation_matrix
from f_run.run_cma_es_search import load_search_config
from f_run.run_liquid import LIQUID_RESULT_DIR, run_liquid
from f_run.run_training import build_cfg, build_network_cfg

RUN_CFG = getattr(cfg_run, "CFG_RUN", {})
RESULTS_ROOT = PROJECT_ROOT / RUN_CFG["RESULTS_DIR"]
OUT_ROOT = RESULTS_ROOT / "fixed_vae_parameter_trial"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_trial_module(name_or_path: str):
    path = Path(name_or_path)
    if path.suffix == ".py" and path.exists():
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import trial config file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    module_name = name_or_path
    if "." not in module_name:
        module_name = f"c_configs.VAE_SEARCH.{module_name}"
    return importlib.import_module(module_name)


def _latest_fixed_encoder_dir() -> Path:
    candidates = []
    root = RESULTS_ROOT / "fixed_vae_encoder_pretrain"
    if root.exists():
        for model_fp in root.glob("**/fixed_encoder_vae/common_vae_model.pt"):
            candidates.append(model_fp.parent)
        for model_fp in root.glob("**/fixed_encoder_vae/vae_model.pt"):
            candidates.append(model_fp.parent)
    if not candidates:
        raise FileNotFoundError(
            "No fixed VAE Encoder was found. Run run_fixed_vae_encoder_pretrain.py first, "
            "or set FIXED_PARAMETER_TRIAL['fixed_encoder_dir']."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _trial_experiment_payload(name: str, values: dict) -> dict:
    safe_name = safe_stem(name)
    return {
        "name": "fixed_vae_parameter_trial",
        "id": safe_stem(f"fixed_vae_parameter_trial__{safe_name}"),
        "trial_id": safe_name,
        "trial_index": 0,
        "target": "liquid",
        "memo": "manual fixed parameter trial for fixed VAE Encoder",
        "overrides": values,
    }


def _save_material_distance_outputs(details: dict, out_dir: Path) -> dict:
    vae_details = details.get("vae", {}) if isinstance(details.get("vae", {}), dict) else {}
    latent_file = (
        details.get("latent_npz_file")
        or details.get("latent_file")
        or vae_details.get("latent_npz_file")
        or vae_details.get("latent_file")
    )
    if not latent_file:
        return {"material_distance_message": "latent npz file was not found in metric details"}
    latent_fp = Path(latent_file)
    if not latent_fp.exists():
        return {"material_distance_message": f"latent npz file does not exist: {latent_fp}"}

    with np.load(latent_fp, allow_pickle=True) as data:
        z_mu = np.asarray(data["z_mu"], dtype=np.float64)
        labels = np.asarray(data["labels"], dtype=int)
        material_names = [str(item) for item in np.asarray(data["material_names"]).tolist()]

    rows = []
    centroids = []
    used_names = []
    class_points = []
    for label_index, material in enumerate(material_names):
        points = z_mu[labels == label_index]
        if points.size == 0:
            continue
        centroid = points.mean(axis=0)
        spread = float(np.mean(np.linalg.norm(points - centroid, axis=1)))
        rows.append({"material": material, "n_samples": int(points.shape[0]), "within_spread": spread})
        centroids.append(centroid)
        used_names.append(material)
        class_points.append(points)

    if len(centroids) < 2:
        return {}

    pairwise = pairwise_separation_matrix(class_points)
    sppw_matrix = np.asarray(pairwise["pairwise_matrix"], dtype=np.float64)

    centroid_arr = np.vstack(centroids)
    diff = centroid_arr[:, None, :] - centroid_arr[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))

    out_dir.mkdir(parents=True, exist_ok=True)
    distance_csv = out_dir / "material_centroid_distances.csv"
    spread_csv = out_dir / "material_within_spread.csv"
    sppw_csv = out_dir / "material_SPpw_pairwise.csv"
    sppw_summary_csv = out_dir / "material_SPpw_summary.csv"
    pd.DataFrame(dist, index=used_names, columns=used_names).to_csv(distance_csv, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(spread_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(sppw_matrix, index=used_names, columns=used_names).to_csv(sppw_csv, encoding="utf-8-sig")
    pd.DataFrame([
        {
            "SPpw_between_mean": float(pairwise["SPpw_between_mean"]),
            "SPpw_within_mean": float(pairwise["SPpw_within_mean"]),
            "SPpw_gap": float(pairwise["SPpw_between_mean"] - pairwise["SPpw_within_mean"]),
        }
    ]).to_csv(sppw_summary_csv, index=False, encoding="utf-8-sig")

    plot_fp = ""
    sppw_plot_fp = ""
    plt = try_import_pyplot()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(8.5, 7.0))
        image = ax.imshow(dist, cmap="viridis")
        ax.set_xticks(np.arange(len(used_names)))
        ax.set_yticks(np.arange(len(used_names)))
        ax.set_xticklabels(used_names, rotation=45, ha="right")
        ax.set_yticklabels(used_names)
        ax.set_title("VAE latent centroid distance by material")
        fig.colorbar(image, ax=ax, label="Euclidean distance")
        fig.tight_layout()
        plot_path = out_dir / "material_centroid_distances.png"
        fig.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        plot_fp = str(plot_path)

        fig, ax = plt.subplots(figsize=(8.5, 7.0))
        image = ax.imshow(sppw_matrix, cmap="magma")
        ax.set_xticks(np.arange(len(used_names)))
        ax.set_yticks(np.arange(len(used_names)))
        ax.set_xticklabels(used_names, rotation=45, ha="right")
        ax.set_yticklabels(used_names)
        ax.set_title("SPpw pairwise separation by material")
        fig.colorbar(image, ax=ax, label="SPpw")
        fig.tight_layout()
        sppw_plot_path = out_dir / "material_SPpw_pairwise.png"
        fig.savefig(sppw_plot_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        sppw_plot_fp = str(sppw_plot_path)

    return {
        "material_distance_csv": str(distance_csv),
        "material_spread_csv": str(spread_csv),
        "material_distance_plot": plot_fp,
        "material_sppw_csv": str(sppw_csv),
        "material_sppw_summary_csv": str(sppw_summary_csv),
        "material_sppw_plot": sppw_plot_fp,
        "SPpw_between_mean": float(pairwise["SPpw_between_mean"]),
        "SPpw_within_mean": float(pairwise["SPpw_within_mean"]),
        "SPpw_gap": float(pairwise["SPpw_between_mean"] - pairwise["SPpw_within_mean"]),
    }


def run_trial(*, search_config: str, trial_config: str) -> dict:
    search_cfg = deepcopy(load_search_config(search_config))
    trial_module = _load_trial_module(trial_config)
    trial_cfg = dict(getattr(trial_module, "FIXED_PARAMETER_TRIAL"))
    trial_name = safe_stem(str(trial_cfg.get("name", "manual_trial")))

    fixed_encoder = str(trial_cfg.get("fixed_encoder_dir", "")).strip()
    fixed_encoder_dir = Path(fixed_encoder) if fixed_encoder else _latest_fixed_encoder_dir()
    if not fixed_encoder_dir.exists():
        raise FileNotFoundError(f"fixed_encoder_dir not found: {fixed_encoder_dir}")

    base_cfg = build_cfg()
    base_cfg = apply_overrides(base_cfg, search_cfg.get("base_overrides", {}))
    params = normalize_parameter_specs(search_cfg["parameters"], base_cfg)
    default_values = unit_vector_to_values(params, initial_unit_vector(params))
    values = dict(default_values)
    values.update(dict(trial_cfg.get("parameter_values", {})))

    encoder_input_samples = trial_cfg.get("encoder_input_samples_per_material")
    if encoder_input_samples is None:
        encoder_input_samples = trial_cfg.get("samples_per_material", 10)

    cfg = build_cfg()
    cfg = apply_overrides(cfg, search_cfg.get("base_overrides", {}))
    cfg = apply_parameter_values(cfg, params, values)
    cfg = apply_overrides(
        cfg,
        {
            "run.LIVE_PLOT_ENABLE": False,
            "run.LIVE_RASTER_ENABLE": False,
            "run.INTERNAL_STATE_ENABLE": True,
            "run.INTERNAL_STATE_PCA_ENABLE": False,
            "run.BRIAN_CODEGEN_TARGET": str(trial_cfg.get("brian_codegen_target", "auto")),
            "liquid.NUM_LIQUID_SAMPLE": int(encoder_input_samples),
        },
    )
    materials = trial_cfg.get("materials")
    if materials:
        cfg = apply_overrides(cfg, {"liquid.LIQUID_MAT": list(materials)})
    cfg["experiment"] = _trial_experiment_payload(trial_name, values)

    metric_cfg = deepcopy(search_cfg["metric"])
    metric_cfg.setdefault("vae", {})["fixed_encoder_dir"] = str(fixed_encoder_dir)
    metric_cfg["common_latent_space"] = False

    out_dir = OUT_ROOT / f"{trial_name}__{now_text().replace(':', '-').replace(' ', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "trial_values.json", values)
    _write_json(out_dir / "trial_config.json", trial_cfg)
    _write_json(out_dir / "metric_config.json", metric_cfg)

    net_cfg = build_network_cfg(cfg)
    run_out_dir = make_run_output_dir(LIQUID_RESULT_DIR, cfg, net_cfg, include_output=False)
    print(f"[fixed-vae-trial] fixed_encoder_dir={fixed_encoder_dir}")
    print(f"[fixed-vae-trial] encoder_input_samples_per_material={int(encoder_input_samples)}")
    print(f"[fixed-vae-trial] run_out_dir={run_out_dir}")
    message = run_liquid(cfg)
    print(f"[fixed-vae-trial] {message}")

    internal_state_dir = run_out_dir / internal_state_config(cfg["run"])["dir_name"]
    metric_result = compute_internal_state_metric(
        internal_state_dir,
        metric_cfg,
        metric_out_dir=out_dir / "metric",
    )
    score = float(metric_result["score"])
    details = dict(metric_result.get("details", {}))
    distance_outputs = _save_material_distance_outputs(details, out_dir / "material_distance")

    summary = {
        "trial_name": trial_name,
        "fixed_encoder_dir": str(fixed_encoder_dir),
        "run_out_dir": str(run_out_dir),
        "internal_state_dir": str(internal_state_dir),
        "metric": metric_result.get("metric", ""),
        "direction": metric_result.get("direction", ""),
        "score": score,
        "encoder_input_samples_per_material": int(encoder_input_samples),
        "J": details.get("J"),
        "silhouette": details.get("silhouette"),
        "DR": details.get("DR"),
        "firing_rate_penalty": details.get("firing_rate_penalty"),
        "synchrony_penalty": details.get("synchrony_penalty"),
        "message": message,
        "out_dir": str(out_dir),
        **distance_outputs,
    }
    _write_json(out_dir / "trial_summary.json", summary)
    pd.DataFrame([summary]).to_csv(out_dir / "trial_summary.csv", index=False, encoding="utf-8-sig")
    print(
        f"[fixed-vae-trial] score={score:.6g} J={summary['J']} "
        f"silhouette={summary['silhouette']} DR={summary['DR']}"
    )
    if distance_outputs.get("material_distance_csv"):
        print(f"[fixed-vae-trial] material_distance_csv={distance_outputs['material_distance_csv']}")
        print(f"[fixed-vae-trial] material_distance_plot={distance_outputs.get('material_distance_plot', '')}")
        print(f"[fixed-vae-trial] material_sppw_csv={distance_outputs.get('material_sppw_csv', '')}")
        print(f"[fixed-vae-trial] material_sppw_plot={distance_outputs.get('material_sppw_plot', '')}")
    print(f"[fixed-vae-trial] out_dir={out_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-config", default="cma_es_internal_state")
    parser.add_argument("--trial-config", default="fixed_parameter_trial_config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_trial(search_config=args.search_config, trial_config=args.trial_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
