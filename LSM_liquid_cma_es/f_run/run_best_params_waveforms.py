"""Run the liquid with CMA-ES best parameters and save debug waveforms.

This script does not train anything. It only applies the liquid hyperparameters
from best_params.json, runs the liquid, and lets run_liquid save the usual
input waveforms, liquid raster plots, membrane voltages, and heatmaps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f_run.run_cma_es_search import apply_liquid_params
from f_run.run_common import build_cfg
from f_run.run_liquid import run_liquid


DEFAULT_SAMPLES_PER_CLASS = 100
DEFAULT_PCA_MAX_SAMPLES_PER_CLASS = 100
DEFAULT_BEST_PARAMS = (
    PROJECT_ROOT
    / "g_tactile_results"
    / "cma_es_search"
    / "liquid_accuracy_spikes_fisher"
    / "best_params.json"
)


def load_best_params(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"{path} does not contain a 'params' object")
    return params


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run liquid-only waveform/debug plots with CMA-ES best params."
    )
    parser.add_argument(
        "--best-params",
        type=Path,
        default=DEFAULT_BEST_PARAMS,
        help="Path to best_params.json.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help=(
            "Number of samples per material. Default is 100. "
            "Debug plots are saved for the first sample."
        ),
    )
    parser.add_argument(
        "--internal-state-bin-ms",
        type=float,
        default=1.0,
        help="Internal-state bin width in ms.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=3,
        help="Number of PCA components to save.",
    )
    parser.add_argument(
        "--pca-max-samples-per-class",
        type=int,
        default=DEFAULT_PCA_MAX_SAMPLES_PER_CLASS,
        help="Maximum samples per material used for PCA. Default is 100.",
    )
    parser.add_argument(
        "--disable-pca",
        action="store_true",
        help="Disable internal-state PCA output.",
    )
    args = parser.parse_args()

    params = load_best_params(args.best_params)
    cfg = apply_liquid_params(build_cfg(), params)
    cfg["liquid"]["NUM_LIQUID_SAMPLE"] = [int(args.samples_per_class)]
    cfg["run"]["INTERNAL_STATE_BIN_MS"] = float(args.internal_state_bin_ms)
    cfg["run"]["INTERNAL_STATE_PCA_ENABLE"] = not bool(args.disable_pca)
    cfg["run"]["INTERNAL_STATE_PCA_COMPONENTS"] = int(args.pca_components)
    cfg["run"]["INTERNAL_STATE_PCA_MAX_SAMPLES_PER_CLASS"] = args.pca_max_samples_per_class
    cfg["experiment"] = {
        "id": "best_params_waveforms",
        "name": "best_params_waveforms",
        "trial_id": args.best_params.parent.name,
    }

    print(f"[best-params] loaded: {args.best_params}")
    n_liq = params.get("n_liq", cfg["network"]["N_liq"][0])
    print(f"[best-params] n_liq={int(n_liq)}")
    print(f"[best-params] samples_per_class={int(args.samples_per_class)}")
    print(f"[best-params] pca_max_samples_per_class={args.pca_max_samples_per_class}")
    print(run_liquid(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
