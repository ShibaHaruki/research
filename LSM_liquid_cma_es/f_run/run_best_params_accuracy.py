"""Evaluate accuracy for the best-parameter liquid run.

This script does not train anything. It uses saved internal states created by
run_best_params_waveforms.py and evaluates eval.py-style Mahalanobis accuracy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f_run.run_random_neuron_accuracy import evaluate_random_neuron_accuracy


LIQUID_RUN_DIR = PROJECT_ROOT / "g_tactile_results" / "liquid_run"
DEFAULT_SELECTED_NEURONS = 1000
DEFAULT_SAMPLES_PER_CLASS = 100
DEFAULT_REPEATS = 20
DEFAULT_FOLDS = 10
DEFAULT_T_N_MS = 25.0
REQUIRED_MATERIALS = [
    "Al_board",
    "buta_omote",
    "buta_ura",
    "cork",
    "denim",
    "rubber_board",
    "washi",
    "wood_board",
]


def internal_state_class_counts(path: Path) -> dict[str, int]:
    counts = {}
    for material_dir in Path(path).iterdir() if Path(path).exists() else []:
        if material_dir.is_dir():
            counts[material_dir.name] = len(list(material_dir.glob("*.npz")))
    return counts


def has_required_internal_states(path: Path, min_samples_per_class: int = 1) -> bool:
    counts = internal_state_class_counts(path)
    return all(counts.get(material, 0) >= min_samples_per_class for material in REQUIRED_MATERIALS)


def find_default_internal_state_dir() -> Path:
    candidates = [
        path
        for path in LIQUID_RUN_DIR.rglob("best_params_waveforms/internal_states")
        if path.is_dir() and has_required_internal_states(path)
    ]
    if not candidates:
        raise FileNotFoundError(
            "No complete best_params_waveforms/internal_states directory was found. "
            "Run f_run/run_best_params_waveforms.py first and let it finish all 8 materials."
        )
    nliq1000 = [
        path
        for path in candidates
        if any(part.startswith("Nliq_1000__") for part in path.parts)
    ]
    selected = nliq1000 if nliq1000 else candidates
    return max(selected, key=lambda path: path.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate best-parameter liquid accuracy from saved internal states."
    )
    parser.add_argument(
        "--internal-state-dir",
        type=Path,
        default=None,
        help="Internal states from run_best_params_waveforms.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default is random_neuron_accuracy_best100 beside internal_states.",
    )
    parser.add_argument("--neurons", type=int, default=DEFAULT_SELECTED_NEURONS)
    parser.add_argument("--samples-per-class", type=int, default=DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-n-ms", type=float, default=DEFAULT_T_N_MS)
    args = parser.parse_args()

    internal_state_dir = (
        Path(args.internal_state_dir)
        if args.internal_state_dir is not None
        else find_default_internal_state_dir()
    )
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else internal_state_dir.parent / "random_neuron_accuracy_best100"
    )

    metrics = evaluate_random_neuron_accuracy(
        internal_state_dir,
        n_neurons=int(args.neurons),
        n_repeats=int(args.repeats),
        n_folds=int(args.folds),
        seed_value=int(args.seed),
        t_n_ms=float(args.t_n_ms),
        max_samples_per_class=int(args.samples_per_class),
        out_dir=out_dir,
    )
    print(
        "[accuracy-best] "
        f"acc8_overall_mean={metrics['accuracy8_overall_mean']:.4f} "
        f"acc8_overall_std={metrics['accuracy8_overall_std']:.4f} "
        f"acc3_overall_mean={metrics['accuracy3_overall_mean']:.4f} "
        f"acc3_overall_std={metrics['accuracy3_overall_std']:.4f}"
    )
    print(f"[accuracy-best] saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
