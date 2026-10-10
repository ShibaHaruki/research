"""Freeze the best liquid and test on unused trials from the same recording.

Training uses only the selected candidate's saved states. Test IDs exclude
every candidate's simulated trials, including trials used for validation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from d_tools.run_paths import jsonable
from d_tools.heldout_plots import save_heldout_plots
from f_run.run_cma_es_search import apply_liquid_params
from f_run.run_common import build_cfg, build_network_cfg, load_tactile_data
from f_run.run_liquid import run_liquid
from f_run.run_random_neuron_accuracy import (
    DIR_NAME, RIDGE_LAMBDA, extract_eval_features,
    fit_ridge_mahalanobis_model, fold_8_to_3,
    load_liquid_states_by_material, mahalanobis_sq_woodbury,
)


def candidate_rows(search_dir: Path) -> list[tuple[float, dict, Path]]:
    csvs = sorted(search_dir.glob("start*/cma_es_results.csv"))
    if not csvs:
        csvs = [search_dir / "cma_es_results.csv"]
    rows = []
    for csv_path in csvs:
        frame = pd.read_csv(csv_path)
        for row in frame.to_dict(orient="records"):
            score = float(row["objective"])
            if not np.isfinite(score) or score >= 1e300:
                continue
            candidate = csv_path.parent / (
                f"gen{int(row['generation']):03d}_cand{int(row['candidate']):03d}"
            )
            rows.append((score, row, candidate))
    if not rows:
        raise ValueError("No finite evaluated candidates were found.")
    return rows


def state_dir(candidate: Path) -> Path:
    paths = list(candidate.glob("*/neuron_*/internal_states"))
    if len(paths) != 1:
        raise ValueError(f"Expected one internal-state directory under {candidate}")
    return paths[0]


def trial_ids(path: Path) -> dict[str, set[int]]:
    result = {material: set() for material in DIR_NAME}
    for material in DIR_NAME:
        for fp in (path / material).glob("*_internal_state_manifest.json"):
            match = re.search(r"_sid(\d+)_", fp.name)
            if not match:
                raise ValueError(f"Cannot identify trial ID: {fp}")
            result[material].add(int(match.group(1)))
    return result


def evaluate_fixed_readout(train_dir: Path, test_dir: Path, out_dir: Path, t_n_ms: float) -> dict:
    train, train_materials, train_files, train_bin = load_liquid_states_by_material(train_dir)
    test, test_materials, test_files, test_bin = load_liquid_states_by_material(test_dir)
    if train_materials != DIR_NAME or test_materials != DIR_NAME:
        raise ValueError("All eight materials must be present in both datasets.")
    if train.shape[2:] != test.shape[2:] or not np.isclose(train_bin, test_bin):
        raise ValueError("Neuron counts, trial durations and bin widths must match.")
    # The legacy loader aligns to the smallest class; refuse silent truncation.
    for directory, files in ((train_dir, train_files), (test_dir, test_files)):
        for material, selected in zip(DIR_NAME, files):
            available = list((directory / material).glob("*_liquid_internal_state_all.npz"))
            if len(selected) != len(available):
                raise ValueError(f"Unequal class counts would discard trials in {directory}")
            for fp in selected:
                with np.load(fp) as data:
                    if data['x_state'].shape != train.shape[2:]:
                        raise ValueError(f"Inconsistent state dimensions: {fp}")
    steps = int(round(t_n_ms / train_bin))
    if steps < 1 or not np.isclose(steps * train_bin, t_n_ms):
        raise ValueError("Feature window must be a positive multiple of the bin width.")
    neurons = np.arange(train.shape[2])
    train_features = extract_eval_features(train, neurons, t_n=steps)
    test_features = extract_eval_features(test, neurons, t_n=steps)
    models = [fit_ridge_mahalanobis_model(x, RIDGE_LAMBDA) for x in train_features]
    confusion = np.zeros((8, 8), dtype=int)
    predictions = []
    for label, features in enumerate(test_features):
        for index, x in enumerate(features):
            distances = [mahalanobis_sq_woodbury(x, model) for model in models]
            pred = int(np.argmin(distances))
            confusion[label, pred] += 1
            predictions.append({"file": test_files[label][index], "true": DIR_NAME[label],
                                "predicted": DIR_NAME[pred], "correct": pred == label})
    confusion3 = fold_8_to_3(confusion)
    metrics = {
        "evaluation": "unused trials from the same recording; readout fitted on search data only",
        "accuracy8": float(np.trace(confusion) / confusion.sum()),
        "accuracy3": float(np.trace(confusion3) / confusion3.sum()),
        "per_class_accuracy": dict(zip(DIR_NAME, (np.diag(confusion) / confusion.sum(axis=1)).tolist())),
        "train_samples_per_class": int(train.shape[1]),
        "test_samples_per_class": int(test.shape[1]),
        "t_n_ms": t_n_ms, "bin_ms": train_bin,
        "train_files": train_files, "test_files": test_files,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(confusion, index=DIR_NAME, columns=DIR_NAME).to_csv(out_dir / "confusion8.csv")
    pd.DataFrame(confusion3).to_csv(out_dir / "confusion3.csv", index=False)
    pd.DataFrame(predictions).to_csv(out_dir / "predictions.csv", index=False)
    np.savez_compressed(out_dir / "readout_models.npz", **{
        f"class{i}_{key}": value for i, model in enumerate(models) for key, value in model.items()
    })
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_heldout_plots(train_features, test_features, DIR_NAME, train_files,
                       test_files, confusion, out_dir.parent / "plots", t_n_ms)
    return metrics


def verify_weights(train_dir: Path, test_dir: Path) -> None:
    original = train_dir.parent / "weights" / "init"
    tested = test_dir.parent / "weights" / "init"
    names = sorted(p.name for p in original.glob("*.npy"))
    if not names or names != sorted(p.name for p in tested.glob("*.npy")):
        raise ValueError("Matching saved network weights are required.")
    for name in names:
        if not np.array_equal(np.load(original / name), np.load(tested / name)):
            raise ValueError(f"Network reconstruction differs from the best candidate: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--plots-only", action="store_true",
                        help="Reuse an existing evaluation and regenerate metrics/PCA without simulation.")
    args = parser.parse_args()
    if args.samples_per_class < 1:
        raise ValueError("samples-per-class must be positive")
    search_dir = args.search_dir.resolve()
    if args.plan_only and args.plots_only:
        raise ValueError("Choose either plan-only or plots-only.")
    if args.plots_only:
        out_dir = args.out_dir.resolve() if args.out_dir else search_dir / "heldout_same_recording"
        plan_path = out_dir / "evaluation_plan.json"
        if not plan_path.is_file():
            parser.error(
                f"No completed unused-trial evaluation was found at {out_dir}. "
                "First run this command without --plots-only to simulate and evaluate unused trials. "
                "If the evaluation is saved elsewhere, specify its directory with --out-dir."
            )
        if not (out_dir / "accuracy" / "metrics.json").is_file():
            parser.error(
                f"The evaluation at {out_dir} has not completed (accuracy/metrics.json is missing). "
                "Wait for the original evaluation to finish before using --plots-only."
            )
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        train_dir = state_dir(Path(plan["candidate"]))
        test_dir = state_dir(out_dir / "liquid")
        if trial_ids(test_dir) != {m: set(ids) for m, ids in plan["test_ids"].items()}:
            raise ValueError("Saved test states do not match the evaluation plan.")
        verify_weights(train_dir, test_dir)
        settings = json.loads((Path(plan["candidate"]).parent / "search_settings.json").read_text(encoding="utf-8"))
        evaluate_fixed_readout(train_dir, test_dir, out_dir / "accuracy", float(settings["T_n_ms"]))
        print(f"[heldout] plots saved to {out_dir / 'plots'}")
        return 0
    if not (search_dir / "best_params.json").exists():
        raise ValueError("Finish the search before evaluating its global best candidate.")
    evaluated = candidate_rows(search_dir)
    for _, _, evaluated_candidate in evaluated:
        if not all(trial_ids(state_dir(evaluated_candidate)).values()):
            raise ValueError(f"Missing trial provenance for evaluated candidate: {evaluated_candidate}")
    score, row, candidate = min(evaluated, key=lambda item: item[0])
    train_dir = state_dir(candidate)
    snapshot = json.loads((train_dir.parent / "config_snapshot.json").read_text(encoding="utf-8"))
    params = json.loads((candidate / "candidate_params.json").read_text(encoding="utf-8"))
    # Use the winning candidate's recorded seed and effective neuron parameters.
    cfg = apply_liquid_params(deepcopy(build_cfg()), params)
    cfg["common"] = snapshot["common"]
    model = snapshot["model_params"]
    cfg["models"].update(model["models"])
    cfg["neuron_models"][cfg["models"]["NEURON_MODEL"]] = model["neuron_model_params"]
    cfg["synapse_models"][cfg["models"]["SYNAPSE_MODEL"]] = model["synapse_model_params"]
    if jsonable(build_network_cfg(cfg)) != snapshot["net_cfg"]:
        raise ValueError("Current network/filter configuration differs from the search snapshot.")
    # Include all simulated candidates, even failed candidates, to prevent reuse.
    roots = sorted(search_dir.glob("start*/gen*_cand*")) + sorted(search_dir.glob("gen*_cand*"))
    used = {m: set() for m in DIR_NAME}
    for root in roots:
        for path in root.glob("*/neuron_*/internal_states"):
            for material, ids in trial_ids(path).items():
                used[material].update(ids)
    selected = {}
    selection_rng = np.random.default_rng(int(cfg["common"]["BASE_SEED"]) + 271828)
    data_root = (PROJECT_ROOT / cfg["run"]["TACTILE_DATA_ROOT"] /
                 cfg["run"]["TACTILE_DATA_DIR_NAME"]).resolve()
    for material in DIR_NAME:
        available = sorted({int(re.match(r"data_(\d+)_", p.name).group(1))
                            for p in (data_root / material).glob("data_*_*.csv")
                            if re.match(r"data_(\d+)_", p.name)})
        unused = [sid for sid in available if sid not in used[material]]
        if len(unused) < args.samples_per_class:
            raise ValueError(f"Not enough unused trials for {material}")
        selected[material] = selection_rng.choice(
            unused, size=args.samples_per_class, replace=False
        ).astype(int).tolist()
        for sid in selected[material]:
            matches = list((data_root / material).glob(f"data_{sid}_*.csv"))
            if len(matches) != 1 or load_tactile_data(material, sid) is None:
                raise ValueError(f"Expected exactly one raw file for {material}, sid={sid}")
    plan = {"candidate": str(candidate), "objective": score,
            "generation": int(row["generation"]), "candidate_index": int(row["candidate"]),
            "params": params, "used_ids": {m: sorted(ids) for m, ids in used.items()},
            "test_ids": selected, "data_root": str(data_root)}
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return 0
    out_dir = args.out_dir.resolve() if args.out_dir else search_dir / "heldout_same_recording"
    if out_dir.exists():
        raise FileExistsError(f"Use a new output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    (out_dir / "evaluation_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    cfg["run"].update({"LIQUID_RESULT_ROOT": str(out_dir / "liquid"),
                        "INCLUDE_EXPERIMENT_DIR": False, "INTERNAL_STATE_PCA_ENABLE": False,
                        "INTERNAL_STATE_BIN_MS": snapshot["run"]["INTERNAL_STATE_BIN_MS"]})
    cfg["liquid"].update({"LIQUID_MAT": DIR_NAME,
                           "NUM_LIQUID_SAMPLE": [args.samples_per_class],
                           "SAMPLE_IDS_BY_MATERIAL": selected})
    run_liquid(cfg)
    test_dir = state_dir(out_dir / "liquid")
    if trial_ids(test_dir) != {m: set(ids) for m, ids in selected.items()}:
        raise ValueError("Test simulation did not produce every planned unused trial.")
    verify_weights(train_dir, test_dir)
    settings_path = candidate.parent / "search_settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    metrics = evaluate_fixed_readout(train_dir, test_dir, out_dir / "accuracy",
                                     float(settings["T_n_ms"]))
    print(f"[heldout] accuracy8={metrics['accuracy8']:.4f} accuracy3={metrics['accuracy3']:.4f}")
    print(f"[heldout] saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
