"""Run one-epoch STDP training for a fixed list of equal plus/minus time constants.

From the LSM project root:
    python f_run/run_stdp_tau_sweep.py
    # --out-dir を指定しない場合、結合確率・gain・学習則の階層へ保存する。
"""
from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from f_run.run_common import build_cfg, encode, snapshot
from a_tactile.paths import readout_rule_directory, readout_configuration_directory
from f_run.run_training import run_training


TAU_CANDIDATES_MS = [2, 5, 8, 11, 15, 20, 25, 30, 40, 50,
                     60, 70, 80, 100, 120, 140, 170, 200, 250]
OUTPUT_NEURONS = 80
OUTPUT_CONNECTION_PROBABILITY = 0.11
OUTPUT_GAIN = 0.2963
TRAIN_TRIALS_PER_MATERIAL = 100
TEST_TRIALS_PER_MATERIAL = 100
EPOCHS = 1
STDP_A_PLUS = 1.00e-7
STDP_A_MINUS = -1.05e-7
WEIGHT_MIN = 0.0
WEIGHT_MAX = 10.0
MATERIALS = [
    "aluminum_board", "outer_pigskin", "back_pigskin", "cork",
    "denim", "rubber_board", "Japanese_paper", "wood_board",
]


def make_candidate_cfg(tau_ms: float, out_dir: Path | None = None) -> dict:
    """Create a matched training config; only the common STDP tau changes."""
    cfg = deepcopy(build_cfg())
    cfg["models"]["LEARNING_RULE_MODEL"] = "STDP"
    cfg["learning_rule_models"]["STDP"]["tau_plus"] = float(tau_ms)
    cfg["learning_rule_models"]["STDP"]["tau_minus"] = float(tau_ms)
    cfg["learning_rule_models"]["STDP"]["A_plus"] = STDP_A_PLUS
    cfg["learning_rule_models"]["STDP"]["A_minus"] = STDP_A_MINUS
    cfg["learning_rule_models"]["STDP"]["wmin"] = WEIGHT_MIN
    cfg["learning_rule_models"]["STDP"]["wmax"] = WEIGHT_MAX
    cfg["network"]["N_out"] = [OUTPUT_NEURONS]
    cfg["network"]["p_liq_to_out_pairs"] = [{
        "EE": OUTPUT_CONNECTION_PROBABILITY, "EI": 0.0, "IE": 0.0, "II": 0.0,
    }]
    cfg["network"]["gain_liq_to_out_pairs"] = [{
        "EE": OUTPUT_GAIN, "EI": 0.0, "IE": 0.0, "II": 0.0,
    }]

    # Use the same liquid, output topology, gain, sample split, and presentation order
    # for every candidate.  Only tau_plus=tau_minus changes between runs.
    cfg["training"]["TRAINING_MAT"] = list(MATERIALS)
    cfg["training"]["NUM_TRAINING_SAMPLE"] = [TRAIN_TRIALS_PER_MATERIAL]
    cfg["training"]["NUM_TRAINING_EPOCH"] = [EPOCHS]
    cfg["test"]["TEST_MAT"] = list(MATERIALS)
    cfg["test"]["NUM_TEST_SAMPLE"] = [TEST_TRIALS_PER_MATERIAL]
    cfg["common"]["NUM_REPEAT"] = 1
    cfg["run"]["READOUT_TRAIN_DIR"] = str(
        readout_configuration_directory(cfg, rule_dir=out_dir).resolve()
    )

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path,
        default=None,
        help="Override STDP rule directory; default includes output probability and gain",
    )
    parser.add_argument(
        "--rep", type=int, default=1,
        help="Training repetition; the same value is used for every tau candidate",
    )
    args = parser.parse_args()
    out_dir = (args.out_dir.resolve() if args.out_dir is not None else
               readout_rule_directory(make_candidate_cfg(TAU_CANDIDATES_MS[0])))
    out_dir.mkdir(parents=True, exist_ok=True)

    overview = {
        "learning_rule": "STDP",
        "tau_plus_equals_tau_minus_ms": TAU_CANDIDATES_MS,
        "output_neurons": OUTPUT_NEURONS,
        "materials": MATERIALS,
        "training_trials_per_material": TRAIN_TRIALS_PER_MATERIAL,
        "test_trials_per_material": TEST_TRIALS_PER_MATERIAL,
        "epochs": EPOCHS,
        "repetition": args.rep,
        "output_connection_probability": OUTPUT_CONNECTION_PROBABILITY,
        "output_gain": OUTPUT_GAIN,
        "A_plus": STDP_A_PLUS,
        "A_minus": STDP_A_MINUS,
        "weight_clip": [WEIGHT_MIN, WEIGHT_MAX],
        "shared_liquid_and_topology": True,
        "only_parameter_changed_between_candidates": "tau_plus=tau_minus",
    }
    (out_dir / "sweep_config.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary_path = out_dir / "sweep_summary.csv"
    columns = ["tau_ms", "status", "training_dir", "weight_change_norm"]
    rows: list[dict] = []
    for tau_ms in TAU_CANDIDATES_MS:
        cfg = make_candidate_cfg(tau_ms, out_dir)
        cfg_file = Path(cfg["run"]["READOUT_TRAIN_DIR"]) / "config_snapshot.json"
        # Save/verify before training, so every candidate's matched conditions are reviewable.
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(
            json.dumps(encode(snapshot(cfg)), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            run_training(args.rep, cfg)
            result_dir = Path(cfg["run"]["READOUT_TRAIN_DIR"])
            summary_file = result_dir / f"training_summary_rep{args.rep}.json"
            result = json.loads(summary_file.read_text(encoding="utf-8"))
            row = {
                "tau_ms": tau_ms,
                "status": "complete",
                "training_dir": str(result_dir),
                "weight_change_norm": result["output_weight_change_norm"],
            }
        except Exception as exc:
            row = {
                "tau_ms": tau_ms,
                "status": f"failed: {type(exc).__name__}: {exc}",
                "training_dir": str(cfg["run"]["READOUT_TRAIN_DIR"]),
                "weight_change_norm": "",
            }
            rows.append(row)
            with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader(); writer.writerows(rows)
            raise
        rows.append(row)
        with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader(); writer.writerows(rows)
        print(f"[tau sweep] tau={tau_ms} ms complete", flush=True)


if __name__ == "__main__":
    main()
