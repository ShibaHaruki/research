import numpy as np
import pandas as pd
from brian2 import *
import glob
import os
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
exp_file = SCRIPT_PATH.parent / "01_tactile" / "experiment_sets.py"  # パスは実際の配置に合わせて調整

spec = importlib.util.spec_from_file_location("experiment_sets", exp_file)
experiment_sets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_sets)

COMMON_SETS   = experiment_sets.COMMON_SETS
TRAINING_SETS = experiment_sets.TRAINING_SETS
TEST_SETS     = experiment_sets.TEST_SETS

SCRIPT_PATH = Path(__file__).resolve()
SAMPLE_SEQ_DIR = SCRIPT_PATH.parent / "sample_seq"
TACTILE_DATA_PATH = SCRIPT_PATH.parents[2]

def load_tactile_data( mat: str, sid: int, tactile_data_path = TACTILE_DATA_PATH):
    pattern = str(tactile_data_path / "tactile_data" / mat / f"data_{sid}_*.csv")
    fp = glob.glob(pattern)
    if not fp:
        print(f"[warn] no file matched: {pattern}")
        return None
    
    return fp[0]

def load_or_make_sample_seq_rep(rep: int, out_dir: Path, base_seed: int, n_samples: int) -> np.ndarray:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"sample_seq_rep{rep}.npy"

    if fp.exists():
        seq = np.load(fp).astype(np.int32).reshape(-1)
        if len(seq) >= n_samples:
            return seq
        print(f"[warn] {fp.name} too short ({len(seq)} < {n_samples}). Regenerating...")

    rng = np.random.default_rng(base_seed + rep)
    seq = np.arange(1, n_samples + 1, dtype=np.int32)
    rng.shuffle(seq)
    np.save(fp, seq)
    print(f"[info] created {fp.name} (len={len(seq)})")
    return seq


def run_training(rep: int, cfg: dict):
    start_scope()

    defaultclock.dt = cfg["dt_ms"] * ms

    np.random.seed(cfg["BASE_SEED"] + rep)
    seed(cfg["BASE_SEED"] + rep)
    rng = np.random.default_rng(cfg["BASE_SEED"] + rep)

    sample_seq = load_or_make_sample_seq_rep(rep=rep, out_dir=SAMPLE_SEQ_DIR, base_seed=cfg["BASE_SEED"], n_samples=cfg["NUM_SAMPLE"])
