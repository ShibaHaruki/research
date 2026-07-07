"""rep ごとのサンプル順序を保存・再利用して再現性を保つ処理。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_or_make_sample_seq(
    *,
    name: str,
    out_dir: Path,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{name}.npy"

    if fp.exists():
        seq = np.load(fp).astype(np.int32).reshape(-1)
        if len(seq) >= n_samples:
            return seq
        print(f"[warn] {fp.name} too short ({len(seq)} < {n_samples}). Regenerating...")

    seq = np.arange(1, n_samples + 1, dtype=np.int32)
    rng.shuffle(seq)
    np.save(fp, seq)
    print(f"[info] created {fp.name} (len={len(seq)})")
    return seq


def load_or_make_sample_seq_rep(
    rep: int,
    out_dir: Path,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    return load_or_make_sample_seq(
        name=f"sample_seq_rep{rep}",
        out_dir=out_dir,
        rng=rng,
        n_samples=n_samples,
    )
