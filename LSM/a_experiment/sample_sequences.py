"""試行番号をファイルで共有し、学習・テストの分割を保存して読み込む。"""

from pathlib import Path
import os
import tempfile

import numpy as np

from a_experiment.paths import SAMPLE_SEQUENCE_ROOT


def _save_once(path, writer):
    """完成したファイルだけを公開し、別repが先に保存したファイルは上書きしない。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer(stream)
        try:
            if os.name == "nt":
                os.rename(temporary, path)
            else:
                os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_sample_order(seed, n_samples=324, *, directory=None):
    """保存済みの全試行順序を読む。ファイルがない初回だけseedから作成する。"""
    folder = SAMPLE_SEQUENCE_ROOT if directory is None else Path(directory)
    path = folder / "sample_seq_liquid.npy"
    if not path.exists():
        order = np.random.default_rng(seed).permutation(np.arange(1, n_samples + 1))
        _save_once(path, lambda stream: np.save(stream, order, allow_pickle=False))
    order = np.load(path, allow_pickle=False)
    if (
        order.ndim != 1
        or not np.issubdtype(order.dtype, np.integer)
        or len(order) != n_samples
        or not np.array_equal(np.sort(order), np.arange(1, n_samples + 1))
    ):
        raise ValueError(f"Expected a permutation of 1..{n_samples}: {path}")
    return order.astype(np.int64, copy=True)


def load_sample_split(seed, n_train, n_test, *, rep=1, n_samples=324, directory=None):
    """共通順序からrep番号付きの分割ファイルを保存し、学習・テスト番号を読む。"""
    if isinstance(rep, bool) or int(rep) != rep or rep < 1:
        raise ValueError("rep must be a positive integer")
    if min(n_train, n_test) < 1 or n_train + n_test > n_samples:
        raise ValueError("Require positive train/test sizes with train + test <= total samples")
    folder = SAMPLE_SEQUENCE_ROOT if directory is None else Path(directory)
    order = load_sample_order(seed, n_samples, directory=folder)
    path = folder / f"sample_seq_rep{int(rep)}.npz"
    expected = {
        "sequence": order,
        "train_ids": order[:n_train],
        "test_ids": order[n_train : n_train + n_test],
        "unused_ids": order[n_train + n_test :],
    }
    if not path.exists():
        _save_once(path, lambda stream: np.savez_compressed(stream, **expected))
    with np.load(path, allow_pickle=False) as saved:
        if any(
            name not in saved or not np.array_equal(saved[name], values)
            for name, values in expected.items()
        ):
            raise ValueError(
                f"Saved split differs from the order or requested train/test sizes: {path}"
            )
        return saved["train_ids"].copy(), saved["test_ids"].copy()
