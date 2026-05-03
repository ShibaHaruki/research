"""学習済み重みを旧コード互換の dense 行列形式で保存・読み込みする処理。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


def _group_size(group) -> int:
    return int(len(group))


def _synapse_name(synapses) -> str:
    return str(getattr(synapses, "name", ""))


def _dense_w_matrix(synapses, n_pre: int, n_post: int) -> np.ndarray:
    matrix = np.zeros((int(n_pre), int(n_post)), dtype=np.float64)
    if "w" not in synapses.variables or len(synapses) == 0:
        return matrix

    pre_idx = np.asarray(synapses.i[:], dtype=np.int64)
    post_idx = np.asarray(synapses.j[:], dtype=np.int64)
    weights = np.asarray(synapses.w[:], dtype=np.float64)
    matrix[pre_idx, post_idx] = weights
    return matrix


def _match_int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def _liq_layer_from_name(name: str) -> int | None:
    return _match_int(r"liq(\d+)", name) or _match_int(r"_L(\d+)$", name)


def _liq_out_from_name(name: str) -> tuple[int, int] | None:
    match = re.search(r"liq(\d+)_to_out(\d+)", name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _assign_dense_weights(synapses, dense_w: np.ndarray) -> None:
    if "w" not in synapses.variables or len(synapses) == 0:
        return

    pre_idx = np.asarray(synapses.i[:], dtype=np.int64)
    post_idx = np.asarray(synapses.j[:], dtype=np.int64)
    if dense_w.ndim != 2:
        raise ValueError(f"dense_w must be 2D, got shape={dense_w.shape}")
    synapses.w = np.asarray(dense_w[pre_idx, post_idx], dtype=np.float64)


def _unlink_if_possible(path: Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        print(f"[warn] cleanup failed for {Path(path)}: {exc}")


def _find_legacy_single_file(weights_dir: Path, pattern: str) -> Path | None:
    matches = sorted(weights_dir.glob(pattern))
    return matches[0] if matches else None


def load_weight_matrices_like_old_code(
    weights_dir: Path,
    objects: dict[str, Any],
) -> None:
    # training で保存した dense 重み行列を読み、現在の Brian2 Synapses の w に戻す。
    weights_dir = Path(weights_dir)

    n_in = _group_size(objects["G_in"])
    n_liq_by_layer = [_group_size(group) for group in objects["G_liq"]]
    n_out_by_layer = [_group_size(group) for group in objects["G_out"]]

    input_dense = {}
    for layer_index in range(1, len(n_liq_by_layer) + 1):
        fp = weights_dir / f"w_in1_liq{layer_index}.npy"
        if fp.exists():
            input_dense[layer_index] = np.load(fp)
            continue
        if len(n_liq_by_layer) == 1 and layer_index == 1:
            legacy_fp = _find_legacy_single_file(weights_dir, "*w_in_rep*.npy")
            if legacy_fp is not None:
                input_dense[layer_index] = np.load(legacy_fp)
                continue
        raise FileNotFoundError(fp)
    for layer_index, dense_w in input_dense.items():
        expected = (n_in, n_liq_by_layer[layer_index - 1])
        if tuple(dense_w.shape) != expected:
            raise ValueError(
                f"Input weight shape mismatch for layer {layer_index}: "
                f"expected {expected}, got {tuple(dense_w.shape)}"
            )

    intra_dense = {}
    for layer_index in range(1, len(n_liq_by_layer) + 1):
        fp = weights_dir / f"w_liq{layer_index}_intra.npy"
        if fp.exists():
            intra_dense[layer_index] = np.load(fp)
            continue
        if len(n_liq_by_layer) == 1 and layer_index == 1:
            legacy_fp = _find_legacy_single_file(weights_dir, "*w_res_rep*.npy")
            if legacy_fp is not None:
                intra_dense[layer_index] = np.load(legacy_fp)
                continue
        raise FileNotFoundError(fp)
    for layer_index, dense_w in intra_dense.items():
        expected = (n_liq_by_layer[layer_index - 1], n_liq_by_layer[layer_index - 1])
        if tuple(dense_w.shape) != expected:
            raise ValueError(
                f"Liquid recurrent weight shape mismatch for layer {layer_index}: "
                f"expected {expected}, got {tuple(dense_w.shape)}"
            )

    output_dense = {
        (liq_index, out_index): np.load(weights_dir / f"w_liq{liq_index}_out{out_index}.npy")
        for liq_index in range(1, len(n_liq_by_layer) + 1)
        for out_index in range(1, len(n_out_by_layer) + 1)
        if (weights_dir / f"w_liq{liq_index}_out{out_index}.npy").exists()
    }
    if not output_dense and len(n_liq_by_layer) == 1 and len(n_out_by_layer) == 1:
        legacy_fp = _find_legacy_single_file(weights_dir, "*w_out_rep*.npy")
        if legacy_fp is not None:
            output_dense[(1, 1)] = np.load(legacy_fp)
    for (liq_index, out_index), dense_w in output_dense.items():
        expected = (n_liq_by_layer[liq_index - 1], n_out_by_layer[out_index - 1])
        if tuple(dense_w.shape) != expected:
            raise ValueError(
                f"Output weight shape mismatch for L{liq_index}->O{out_index}: "
                f"expected {expected}, got {tuple(dense_w.shape)}"
            )

    for synapses in objects.get("S_in", []):
        layer_index = _liq_layer_from_name(_synapse_name(synapses))
        if layer_index is None:
            continue
        _assign_dense_weights(synapses, input_dense[layer_index])

    for synapses in objects.get("S_intra", []):
        layer_index = _liq_layer_from_name(_synapse_name(synapses))
        if layer_index is None:
            continue
        _assign_dense_weights(synapses, intra_dense[layer_index])

    for synapses in objects.get("S_lo", []):
        key = _liq_out_from_name(_synapse_name(synapses))
        if key is None:
            continue
        if key not in output_dense:
            raise FileNotFoundError(weights_dir / f"w_liq{key[0]}_out{key[1]}.npy")
        _assign_dense_weights(synapses, output_dense[key])


def dense_output_weight_matrix(objects: dict[str, Any]) -> np.ndarray:
    n_liq_by_layer = [_group_size(group) for group in objects.get("G_liq", [])]
    n_out_by_layer = [_group_size(group) for group in objects.get("G_out", [])]
    total_liq = int(sum(n_liq_by_layer))
    total_out = int(sum(n_out_by_layer))
    dense = np.zeros((total_liq, total_out), dtype=np.float64)

    liq_offsets = np.cumsum([0] + n_liq_by_layer[:-1]).astype(int)
    out_offsets = np.cumsum([0] + n_out_by_layer[:-1]).astype(int)

    for synapses in objects.get("S_lo", []):
        key = _liq_out_from_name(_synapse_name(synapses))
        if key is None or "w" not in synapses.variables or len(synapses) == 0:
            continue

        liq_index, out_index = key
        liq_offset = int(liq_offsets[liq_index - 1])
        out_offset = int(out_offsets[out_index - 1])
        pre_idx = np.asarray(synapses.i[:], dtype=np.int64) + liq_offset
        post_idx = np.asarray(synapses.j[:], dtype=np.int64) + out_offset
        dense[pre_idx, post_idx] = np.asarray(synapses.w[:], dtype=np.float64)

    return dense


def save_output_weight_matrix(
    out_dir: Path,
    objects: dict[str, Any],
    rep: int,
    *,
    prefix: str | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_w_out_rep{rep}.npy" if prefix else f"w_out_rep{rep}.npy"
    out_fp = out_dir / stem
    np.save(out_fp, dense_output_weight_matrix(objects))
    return out_fp


def save_weight_matrices_like_old_code(
    out_dir: Path,
    objects: dict[str, Any],
    net_cfg: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
    sample_seq: np.ndarray | None = None,
    rep: int | None = None,
) -> list[Path]:
    """Save Brian2 weight matrices using old monolithic script style file names.

    Examples:
    - w_in1_liq1.npy: input layer 1 -> liquid layer 1
    - w_liq1_intra.npy: liquid layer 1 recurrent weights
    - w_liq1_out1.npy: liquid layer 1 -> output layer 1
    """

    # 旧コードの run_test.py でも読めるよう、結合行列を dense な .npy として保存する。
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for legacy_fp in out_dir.glob("w_intra*.npy"):
        _unlink_if_possible(legacy_fp)
    for stale_fp in (
        out_dir / "params1.txt",
        out_dir / "weight_files_manifest.txt",
    ):
        if stale_fp.exists():
            _unlink_if_possible(stale_fp)
    for stale_sample_seq in out_dir.glob("sample_seq*.npy"):
        _unlink_if_possible(stale_sample_seq)

    paths: list[Path] = []

    def save_array(name: str, array: np.ndarray) -> None:
        path = out_dir / name
        np.save(path, array)
        paths.append(path)

    n_in = _group_size(objects["G_in"])
    n_liq_by_layer = [_group_size(group) for group in objects["G_liq"]]
    n_out_by_layer = [_group_size(group) for group in objects["G_out"]]

    input_by_liq = {
        layer_index: np.zeros((n_in, n_liq), dtype=np.float64)
        for layer_index, n_liq in enumerate(n_liq_by_layer, start=1)
    }
    for synapses in objects.get("S_in", []):
        layer_index = _liq_layer_from_name(_synapse_name(synapses))
        if layer_index in input_by_liq:
            input_by_liq[layer_index] += _dense_w_matrix(
                synapses,
                n_in,
                n_liq_by_layer[layer_index - 1],
            )

    for layer_index, matrix in sorted(input_by_liq.items()):
        save_array(f"w_in1_liq{layer_index}.npy", matrix)

    intra_by_liq = {
        layer_index: np.zeros((n_liq, n_liq), dtype=np.float64)
        for layer_index, n_liq in enumerate(n_liq_by_layer, start=1)
    }
    for synapses in objects.get("S_intra", []):
        layer_index = _liq_layer_from_name(_synapse_name(synapses))
        if layer_index in intra_by_liq:
            intra_by_liq[layer_index] += _dense_w_matrix(
                synapses,
                n_liq_by_layer[layer_index - 1],
                n_liq_by_layer[layer_index - 1],
            )

    for layer_index, matrix in sorted(intra_by_liq.items()):
        save_array(f"w_liq{layer_index}_intra.npy", matrix)

    liq_to_out = {
        (liq_index, out_index): np.zeros(
            (n_liq_by_layer[liq_index - 1], n_out_by_layer[out_index - 1]),
            dtype=np.float64,
        )
        for liq_index in range(1, len(n_liq_by_layer) + 1)
        for out_index in range(1, len(n_out_by_layer) + 1)
    }
    for synapses in objects.get("S_lo", []):
        key = _liq_out_from_name(_synapse_name(synapses))
        if key in liq_to_out:
            liq_index, out_index = key
            liq_to_out[key] += _dense_w_matrix(
                synapses,
                n_liq_by_layer[liq_index - 1],
                n_out_by_layer[out_index - 1],
            )

    for (liq_index, out_index), matrix in sorted(liq_to_out.items()):
        save_array(f"w_liq{liq_index}_out{out_index}.npy", matrix)

    return paths


def save_liquid_weight_matrices(
    out_dir: Path,
    objects: dict[str, Any],
) -> list[Path]:
    """Save input-to-liquid and liquid recurrent matrices for liquid-only runs."""

    # run_liquid は出力層を持たないので、入力→リキッドとリキッド再帰だけ保存する。
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_fp in out_dir.glob("*.npy"):
        _unlink_if_possible(stale_fp)

    paths: list[Path] = []

    def save_array(name: str, array: np.ndarray) -> None:
        path = out_dir / name
        np.save(path, array)
        paths.append(path)

    n_in = _group_size(objects["G_in"])
    n_liq_by_layer = [_group_size(group) for group in objects["G_liq"]]

    input_by_liq = {
        layer_index: np.zeros((n_in, n_liq), dtype=np.float64)
        for layer_index, n_liq in enumerate(n_liq_by_layer, start=1)
    }
    for synapses in objects.get("S_in", []):
        layer_index = _liq_layer_from_name(_synapse_name(synapses))
        if layer_index in input_by_liq:
            input_by_liq[layer_index] += _dense_w_matrix(
                synapses,
                n_in,
                n_liq_by_layer[layer_index - 1],
            )

    for layer_index, matrix in sorted(input_by_liq.items()):
        save_array(f"w_in1_liq{layer_index}.npy", matrix)

    intra_by_liq = {
        layer_index: np.zeros((n_liq, n_liq), dtype=np.float64)
        for layer_index, n_liq in enumerate(n_liq_by_layer, start=1)
    }
    for synapses in objects.get("S_intra", []):
        layer_index = _liq_layer_from_name(_synapse_name(synapses))
        if layer_index in intra_by_liq:
            intra_by_liq[layer_index] += _dense_w_matrix(
                synapses,
                n_liq_by_layer[layer_index - 1],
                n_liq_by_layer[layer_index - 1],
            )

    for layer_index, matrix in sorted(intra_by_liq.items()):
        save_array(f"w_liq{layer_index}_intra.npy", matrix)

    return paths
