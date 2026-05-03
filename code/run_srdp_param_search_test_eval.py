# -*- coding: utf-8 -*-
"""Run test simulation and classification eval for SRDP parameter searches.

This script is intentionally self-contained because the existing test/eval
scripts assume that weights live directly under ``code/``.  Parameter-search
outputs live one directory deeper, e.g.
``code/SRDP_1_param_search_parallel/search_001``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Keep parallel runs from multiplying BLAS threads inside every worker.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from tqdm import tqdm


DIR_NAME = [
    "Al_board",
    "buta_omote",
    "buta_ura",
    "cork",
    "denim",
    "rubber_board",
    "washi",
    "wood_board",
]

WEIGHT_RE = re.compile(r"^(?P<prefix>.+)_w_in_rep(?P<rep>\d+)\.npy$")
SOUT_RE = re.compile(r"^(?P<dataset>.+)_sout_rec_rep(?P<rep>\d+)\.npy$")

DEFAULT_N_TRAIN = 100
DEFAULT_N_SAMPLE = 100
DEFAULT_N_BINS = 500
DEFAULT_TN = [25]
DEFAULT_N_FOLDS = 10
DEFAULT_BASE_SEED = 1
DEFAULT_RIDGE = 1e-6
NAN_FILL_VALUE = 0.0


@dataclass(frozen=True)
class TestTask:
    root: str
    search_id: str
    search_dir: str
    data_root: str
    prefix: str
    rep: int
    w_in: str
    w_res: str
    w_out: str
    sample_seq: str
    out_sout: str
    n_train: int
    n_sample: int
    n_bins: int
    sout_dtype: str
    codegen_target: str
    cython_cache_dir: str
    cpp_compiler: str


@dataclass(frozen=True)
class EvalTask:
    root: str
    search_id: str
    search_dir: str
    sout_path: str
    dataset: str
    rep: int
    t_n: int
    n_folds: int
    base_seed: int
    ridge: float
    save_matrix_xlsx: bool


def calc_meissner(data: np.ndarray, t: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros((4, len(t)))
    for i in range(len(t)):
        if i != 0:
            d_f_dt = np.abs(data[i] - data[i - 1]) / (t[i] - t[i - 1])
            out[0, i] = out[0, i - 1] + 1.0 * d_f_dt + (-out[0, i - 1] * dt / (8 * 1e-3))
            out[1, i] = out[1, i - 1] + 0.24 * d_f_dt + (-(out[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1e-3))
            out[2, i] = out[2, i - 1] + 0.07 * d_f_dt + (-out[2, i - 1] * dt / (1744.6 * 1e-3))
            out[3, i] = out[0, i]
    return out[3, :]


def calc_merkel(data: np.ndarray, t: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros((4, len(t)))
    for i in range(len(t)):
        if i != 0:
            d_f_dt = np.abs(data[i] - data[i - 1]) / (t[i] - t[i - 1])
            if d_f_dt < 0:
                d_f_dt = 0
            out[0, i] = out[0, i - 1] + 0.74 * d_f_dt + (-out[0, i - 1] * dt / (8 * 1e-3))
            out[1, i] = out[1, i - 1] + 0.24 * d_f_dt + (-(out[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1e-3))
            out[2, i] = out[2, i - 1] + 0.07 * d_f_dt + (-out[2, i - 1] * dt / (1744.6 * 1e-3))
            out[3, i] = out[0, i] + out[1, i] + out[2, i]
    return out[3, :]


def find_code_dir() -> Path:
    return Path(__file__).resolve().parent


def find_data_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "tactile_data").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find tactile_data above {start}")


def natural_search_key(path: Path) -> tuple[str, int]:
    match = re.search(r"(\d+)$", path.name)
    return (path.name[: match.start()] if match else path.name, int(match.group(1)) if match else -1)


def default_search_roots(code_dir: Path) -> list[Path]:
    roots = [p for p in code_dir.glob("SRDP*_param_search*") if p.is_dir()]
    return sorted(roots, key=lambda p: p.name)


def read_params(search_dir: Path) -> dict[str, Any]:
    params_path = search_dir / "srdp_params.json"
    if not params_path.exists():
        return {}
    with open(params_path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_search_dirs(
    roots: list[Path],
    search_ids: set[str] | None,
    max_searches: int | None,
) -> list[tuple[Path, Path]]:
    found: list[tuple[Path, Path]] = []
    for root in roots:
        search_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("search_")]
        for search_dir in sorted(search_dirs, key=natural_search_key):
            if search_ids and search_dir.name not in search_ids:
                continue
            found.append((root, search_dir))
            if max_searches is not None and len(found) >= max_searches:
                return found
    return found


def make_sout_path(search_dir: Path, prefix: str, rep: int, output_tag: str) -> Path:
    tag = f"_{output_tag}" if output_tag else ""
    return search_dir / f"{prefix}{tag}_sout_rec_rep{rep}.npy"


def discover_test_tasks(
    roots: list[Path],
    *,
    search_ids: set[str] | None,
    max_searches: int | None,
    max_tasks: int | None,
    output_tag: str,
    force: bool,
    n_train: int,
    n_sample: int,
    n_bins: int,
    sout_dtype: str,
    codegen_target: str,
    cython_cache_dir: str,
    cpp_compiler: str,
) -> tuple[list[TestTask], list[dict[str, Any]]]:
    tasks: list[TestTask] = []
    skipped: list[dict[str, Any]] = []

    for root, search_dir in discover_search_dirs(roots, search_ids, max_searches):
        data_root = find_data_root(search_dir)
        for w_in_path in sorted(search_dir.glob("*_w_in_rep*.npy"), key=lambda p: p.name):
            match = WEIGHT_RE.match(w_in_path.name)
            if not match:
                continue

            prefix = match.group("prefix")
            rep = int(match.group("rep"))
            w_res_path = search_dir / f"{prefix}_w_res_rep{rep}.npy"
            w_out_path = search_dir / f"{prefix}_w_out_rep{rep}.npy"
            sample_seq_path = search_dir / f"sample_seq_rep{rep}.npy"
            missing = [p.name for p in [w_res_path, w_out_path, sample_seq_path] if not p.exists()]
            if missing:
                skipped.append(
                    {
                        "root": str(root),
                        "search_id": search_dir.name,
                        "prefix": prefix,
                        "rep": rep,
                        "status": "missing_input",
                        "missing": ";".join(missing),
                    }
                )
                continue

            out_sout = make_sout_path(search_dir, prefix, rep, output_tag)
            if out_sout.exists() and not force:
                skipped.append(
                    {
                        "root": str(root),
                        "search_id": search_dir.name,
                        "prefix": prefix,
                        "rep": rep,
                        "status": "sout_exists",
                        "out_sout": str(out_sout),
                    }
                )
                continue

            tasks.append(
                TestTask(
                    root=str(root),
                    search_id=search_dir.name,
                    search_dir=str(search_dir),
                    data_root=str(data_root),
                    prefix=prefix,
                    rep=rep,
                    w_in=str(w_in_path),
                    w_res=str(w_res_path),
                    w_out=str(w_out_path),
                    sample_seq=str(sample_seq_path),
                    out_sout=str(out_sout),
                    n_train=n_train,
                    n_sample=n_sample,
                    n_bins=n_bins,
                    sout_dtype=sout_dtype,
                    codegen_target=codegen_target,
                    cython_cache_dir=cython_cache_dir,
                    cpp_compiler=cpp_compiler,
                )
            )
            if max_tasks is not None and len(tasks) >= max_tasks:
                return tasks, skipped

    return tasks, skipped


def run_test_worker(task_dict: dict[str, Any]) -> dict[str, Any]:
    task = TestTask(**task_dict)
    started = time.time()
    try:
        from brian2 import (
            Hz,
            Network,
            NeuronGroup,
            SpikeMonitor,
            Synapses,
            TimedArray,
            defaultclock,
            float64,
            ms,
            prefs,
            seed,
            start_scope,
        )

        prefs.core.default_float_dtype = float64
        prefs.codegen.target = task.codegen_target
        if task.codegen_target == "cython":
            prefs.codegen.runtime.cython.cache_dir = task.cython_cache_dir
            prefs.codegen.runtime.cython.multiprocess_safe = True
            if task.cpp_compiler:
                prefs.codegen.cpp.compiler = task.cpp_compiler

        w_in = np.load(task.w_in)
        w_res_init = np.load(task.w_res)
        w_out_init = np.load(task.w_out)
        sample_seq = np.load(task.sample_seq).astype(int)
        test_seq = sample_seq[task.n_train :]
        if len(test_seq) < task.n_sample:
            raise ValueError(
                f"test_seq too short: need {task.n_sample}, got {len(test_seq)} "
                f"for {task.search_id} rep{task.rep}"
            )
        test_seq = test_seq[: task.n_sample]

        if w_res_init.ndim != 2 or w_res_init.shape[0] != w_res_init.shape[1]:
            raise ValueError(f"w_res shape invalid: {w_res_init.shape}")
        n_res = w_res_init.shape[0]
        if w_out_init.ndim != 2 or w_out_init.shape[0] != n_res:
            raise ValueError(f"w_out shape invalid: {w_out_init.shape}")
        n_out = w_out_init.shape[1]
        if w_in.ndim != 2 or w_in.shape[1] != n_res:
            raise ValueError(f"w_in must be (N_in,N_res), got {w_in.shape}")
        n_in = w_in.shape[0]

        start_scope()
        np.random.seed(2 + (task.rep - 1))
        seed(2 + (task.rep - 1))

        v_reset = -65
        v_thr = -40
        tau_r = 2 * ms
        tau_d = 20 * ms
        bias = -65
        gain = 0.25

        dt_ms = 0.1
        dt_s = dt_ms * 1e-3
        defaultclock.dt = dt_ms * ms

        tau_m_res = np.ones(n_res) * 10
        t_ref_res = np.ones(n_res) * 2
        tau_m_out = np.ones(n_out) * 10
        t_ref_out = np.ones(n_out) * 2

        lif = """
        dv/dt = (-v + BIAS + I_exc - I_inh + I_syn) / tau_m : 1 (unless refractory)
        I_exc : 1
        I_inh : 1
        tau_m : second
        t_ref : second
        """

        double_exp_res = """
        dR/dt = -R / tau_d + H : 1
        dH/dt = -H / tau_r : Hz
        I_syn = G * R : 1
        """
        on_pre_res = """
        H_post += (w_res / (tau_r * tau_d)) / Hz
        """

        double_exp_out = """
        dR/dt = -R / tau_d + H : 1
        dH/dt = -H / tau_r : Hz
        I_syn = R : 1
        """
        on_pre_out = """
        H_post += (w_out / (tau_r * tau_d)) / Hz
        """

        g_res = NeuronGroup(
            n_res,
            double_exp_res + lif,
            threshold="v >= v_thr",
            reset="v = v_reset",
            refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
            method="exact",
        )
        g_res.tau_m = tau_m_res * ms
        g_res.t_ref = t_ref_res * ms
        g_res.I_inh = 0

        g_out = NeuronGroup(
            n_out,
            double_exp_out + lif,
            threshold="v >= v_thr",
            reset="v = v_reset",
            refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
            method="exact",
        )
        g_out.tau_m = tau_m_out * ms
        g_out.t_ref = t_ref_out * ms
        g_out.I_inh = 0

        s_res = Synapses(g_res, g_res, model="w_res : 1", on_pre=on_pre_res, method="euler")
        s_res.connect(condition="i != j")
        s_res.w_res = w_res_init[s_res.i, s_res.j]
        s_res.delay = 0 * ms

        pre_idx, post_idx = np.where(w_out_init != 0)
        s_out = Synapses(g_res, g_out, model="w_out : 1", on_pre=on_pre_out, method="euler")
        s_out.connect(i=pre_idx, j=post_idx)
        s_out.w_out = w_out_init[s_out.i, s_out.j]
        s_out.delay = 0 * ms

        input_ta = TimedArray(np.zeros((1, n_in)), dt=dt_ms * ms)
        g_in = NeuronGroup(
            n_in,
            """
            t_start : second (shared)
            I = input_ta(t - t_start, i) : 1
            """,
            method="euler",
        )
        g_in.t_start = 0 * ms

        s_in = Synapses(
            g_in,
            g_res,
            model="""
            w : 1
            I_exc_post = w * I_pre : 1 (summed)
            """,
            method="euler",
        )
        pre_in, post_in = np.nonzero(w_in)
        s_in.connect(i=pre_in, j=post_in)
        s_in.w = w_in[pre_in, post_in]

        mr_out = SpikeMonitor(g_out)
        net = Network(g_in, g_res, g_out, s_in, s_res, s_out, mr_out)

        sout_rec = np.zeros(
            (len(DIR_NAME), task.n_sample, n_out, task.n_bins),
            dtype=np.dtype(task.sout_dtype),
        )

        data_root = Path(task.data_root)
        t0 = 0 * ms
        for i_mat, mat in enumerate(DIR_NAME):
            mat_dir = data_root / "tactile_data" / mat
            for j in range(task.n_sample):
                sid = int(test_seq[j])
                files = sorted(mat_dir.glob(f"data_{sid}_*"))
                if not files:
                    raise FileNotFoundError(f"data not found: {mat} sid={sid}")

                df = pd.read_table(files[0], header=None)
                df_np = df.to_numpy().T
                in_data_0 = df_np[:3, 3000:8000]
                nt = in_data_0.shape[1]
                t_array_s = np.arange(nt) * dt_s

                input_current = np.zeros((n_in, nt), dtype=float)
                in_data = in_data_0[0, :]
                i_merkel = calc_merkel(in_data, t_array_s, dt_s)
                i_meissner = calc_meissner(in_data, t_array_s, dt_s)
                input_current[0, :] = 0.4 * i_merkel * 0.02
                input_current[1, :] = 0.6 * 7.3 * i_meissner * 0.02

                vals = input_current.T
                vals = np.vstack([vals, vals[-1]])
                input_ta = TimedArray(vals, dt=dt_ms * ms)
                g_in.t_start = t0

                g_res.v = v_reset + (v_thr - v_reset) * np.random.rand(n_res)
                g_out.v = v_reset + (v_thr - v_reset) * np.random.rand(n_out)
                g_res.R = 0
                g_res.H = 0
                g_out.R = 0
                g_out.H = 0

                start_t = t0
                start_idx = len(mr_out.t)
                namespace = {
                    "input_ta": input_ta,
                    "tau_r": tau_r,
                    "tau_d": tau_d,
                    "G": gain,
                    "BIAS": bias,
                    "v_reset": v_reset,
                    "v_thr": v_thr,
                }

                duration = (nt * dt_ms) * ms
                net.run(duration, namespace=namespace)

                end_idx = len(mr_out.t)
                t0 += duration

                if end_idx > start_idx:
                    t_sp = mr_out.t[start_idx:end_idx]
                    i_sp = mr_out.i[start_idx:end_idx]
                    mask = (t_sp > start_t) & (t_sp <= t0)

                    if np.any(mask):
                        rel_times_ms = (t_sp[mask] - start_t) / ms
                        ids = np.asarray(i_sp[mask], dtype=int)
                        bin_edges = np.linspace(0, nt * dt_ms, task.n_bins + 1)
                        for n in range(n_out):
                            counts, _ = np.histogram(rel_times_ms[ids == n], bins=bin_edges)
                            sout_rec[i_mat, j, n, :] = counts.astype(sout_rec.dtype, copy=False)

        out_path = Path(task.out_sout)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, sout_rec)
        return {
            "stage": "test",
            "status": "success",
            "elapsed_sec": time.time() - started,
            **asdict(task),
        }

    except Exception as exc:
        search_dir = Path(task.search_dir)
        log_path = search_dir / f"test_error_rep{task.rep}.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        return {
            "stage": "test",
            "status": "error",
            "elapsed_sec": time.time() - started,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error_log": str(log_path),
            **asdict(task),
        }


def extract_features(sout_rec: np.ndarray, t_n: int) -> np.ndarray:
    n_sozai, n_sample, n_out, t = sout_rec.shape
    if t % t_n != 0:
        raise ValueError(f"T={t} is not divisible by T_n={t_n}")

    n_interval = t // t_n
    reshaped = sout_rec.reshape(n_sozai, n_sample, n_out, n_interval, t_n)
    spike_sum = reshaped.sum(axis=-1)
    rate = spike_sum / (t_n / 1000.0)
    features = rate.reshape(n_sozai, n_sample, n_out * n_interval).astype(np.float64, copy=False)

    if not np.isfinite(features).all():
        features = np.nan_to_num(
            features,
            nan=NAN_FILL_VALUE,
            posinf=NAN_FILL_VALUE,
            neginf=NAN_FILL_VALUE,
        )
    return features


def fit_ridge_mahalanobis_model(train_data: np.ndarray, ridge: float) -> dict[str, np.ndarray]:
    train_data = np.asarray(train_data, dtype=np.float64)
    if not np.isfinite(train_data).all():
        train_data = np.nan_to_num(train_data, nan=0.0, posinf=0.0, neginf=0.0)

    mean = np.mean(train_data, axis=0)
    centered = train_data - mean
    denom = max(train_data.shape[0] - 1, 1)

    if centered.size == 0:
        return {
            "mean": mean,
            "components": np.zeros((0, train_data.shape[1]), dtype=np.float64),
            "coeff": np.zeros(0, dtype=np.float64),
        }

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    eigvals = (singular_values * singular_values) / denom
    keep = eigvals > 0
    components = vt[keep, :]
    eigvals = eigvals[keep]
    coeff = eigvals / (ridge * (eigvals + ridge))
    return {"mean": mean, "components": components, "coeff": coeff}


def mahalanobis_sq_batch(x: np.ndarray, model: dict[str, np.ndarray], ridge: float) -> np.ndarray:
    diff = np.asarray(x, dtype=np.float64) - model["mean"]
    base = np.einsum("ij,ij->i", diff, diff) / ridge
    components = model["components"]
    if components.size:
        proj = diff @ components.T
        base -= (proj * proj) @ model["coeff"]
    return np.maximum(base, 0.0)


def fold_8_to_3(conf_8_fold: np.ndarray) -> np.ndarray:
    mtrx1 = np.zeros((8, 3))
    mtrx1[:, 0] = conf_8_fold[:, 0] + conf_8_fold[:, 5] + conf_8_fold[:, 7]
    mtrx1[:, 1] = conf_8_fold[:, 3] + conf_8_fold[:, 4] + conf_8_fold[:, 6]
    mtrx1[:, 2] = conf_8_fold[:, 1] + conf_8_fold[:, 2]

    mtrx2 = np.zeros((3, 3))
    mtrx2[0, :] = mtrx1[0, :] + mtrx1[5, :] + mtrx1[7, :]
    mtrx2[1, :] = mtrx1[3, :] + mtrx1[4, :] + mtrx1[6, :]
    mtrx2[2, :] = mtrx1[1, :] + mtrx1[2, :]
    return mtrx2


def eval_10fold(features: np.ndarray, rng: np.random.Generator, n_folds: int, ridge: float) -> tuple[Any, ...]:
    n_sozai, n_sample, _ = features.shape
    if n_sample < n_folds:
        raise ValueError(f"Need at least {n_folds} samples per class, but n_sample={n_sample}")

    all_indices = np.arange(n_sample)
    rng.shuffle(all_indices)
    fold_indices = np.array_split(all_indices, n_folds)

    conf_8_total = np.zeros((n_sozai, n_sozai))
    conf_3_total = np.zeros((3, 3))
    acc_list8: list[float] = []
    acc_list3: list[float] = []

    for fold in range(n_folds):
        test_idx = np.array(fold_indices[fold])
        train_idx = np.setdiff1d(all_indices, test_idx)
        models = [fit_ridge_mahalanobis_model(features[c, train_idx, :], ridge) for c in range(n_sozai)]

        conf_8_fold = np.zeros((n_sozai, n_sozai))
        for true_c in range(n_sozai):
            x = features[true_c, test_idx, :]
            if not np.isfinite(x).all():
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            distances = np.column_stack([mahalanobis_sq_batch(x, model, ridge) for model in models])
            pred = np.argmin(distances, axis=1)
            for pred_c in pred:
                conf_8_fold[true_c, int(pred_c)] += 1

        total_samples_fold = np.sum(conf_8_fold)
        acc_list8.append(float(np.trace(conf_8_fold) / total_samples_fold))

        conf_3_fold = fold_8_to_3(conf_8_fold)
        correct_fold_3 = conf_3_fold[0, 0] + conf_3_fold[1, 1] + conf_3_fold[2, 2]
        acc_list3.append(float(correct_fold_3 / total_samples_fold))

        conf_8_total += conf_8_fold
        conf_3_total += conf_3_fold

    total_samples = np.sum(conf_8_total)
    correct_8 = np.trace(conf_8_total)
    correct_3 = conf_3_total[0, 0] + conf_3_total[1, 1] + conf_3_total[2, 2]
    return (
        conf_8_total,
        conf_3_total,
        float(correct_8 / total_samples),
        float(np.mean(acc_list8)),
        float(correct_3 / total_samples),
        float(np.mean(acc_list3)),
        int(total_samples),
    )


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def save_matrix_workbook(
    out_dir: Path,
    task: EvalTask,
    conf8: np.ndarray,
    conf3: np.ndarray,
    metrics: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{task.search_id}_rep{task.rep:02d}_Tn_{task.t_n}_"
        f"{short_hash(task.dataset)}_conf_matrices.xlsx"
    )
    out_path = out_dir / filename
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(conf8, index=DIR_NAME, columns=DIR_NAME).to_excel(writer, sheet_name="conf_8cls")
        pd.DataFrame(conf3).to_excel(writer, sheet_name="conf_3cls", index=False)
        pd.DataFrame([metrics]).to_excel(writer, sheet_name="accuracy", index=False)
    return out_path


def run_eval_worker(task_dict: dict[str, Any]) -> dict[str, Any]:
    task = EvalTask(**task_dict)
    started = time.time()
    try:
        sout_rec = np.load(task.sout_path)
        if sout_rec.ndim != 4:
            raise ValueError(f"invalid sout_rec shape: {sout_rec.shape}")

        features = extract_features(sout_rec, task.t_n)
        rng = np.random.default_rng(task.base_seed + task.rep)
        conf8, conf3, acc8_overall, acc8_mean, acc3_overall, acc3_mean, total_samples = eval_10fold(
            features,
            rng,
            task.n_folds,
            task.ridge,
        )

        n_sozai, n_sample, n_out, t = sout_rec.shape
        params = read_params(Path(task.search_dir))
        row: dict[str, Any] = {
            "stage": "eval",
            "status": "success",
            "elapsed_sec": time.time() - started,
            "root": task.root,
            "search_id": task.search_id,
            "search_dir": task.search_dir,
            "dataset": task.dataset,
            "rep": task.rep,
            "T_n": task.t_n,
            "n_folds": task.n_folds,
            "accuracy8_overall": acc8_overall,
            "accuracy8_mean": acc8_mean,
            "accuracy3_overall": acc3_overall,
            "accuracy3_mean": acc3_mean,
            "n_sozai": n_sozai,
            "n_sample_per_class": n_sample,
            "N_out": n_out,
            "T": t,
            "total_samples": total_samples,
            "sout_path": task.sout_path,
            **params,
        }

        if task.save_matrix_xlsx:
            out_dir = Path(task.root) / "eval_results_10fold" / task.search_id
            matrix_path = save_matrix_workbook(out_dir, task, conf8, conf3, row)
            row["matrix_xlsx"] = str(matrix_path)

        return row

    except Exception as exc:
        search_dir = Path(task.search_dir)
        log_path = search_dir / f"eval_error_rep{task.rep}_Tn_{task.t_n}.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        return {
            "stage": "eval",
            "status": "error",
            "elapsed_sec": time.time() - started,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error_log": str(log_path),
            **asdict(task),
        }


def discover_eval_tasks(
    roots: list[Path],
    *,
    search_ids: set[str] | None,
    max_searches: int | None,
    output_tag: str,
    tns: list[int],
    n_folds: int,
    base_seed: int,
    ridge: float,
    save_matrix_xlsx: bool,
) -> list[EvalTask]:
    tasks: list[EvalTask] = []
    for root, search_dir in discover_search_dirs(roots, search_ids, max_searches):
        weight_prefixes: set[str] = set()
        for w_in_path in search_dir.glob("*_w_in_rep*.npy"):
            weight_match = WEIGHT_RE.match(w_in_path.name)
            if weight_match:
                weight_prefixes.add(weight_match.group("prefix"))

        for sout_path in sorted(search_dir.glob("*_sout_rec_rep*.npy"), key=lambda p: p.name):
            match = SOUT_RE.match(sout_path.name)
            if not match:
                continue

            dataset = match.group("dataset")
            if output_tag:
                allowed = {f"{prefix}_{output_tag}" for prefix in weight_prefixes}
            else:
                allowed = weight_prefixes
            if allowed and dataset not in allowed:
                continue

            rep = int(match.group("rep"))
            for t_n in tns:
                tasks.append(
                    EvalTask(
                        root=str(root),
                        search_id=search_dir.name,
                        search_dir=str(search_dir),
                        sout_path=str(sout_path),
                        dataset=dataset,
                        rep=rep,
                        t_n=t_n,
                        n_folds=n_folds,
                        base_seed=base_seed,
                        ridge=ridge,
                        save_matrix_xlsx=save_matrix_xlsx,
                    )
                )
    return tasks


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if path.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="summary", index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def write_eval_summaries(roots: list[Path], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)
    for root in roots:
        root_rows = df[df["root"] == str(root)].copy()
        if root_rows.empty:
            continue

        out_dir = root / "eval_results_10fold"
        out_dir.mkdir(parents=True, exist_ok=True)
        sort_cols = [
            col
            for col in ["accuracy3_overall", "accuracy8_overall", "search_id", "rep", "T_n"]
            if col in root_rows.columns
        ]
        ascending = [False if col.startswith("accuracy") else True for col in sort_cols]
        if sort_cols:
            root_rows = root_rows.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

        csv_path = out_dir / "summary_all_searches.csv"
        xlsx_path = out_dir / "summary_all_searches.xlsx"
        root_rows.to_csv(csv_path, index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            root_rows.to_excel(writer, sheet_name="summary", index=False)
            metric_cols = [
                "search_id",
                "rep",
                "T_n",
                "accuracy8_overall",
                "accuracy8_mean",
                "accuracy3_overall",
                "accuracy3_mean",
                "dataset",
                "sout_path",
                "matrix_xlsx",
            ]
            metric_cols = [c for c in metric_cols if c in root_rows.columns]
            root_rows[metric_cols].head(50).to_excel(writer, sheet_name="top50", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test and evaluate all SRDP parameter-search results under code/.",
    )
    parser.add_argument("--search-root", action="append", type=Path, help="Search root. May be passed multiple times.")
    parser.add_argument("--search-id", action="append", help="Limit to a search id such as search_001.")
    parser.add_argument("--max-searches", type=int, help="Limit number of search_* directories.")
    parser.add_argument("--max-tasks", type=int, help="Limit test tasks, useful for smoke checks.")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)), help="Test workers.")
    parser.add_argument("--eval-workers", type=int, default=max(1, min(4, os.cpu_count() or 1)), help="Eval workers.")
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument("--n-sample", type=int, default=DEFAULT_N_SAMPLE)
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument("--tn", action="append", type=int, dest="tns", help="T_n value. May be passed multiple times.")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    parser.add_argument("--sout-dtype", default="uint16")
    parser.add_argument("--codegen-target", default="numpy")
    parser.add_argument(
        "--cython-cache-dir",
        type=Path,
        help="Brian2 Cython cache directory. Default: code/brian2_cython_cache.",
    )
    parser.add_argument(
        "--cpp-compiler",
        default="",
        help="Compiler for Brian2 C++/Cython codegen, e.g. msvc or unix. Empty lets Brian2 choose.",
    )
    parser.add_argument("--output-tag", default="", help="Append a tag before _sout_rec_repN.npy.")
    parser.add_argument("--force-test", action="store_true", help="Regenerate sout_rec even if it already exists.")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--no-matrix-xlsx", action="store_true", help="Only write summary CSV/XLSX.")
    parser.add_argument("--dry-run", action="store_true", help="Show discovered work without running it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code_dir = find_code_dir()
    roots = [p.resolve() for p in args.search_root] if args.search_root else default_search_roots(code_dir)
    roots = [p for p in roots if p.exists() and p.is_dir()]
    if not roots:
        raise FileNotFoundError("No SRDP parameter search roots were found.")

    output_tag = args.output_tag
    if not output_tag and (args.n_sample != DEFAULT_N_SAMPLE or args.n_bins != DEFAULT_N_BINS):
        output_tag = f"N{args.n_sample}_B{args.n_bins}"
        print(f"[INFO] output_tag was empty; using '{output_tag}' to avoid overwriting full-size outputs.")

    search_ids = set(args.search_id) if args.search_id else None
    tns = args.tns or DEFAULT_TN
    cython_cache_dir = (args.cython_cache_dir or (code_dir / "brian2_cython_cache")).resolve()
    if args.codegen_target == "cython":
        cython_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CYTHON] cache_dir={cython_cache_dir}")
        if args.cpp_compiler:
            print(f"[CYTHON] cpp_compiler={args.cpp_compiler}")

    print("[ROOTS]")
    for root in roots:
        print(f"  - {root}")

    skipped_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    if not args.skip_test:
        test_tasks, skipped_rows = discover_test_tasks(
            roots,
            search_ids=search_ids,
            max_searches=args.max_searches,
            max_tasks=args.max_tasks,
            output_tag=output_tag,
            force=args.force_test,
            n_train=args.n_train,
            n_sample=args.n_sample,
            n_bins=args.n_bins,
            sout_dtype=args.sout_dtype,
            codegen_target=args.codegen_target,
            cython_cache_dir=str(cython_cache_dir),
            cpp_compiler=args.cpp_compiler,
        )
        print(f"[TEST] pending={len(test_tasks)} skipped={len(skipped_rows)} workers={args.workers}")

        if args.dry_run:
            for task in test_tasks[:10]:
                print(f"  test: {task.search_id} rep{task.rep} -> {task.out_sout}")
        elif test_tasks:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_task = {executor.submit(run_test_worker, asdict(task)): task for task in test_tasks}
                for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="SRDP test"):
                    row = future.result()
                    test_rows.append(row)
                    if row["status"] == "error":
                        print(f"[TEST ERROR] {row['search_id']} rep{row['rep']}: {row.get('error_message')}")

        if skipped_rows:
            write_table(code_dir / "srdp_param_search_test_skipped.csv", skipped_rows)
        if test_rows:
            write_table(code_dir / "srdp_param_search_test_results.csv", test_rows)

    eval_rows: list[dict[str, Any]] = []
    if not args.skip_eval:
        eval_tasks = discover_eval_tasks(
            roots,
            search_ids=search_ids,
            max_searches=args.max_searches,
            output_tag=output_tag,
            tns=tns,
            n_folds=args.n_folds,
            base_seed=args.base_seed,
            ridge=args.ridge,
            save_matrix_xlsx=not args.no_matrix_xlsx,
        )
        print(f"[EVAL] pending={len(eval_tasks)} workers={args.eval_workers} Tn={tns}")

        if args.dry_run:
            for task in eval_tasks[:10]:
                print(f"  eval: {task.search_id} rep{task.rep} Tn={task.t_n} <- {task.sout_path}")
            return

        if eval_tasks:
            with ProcessPoolExecutor(max_workers=args.eval_workers) as executor:
                future_to_task = {executor.submit(run_eval_worker, asdict(task)): task for task in eval_tasks}
                for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="SRDP eval"):
                    row = future.result()
                    eval_rows.append(row)
                    if row["status"] == "error":
                        print(
                            f"[EVAL ERROR] {row.get('search_id')} rep{row.get('rep')} "
                            f"Tn={row.get('t_n')}: {row.get('error_message')}"
                        )

            ok_rows = [row for row in eval_rows if row.get("status") == "success"]
            write_eval_summaries(roots, ok_rows)
            if eval_rows:
                write_table(code_dir / "srdp_param_search_eval_results.csv", eval_rows)

    n_test_ok = sum(1 for row in test_rows if row.get("status") == "success")
    n_test_err = sum(1 for row in test_rows if row.get("status") == "error")
    n_eval_ok = sum(1 for row in eval_rows if row.get("status") == "success")
    n_eval_err = sum(1 for row in eval_rows if row.get("status") == "error")
    print("[DONE]")
    print(f"  test success={n_test_ok} error={n_test_err}")
    print(f"  eval success={n_eval_ok} error={n_eval_err}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
