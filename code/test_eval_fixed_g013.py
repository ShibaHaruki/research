# -*- coding: utf-8 -*-
import argparse
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DATA_ROOT = SCRIPT_PATH.parents[1]
DATA_PATH = str(DATA_ROOT) + "/"

PREFIX = "SRDP_1"
DATASET_NAME = "SRDP_1.g0.13".replace(".", "_DOT_")
EVAL_DATASET_NAME = "SRDP_1_g0.13"
REP = 1
G_TAG = "g0.13"

W_IN_PATH = SCRIPT_DIR / f"{PREFIX}_w_in_rep{REP}_{G_TAG}.npy"
W_RES_PATH = SCRIPT_DIR / f"{PREFIX}_w_res_rep{REP}_{G_TAG}.npy"
W_OUT_PATH = SCRIPT_DIR / f"{PREFIX}_w_out_rep{REP}_{G_TAG}.npy"
SAMPLE_SEQ_PATH = SCRIPT_DIR / f"sample_seq_rep{REP}.npy"
SOUT_PATH = SCRIPT_DIR / f"{EVAL_DATASET_NAME}_sout_rec_rep{REP}.npy"

N_TRAIN = 100
N_SAMPLE = 100
N_BINS = 500
SOUT_DTYPE = np.uint16
DEFAULT_CODEGEN_TARGET = "numpy"
CYTHON_CACHE_DIR = SCRIPT_DIR / "brian2_cython_cache"

DIR_NAME = [
    "Al_board", "buta_omote", "buta_ura", "cork",
    "denim", "rubber_board", "washi", "wood_board",
]


def calc_meissner(data, t, dt):
    current = np.zeros((4, len(t)))
    for i in range(len(t)):
        if i != 0:
            dF_dt = np.abs(data[i] - data[i - 1]) / (t[i] - t[i - 1])
            current[0, i] = current[0, i - 1] + dF_dt + (-current[0, i - 1] * dt / (8 * 1e-3))
            current[1, i] = (
                current[1, i - 1]
                + 0.24 * dF_dt
                + (-(current[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1e-3))
            )
            current[2, i] = current[2, i - 1] + 0.07 * dF_dt + (-current[2, i - 1] * dt / (1744.6 * 1e-3))
            current[3, i] = current[0, i]
    return current[3, :]


def calc_merkel(data, t, dt):
    current = np.zeros((4, len(t)))
    for i in range(len(t)):
        if i != 0:
            dF_dt = np.abs(data[i] - data[i - 1]) / (t[i] - t[i - 1])
            if dF_dt < 0:
                dF_dt = 0
            current[0, i] = current[0, i - 1] + 0.74 * dF_dt + (-current[0, i - 1] * dt / (8 * 1e-3))
            current[1, i] = (
                current[1, i - 1]
                + 0.24 * dF_dt
                + (-(current[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1e-3))
            )
            current[2, i] = current[2, i - 1] + 0.07 * dF_dt + (-current[2, i - 1] * dt / (1744.6 * 1e-3))
            current[3, i] = current[0, i] + current[1, i] + current[2, i]
    return current[3, :]


def load_input_current(material: str, sample_id: int, n_in: int, dt_s: float):
    files = glob.glob(DATA_PATH + "tactile_data/" + material + f"/data_{sample_id}_*")
    if len(files) == 0:
        raise FileNotFoundError(f"data not found: material={material}, sample_id={sample_id}")

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
    return input_current, nt


def generate_sout_rec(force: bool, n_sample: int, codegen_target: str):
    if SOUT_PATH.exists() and not force:
        print(f"[skip test] exists: {SOUT_PATH}")
        return SOUT_PATH

    from brian2 import (
        prefs, float64, ms, Hz, defaultclock, seed, start_scope,
        NeuronGroup, Synapses, SpikeMonitor, TimedArray, Network,
    )

    prefs.core.default_float_dtype = float64
    prefs.codegen.target = codegen_target
    if codegen_target == "cython":
        CYTHON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        prefs.codegen.runtime.cython.cache_dir = str(CYTHON_CACHE_DIR)
        prefs.codegen.runtime.cython.multiprocess_safe = True

    for path in [W_IN_PATH, W_RES_PATH, W_OUT_PATH, SAMPLE_SEQ_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    w_in = np.load(W_IN_PATH)
    w_res_init = np.load(W_RES_PATH)
    w_out_init = np.load(W_OUT_PATH)
    sample_seq = np.load(SAMPLE_SEQ_PATH).astype(int)
    test_seq = sample_seq[N_TRAIN:N_TRAIN + n_sample]

    n_in, n_res = w_in.shape
    n_out = w_out_init.shape[1]
    if w_res_init.shape != (n_res, n_res):
        raise ValueError(f"w_res shape mismatch: {w_res_init.shape}")
    if w_out_init.shape[0] != n_res:
        raise ValueError(f"w_out shape mismatch: {w_out_init.shape}")

    start_scope()
    np.random.seed(2 + (REP - 1))
    rng = np.random.default_rng(2 + (REP - 1))
    seed(2 + (REP - 1))

    v_reset = -65
    v_thr = -40
    tau_r = 2 * ms
    tau_d = 20 * ms
    bias = -65
    gain = 0.25
    dt_ms = 0.1
    dt_s = dt_ms * 1e-3
    defaultclock.dt = dt_ms * ms

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
    double_exp_out = """
    dR/dt = -R / tau_d + H : 1
    dH/dt = -H / tau_r : Hz
    I_syn = R : 1
    """
    on_pre_res = "H_post += (w_res / (tau_r * tau_d)) / Hz"
    on_pre_out = "H_post += (w_out / (tau_r * tau_d)) / Hz"

    g_res = NeuronGroup(
        n_res, double_exp_res + lif,
        threshold="v >= v_thr",
        reset="v = v_reset",
        refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
        method="exact",
    )
    g_res.tau_m = 10 * ms
    g_res.t_ref = 2 * ms
    g_res.I_inh = 0

    g_out = NeuronGroup(
        n_out, double_exp_out + lif,
        threshold="v >= v_thr",
        reset="v = v_reset",
        refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
        method="exact",
    )
    g_out.tau_m = 10 * ms
    g_out.t_ref = 2 * ms
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
        g_in, g_res,
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

    sout_rec = np.zeros((len(DIR_NAME), n_sample, n_out, N_BINS), dtype=SOUT_DTYPE)
    namespace = {
        "input_ta": input_ta,
        "tau_r": tau_r,
        "tau_d": tau_d,
        "G": gain,
        "BIAS": bias,
        "v_reset": v_reset,
        "v_thr": v_thr,
    }

    t0 = 0 * ms
    for i_mat, material in enumerate(DIR_NAME):
        for j, sample_id in enumerate(tqdm(test_seq, desc=material)):
            input_current, nt = load_input_current(material, int(sample_id), n_in, dt_s)
            vals = input_current.T
            vals = np.vstack([vals, vals[-1]])
            input_ta = TimedArray(vals, dt=dt_ms * ms)
            namespace["input_ta"] = input_ta
            g_in.t_start = t0

            g_res.v = v_reset + (v_thr - v_reset) * rng.random(n_res)
            g_out.v = v_reset + (v_thr - v_reset) * rng.random(n_out)
            g_res.R = 0
            g_res.H = 0
            g_out.R = 0
            g_out.H = 0

            start_t = t0
            start_idx = len(mr_out.t)
            duration = (nt * dt_ms) * ms
            net.run(duration, namespace=namespace)
            end_idx = len(mr_out.t)
            t0 += duration

            if end_idx <= start_idx:
                continue

            t_sp = mr_out.t[start_idx:end_idx]
            i_sp = mr_out.i[start_idx:end_idx]
            mask = (t_sp > start_t) & (t_sp <= t0)
            if not np.any(mask):
                continue

            rel_times_ms = (t_sp[mask] - start_t) / ms
            ids = np.asarray(i_sp[mask], dtype=int)
            bin_edges = np.linspace(0, nt * dt_ms, N_BINS + 1)
            for n in range(n_out):
                counts, _ = np.histogram(rel_times_ms[ids == n], bins=bin_edges)
                sout_rec[i_mat, j, n, :] = counts.astype(SOUT_DTYPE, copy=False)

    np.save(SOUT_PATH, sout_rec)
    print(f"[saved test] {SOUT_PATH} shape={sout_rec.shape}")
    return SOUT_PATH


def run_eval(t_n: int):
    cmd = [sys.executable, str(SCRIPT_DIR / "eval.py"), str(t_n), EVAL_DATASET_NAME]
    print("[run eval]", " ".join(cmd))
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sample", type=int, default=N_SAMPLE)
    parser.add_argument("--t-n", type=int, default=25)
    parser.add_argument("--force-test", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--codegen",
        choices=["numpy", "cython"],
        default=DEFAULT_CODEGEN_TARGET,
        help="Brian2 code generation target for the test simulation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.skip_test:
        generate_sout_rec(force=args.force_test, n_sample=args.n_sample, codegen_target=args.codegen)
    if not args.skip_eval:
        run_eval(args.t_n)


if __name__ == "__main__":
    main()
