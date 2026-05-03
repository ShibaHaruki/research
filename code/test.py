# -*- coding: utf-8 -*-
import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================
# settings
# =========================
N_TRAIN  = 100
N_SAMPLE = 100
N_BINS   = 500

# 並列数（メモリが重いので CPU 全開より控えめ推奨）
MAX_WORKERS = min(40, os.cpu_count() or 1)

# Brian2 codegen: 以前の Python.h 問題があるなら numpy 推奨
CODEGEN_TARGET = "numpy"   # "cython" にしたいなら python-dev 入ってる前提

# sout_rec はカウントなので int の方が軽い（float64だとメモリ爆増）
SOUT_DTYPE = np.uint16     # カウントが大きいなら np.uint32

# ファイル名フォーマット（あなたの命名に合わせる）
W_IN_FMT   = "{prefix}_w_in_rep{rep}.npy"
W_RES_FMT  = "{prefix}_w_res_rep{rep}.npy"
W_OUT_FMT  = "{prefix}_w_out_rep{rep}.npy"
SAMPLE_SEQ_FMT = "sample_seq_rep{rep}.npy"
OUT_SOUT_FMT   = "{prefix}_sout_rec_rep{rep}.npy"

# =========================
# tactile input filters
# =========================
def calc_meissner(data, t, dt):
    I = np.zeros((4, len(t)))
    for i in range(len(t)):
        if i != 0:
            dF_dt = np.abs(data[i] - data[i - 1]) / (t[i] - t[i - 1])
            I[0, i] = I[0, i - 1] + 1 * dF_dt + (-I[0, i - 1] * dt / (8 * 1 * 1e-3))
            I[1, i] = I[1, i - 1] + 0.24 * dF_dt + (-(I[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1e-3))
            I[2, i] = I[2, i - 1] + 0.07 * dF_dt + (-I[2, i - 1] * dt / (1744.6 * 1e-3))
            I[3, i] = I[0, i]
    return I[3, :]

def calc_merkel(data, t, dt):
    I = np.zeros((4, len(t)))
    for i in range(len(t)):
        if i != 0:
            dF_dt = np.abs(data[i] - data[i - 1]) / (t[i] - t[i - 1])
            if dF_dt < 0:
                dF_dt = 0
            I[0, i] = I[0, i - 1] + 0.74 * dF_dt + (-I[0, i - 1] * dt / (8 * 1 * 1e-3))
            I[1, i] = I[1, i - 1] + 0.24 * dF_dt + (-(I[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1e-3))
            I[2, i] = I[2, i - 1] + 0.07 * dF_dt + (-I[2, i - 1] * dt / (1744.6 * 1e-3))
            I[3, i] = I[0, i] + I[1, i] + I[2, i]
    return I[3, :]

# =========================
# discover tasks (prefix, rep)
# =========================
def discover_tasks(script_dir: Path):
    # *_w_in_rep{rep}.npy から prefix を抽出
    rex = re.compile(r"(.+)_w_in_rep(\d+)\.npy$")
    tasks = []

    for fp in glob.glob(str(script_dir / "*_w_in_rep*.npy")):
        name = Path(fp).name
        m = rex.match(name)
        if not m:
            continue
        prefix = m.group(1)
        rep = int(m.group(2))

        # 必須ファイルが揃っているかチェック
        ok = True
        for need in [
            script_dir / W_IN_FMT.format(prefix=prefix, rep=rep),
            script_dir / W_RES_FMT.format(prefix=prefix, rep=rep),
            script_dir / W_OUT_FMT.format(prefix=prefix, rep=rep),
            script_dir / SAMPLE_SEQ_FMT.format(rep=rep),
        ]:
            if not need.exists():
                ok = False
                break

        if ok:
            tasks.append((prefix, rep))

    # 重複除去してソート
    tasks = sorted(list(set(tasks)), key=lambda x: (x[0], x[1]))
    return tasks

# =========================
# worker: evaluate one (prefix, rep)
# =========================
def run_once_worker(prefix: str, rep: int, script_dir_str: str):
    # Brian2 はプロセス内で import/設定する方が安全
    from brian2 import (
        prefs, float64, ms, Hz, defaultclock, seed, start_scope,
        NeuronGroup, Synapses, SpikeMonitor, TimedArray, Network
    )

    prefs.core.default_float_dtype = float64
    prefs.codegen.target = CODEGEN_TARGET

    script_dir = Path(script_dir_str)
    script_path = script_dir / "dummy.py"  # 使わないが一応

    # tactile_data の場所（あなたのコードの SCRIPT_PATH.parents[1] 相当）
    # ここは「評価スクリプトの1つ上に tactile_data がある」想定
    data_root = script_dir.parent
    path = str(data_root) + "/"

    dir_name = ["Al_board", "buta_omote", "buta_ura", "cork",
                "denim", "rubber_board", "washi", "wood_board"]

    # --- sample seq ---
    sample_seq = np.load(script_dir / SAMPLE_SEQ_FMT.format(rep=rep)).astype(int)
    test_seq = sample_seq[N_TRAIN:]
    if len(test_seq) < N_SAMPLE:
        raise ValueError(f"[{prefix} rep{rep}] test_seq too short: {len(test_seq)}")
    test_seq = test_seq[:N_SAMPLE]

    # --- weights ---
    w_in       = np.load(script_dir / W_IN_FMT.format(prefix=prefix, rep=rep))
    w_res_init = np.load(script_dir / W_RES_FMT.format(prefix=prefix, rep=rep))
    w_out_init = np.load(script_dir / W_OUT_FMT.format(prefix=prefix, rep=rep))

    if w_res_init.ndim != 2 or w_res_init.shape[0] != w_res_init.shape[1]:
        raise ValueError(f"[{prefix} rep{rep}] w_res shape invalid: {w_res_init.shape}")
    N_res = w_res_init.shape[0]

    if w_out_init.ndim != 2 or w_out_init.shape[0] != N_res:
        raise ValueError(f"[{prefix} rep{rep}] w_out shape invalid: {w_out_init.shape}")
    N_out = w_out_init.shape[1]

    if w_in.ndim != 2 or w_in.shape[1] != N_res:
        raise ValueError(f"[{prefix} rep{rep}] w_in must be (N_in,N_res), got {w_in.shape}")
    N_in = w_in.shape[0]

    # --- build network ---
    start_scope()
    np.random.seed(2 + (rep - 1))
    seed(2 + (rep - 1))

    v_reset = -65
    v_thr   = -40
    tau_r = 2 * ms
    tau_d = 20 * ms
    BIAS = -65
    G = 0.25

    dt_ms = 0.1
    dt_s  = dt_ms * 1e-3
    defaultclock.dt = dt_ms * ms

    neuron_array_res = np.ones(N_res)
    tau_m_res = np.where(neuron_array_res == 1, 10, 10)
    t_ref_res = np.where(neuron_array_res == 1, 2, 2)

    neuron_array_out = np.ones(N_out)
    tau_m_out = np.where(neuron_array_out == 1, 10, 10)
    t_ref_out = np.where(neuron_array_out == 1, 2, 2)

    LIF = """
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

    eqs_res = double_exp_res + LIF
    eqs_out = double_exp_out + LIF

    G_res = NeuronGroup(
        N_res, eqs_res,
        threshold="v >= v_thr",
        reset="v = v_reset",
        refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
        method="exact",
    )
    G_res.tau_m = tau_m_res * ms
    G_res.t_ref = t_ref_res * ms
    G_res.I_inh = 0

    G_out = NeuronGroup(
        N_out, eqs_out,
        threshold="v >= v_thr",
        reset="v = v_reset",
        refractory="timestep(t - lastspike, dt) <= timestep(t_ref, dt)",
        method="exact",
    )
    G_out.tau_m = tau_m_out * ms
    G_out.t_ref = t_ref_out * ms
    G_out.I_inh = 0

    S_res = Synapses(G_res, G_res, model="w_res : 1", on_pre=on_pre_res, method="euler")
    S_res.connect(condition="i != j")
    S_res.w_res = w_res_init[S_res.i, S_res.j]
    S_res.delay = 0 * ms

    pre_idx, post_idx = np.where(w_out_init != 0)
    S_out = Synapses(G_res, G_out, model="w_out : 1", on_pre=on_pre_out, method="euler")
    S_out.connect(i=pre_idx, j=post_idx)
    S_out.w_out = w_out_init[S_out.i, S_out.j]
    S_out.delay = 0 * ms

    input_ta = TimedArray(np.zeros((1, N_in)), dt=dt_ms * ms)
    G_in = NeuronGroup(
        N_in,
        """
        t_start : second (shared)
        I = input_ta(t - t_start, i) : 1
        """,
        method="euler",
    )
    G_in.t_start = 0 * ms

    S_in = Synapses(
        G_in, G_res,
        model="""
        w : 1
        I_exc_post = w * I_pre : 1 (summed)
        """,
        method="euler",
    )
    pre_in, post_in = np.nonzero(w_in)
    S_in.connect(i=pre_in, j=post_in)
    S_in.w = w_in[pre_in, post_in]

    Mr_out = SpikeMonitor(G_out)
    net = Network(G_in, G_res, G_out, S_in, S_res, S_out, Mr_out)

    sout_rec = np.zeros((len(dir_name), N_SAMPLE, N_out, N_BINS), dtype=SOUT_DTYPE)

    t0 = 0 * ms
    for i_mat, mat in enumerate(dir_name):
        for j in range(N_SAMPLE):
            sid = int(test_seq[j])
            files = glob.glob(path + "tactile_data/" + mat + f"/data_{sid}_*")
            if len(files) == 0:
                raise FileNotFoundError(f"[{prefix} rep{rep}] data not found: {mat} sid={sid}")

            df = pd.read_table(files[0], header=None)
            df_np = df.to_numpy().T
            in_data_0 = df_np[:3, 3000:8000]
            nt = in_data_0.shape[1]
            t_array_s = np.arange(nt) * dt_s

            input_current = np.zeros((N_in, nt), dtype=float)

            in_data = in_data_0[0, :]
            I_merkel   = calc_merkel(in_data, t_array_s, dt_s)
            I_meissner = calc_meissner(in_data, t_array_s, dt_s)
            input_current[0, :] = 0.4 * I_merkel * 0.02
            input_current[1, :] = 0.6 * 7.3 * I_meissner * 0.02

            vals = input_current.T
            vals = np.vstack([vals, vals[-1]])
            input_ta = TimedArray(vals, dt=dt_ms * ms)
            G_in.t_start = t0

            G_res.v = v_reset + (v_thr - v_reset) * np.random.rand(N_res)
            G_out.v = v_reset + (v_thr - v_reset) * np.random.rand(N_out)
            G_res.R = 0; G_res.H = 0
            G_out.R = 0; G_out.H = 0

            start_t   = t0
            start_idx = len(Mr_out.t)

            ns = {"input_ta": input_ta, "tau_r": tau_r, "tau_d": tau_d, "G": G,
                  "BIAS": BIAS, "v_reset": v_reset, "v_thr": v_thr}

            duration = (nt * dt_ms) * ms
            net.run(duration, namespace=ns)

            end_idx = len(Mr_out.t)
            t0 += duration

            if end_idx > start_idx:
                t_sp = Mr_out.t[start_idx:end_idx]
                i_sp = Mr_out.i[start_idx:end_idx]
                mask = (t_sp > start_t) & (t_sp <= t0)

                if np.any(mask):
                    rel_times_ms = (t_sp[mask] - start_t) / ms
                    ids = np.asarray(i_sp[mask], dtype=int)
                    bin_edges = np.linspace(0, nt * dt_ms, N_BINS + 1)
                    for n in range(N_out):
                        counts, _ = np.histogram(rel_times_ms[ids == n], bins=bin_edges)
                        sout_rec[i_mat, j, n, :] = counts.astype(SOUT_DTYPE, copy=False)

    out_path = script_dir / OUT_SOUT_FMT.format(prefix=prefix, rep=rep)
    np.save(out_path, sout_rec)
    return str(out_path)

# =========================
# main (parallel)
# =========================
def main():
    script_dir = Path(__file__).resolve().parent
    tasks = discover_tasks(script_dir)

    if len(tasks) == 0:
        print("No tasks found. Check filenames like '*_w_in_rep{rep}.npy' etc.")
        return

    print(f"Found {len(tasks)} tasks. max_workers={MAX_WORKERS}, codegen={CODEGEN_TARGET}")
    # (参考) sout_rec 1本のサイズ概算
    est_bytes = (8 * N_SAMPLE * 40 * N_BINS) * np.dtype(SOUT_DTYPE).itemsize
    print(f"Estimated sout_rec bytes per task (rough): ~{est_bytes/1e6:.1f} MB (dtype={SOUT_DTYPE})")

    futures = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for prefix, rep in tasks:
            fut = ex.submit(run_once_worker, prefix, rep, str(script_dir))
            futures[fut] = (prefix, rep)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="all rules parallel"):
            prefix, rep = futures[fut]
            try:
                out = fut.result()
                # ここでログ
                # print(f"[done] {prefix} rep{rep} -> {out}")
            except Exception as e:
                print(f"[ERROR] {prefix} rep{rep}: {e}")

if __name__ == "__main__":
    main()

