# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import glob
from pathlib import Path
from tqdm import tqdm
from brian2 import *

# =========================
# Brian2 settings
# =========================
prefs.core.default_float_dtype = float64
prefs.codegen.target = "numpy"

# =========================
# Paths
# =========================
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
SAVE_DIR = SCRIPT_DIR

DATA_ROOT = SCRIPT_PATH.parents[1]
path = str(DATA_ROOT) + "/"

dir_name = ["Al_board", "buta_omote", "buta_ura", "cork",
            "denim", "rubber_board", "washi", "wood_board"]

# =========================
# input filters
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
            I[1, i] = I[1, i - 1] + 0.24 * dF_dt + (-(I[1, i - 1] - 0.24 * 0.13) * dt / (200 * 1 * 1e-3))
            I[2, i] = I[2, i - 1] + 0.07 * dF_dt + (-I[2, i - 1] * dt / (1744.6 * 1 * 1e-3))
            I[3, i] = I[0, i] + I[1, i] + I[2, i]
    return I[3, :]


# =========================
# Save weights (1-liquid)
# =========================
OUT_PREFIX = "T_STDP_1"

def save_weights(rep: int,
                 w_in: np.ndarray,
                 w_res: np.ndarray,
                 S_out: Synapses,
                 N_res: int,
                 N_out: int,
                 out_dir: Path):
    """
    保存名:
      T_STDP_1_w_in_rep{rep}.npy
      T_STDP_1_w_res_rep{rep}.npy
      T_STDP_1_w_out_rep{rep}.npy (dense)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{OUT_PREFIX}_w_in_rep{rep}.npy", w_in)
    np.save(out_dir / f"{OUT_PREFIX}_w_res_rep{rep}.npy", w_res)

    w_out_dense = np.zeros((N_res, N_out), dtype=float)
    ii = np.array(S_out.i[:], dtype=int)
    jj = np.array(S_out.j[:], dtype=int)
    ww = np.array(S_out.w_out[:], dtype=float)
    w_out_dense[ii, jj] = ww
    np.save(out_dir / f"{OUT_PREFIX}_w_out_rep{rep}.npy", w_out_dense)


# =========================
# Repeat settings
# =========================
N_REPEAT = 10
BASE_SEED = 2
N_EPOCH = 3
N_TRAIN = 100


def run_once(rep: int):
    start_scope()

    seed_val = BASE_SEED + (rep - 1)
    np.random.seed(seed_val)
    rng = np.random.default_rng(seed_val)
    seed(seed_val)

    # sample order
    sample_seq = np.arange(1, 325, dtype=int)
    rng.shuffle(sample_seq)
    np.save(SAVE_DIR / f"sample_seq_rep{rep}.npy", sample_seq)

    # hyper-parameters
    N_in = 2
    N_res = 1000
    N_out = 40

    p_in = 0.2
    p_res = 0.5
    p_out = 0.5

    v_reset = -65
    v_thr = -40

    # time
    dt_ms = 0.1
    dt_s = dt_ms * 1e-3
    defaultclock.dt = dt_ms * ms
    t0 = 0 * ms

    # double_exponential_synapse
    tau_r = 2 * ms
    tau_d = 20 * ms

    # T-STDP params
    wmin = -1.0
    wmax = 1.0

    tau_s1 = 11.5 * ms
    tau_s2 = 15.0 * ms
    tau_t1 = 14.0 * ms
    tau_t2 = 15.0 * ms

    A2_plus  = 0.0007
    A3_plus  = 0.00000005
    A2_minus = 0.0006
    A3_minus = 0.00000005

    BIAS = -65
    G = 0.25

    # neuron params
    neuron_array_res = np.ones(N_res)
    t_ref_res = np.where(neuron_array_res == 1, 2, 2)
    tau_m_res = np.where(neuron_array_res == 1, 10, 10)

    neuron_array_out = np.ones(N_out)
    t_ref_out = np.where(neuron_array_out == 1, 2, 2)
    tau_m_out = np.where(neuron_array_out == 1, 10, 10)

    # weights
    w_in = rng.standard_normal((N_in, N_res)) * (rng.random((N_in, N_res)) < p_in) / np.sqrt(N_in * p_in)

    variance = (N_res * p_res**2) ** -1
    w_res_init = rng.standard_normal((N_res, N_res)) * (rng.random((N_res, N_res)) < p_res) * np.sqrt(variance)
    for k in range(N_res):
        QS = np.where(np.abs(w_res_init[:, k]) > 0)[0]
        if len(QS) > 0:
            w_res_init[QS, k] -= np.mean(w_res_init[QS, k])

    mask_out = (rng.random((N_res, N_out)) < p_out).astype(float)
    w_out_init = rng.standard_normal((N_res, N_out)) * mask_out / np.sqrt(N_out * p_out) * G

    # models
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

    T_STDP = """
    dApre1/dt   = -Apre1/tau_s1   : 1 (event-driven)
    dApre2/dt   = -Apre2/tau_s2   : 1 (event-driven)
    dApost1/dt  = -Apost1/tau_t1  : 1 (event-driven)
    dApost2/dt  = -Apost2/tau_t2  : 1 (event-driven)

    w_out : 1
    eps_w : 1

    tau_s1 : second (constant)
    tau_s2 : second (constant)
    tau_t1 : second (constant)
    tau_t2 : second (constant)

    A2_plus  : 1 (constant)
    A3_plus  : 1 (constant)
    A2_minus : 1 (constant)
    A3_minus : 1 (constant)
    """

    T_STDP_pre = """
    Apre1 += 1.
    Apre2 += 1.
    w_out = clip(w_out - int(w_out > eps_w) * Apost1 * (A2_minus + A3_minus * Apre2), wmin, wmax)
    """

    T_STDP_post = """
    Apost1 += 1.
    Apost2 += 1.
    w_out = clip(w_out + int(w_out > eps_w) * Apre1 * (A2_plus + A3_plus * Apost2), wmin, wmax)
    """

    eqs_res = double_exp_res + LIF
    eqs_out = double_exp_out + LIF

    # groups (1 liquid)
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

    # synapses
    S_res = Synapses(G_res, G_res, model="w_res : 1", on_pre=on_pre_res, method="euler")
    S_res.connect(condition="i != j")
    S_res.w_res = w_res_init[S_res.i, S_res.j]
    S_res.delay = 0 * ms

    pre_idx, post_idx = np.where(mask_out > 0)
    S_out = Synapses(
        G_res, G_out,
        model=T_STDP,
        on_pre=on_pre_out + T_STDP_pre,
        on_post=T_STDP_post,
        method="euler",
    )
    S_out.connect(i=pre_idx, j=post_idx)
    S_out.w_out = w_out_init[S_out.i, S_out.j]
    S_out.eps_w = 1e-12
    S_out.delay = 0 * ms

    # set constants for T-STDP (Synapses variables)
    S_out.tau_s1 = tau_s1
    S_out.tau_s2 = tau_s2
    S_out.tau_t1 = tau_t1
    S_out.tau_t2 = tau_t2

    S_out.A2_plus  = A2_plus
    S_out.A3_plus  = A3_plus
    S_out.A2_minus = A2_minus
    S_out.A3_minus = A3_minus

    # input (trial中に更新)
    input_current = np.zeros((N_in, 1), dtype=float)

    @network_operation(dt=dt_ms * ms)
    def apply_input():
        nonlocal input_current, t0
        idx = int(((defaultclock.t - t0) / (dt_ms * ms)))
        if idx < 0:
            idx = 0
        if idx >= input_current.shape[1]:
            idx = input_current.shape[1] - 1
        I_input = input_current[:, idx] @ w_in
        G_res.I_exc = I_input
        G_res.I_inh = 0

    net = Network(G_res, G_out, S_res, S_out, apply_input)

    # namespace（STDP_1 のスタイルに合わせる：学習定数もnsに入れる）
    ns = {
        "v_reset": v_reset,
        "v_thr": v_thr,
        "BIAS": BIAS,
        "G": G,
        "tau_r": tau_r,
        "tau_d": tau_d,
        "wmin": wmin,
        "wmax": wmax,
        "tau_s1": tau_s1,
        "tau_s2": tau_s2,
        "tau_t1": tau_t1,
        "tau_t2": tau_t2,
        "A2_plus": A2_plus,
        "A3_plus": A3_plus,
        "A2_minus": A2_minus,
        "A3_minus": A3_minus,
    }

    # training
    for epo in range(1, N_EPOCH + 1):
        for i_size in tqdm(range(N_TRAIN), desc=f"rep{rep}-epo{epo}"):
            sid = int(sample_seq[i_size])

            for mat in dir_name:
                files = glob.glob(path + "tactile_data/" + mat + f"/data_{sid}_*")
                if len(files) == 0:
                    raise FileNotFoundError(f"data not found: {mat} sample={sid}")
                df = pd.read_table(files[0], header=None)

                df_np = df.to_numpy().T
                in_data_0 = df_np[:3, 3000:8000]
                nt = in_data_0.shape[1]
                t_array_s = np.arange(nt) * dt_s

                input_current = np.zeros((N_in, nt), dtype=float)

                # STDP_1 に合わせて ch==0 だけ使う
                ch = 0
                in_data = in_data_0[ch, :]
                I_merkel = calc_merkel(in_data, t_array_s, dt_s)
                I_meissner = calc_meissner(in_data, t_array_s, dt_s)
                input_current[ch * 2, :] = 0.4 * I_merkel * 0.02
                input_current[ch * 2 + 1, :] = 0.6 * 7.3 * I_meissner * 0.02

                # initialize_state
                G_res.v = v_reset + (v_thr - v_reset) * rng.random(N_res)
                G_out.v = v_reset + (v_thr - v_reset) * rng.random(N_out)

                G_res.R = 0; G_res.H = 0
                G_out.R = 0; G_out.H = 0

                # run
                net.run((nt * dt_ms) * ms, namespace=ns)
                t0 += (nt * dt_ms) * ms

    # save weights
    save_weights(
        rep=rep,
        w_in=w_in,
        w_res=w_res_init,
        S_out=S_out,
        N_res=N_res,
        N_out=N_out,
        out_dir=Path(SAVE_DIR),
    )
    print(f"[saved] rep{rep} -> {Path(SAVE_DIR)}")


if __name__ == "__main__":
    for rep in range(1, N_REPEAT + 1):
        run_once(rep)

