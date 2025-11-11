from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import time
from pathlib import Path
from tqdm import tqdm

from brian2 import *
prefs.core.default_float_dtype = float64
prefs.codegen.target = 'numpy'


# ==============================
# グローバル共通：パス・素材名など
# ==============================
# 実行ファイルの1つ上のディレクトリを基準に tactile_data を探す
path = str(Path(__file__).resolve().parents[1]) + "/"
dir_name = ["Al_board", "buta_omote", "buta_ura",
            "cork", "denim", "rubber_board", "washi", "wood_board"]

# サンプル番号のシャッフル（1〜324）
rng_global = np.random.default_rng(2)
sample_seq = np.arange(1, 325)
rng_global.shuffle(sample_seq)
np.save("sample_seq.npy", sample_seq)

# 時間刻み（共通）
dt_ms = 0.1
dt_s = dt_ms * 1e-3  # [s]


# ==============================
# 触覚フィルタ関数（グローバル）
# ==============================
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


# ==============================
# Separation property 用の補助関数（グローバル）
# ==============================
def exp_conv(t_ms, spikes_ms, tau_ms):
    """
    t_ms: 時間軸 (1D array, ms)
    spikes_ms: スパイク時刻配列 (ms)
    tau_ms: 時定数 (ms)
    """
    t_ms = np.asarray(t_ms, dtype=float)
    spikes_ms = np.asarray(spikes_ms, dtype=float)
    if spikes_ms.size == 0:
        return np.zeros_like(t_ms, dtype=float)
    conv = np.zeros_like(t_ms, dtype=float)
    for st in spikes_ms:
        conv += np.exp(-(t_ms - st) / tau_ms) * (t_ms >= st)
    return conv


def liquid_distance(state_u, state_v):
    """
    state_u, state_v : shape (N_res, T)
    各時刻 t ごとに (平均二乗誤差) を返す => shape (T,)
    """
    return np.mean((state_u - state_v) ** 2, axis=0)


# ====================================================
# 1回分の実験を実行し、SPスカラーを返す関数
# ====================================================
def run_one_experiment(N_res, seed=0):
    """
    指定した N_res で1回だけシミュレーションし、
    Reservoir Separation Property を1つのスカラーにして返す。
    """
    print(f"  [run_one_experiment] N_res={N_res}, seed={seed}")

    # Brian2 のネットワークをリセット
    start_scope()

    # numpy RNG（重み初期化・電位初期化などに使用）
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # ===== hyper-parameters =====
    N_in = 2
    N_out = 40

    # connection probability
    p_in = 0.4
    p_res = 0.1
    p_out = 0.25

    # LIF params
    v_reset = -65
    v_peak = 30
    v_thr = -40

    # reservoir neuron types（ここでは全部+1=興奮性としている）
    neuron_array_res = np.ones(N_res)
    t_ref_res = np.where(neuron_array_res == 1, 2, 2)  # [ms]
    tau_m_res = np.where(neuron_array_res == 1, 10, 10)  # [ms]

    # output neurons
    neuron_array_out = np.ones(N_out)
    t_ref_out = np.where(neuron_array_out == 1, 2, 2)
    tau_m_out = np.where(neuron_array_out == 1, 10, 10)

    # double_exponential_synapse
    tau_r = 2 * ms
    tau_d = 20 * ms
    tau_s = 4.65 * ms
    tau_t = 9.5 * ms

    # STDP
    A_B_LTP = 0.006
    A_B_LTD = -0.003
    wmin = -1e9
    wmax = 1e9

    BIAS = -40
    G = 0.01
    A = 0.01

    # ===== time =====
    defaultclock.dt = dt_ms * ms

    # ===== weights =====
    # 入力結合
    w_in = np.random.randn(N_in, N_res) * (np.random.rand(N_in, N_res) < p_in) / (np.sqrt(N_in) * p_in)

    # Reservoir 内部結合
    variance = (N_res * p_res ** 2) ** -1
    w_res_init = np.random.randn(N_res, N_res) * (np.random.rand(N_res, N_res) < p_res) * np.sqrt(variance)
    for k in range(N_res):
        QS = np.where(np.abs(w_res_init[:, k]) > 0)[0]
        if len(QS) > 0:
            w_res_init[QS, k] -= np.mean(w_res_init[QS, k])

    # Reservoir -> Output
    mask_out = (np.random.rand(N_res, N_out) < p_out).astype(float)
    w_out_init = np.random.randn(N_res, N_out) * mask_out / (np.sqrt(N_out) * p_out) * A

    # ===== models =====
    LIF = '''
    dv/dt = (-v + BIAS + I_exc - I_inh + I_syn) / tau_m : 1 (unless refractory)
    spiked : 1
    I_exc : 1
    I_inh : 1
    tau_m : second
    t_ref : second
    '''

    double_exp_res = '''
    dR/dt = -R / tau_d + H : 1
    dH/dt = -H / tau_r : Hz
    I_syn = G * R : 1
    '''

    on_pre_res = '''
    H_post += (w_res / (tau_r * tau_d)) / Hz
    '''

    double_exp_out = '''
    dR/dt = -R / tau_d + H : 1
    dH/dt = -H / tau_r : Hz
    I_syn = R : 1
    '''

    on_pre_out = '''
    H_post += (w_out / (tau_r * tau_d)) / Hz
    '''

    STDP_eqs = '''
    dApre/dt  = -Apre/tau_s  : 1 (event-driven)
    dApost/dt = -Apost/tau_t : 1 (event-driven)
    w_out : 1
    eps_w : 1
    plastic : 1 (shared)
    '''

    STDP_pre = '''
    Apre +=   A_B_LTP * plastic
    w_out = clip(w_out + int(w_out > eps_w) * Apost * plastic, wmin, wmax)
    '''

    STDP_post = '''
    Apost +=  A_B_LTD * plastic
    w_out  = clip(w_out + int(w_out > eps_w) * Apre * plastic,  wmin, wmax)
    '''

    eqs_res = double_exp_res + LIF
    eqs_out = double_exp_out + LIF

    # ===== groups =====
    G_res = NeuronGroup(
        N_res, eqs_res,
        threshold='v >= v_thr',
        reset='v = v_reset',
        refractory='timestep(t - lastspike, dt) <= timestep(t_ref, dt)',
        method='exact'
    )
    G_res.tau_m = tau_m_res * ms
    G_res.t_ref = t_ref_res * ms

    G_out = NeuronGroup(
        N_out, eqs_out,
        threshold='v >= v_thr',
        reset='v = v_reset',
        refractory='timestep(t - lastspike, dt) <= timestep(t_ref, dt)',
        method='exact'
    )
    G_out.tau_m = tau_m_out * ms
    G_out.t_ref = t_ref_out * ms

    # Reservoir 内シナプス
    S_res = Synapses(G_res, G_res, model='w_res : 1', on_pre=on_pre_res, method='euler')
    S_res.connect(condition='i != j')
    S_res.w_res = w_res_init[S_res.i, S_res.j]
    S_res.delay = 0 * ms

    # Reservoir → Output（STDP付き）
    S_out = Synapses(G_res, G_out, model=STDP_eqs, on_pre=on_pre_out + STDP_pre,
                     on_post=STDP_post, method='euler')
    S_out.connect(condition='i != j')
    S_out.w_out = w_out_init[S_out.i, S_out.j]
    S_out.eps_w = 1e-12
    S_out.delay = 0 * ms
    S_out.plastic = 1
    S_out.Apre = 0
    S_out.Apost = 0

    # ===== monitors =====
    Msp_res = SpikeMonitor(G_res)
    # Separation property は Reservoir のスパイクだけあればよいので、
    # Msp_out や StateMonitor は付けない（メモリ節約）

    # ===== 入力を与える network_operation =====
    # この中から、外側スコープの t0, input_current, w_in, G_res を参照する
    t0 = 0 * ms
    input_current = np.zeros((N_in, 1))  # ダミーで1ステップ

    @network_operation(dt=dt_ms * ms)
    def apply_input():
        nonlocal t0, input_current  # t0, input_current を外側スコープから参照
        # 現在の時間から、この trial 内でのインデックスを計算
        idx = int(((defaultclock.t - t0) / (dt_ms * ms)))
        if 0 <= idx < input_current.shape[1]:
            I_in = input_current[:, idx] @ w_in   # shape (N_res,)
            G_res.I_exc = I_in
        else:
            # trial の外側では電流ゼロ
            G_res.I_exc = 0
        G_res.I_inh = 0

    # ===== Separation property 用の trial 情報 =====
    trial_start_times_ms = []
    trial_labels = []
    trial_duration_ms = None

    # ===== 実行ループ =====
    start_time = time.perf_counter()

    for epo in range(1):
        # i_size: ここでは 1 つのサンプルだけ使う（必要なら range(3) に変える）
        for i_size in range(1):
            sample_id = int(sample_seq[i_size])

            for mat in dir_name:
                # tactile データ読み込み
                pattern = path + f"tactile_data/{mat}/data_{sample_id}_*"
                files = glob.glob(pattern)
                if len(files) == 0:
                    print(f"[WARN] file not found: {pattern}")
                    continue

                df = pd.read_table(files[0], header=None)
                df_np = df.to_numpy().T  # shape (channel, time)
                in_data_0 = df_np[:3, 3000:8000]  # 3ch, 対象区間
                nt = in_data_0.shape[1]
                t_array_s = np.arange(nt) * dt_s

                # 入力電流（N_in=2）
                input_current = np.zeros((N_in, nt))
                for j in range(3):
                    in_data = in_data_0[j, :]
                    I_merkel = calc_merkel(in_data, t_array_s, dt_s)
                    I_meissner = calc_meissner(in_data, t_array_s, dt_s)
                    if j == 0:
                        input_current[j * 2, :] = 0.01 * I_merkel
                        input_current[j * 2 + 1, :] = 0.04 * I_meissner

                # ニューロン・シナプス状態の初期化
                G_res.v = v_reset + (v_thr - v_reset) * rng.random(N_res)
                G_out.v = v_reset + (v_thr - v_reset) * rng.random(N_out)
                G_res.R = 0
                G_res.H = 0
                G_out.R = 0
                G_out.H = 0
                S_out.Apre = 0
                S_out.Apost = 0

                # trial 情報記録
                trial_start_times_ms.append(float(defaultclock.t / ms))
                trial_labels.append(mat)
                if trial_duration_ms is None:
                    trial_duration_ms = nt * dt_ms  # [ms]

                # STDP を有効にするかどうか（Reservoir の SP には直接影響しない）
                S_out.plastic = 1

                # この trial の開始時刻として t0 を記録
                t0 = defaultclock.t

                # 実行
                run(nt * dt_ms * ms)

            print(f"    i_size={i_size} done.")

    end_time = time.perf_counter()
    print(f"  [TIME] N_res={N_res}, seed={seed} : {end_time - start_time:.3f} s")

    # ===== Separation property (Reservoir) 評価 =====
    n_trials = len(trial_start_times_ms)
    if n_trials == 0:
        return np.nan

    T_ms = trial_duration_ms
    # メモリを抑えるため、SP評価用の時間刻みは 1ms とする
    TS = np.arange(0.0, T_ms, 1.0)  # [ms]

    all_t_res_ms = np.array(Msp_res.t / ms)  # 全スパイク時刻
    all_i_res = np.array(Msp_res.i)          # 対応するニューロン index

    # 各 trial の liquid state: list of (N_res, len(TS))
    liquid_states_res = []

    for k in range(n_trials):
        t_start = trial_start_times_ms[k]
        t_end = t_start + T_ms

        mask = (all_t_res_ms >= t_start) & (all_t_res_ms < t_end)
        t_trial_ms = all_t_res_ms[mask] - t_start
        i_trial = all_i_res[mask]

        state_k = np.zeros((N_res, len(TS)), dtype=float)
        for n in range(N_res):
            spikes_n = t_trial_ms[i_trial == n]
            if spikes_n.size > 0:
                state_k[n, :] = exp_conv(TS, spikes_n, tau_ms=30.0)
        liquid_states_res.append(state_k)

    # 素材ペアごとの距離
    d_scalar_by_pair = defaultdict(list)

    for a in range(n_trials):
        for b in range(a + 1, n_trials):
            label_a = trial_labels[a]
            label_b = trial_labels[b]
            pair_key = tuple(sorted((label_a, label_b)))

            d_ab_ts = liquid_distance(liquid_states_res[a], liquid_states_res[b])  # (T,)
            d_scalar_by_pair[pair_key].append(np.mean(d_ab_ts))  # 時間平均

    # 素材×素材 行列 dist_mat
    unique_labels = sorted(list(set(trial_labels)))
    n_labels = len(unique_labels)
    dist_mat = np.full((n_labels, n_labels), np.nan, dtype=float)

    for i_lab, la in enumerate(unique_labels):
        for j_lab, lb in enumerate(unique_labels):
            pair_key = tuple(sorted((la, lb)))
            if pair_key in d_scalar_by_pair:
                dist_mat[i_lab, j_lab] = np.mean(d_scalar_by_pair[pair_key])

    # 対角成分を除いた平均（二乗平均距離の平均）を SP スカラーとする
    idx = np.triu_indices(n_labels, k=1)
    sp_scalar = np.nanmean(dist_mat[idx])

    print(f"  [SP] N_res={N_res}, seed={seed} -> SP={sp_scalar:.4e}")

    return sp_scalar


# ====================================================
# メイン：N_res を 100〜1000 まで振って SP をプロット
# ====================================================
if __name__ == "__main__":
    # N_res のリスト
    N_res_list = np.arange(100, 1001, 50)  # 100,150,...,1000
    n_repeats = 10

    sp_mean_list = []
    sp_std_list = []

    for N in N_res_list:
        sp_values = []
        for rep in range(n_repeats):
            sp = run_one_experiment(N_res=N, seed=rep)
            sp_values.append(sp)
        sp_values = np.array(sp_values)
        sp_mean_list.append(sp_values.mean())
        sp_std_list.append(sp_values.std())

        print(f"[SUMMARY] N_res={N}: mean SP={sp_values.mean():.4e}, std={sp_values.std():.4e}")

    sp_mean_list = np.array(sp_mean_list)
    sp_std_list = np.array(sp_std_list)

    # ===== プロット =====
    plt.figure(figsize=(8, 5))
    plt.errorbar(N_res_list, sp_mean_list, yerr=sp_std_list,
                 marker='o', capsize=3)
    plt.xlabel("Number of reservoir neurons N_res")
    plt.ylabel("Separation Property (mean squared distance)")
    plt.title("Reservoir Separation Property vs N_res (10 trials mean)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("SP_vs_Nres.png", dpi=300)
    plt.show()
