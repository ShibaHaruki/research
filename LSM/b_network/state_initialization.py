"""結合用の乱数列を変えず、試行開始時の膜電位を初期化する。"""

import hashlib
import numpy as np


def trial_voltage_values(current, n_neurons, *, seed, population, layer=0):
    """入力波形・seed・層から再現可能な[0, 1)の一様乱数を生成する。

    同じ試行・seedでは同じ初期値を使い、gainや時定数候補の比較を揃える。
    入力はフィルタ後・結合重みを掛ける前の波形で指定する。
    """
    wave = np.ascontiguousarray(current, dtype="<f8")
    digest = hashlib.blake2b(wave.tobytes(), digest_size=16).digest()
    words = np.frombuffer(digest, dtype="<u4").tolist()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), population, layer, *words]))
    return rng.random(n_neurons)


def initialize_trial_voltages(objects, current):
    """試行開始時だけ、各ニューロンのリセット電位以上・閾値未満へ設定する。"""
    settings = objects.get("voltage_initialization", {})
    if settings.get("policy") != "uniform_reset_to_threshold":
        return
    for population, key in enumerate(("G_liq", "G_out")):
        for index, group in enumerate(objects.get(key, [])):
            seed = settings["base_seed"] if population == 0 else settings["output_seeds"][index]
            values = trial_voltage_values(
                current, len(group), seed=seed, population=population, layer=index if population == 0 else 0
            )
            lower = np.asarray(group.v_reset[:], dtype=float)
            upper = np.asarray(group.v_thr[:], dtype=float)
            if np.any(upper <= lower):
                raise ValueError("Membrane threshold must be above reset potential")
            group.v = np.minimum(lower + (upper - lower) * values, np.nextafter(upper, lower))
