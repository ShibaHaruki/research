import numpy as np

_EPS = 1e-12


def init_in_to_liq(
    rng: np.random.Generator,
    n_syn: int,
    scale: float,
    N_post: int,
) -> np.ndarray:
    """
    入力 -> Liquid の重み初期化（常に √ 正規化）
    Kbar = n_syn / N_post をこの関数内で計算する
    """
    n_syn = int(n_syn)
    N_post = int(N_post)

    x = rng.standard_normal(n_syn).astype(float)

    denom = max(N_post, 1)
    Kbar = max(float(n_syn) / float(denom), _EPS)

    return x * float(scale) / np.sqrt(Kbar)


def init_liq_intra(
    rng: np.random.Generator,
    n_syn: int,
    gain: float,
    N_post: int,
) -> np.ndarray:
    """
    Liquid 内部の重み初期化（常に √ 正規化）
    Kbar = n_syn / N_post をこの関数内で計算する
    """
    n_syn = int(n_syn)
    N_post = int(N_post)

    x = rng.standard_normal(n_syn).astype(float)

    denom = max(N_post, 1)
    Kbar = max(float(n_syn) / float(denom), _EPS)

    return np.abs(x) * float(gain) / np.sqrt(Kbar)


def init_liq_to_out(
    rng: np.random.Generator,
    n_syn: int,
    gain: float,
    N_post: int,
) -> np.ndarray:
    """
    Liquid -> Output の重み初期化（常に √ 正規化）
    Kbar = n_syn / N_post をこの関数内で計算する
    """
    n_syn = int(n_syn)
    N_post = int(N_post)

    x = rng.standard_normal(n_syn).astype(float)

    denom = max(N_post, 1)
    Kbar = max(float(n_syn) / float(denom), _EPS)

    return np.abs(x) * float(gain) / np.sqrt(Kbar)