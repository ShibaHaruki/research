"""シナプス重みを初期化するモジュール"""

import numpy as np

_EPS = 1e-12

#入力層からLiquid層へのシナプス重みを初期化
def init_in_to_liq(rng: np.random.Generator,n_syn: int,scale: float,N_post: int,) -> np.ndarray:
    n_syn = int(n_syn)
    N_post = int(N_post)

    x = rng.standard_normal(n_syn).astype(float)
    kbar = max(float(n_syn) / float(max(N_post, 1)), _EPS)
    return np.abs(x) * float(scale) / np.sqrt(kbar)

#Liquid層内のシナプス重みを初期化
def init_liq_intra(rng: np.random.Generator,n_syn: int,gain: float,N_post: int,) -> np.ndarray:
    n_syn = int(n_syn)
    N_post = int(N_post)

    x = rng.standard_normal(n_syn).astype(float)
    kbar = max(float(n_syn) / float(max(N_post, 1)), _EPS)
    return np.abs(x) * float(gain) / np.sqrt(kbar)
