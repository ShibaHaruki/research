"""Weight initialization for input-to-liquid and liquid recurrent synapses."""

import numpy as np

_EPS = 1e-12


def init_in_to_liq(
    rng: np.random.Generator,
    n_syn: int,
    scale: float,
    N_post: int,
) -> np.ndarray:
    """Initialize non-negative input weights with fan-in normalization."""
    n_syn = int(n_syn)
    N_post = int(N_post)

    x = rng.standard_normal(n_syn).astype(float)
    kbar = max(float(n_syn) / float(max(N_post, 1)), _EPS)
    return np.abs(x) * float(scale) / np.sqrt(kbar)


def init_liq_intra(
    rng: np.random.Generator,
    n_syn: int,
    gain: float,
    N_post: int,
) -> np.ndarray:
    n_syn = int(n_syn)
    N_post = int(N_post)

    x = rng.standard_normal(n_syn).astype(float)
    kbar = max(float(n_syn) / float(max(N_post, 1)), _EPS)
    return np.abs(x) * float(gain) / np.sqrt(kbar)
