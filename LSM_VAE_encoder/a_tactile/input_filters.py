"""Tactile input filters used before injecting currents into the LSM."""

import numpy as np


DERIVATIVE_WIDTH = 1
DERIVATIVE_EMA_TAU_MS = 2.0
FAST_FILTER_TAU_S = 0.2 * 1e-3


def _ema(data, dt, tau_ms=DERIVATIVE_EMA_TAU_MS):
    alpha = 1.0 - np.exp(-float(dt) / (float(tau_ms) * 1e-3))
    smoothed = np.zeros_like(data, dtype=float)
    if len(data) == 0:
        return smoothed
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = smoothed[i - 1] + alpha * (data[i] - smoothed[i - 1])
    return smoothed


def _abs_derivative(data, t, i, width=DERIVATIVE_WIDTH):
    j = max(0, i - int(width))
    if j == i:
        return 0.0
    return np.abs(data[i] - data[j]) / (t[i] - t[j])


def _filter_components(data, t, dt):
    """Return RI, USI, SI, and their summed legacy Merkel-like response."""

    I = np.zeros((4, len(t)))
    data_smooth = _ema(data, dt)

    for i in range(len(t)):
        if i == 0:
            continue
        dF_dt = _abs_derivative(data_smooth, t, i)
        I[0, i] = I[0, i - 1] + 0.74 * dF_dt - I[0, i - 1] * dt / FAST_FILTER_TAU_S
        I[1, i] = (
            I[1, i - 1]
            + 0.24 * dF_dt
            - (I[1, i - 1] - 0.24 * 0.13) * dt / (5 * 1e-3)
        )
        I[2, i] = (
            I[2, i - 1]
            + 0.07 * dF_dt
            - I[2, i - 1] * dt / (80 * 1e-3)
        )
        I[3, i] = I[0, i] + I[1, i] + I[2, i]

    return I


def calc_RI(data, t, dt):
    return _filter_components(data, t, dt)[0, :]


def calc_USI(data, t, dt):
    return _filter_components(data, t, dt)[1, :]


def calc_SI(data, t, dt):
    return _filter_components(data, t, dt)[2, :]


def calc_meissner(data, t, dt):
    return calc_RI(data, t, dt)


def calc_merkel(data, t, dt):
    return _filter_components(data, t, dt)[3, :]
