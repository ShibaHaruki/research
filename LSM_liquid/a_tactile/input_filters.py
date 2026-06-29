"""Tactile waveform filters for Merkel / Meissner input currents."""

from __future__ import annotations
import numpy as np


DERIVATIVE_WIDTH = 1

COMPONENT_NAMES = ("RI", "SI", "USI")


EXP_FILTER_tau = 2.0 * 1e-3

RI_tau = 8.0 * 1e-3
SI_tau = 200 * 1e-3
USI_tau = 1744.6 * 1e-3

RI_gain = 0.74
SI_gain = 0.24
USI_gain = 0.07

SENSOR_GAIN = {
    0: 1 / 1.19,
    1: 1 / 2.18,
    2: 1 / 2.03,
}


FILTER_GAIN = {
    "RI": (1 / 5.28)*70,
    "SI": (1 / 22.57)*20*7,
    "USI": (1 / 532.04)*20,
    "merkel": 0.008,
    "meissner": 0.0876,
}

def _exp_filter(data, dt):
    filtered = np.zeros_like(data, dtype=float)
    if len(data) == 0:
        return filtered

    filtered[0] = data[0]
    alpha = dt / EXP_FILTER_tau
    for i in range(1, len(data)):
        filtered[i] = filtered[i - 1] + alpha * (data[i] - filtered[i - 1])
    return filtered

def _abs_derivative(data, t, i, width=DERIVATIVE_WIDTH):
    j = max(0, i - int(width))
    if j == i:
        return 0.0
    return np.abs(data[i] - data[j]) / (t[i] - t[j])

def _calc_components(data, t, dt):
    components = np.zeros((3, len(t)))
    filtered_data = _exp_filter(data, dt)

    for i in range(len(t)):
        if i == 0:
            continue

        dF_dt = _abs_derivative(filtered_data, t, i)
        components[0, i] = _RI_step(components, i, dF_dt, dt)
        components[1, i] = _SI_step(components, i, dF_dt, dt)
        components[2, i] = _USI_step(components, i, dF_dt, dt)

    return components

def _RI_step(components, i, dF_dt, dt):
    return components[0, i - 1] + dF_dt + (-components[0, i - 1] * dt / RI_tau)


def _SI_step(components, i, dF_dt, dt):
    return components[1, i - 1] +  dF_dt + (-(components[1, i - 1] - 0.24 * 0.13) * dt / SI_tau)


def _USI_step(components, i, dF_dt, dt):
    return components[2, i - 1] + dF_dt + (-components[2, i - 1] * dt / USI_tau)


def calc_meissner_components(data, t, dt):
    components = _calc_components(data, t, dt)
    return {name: components[index, :] for index, name in enumerate(COMPONENT_NAMES)}


def calc_merkel_components(data, t, dt):
    components = _calc_components(data, t, dt)
    return {name: components[index, :] for index, name in enumerate(COMPONENT_NAMES)}


def RI(data, t, dt):
    return _calc_components(data, t, dt)[0, :]


def SI(data, t, dt):
    return _calc_components(data, t, dt)[1, :]


def USI(data, t, dt):
    return _calc_components(data, t, dt)[2, :]


def calc_meissner(data, t, dt):
    return RI_gain * RI(data, t, dt)


def calc_merkel(data, t, dt):
    return RI_gain * RI(data, t, dt) + SI_gain * SI(data, t, dt) + USI_gain * USI(data, t, dt)
