"""Firing rates and distinct per-trial / across-trial inactivity metrics."""

from __future__ import annotations

import numpy as np


def summarize_activity(spike_counts, *, bin_ms: float, neuron_ids=None) -> dict:
    """Summarize trial x neuron x time-bin counts for selected neurons.

    neuron_ids selects columns and is also used to label never-active neurons.
    Rate arrays are in Hz; percentage fields range from 0 to 100.
    """
    bins = np.asarray(spike_counts)
    if bins.ndim != 3 or any(size == 0 for size in bins.shape):
        raise ValueError("Expected nonempty (trials, neurons, time_bins) spike counts")
    if not np.isfinite(bins).all() or np.any(bins < 0):
        raise ValueError("Spike counts must be finite and nonnegative")
    if not np.isfinite(bin_ms) or bin_ms <= 0:
        raise ValueError("bin_ms must be finite and positive")
    ids = np.arange(bins.shape[1]) if neuron_ids is None else np.asarray(neuron_ids)
    if (
        ids.ndim != 1
        or ids.size == 0
        or not np.issubdtype(ids.dtype, np.integer)
        or np.any(ids < 0)
        or np.any(ids >= bins.shape[1])
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("neuron_ids must be nonempty, unique valid integer neuron indices")
    counts = bins[:, ids].sum(2)
    duration = bins.shape[2] * float(bin_ms) / 1000.0
    silent = counts == 0
    never = np.all(silent, axis=0)
    return {
        "n_trials": len(counts),
        "n_neurons": len(ids),
        "duration_s": duration,
        "neuron_ids": ids,
        "trial_neuron_rate_hz": counts / duration,
        "neuron_mean_rate_hz": counts.mean(0) / duration,
        "trial_mean_rate_hz": counts.mean(1) / duration,
        "mean_rate_hz": float(counts.mean() / duration),
        "silent_neuron_count_per_trial": silent.sum(1),
        "silent_neuron_percent_per_trial": silent.mean(1) * 100,
        "mean_silent_neuron_percent": float(silent.mean() * 100),
        "all_active_trials": int(np.all(~silent, axis=1).sum()),
        "never_active_neuron_ids": ids[never].tolist(),
        "never_active_neuron_count": int(never.sum()),
        "never_active_neuron_percent": float(never.mean() * 100),
    }


def summarize_ei_activity(spike_counts, *, n_excitatory: int, bin_ms: float) -> dict:
    """E-first/I-second summary retaining the calibration decision fields."""
    bins = np.asarray(spike_counts)
    if (
        bins.ndim != 3
        or not isinstance(n_excitatory, (int, np.integer))
        or not 0 < n_excitatory < bins.shape[1]
    ):
        raise ValueError("Expected E-first/I-second states with nonempty E and I groups")
    exc = summarize_activity(bins, bin_ms=bin_ms, neuron_ids=np.arange(n_excitatory))
    inh = summarize_activity(bins, bin_ms=bin_ms, neuron_ids=np.arange(n_excitatory, bins.shape[1]))
    return {
        "n_trials": exc["n_trials"],
        "mean_E_rate_hz": exc["mean_rate_hz"],
        "mean_I_rate_hz": inh["mean_rate_hz"],
        "all_E_trials": exc["all_active_trials"],
        "all_I_trials": inh["all_active_trials"],
        "never_active_E": exc["never_active_neuron_ids"],
        "never_active_I": inh["never_active_neuron_ids"],
        "mean_silent_E_percent": exc["mean_silent_neuron_percent"],
    }
