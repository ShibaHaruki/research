"""Figure export separated from simulation, classification, and evaluation."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def save_confusion_matrix(counts, labels, path, *, title: str, dpi=150) -> Path:
    """Save counts and true-class percentages using a fixed 0–100% color bar."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from d_tools.evaluation import confusion_percentages
    cm = np.asarray(counts)
    pct = confusion_percentages(cm)
    labels = list(labels)
    if len(labels) != len(cm):
        raise ValueError('One label is required for each confusion-matrix row')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 9), layout='constrained')
    try:
        im = ax.imshow(pct, cmap='Blues', vmin=0, vmax=100)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f'{cm[i,j]}\n{pct[i,j]:.1f}%', ha='center', va='center',
                        color='white' if pct[i,j] > 50 else 'black', fontsize=8)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right')
        ax.set_yticks(range(len(labels)), labels)
        ax.set(xlabel='Predicted material', ylabel='True material', title=title)
        fig.colorbar(im, ax=ax, label='Percentage of true class (%)')
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)
    return path


def save_liquid_debug(data, path, *, title, n_excitatory=80):
    """代表1試行の全ニューロン発火・集団発火率・選択ニューロン膜電位を保存する。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    counts = np.asarray(data['counts_25ms'])
    bin_ms = float(data['bin_ms'])
    times = np.asarray(data['spike_times_ms'])
    ids = np.asarray(data['spike_neuron_ids'])
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout='constrained')
    try:
        for mask, color, label in [(ids < n_excitatory, 'tab:blue', 'E'),
                                    (ids >= n_excitatory, 'tab:red', 'I')]:
            axes[0].scatter(times[mask], ids[mask], s=3, color=color, label=label)
        axes[0].axhline(n_excitatory-.5, color='gray', linewidth=.7)
        axes[0].set(ylabel='Neuron index', ylim=(-1, counts.shape[0]))
        axes[0].legend(loc='upper right')
        centers = (np.arange(counts.shape[1])+.5)*bin_ms
        for block, color, label in [(counts[:n_excitatory], 'tab:blue', 'E'),
                                     (counts[n_excitatory:], 'tab:red', 'I')]:
            if len(block):
                axes[1].plot(centers, block.mean(axis=0)/(bin_ms/1000), color=color, label=label)
        axes[1].set(ylabel='Mean firing rate (Hz)')
        axes[1].legend(loc='upper right')
        for neuron, trace in zip(data['voltage_neuron_ids'], data['voltage_mV']):
            axes[2].plot(data['voltage_times_ms'], trace, linewidth=.7, label=f'N{neuron}')
        axes[2].set(xlabel='Time within trial (ms)', ylabel='Membrane potential (mV)',
                    xlim=(0, counts.shape[1]*bin_ms))
        axes[2].legend(ncol=5, fontsize=7, loc='upper right')
        fig.suptitle(title)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
    return path
