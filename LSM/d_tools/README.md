# Tools

Reusable implementation lives here. Keep executable scripts in `f_run/` and
configuration in `c_configs/`.

```text
compat.py                       compatibility helpers for older config names
experiments.py                  experiment expansion and override application
run_paths.py                    result folder naming and parameter snapshots
sample_sequence.py              shared sample order handling

internal_state.py               internal-state capture and save logic
internal_state_visualization.py internal-state heatmaps and overview plots
internal_state_vae.py           1D-CNN VAE, latent metrics, latent PCA plots
pca.py                          PCA for saved internal states
separation_metrics.py           DR, Sb, Sw, SPpw, SPlin metrics
cma_es_search.py                CMA-ES and objective calculation helpers

visualization.py                training/test debug plots
live_visualization.py           realtime plot viewers
plotting.py                     optional matplotlib import helper
weight_change.py                average weight-change tracking and plots
weight_export.py                dense weight export helpers
```

Rule of thumb:

```text
f_run/       script entry points and CLI parsing
d_tools/     reusable functions/classes
c_configs/   parameters and search definitions
g_tactile_results/ generated outputs only
```
