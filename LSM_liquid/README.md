# LSM Liquid Project Layout

This repository now runs the network only up to the liquid layer.

```text
LSM/
  a_tactile/          input data definitions and tactile filters
  b_network/          liquid network construction and weight initialization
  c_configs/          fixed base configs and liquid experiment definitions
  d_tools/            reusable tools for saving, plotting, paths, experiments
  e_evaluation/       notebooks and analysis artifacts
  f_run/              runnable liquid scripts
  g_tactile_results/  generated results, manifests, weights, debug plots
  run_experiment.py   main liquid experiment runner
```

Use `c_configs/FIXED` as the base configuration. Put liquid parameter sweeps
and trial-specific overrides in `c_configs/EXPERIMENTS`.

Common commands:

```powershell
python .\run_experiment.py exp_001_base --dry-run
python .\run_experiment.py exp_001_base
python .\run_experiment.py exp_002_liquid_gain_sweep_example --trial liq_gain_1p0
python .\f_run\run_liquid.py
```

Generated cache cleanup:

```powershell
python .\f_run\cleanup_generated.py
python .\f_run\cleanup_generated.py --apply
```

Result folders are written as:

```text
g_tactile_results/
  sample_seq/       shared sample orders
  liquid_run/       liquid internal states, PCA, VAE artifacts
  _runtime_cache/   local temporary cache, safe to regenerate
```

Network folder names include the visible liquid topology:

```text
Nliq_<sizes>__in2liq_<liquid targets>__liqRec_<recurrent layers>__<connection>__<hash>
```
