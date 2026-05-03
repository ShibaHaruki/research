# LSM Project Layout

This repository is organized by role:

```text
LSM/
  a_tactile/          input data definitions and tactile filters
  b_network/          network construction, model registry, weight initialization
  c_configs/          fixed base configs and experiment definitions
  d_tools/            reusable tools for saving, plotting, paths, experiments
  e_evaluation/       notebooks and analysis outputs
  f_run/              runnable training/liquid scripts
  g_tactile_results/  generated results, manifests, weights, debug plots
  run_experiment.py   main experiment runner
```

Use `c_configs/FIXED` as the base configuration. Put parameter sweeps and
trial-specific overrides in `c_configs/EXPERIMENTS`.

Common commands:

```powershell
python .\run_experiment.py exp_001_base --dry-run
python .\run_experiment.py exp_001_base
python .\run_experiment.py exp_002_gain_sweep_example --trial liq_out_gain_1p0
python .\run_experiment.py exp_001_base --target liquid
```

Recent focused runners:

```powershell
python .\f_run\run_training.py
python .\f_run\run_liquid.py
```

VAE / CMA-ES parameter search has been split into a sibling project:

```text
..\LSM_VAE_Search\
```

Keep this LSM folder for the standard training, liquid-state recording, test,
and evaluation workflow. Use `..\LSM_VAE_Search` when training a shared VAE
latent space or running CMA-ES with the fixed Encoder.

Generated cache cleanup:

```powershell
python .\f_run\cleanup_generated.py
python .\f_run\cleanup_generated.py --apply
```

The cleanup script only targets generated Python caches, notebook checkpoints,
and `g_tactile_results/_runtime_cache`. It does not remove experiment results
such as `training_run`, `liquid_run`, or `cma_es_search`.

Result folders are written as:

```text
g_tactile_results/
  training_run/
    <network_structure>/
      network_params.json
      <model_name>/
        model_params.json
        <experiment_id>/
```

Network folder names include the visible layer topology:

```text
Nliq_<sizes>__Nout_<sizes>__in2liq_<liquid targets>__liqRec_<recurrent layers>__liq2out_<routes>__<connection>__<hash>
```

Example:

```text
Nliq_10-1000__Nout_100__in2liq_L1-L2__liqRec_L1-L2__liq2out_L1-L2_to_O1__random__d70500c2bb
```

Each network folder saves the parameters shared by that network in
`network_params.json`. Each model folder saves the selected neuron/synapse/
learning-rule parameters in `model_params.json`.

Each experiment run saves `config_snapshot.json`, `experiment_trial.json`,
spike counts, debug plots, weight matrices, and weight-change plots when
available.

Current result folders:

```text
g_tactile_results/
  sample_seq/       shared sample orders
  training_run/     training weights, debug plots, weight changes
  liquid_run/       liquid internal states, PCA, VAE outputs
  _runtime_cache/   local temporary cache, safe to regenerate
```
