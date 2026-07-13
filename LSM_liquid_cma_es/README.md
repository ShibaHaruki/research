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

CMA-ES search-space, objective defaults, and execution defaults are defined in
`c_configs/CMA_ES/cfg_search.py`. Edit that file to change the parameters,
their bounds, initial values, objective coefficients, generations, population
size, parallelism, or initial-center randomization. The first CMA-ES center is
sampled reproducibly from the configured search space using `--seed`.

All CMA-ES coordinates are normalized to `[0, 1]`; the original physical
values are restored from each parameter's `low` and `high` bounds.

Filter-specific search parameters are selected automatically from the active
`IN_ROUTE` entries. A filter route with zero input probability is excluded
from the CMA-ES search space.

By default, filter parameters are shared across sensors: for example,
`merkel_p` and `merkel_gain` are applied to every active Merkel route. This is
controlled by `share_filter_input_params_across_sensors` in `cfg_search.py`.

For ordinary network configuration, the same sharing can be enabled in
`c_configs/FIXED/cfg_network.py` by setting `SHARED_IN_ROUTE["enabled"]` to
`True` and editing one filter entry under `SHARED_IN_ROUTE["filters"]`.

The CMA-ES filter list is controlled by `search_input_filters` in
`c_configs/CMA_ES/cfg_search.py`. It currently includes Merkel, Meissner, RI,
SI, and USI, so their `p` and `gain` values are all searched. Set it to `None`
to return to automatic exclusion based on inactive `IN_ROUTE` probabilities.

Common commands:

```powershell
python -m pip install cma
python -m pip install mlflow
python .\run_experiment.py exp_001_base --dry-run
python .\run_experiment.py exp_001_base
python .\run_experiment.py exp_002_liquid_gain_sweep_example --trial liq_gain_1p0
python .\f_run\run_liquid.py
```

## MLflow tracking for CMA-ES

Install MLflow once, then add `--mlflow` to the CMA-ES command. The parent
run records the search settings and best objective. Each candidate is recorded
as a nested run with its decoded liquid parameters and evaluation metrics.

```powershell
python .\f_run\run_cma_es_search.py `
  --mlflow `
  --search-name liquid_accuracy_spikes_fisher_mlflow `
  --generations 5 `
  --population-size 8
```

By default, the local tracking database is created at
`g_tactile_results/mlflow.db`. To inspect it in the MLflow UI:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///./g_tactile_results/mlflow.db
```

Use `--mlflow-tracking-uri` for a shared MLflow server or another local
tracking directory, and `--mlflow-experiment` to choose the experiment name.

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
