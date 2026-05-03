# Run Scripts

Use this folder for executable scripts only. Reusable logic should live in
`d_tools/`.

## Core runs

```text
run_training.py            train liquid + output layers
run_liquid.py              liquid-only internal-state recording
run_test.py                test with saved training weights
run_train_test.py          training + test
run_train_test_eval.py     training + test + classification evaluation
```

## Analysis and visualization

```text
run_test_classification.py classify saved test spike records
run_separation_metrics.py  compute DR, Sb, Sw, SPpw, SPlin from internal states
plot_internal_states.py    redraw saved internal-state heatmaps
```

## Parameter search

```text
..\LSM_VAE_Search\         VAE and CMA-ES search project
```

VAE training, fixed Encoder creation, CMA-ES search, and CMA-ES progress plots
are separated into the sibling `LSM_VAE_Search` folder.

## Maintenance

```text
cleanup_generated.py       remove local generated caches only
```

For normal experiment management, prefer the repository-level runner when it
covers the task:

```powershell
python .\run_experiment.py exp_001_base
python .\run_experiment.py exp_001_base --target liquid
python .\run_experiment.py exp_001_base --target test
python .\run_experiment.py exp_001_base --target train_test
```
