# Run Scripts

Use this folder for executable scripts only. Reusable logic should live in
`d_tools/`.

## Core Run

```text
run_liquid.py              liquid internal-state recording
```

## Analysis and Visualization

```text
run_separation_metrics.py  compute DR, Sb, Sw, SPpw, SPlin from internal states
plot_internal_states.py    redraw saved internal-state heatmaps
```

## Maintenance

```text
cleanup_generated.py       remove local generated caches only
```

Repository-level runs:

```powershell
python .\run_experiment.py exp_001_base
python .\run_experiment.py exp_001_base --target liquid
python .\f_run\run_liquid.py
```
