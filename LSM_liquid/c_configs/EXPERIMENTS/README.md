# Experiment Configs

Put liquid parameter sweeps and trial-specific overrides here.

Each experiment module should define `EXPERIMENT`:

```python
EXPERIMENT = {
    "name": "exp_001_base",
    "target": "liquid",
    "trials": [
        {
            "id": "base",
            "overrides": {},
        },
    ],
}
```

Override paths use dot notation:

```python
"network.N_liq": [100, 1000]
"network.liq_intra_gain_pairs.EE": 0.5
"liquid.NUM_LIQUID_SAMPLE": 100
```
