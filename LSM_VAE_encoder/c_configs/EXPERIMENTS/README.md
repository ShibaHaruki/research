# Experiment Configs

Put parameter sweeps and trial-specific overrides here.

Each experiment module should define `EXPERIMENT`:

```python
EXPERIMENT = {
    "name": "exp_001_base",
    "target": "training",
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
"models.LEARNING_RULE_MODEL": "STDP"
"training.NUM_TRAINING_SAMPLE": [100]
```
