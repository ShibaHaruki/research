# Learning-rule CMA-ES search

This folder is the entry point for searching learning-rule parameters together
with liquid-to-output connectivity parameters.

The search is deliberately separated from the existing `off`, `STDP`, `SRDP`,
and `T_STDP` folders. A candidate is represented by a JSON file and is passed
to an evaluator. The evaluator must run training/testing for that candidate
and print a JSON object containing at least `objective` and `accuracy8`.

Example command:

```powershell
python .\code\cma_es_learning_rule_search\run_cma_es.py `
  --rule SRDP `
  --evaluator .\code\cma_es_learning_rule_search\evaluate_candidate.py `
  --generations 10 `
  --population-size 10
```

The current files define the CMA-ES interface and search space. The existing
training scripts still contain hard-coded constants, so the evaluator must
apply the JSON candidate to `run_once()` before a real search is started.
This prevents a false search in which CMA-ES changes numbers that never reach
the Brian2 model.

Results are written to `code/cma_es_learning_rule_search/results/<search_name>`:

- `cma_es_results.csv`: one row per candidate
- `best_candidate.json`: best candidate and objective
- `search_config.json`: immutable search settings
