"""Validate saved CMA-ES candidates, then resume a single-start search.

Run with the original search environment and unchanged model/data files.
--check-only performs no simulation and writes no files.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path, value, search):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(search.jsonable(value), indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_results(root, population):
    results = {}
    for path in sorted(root.glob("gen*_cand*/candidate_result.json")):
        result = read_json(path)
        key = (int(result["generation"]), int(result["candidate"]))
        if key[0] < 1 or not 1 <= key[1] <= population:
            raise ValueError(f"Invalid candidate number: {path}")
        if path.parent.name != f"gen{key[0]:03d}_cand{key[1]:03d}" or key in results:
            raise ValueError(f"Invalid or duplicate candidate: {path}")
        float(result["objective"])
        results[key] = result
    if not results:
        raise ValueError("No saved candidate results found")
    last = max(g for g, _ in results)
    for generation in range(1, last):
        if any((generation, c) not in results for c in range(1, population + 1)):
            raise ValueError(f"Generation {generation} is incomplete but later results exist")
    return results, last


def verify_candidate(search, raw, result):
    import numpy as np

    expected = search.decode_vector(np.asarray(raw, dtype=float))
    actual = result["params"]
    if set(expected) != set(actual) or any(
        not np.isclose(float(value), float(actual[name]), rtol=1e-10, atol=1e-9)
        for name, value in expected.items()
    ):
        raise ValueError(
            f"Candidate mismatch at gen={result['generation']} cand={result['candidate']}. "
            "Nothing will be recalculated; check original code, seed, sigma0 and package versions."
        )


def save_csv(root, results, search):
    rows = []
    fields = []
    for key in sorted(results):
        result = results[key]
        row = {"start": 1, "generation": key[0], "candidate": key[1], "objective": result["objective"]}
        for name, value in (result.get("metrics") or {}).items():
            row[name] = json.dumps(search.jsonable(value), ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
        row.update(result["params"])
        fields.extend(name for name in row if name not in fields)
        rows.append(row)
    path = root / "cma_es_results.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    import numpy as np

    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from f_run import run_cma_es_search as search

    check_only = "--check-only" in sys.argv
    sys.argv = [arg for arg in sys.argv if arg != "--check-only"]
    args = search.parse_args()
    if args.dry_run or args.mlflow or int(args.n_starts) != 1:
        raise ValueError("This recovery script supports one start, without dry-run or MLflow")
    root = search.CMA_DIR / args.search_name
    settings = read_json(root / "search_settings.json")
    # Old searches predate this option and used normal CMA sampling. Recover
    # the recorded initialization unless the caller explicitly overrides it.
    if not any(arg.split("=", 1)[0] == "--initial-population-distribution" for arg in sys.argv):
        args.initial_population_distribution = settings.get("initial_population_distribution", "gaussian")
    current = search.jsonable(search.build_search_settings(args))
    if "initial_population_distribution" not in settings:
        if args.initial_population_distribution != "gaussian":
            raise ValueError("Saved search used gaussian initialization; uniform cannot replay it")
        current.pop("initial_population_distribution")
    if current != settings:
        differing = sorted(key for key in set(settings) | set(current) if settings.get(key) != current.get(key))
        raise ValueError(f"Search settings differ from saved settings: {differing}")
    centers = read_json(root / "initial_centers.json")
    if len(centers) != 1 or int(centers[0]["start"]) != 1:
        raise ValueError("Expected one saved initial center")
    x0 = np.asarray(centers[0]["raw_x0"], dtype=float)
    if len(x0) != len(search.PARAMS):
        raise ValueError("Initial center dimension mismatch")
    population = int(args.population_size)
    results, last = load_results(root, population)
    if int(args.generations) < last:
        raise ValueError("Target generations is less than saved generations")
    strategy = search.cma.CMAEvolutionStrategy(x0.tolist(), float(args.sigma0), {
        "popsize": population, "seed": int(args.seed) + 100000,
        "verb_disp": 0, "verbose": -9,
    })
    best_x = x0.copy()
    best_obj = float("inf")
    best_result = None

    def tell(generation, solutions):
        nonlocal best_x, best_obj, best_result
        objectives = []
        for candidate, raw in enumerate(solutions, 1):
            result = results[(generation, candidate)]
            objective = float(result["objective"])
            objectives.append(objective if math.isfinite(objective) else 1e300)
            if objective < best_obj:
                best_x, best_obj, best_result = np.asarray(raw).copy(), objective, result
        strategy.tell(solutions, objectives)

    # Verify ALL saved generations before starting any expensive evaluation.
    pending = None
    for generation in range(1, last + 1):
        solutions = search.ask_search_population(
            strategy, generation=generation, seed=int(args.seed) + 100000,
            initial_population_distribution=args.initial_population_distribution,
        )
        count = 0
        for candidate, raw in enumerate(solutions, 1):
            result = results.get((generation, candidate))
            if result is not None:
                verify_candidate(search, raw, result)
                count += 1
        print(f"[replay] generation={generation}: verified {count}/{population}", flush=True)
        if count == population:
            tell(generation, solutions)
        else:
            pending = (generation, solutions)
    next_generation = pending[0] if pending else last + 1
    print(f"[resume] saved={len(results)} next_generation={next_generation} target={args.generations}", flush=True)
    if check_only:
        print("[check-only] Verification passed. No files changed and no simulations run.")
        return 0

    # CSV is derived from candidate JSON; retain the previous CSV before repair.
    csv_path = root / "cma_es_results.csv"
    backup = root / "cma_es_results.before_resume.csv"
    if csv_path.exists() and not backup.exists():
        with backup.open("xb") as handle:
            handle.write(csv_path.read_bytes())
    save_csv(root, results, search)
    jobs = max(1, min(int(args.jobs or population), population, search._max_process_workers()))
    for generation in range(next_generation, int(args.generations) + 1):
        solutions = pending[1] if pending and generation == pending[0] else search.ask_search_population(
            strategy, generation=generation, seed=int(args.seed) + 100000,
            initial_population_distribution=args.initial_population_distribution,
        )
        payloads = [
            (np.asarray(raw, dtype=float), generation, candidate, str(root), args)
            for candidate, raw in enumerate(solutions, 1)
            if (generation, candidate) not in results
        ]
        # Separate workers keep simulator RNG changes out of CMA's RNG state.
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(search.evaluate_candidate_worker, payload): payload[2] for payload in payloads}
            for future in as_completed(futures):
                result = future.result()
                candidate = futures[future]
                if (int(result["generation"]), int(result["candidate"])) != (generation, candidate):
                    raise ValueError("Worker returned the wrong candidate")
                verify_candidate(search, solutions[candidate - 1], result)
                results[(generation, candidate)] = result
                save_csv(root, results, search)
                print(f"[cma] gen={generation} cand={candidate} objective={result['objective']}", flush=True)
        tell(generation, solutions)
        best = {
            "start": 1, "objective": best_obj, "params": search.decode_vector(best_x),
            "initial_center": search.decode_vector(x0), "raw_initial_center": x0.tolist(),
            "objective_settings": settings,
            "metrics": None if best_result is None else best_result.get("metrics"),
        }
        atomic_json(root / "best_params.json", best, search)
    print(f"[resume] Finished. Best objective={best_obj}; results={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
