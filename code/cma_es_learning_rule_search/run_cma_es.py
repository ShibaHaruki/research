"""CMA-ES driver for learning-rule and output-connectivity parameters."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import cma
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit("pycma is required: python -m pip install cma") from exc

from search_config import candidate_from_vector, parameters_for


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", choices=["off", "STDP", "SRDP", "T_STDP"], required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--search-name", default="search_001")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--sigma0", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


def _encode(spec, value: float) -> float:
    value = float(np.clip(value, spec.low, spec.high))
    if spec.kind == "log10":
        return float(np.log10(value))
    return value


def _decode(spec, value: float) -> float:
    if spec.kind == "log10":
        value = 10.0 ** float(value)
    return float(np.clip(value, spec.low, spec.high))


def _evaluate(evaluator: Path, candidate: dict, output_dir: Path, seed: int) -> dict:
    candidate_path = output_dir / "candidate.json"
    candidate_path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    command = [sys.executable, str(evaluator), "--candidate", str(candidate_path), "--output-dir", str(output_dir), "--seed", str(seed)]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Evaluator printed no JSON result.")
    result = json.loads(lines[-1])
    if "objective" not in result:
        raise RuntimeError("Evaluator result must contain 'objective'.")
    return result


def main() -> int:
    args = parse_args()
    specs = parameters_for(args.rule)
    rng = np.random.default_rng(args.seed)
    center = np.array([_encode(spec, spec.initial) for spec in specs], dtype=float)
    low = np.array([_encode(spec, spec.low) for spec in specs], dtype=float)
    high = np.array([_encode(spec, spec.high) for spec in specs], dtype=float)
    root = Path(__file__).resolve().parent / "results" / args.search_name
    root.mkdir(parents=True, exist_ok=True)
    (root / "search_config.json").write_text(json.dumps({"rule": args.rule, "parameters": [spec.__dict__ for spec in specs], "generations": args.generations, "population_size": args.population_size, "sigma0": args.sigma0, "seed": args.seed}, indent=2), encoding="utf-8")

    rows = []
    best = None
    options = {
        "bounds": [low.tolist(), high.tolist()],
        "popsize": args.population_size,
        "maxiter": args.generations,
        "seed": args.seed,
        "verb_disp": 0,
    }
    es = cma.CMAEvolutionStrategy(center.tolist(), args.sigma0, options)
    # The evaluator is intentionally external so every candidate can be run
    # through the same training and evaluation pipeline as the existing code.
    generation = 0
    while not es.stop():
        generation += 1
        population = es.ask()
        evaluated = []
        for candidate_index, point in enumerate(population, 1):
            candidate = candidate_from_vector(args.rule, [_decode(spec, value) for spec, value in zip(specs, point)])
            candidate_dir = root / f"gen{generation:03d}_cand{candidate_index:03d}"
            result = _evaluate(args.evaluator, candidate, candidate_dir, args.seed + generation * 1000 + candidate_index)
            row = {"generation": generation, "candidate": candidate_index, **candidate, **result}
            rows.append(row)
            evaluated.append((float(result["objective"]), point))
            if best is None or float(result["objective"]) < best["objective"]:
                best = {"objective": float(result["objective"]), "generation": generation, "candidate": candidate_index, "params": candidate, "metrics": result}
            print(f"[cma] rule={args.rule} gen={generation} cand={candidate_index} objective={result['objective']}")
        es.tell([point for _, point in evaluated], [objective for objective, _ in evaluated])

    with (root / "cma_es_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (root / "best_candidate.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"[saved] {root / 'cma_es_results.csv'}")
    print(f"[best] {json.dumps(best, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
