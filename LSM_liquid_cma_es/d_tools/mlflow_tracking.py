"""Small MLflow adapter for CMA-ES searches.

MLflow is imported lazily so the existing scripts keep working when tracking
is not enabled and the optional dependency is not installed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def _value_text(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 500 else text[:497] + "..."


def _scalar_metrics(values: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            metrics[str(key).replace(" ", "_")] = number
    return metrics


class MLflowSearchTracker:
    """Track one complete CMA-ES search and its candidate evaluations."""

    def __init__(
        self,
        *,
        tracking_uri: str | None,
        experiment_name: str,
        run_name: str | None,
        search_dir: Path,
    ) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow tracking was requested, but mlflow is not installed. "
                "Install it with: python -m pip install mlflow"
            ) from exc

        self.mlflow = mlflow
        self.search_dir = Path(search_dir)
        # Use SQLite because recent MLflow versions reject the legacy file store.
        default_db = (self.search_dir.parents[1] / "mlflow.db").resolve()
        self.tracking_uri = tracking_uri or f"sqlite:///{default_db.as_posix()}"
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._trial_step = 0
        self._best_objective = float("inf")
        self._logged_candidates: set[tuple[int, int, int]] = set()

    def start(self, search_settings: dict[str, Any]) -> None:
        self.mlflow.set_tracking_uri(self.tracking_uri)
        self.mlflow.set_experiment(self.experiment_name)
        self.mlflow.start_run(run_name=self.run_name)
        params = {
            f"search.{key}": _value_text(value)
            for key, value in search_settings.items()
            if key != "parameters"
        }
        params["search.parameter_names"] = ",".join(
            str(item["name"]) for item in search_settings.get("parameters", [])
        )
        self.mlflow.log_params(params)
        self.mlflow.set_tags({"tracker": "LSM_liquid_cma_es", "search_dir": str(self.search_dir)})

    def log_candidate(self, result: dict[str, Any]) -> None:
        generation = int(result.get("generation", 0))
        candidate = int(result.get("candidate", 0))
        start = int(result.get("start", 1))
        key = (start, generation, candidate)
        if key in self._logged_candidates:
            return
        self._logged_candidates.add(key)

        objective = result.get("objective")
        metrics = result.get("metrics") or {}
        parent_metric_values: dict[str, Any] = {
            "candidate_generation": generation,
            "candidate_index": candidate,
            "candidate_objective": objective,
            "candidate_best_objective": self._best_objective,
        }
        if objective is not None and math.isfinite(float(objective)):
            self._best_objective = min(self._best_objective, float(objective))
            parent_metric_values["candidate_best_objective"] = self._best_objective
        # Prefixing keeps the parent-run charts easy to find in MLflow.
        parent_metric_values.update(
            {
                f"candidate_{key}": value
                for key, value in metrics.items()
            }
        )
        parent_metric_values.update(
            {
                f"param_{key}": value
                for key, value in (result.get("params") or {}).items()
            }
        )
        parent_metrics = _scalar_metrics(parent_metric_values)
        if parent_metrics:
            self.mlflow.log_metrics(parent_metrics, step=self._trial_step)
        self._trial_step += 1

        params = {
            str(key): _value_text(value)
            for key, value in (result.get("params") or {}).items()
        }
        params.update({
            "start": str(start),
            "generation": str(generation),
            "candidate": str(candidate),
        })
        child_metrics = {"objective": objective}
        child_metrics.update(metrics)
        with self.mlflow.start_run(
            run_name=f"start{start:03d}_gen{generation:03d}_cand{candidate:03d}",
            nested=True,
        ):
            self.mlflow.log_params(params)
            scalar_metrics = _scalar_metrics(child_metrics)
            if scalar_metrics:
                self.mlflow.log_metrics(scalar_metrics)

    def log_remaining_candidates(self) -> None:
        """Import results produced by independent start worker processes."""
        for result_path in sorted(self.search_dir.rglob("candidate_result.json")):
            try:
                import json

                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            relative_parts = result_path.relative_to(self.search_dir).parts
            if relative_parts and relative_parts[0].startswith("start"):
                result["start"] = int(relative_parts[0].removeprefix("start"))
            self.log_candidate(result)

    def finish(self, best: dict[str, Any]) -> None:
        objective = best.get("objective")
        metrics = _scalar_metrics({"best_objective": objective})
        if metrics:
            self.mlflow.log_metrics(metrics)
        for filename in (
            "search_settings.json",
            "initial_centers.json",
            "best_params.json",
        ):
            path = self.search_dir / filename
            if path.exists():
                self.mlflow.log_artifact(str(path), artifact_path="search")
        for path in sorted(self.search_dir.rglob("cma_es_results.csv")):
            self.mlflow.log_artifact(str(path), artifact_path="search")

    def end(self, *, status: str = "FINISHED") -> None:
        if self.mlflow.active_run() is not None:
            self.mlflow.end_run(status=status)
