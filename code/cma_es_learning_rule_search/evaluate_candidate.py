"""Adapter contract for one CMA-ES candidate.

This file is intentionally explicit: the existing training scripts hard-code
their constants, so they must be refactored to consume the candidate JSON
before this adapter can run a real experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    raise RuntimeError(
        "The candidate adapter is not connected yet. Apply this candidate to "
        "the selected *_training.py run_once() and then return JSON containing "
        "objective, accuracy8, and any other metrics. Candidate: "
        + json.dumps(candidate)
    )


if __name__ == "__main__":
    raise SystemExit(main())
