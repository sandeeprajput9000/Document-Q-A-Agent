"""Run the controlled benchmark and combined configuration grid."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.evaluation import load_questions
from src.experiments import run_experiments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--no-combined-grid", action="store_true", help="Run only baseline and one-factor-at-a-time")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings(args.config)
    questions = load_questions(settings.path("benchmark_file"))
    if len(questions) != 50:
        logging.error(
            "Expected 50 manually verified questions in %s; found %d. "
            "Review the draft before running experiments.",
            settings.path("benchmark_file"),
            len(questions),
        )
        return 2
    run_experiments(settings, questions, include_combined_grid=not args.no_combined_grid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

