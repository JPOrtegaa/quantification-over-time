"""
CLI entry point for textual quantification experiments.

Orchestration lives in `utils.main_functions_experiments` (`master_textual_experiment`,
`run_textual_experiments_grid`, `TextualExperimentConfig`).
"""

import argparse

from utils.main_functions_experiments import (
    TextualExperimentConfig,
    run_textual_experiments_grid,
)

# Default grid configuration (edit here or construct another TextualExperimentConfig).
DEFAULT_CONFIG = TextualExperimentConfig()


def _cli_log(msg: str) -> None:
    print(f"{DEFAULT_CONFIG.log_prefix} {msg}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Global Covid19 textual quantification experiments."
    )
    parser.add_argument(
        "--run",
        choices=["global_textual"],
        default="global_textual",
        help="Run global textual quantification experiment",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke test: 1 seed, DyS only, QFY only for original (+ TOMS with QFY). "
            "Writes MAE_quanti_results_mean_global_textual_quick.csv"
        ),
    )
    args = parser.parse_args()
    _cli_log(f"__main__: args.run={args.run!r}, quick={args.quick}")
    run_textual_experiments_grid(DEFAULT_CONFIG, quick=args.quick)
    _cli_log("Main run finished.")
