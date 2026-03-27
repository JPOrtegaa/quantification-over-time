"""
CLI entry point for textual quantification experiments.

Builds ``TextualExperimentConfig`` from command-line flags and calls
``run_textual_experiments_grid``. Orchestration remains in
``utils.main_functions_experiments``.
"""

from __future__ import annotations

import argparse
from typing import Tuple

from utils.main_functions_experiments import (
    TextualExperimentConfig,
    run_textual_experiments_grid,
)

# Registered in data_loading_old.loading (textual + tabular).
DATASET_CHOICES = [
    "global_covid19_tweets",
    "nepali_dataset_eng",
    "Apple-Twitter-Sentiment-DFE",
    "bike",
    "energy",
    "news",
] + [f"hotel{i}" for i in range(1, 10)]

DEFAULT_CLASSIFIERS = "amansolanki/autonlp-Tweet-Sentiment-Extraction-20114061"


def _split_csv_nonempty(s: str) -> Tuple[str, ...]:
    parts = tuple(p.strip() for p in s.split(",") if p.strip())
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one comma-separated item")
    return parts


def _split_csv_int(s: str) -> Tuple[int, ...]:
    parts = _split_csv_nonempty(s)
    try:
        return tuple(int(x) for x in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid integer list: {s!r}") from e


def _build_tsmn_kwargs(args: argparse.Namespace) -> dict:
    kw = {"tsmn_mode": args.tsmn_mode, "tsmn_degree": args.tsmn_degree}
    if args.tsmn_period is not None:
        kw["tsmn_period"] = args.tsmn_period
    return kw


def build_config(args: argparse.Namespace) -> TextualExperimentConfig:
    if args.quick:
        seeds: Tuple[int, ...] = (1,)
        qua_methods = ("ACC",)
        tsa_methods = ("QFY",)
    else:
        seeds = args.seeds
        qua_methods = args.qua_methods
        tsa_methods = args.tsa_methods

    return TextualExperimentConfig(
        seeds=seeds,
        val_length=args.val_length,
        max_test_chunks=args.max_test_chunks,
        dataset_name=args.dataset,
        classifiers=args.classifiers,
        qua_methods=qua_methods,
        tsa_methods=tsa_methods,
        exp_types=args.exp_types,
        regressor_label=args.regressor_label,
        regressor_time_column=args.regressor_time_column,
        regressor_tsmn_kwargs=_build_tsmn_kwargs(args),
        unified_window=args.unified_window,
        log_prefix=args.log_prefix,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Grid of textual quantification experiments: seeds × exp_types × quantifiers "
            "× classifiers × (TSA when exp_type=original). "
            "Config is driven entirely from these flags (see also utils.main_functions_experiments.TextualExperimentConfig)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Datasets (--dataset): {", ".join(DATASET_CHOICES)}

Examples:
  python run_experiment.py --quick
  python run_experiment.py --dataset hotel3 --val-length 20 --seeds 1,2
  python run_experiment.py --dataset global_covid19_tweets \\
      --qua-methods DyS,DyS-Opt --exp-types TOMS,original \\
      --classifiers {DEFAULT_CLASSIFIERS}

With --quick, --seeds / --qua-methods / --tsa-methods are ignored (smoke preset).
""",
    )
    parser.add_argument(
        "--run",
        choices=["global_textual"],
        default="global_textual",
        help="Entrypoint (only the textual grid is implemented).",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="global_covid19_tweets",
        metavar="NAME",
        help="Key passed to data_loading.loading.",
    )
    parser.add_argument(
        "--seeds",
        type=_split_csv_int,
        metavar="S,...",
        default=(1, 2, 3),
        help="Comma-separated random seeds (ignored if --quick). Default: 1,2,3",
    )
    parser.add_argument(
        "--val-length",
        type=int,
        default=15,
        metavar="N",
        help="Number of first time chunks used as validation windows.",
    )
    parser.add_argument(
        "--max-test-chunks",
        type=int,
        default=5000,
        metavar="N",
        help="After val_length, cap the number of test chunks (truncation).",
    )
    parser.add_argument(
        "--classifiers",
        type=_split_csv_nonempty,
        default=_split_csv_nonempty(DEFAULT_CLASSIFIERS),
        metavar="ID,...",
        help="Comma-separated classifier ids (HF model names or 'vader' where supported).",
    )
    parser.add_argument(
        "--qua-methods",
        type=_split_csv_nonempty,
        default=_split_csv_nonempty("DyS,DyS-Opt"),
        metavar="NAME,...",
        help="Quantifiers (e.g. DyS, DyS-Opt, ACC, GPAC, EDy, CC, ReadMe2). Ignored if --quick.",
    )
    parser.add_argument(
        "--tsa-methods",
        type=_split_csv_nonempty,
        default=_split_csv_nonempty("QFY,MA,KFMA"),
        metavar="NAME,...",
        help="Temporal adjustment methods for exp_type=original (QFY, MA, KFMA). Ignored if --quick.",
    )
    parser.add_argument(
        "--exp-types",
        type=_split_csv_nonempty,
        default=_split_csv_nonempty("TOMS"),
        metavar="TYPE,...",
        help="Experiment modes: TOMS and/or original.",
    )
    parser.add_argument(
        "--regressor-label",
        default="TSMN",
        help="Regressor backend when training TOMS ('TSMN' or 'LR').",
    )
    parser.add_argument(
        "--regressor-time-column",
        default="TweetAt",
        metavar="COL",
        help="DataFrame column for times (use 'date' for hotel-*; Covid uses TweetAt).",
    )
    parser.add_argument(
        "--tsmn-mode",
        default="linear",
        choices=["linear", "polynomial", "cyclic", "identity"],
        help="TimeSeriesMultinomialRegressor feature mode (TOMS / TSMN).",
    )
    parser.add_argument(
        "--tsmn-degree",
        type=int,
        default=3,
        help="Polynomial degree when --tsmn-mode polynomial.",
    )
    parser.add_argument(
        "--tsmn-period",
        type=float,
        default=None,
        metavar="P",
        help="Period for cyclic mode (optional; default: model default).",
    )
    parser.add_argument(
        "--unified-window",
        type=int,
        default=4,
        metavar="W",
        help="Unified window parameter for initial prevalence slice.",
    )
    parser.add_argument(
        "--log-prefix",
        default="[run_experiment]",
        help="Prefix printed on log lines.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke test: seed=1, quantifier=ACC, TSA=QFY only (overrides --seeds, "
            "--qua-methods, --tsa-methods). Writes …_quick.csv."
        ),
    )

    args = parser.parse_args()
    cfg = build_config(args)

    def cli_log(msg: str) -> None:
        print(f"{cfg.log_prefix} {msg}", flush=True)

    cli_log(
        "__main__: "
        f"run={args.run!r}, dataset={args.dataset!r}, quick={args.quick}, "
        f"val_length={cfg.val_length}, seeds={cfg.seeds if not args.quick else '(quick:1)'}, "
        f"qua={cfg.qua_methods if not args.quick else '(quick:ACC)'}, "
        f"exp_types={cfg.exp_types}"
    )
    run_textual_experiments_grid(cfg, quick=args.quick)
    cli_log("Main run finished.")


if __name__ == "__main__":
    main()
