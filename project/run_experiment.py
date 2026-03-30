"""
Experiment entry point. Edit ``VARIABLES.py``, then from ``project/``::

    python run_experiment.py

* Textual series: ``utils.main_functions_experiments``.
* Tabular (bike, energy, news): ``utils.tubular_experiments``.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import VARIABLES as V
from utils.main_functions_experiments import (
    TextualExperimentConfig,
    run_textual_experiments_grid,
)
from utils.tubular_experiments import run_tubular_experiments_grid

DATASET_CHOICES = [
    "global_covid19_tweets",
    "nepali_dataset_eng",
    "Apple-Twitter-Sentiment-DFE",
    "bike",
    "energy",
    "news",
] + [f"hotel{i}" for i in range(1, 10)]

TUBULAR_DATASETS = frozenset({"bike", "energy", "news"})
_VALID_RUN = frozenset({"global_textual", "tubular", "auto"})

_VALID_TSMN_MODES = frozenset({"linear", "polynomial", "cyclic", "identity"})
_VALID_REGRESSOR_TIME_ENCODING = frozenset({"scalar", "week"})


def _resolve_run_mode() -> str:
    if V.RUN not in _VALID_RUN:
        raise ValueError(
            f"VARIABLES.RUN must be one of {sorted(_VALID_RUN)}, got {V.RUN!r}"
        )
    if V.RUN == "auto":
        return "tubular" if V.DATASET in TUBULAR_DATASETS else "global_textual"
    if V.RUN == "tubular":
        if V.DATASET not in TUBULAR_DATASETS:
            raise ValueError(
                f"RUN='tubular' requires VARIABLES.DATASET in {sorted(TUBULAR_DATASETS)}, "
                f"got {V.DATASET!r}"
            )
        return "tubular"
    if V.DATASET in TUBULAR_DATASETS:
        raise ValueError(
            f"DATASET={V.DATASET!r} is tabular: set VARIABLES.RUN to 'tubular' or 'auto'."
        )
    return "global_textual"


def _tsmn_kwargs() -> dict:
    mode = V.TSMN_MODE
    if mode not in _VALID_TSMN_MODES:
        raise ValueError(
            f"VARIABLES.TSMN_MODE must be one of {sorted(_VALID_TSMN_MODES)}, got {mode!r}"
        )
    kw: dict = {"tsmn_mode": mode, "tsmn_degree": int(V.TSMN_DEGREE)}
    if V.TSMN_PERIOD is not None:
        kw["tsmn_period"] = float(V.TSMN_PERIOD)
    return kw


def _as_tuple_str(name: str, value) -> Tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"VARIABLES.{name} must be a tuple/list of strings, not a single str")
    return tuple(str(x).strip() for x in value if str(x).strip())


def _as_tuple_int(name: str, value) -> Tuple[int, ...]:
    return tuple(int(x) for x in value)


def build_textual_config() -> TextualExperimentConfig:
    if V.DATASET not in DATASET_CHOICES:
        raise ValueError(
            f"VARIABLES.DATASET={V.DATASET!r} is not valid. "
            f"Choose one of: {', '.join(DATASET_CHOICES)}"
        )

    enc = str(getattr(V, "REGRESSOR_TIME_ENCODING", "scalar"))
    if enc not in _VALID_REGRESSOR_TIME_ENCODING:
        raise ValueError(
            f"VARIABLES.REGRESSOR_TIME_ENCODING must be one of {sorted(_VALID_REGRESSOR_TIME_ENCODING)}, "
            f"got {enc!r}"
        )

    if V.QUICK:
        seeds: Tuple[int, ...] = (1,)
        qua_methods = ("ACC",)
        tsa_methods = ("QFY",)
    else:
        seeds = _as_tuple_int("SEEDS", V.SEEDS)
        qua_methods = _as_tuple_str("QUA_METHODS", V.QUA_METHODS)
        tsa_methods = _as_tuple_str("TSA_METHODS", V.TSA_METHODS)

    return TextualExperimentConfig(
        seeds=seeds,
        val_length=int(V.VAL_LENGTH),
        max_test_chunks=int(V.MAX_TEST_CHUNKS),
        dataset_name=V.DATASET,
        classifiers=_as_tuple_str("CLASSIFIERS", V.CLASSIFIERS),
        qua_methods=qua_methods,
        tsa_methods=tsa_methods,
        exp_types=_as_tuple_str("EXP_TYPES", V.EXP_TYPES),
        regressor_label=str(V.REGRESSOR_LABEL),
        regressor_time_column=str(V.REGRESSOR_TIME_COLUMN),
        regressor_time_encoding=enc,
        regressor_tsmn_kwargs=_tsmn_kwargs(),
        unified_window=int(V.UNIFIED_WINDOW),
        log_prefix=str(V.LOG_PREFIX),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quantification grid driven by VARIABLES.py (no experiment flags on the CLI)."
        ),
    )
    parser.parse_args()

    if V.DATASET not in DATASET_CHOICES:
        raise ValueError(
            f"VARIABLES.DATASET={V.DATASET!r} is not valid. "
            f"Choose one of: {', '.join(DATASET_CHOICES)}"
        )

    mode = _resolve_run_mode()
    log_prefix = str(V.LOG_PREFIX)

    def cli_log(msg: str) -> None:
        print(f"{log_prefix} {msg}", flush=True)

    if mode == "tubular":
        seeds = (1,) if V.QUICK else _as_tuple_int("SEEDS", V.SEEDS)
        qua_m = ("ACC",) if V.QUICK else _as_tuple_str("QUA_METHODS", V.QUA_METHODS)
        tsa_m = ("QFY",) if V.QUICK else _as_tuple_str("TSA_METHODS", V.TSA_METHODS)
        clf = _as_tuple_str("CLASSIFIERS", V.CLASSIFIERS)
        exp_types = _as_tuple_str("EXP_TYPES", V.EXP_TYPES)
        cfg_tub = build_textual_config()
        cli_log(
            "__main__: "
            f"mode=tubular, dataset={V.DATASET!r}, quick={V.QUICK}, "
            f"val_length={V.VAL_LENGTH}, seeds={seeds}, qua={qua_m}, "
            f"classifiers={clf}, exp_types={exp_types}, "
            f"regressor_time_column={cfg_tub.regressor_time_column!r}"
        )
        run_tubular_experiments_grid(
            dataset_name=V.DATASET,
            val_length=int(V.VAL_LENGTH),
            seeds=seeds,
            qua_methods=qua_m,
            tsa_methods=tsa_m,
            classifiers=clf,
            exp_types=exp_types,
            cfg_toms=cfg_tub,
            time_column=str(cfg_tub.regressor_time_column),
            unified_window=int(V.UNIFIED_WINDOW),
            log_prefix=log_prefix,
            quick=V.QUICK,
        )
        cli_log("Main run finished.")
        return

    cfg = build_textual_config()
    cli_log(
        "__main__: "
        f"mode=global_textual, dataset={V.DATASET!r}, quick={V.QUICK}, "
        f"val_length={cfg.val_length}, "
        f"seeds={cfg.seeds if not V.QUICK else '(quick: 1)'}, "
        f"qua={cfg.qua_methods if not V.QUICK else '(quick: ACC)'}, "
        f"exp_types={cfg.exp_types}"
    )
    run_textual_experiments_grid(cfg, quick=V.QUICK)
    cli_log("Main run finished.")


if __name__ == "__main__":
    main()
