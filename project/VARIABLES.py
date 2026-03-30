"""
Configure what to run, then execute from the ``project/`` folder::

    python run_experiment.py

With ``QUICK = True``, only one smoke experiment runs (1 seed, ACC, QFY,
file ``…_quick.csv``); ``SEEDS``, ``QUA_METHODS`` and ``TSA_METHODS`` are
ignored in that mode.
"""

from __future__ import annotations

# --- quick mode (same as legacy ``--quick``) ---
QUICK = False

# --- grid (used when ``QUICK`` is False) ---
# SEEDS = (1, 2, 3)
# QUA_METHODS = ("DyS", "DyS-Opt")
# TSA_METHODS = ("QFY", "MA", "KFMA")

# --- data and windows ---
# DATASET = "global_covid19_tweets"
# VAL_LENGTH = 15
# MAX_TEST_CHUNKS = 5000

# CLASSIFIERS = ("amansolanki/autonlp-Tweet-Sentiment-Extraction-20114061",)

# EXP_TYPES = ("TOMS",)

# REGRESSOR_LABEL = "TSMN"
# REGRESSOR_TIME_COLUMN = "TweetAt"

# TSMN_MODE = "linear"
# TSMN_DEGREE = 3
# TSMN_PERIOD = None  # float or None (only used in cyclic mode)

# UNIFIED_WINDOW = 4
# LOG_PREFIX = "[run_experiment]"

# RUN = "global_textual"

# =============================================================================
# Commented example — full grid (many seeds and quantifiers)
# =============================================================================
# Copy blocks below upward, or use as reference.
#
# QUICK = False
# RUN = "global_textual"
# DATASET = "global_covid19_tweets"
# SEEDS = (1, 2, 3)
# VAL_LENGTH = 15
# MAX_TEST_CHUNKS = 5000
# CLASSIFIERS = ("amansolanki/autonlp-Tweet-Sentiment-Extraction-20114061",)
# QUA_METHODS = ("DyS", "DyS-Opt", "ACC", "GPAC", "EDy", "CC")
# TSA_METHODS = ("QFY", "MA", "KFMA")
# EXP_TYPES = ("TOMS", "original")
# REGRESSOR_LABEL = "TSMN"
# REGRESSOR_TIME_COLUMN = "TweetAt"
# TSMN_MODE = "linear"
# TSMN_DEGREE = 3
# TSMN_PERIOD = None
# UNIFIED_WINDOW = 4
# LOG_PREFIX = "[run_experiment]"

# =============================================================================
# Commented example — single experiment (like ``--quick``)
# =============================================================================
QUICK = False
RUN = "tubular"
SEEDS = [1]
QUA_METHODS = ("DyS", "ACC")
TSA_METHODS = ("QFY", "MA", "KFMA")
DATASET = "energy"
VAL_LENGTH = 15
MAX_TEST_CHUNKS = 5000
CLASSIFIERS = ("LR", "RF")
EXP_TYPES = ["TOMS", "original"]
REGRESSOR_LABEL = "TSMN"
# energy: date column in CSV (YYYY-MM-DD after load)
REGRESSOR_TIME_COLUMN = "date"
# TOMS: "scalar" = continuous time (Unix seconds per row; median per window at inference);
# "week" = 7 one-hot weekday features (pandas: Mon=0 … Sun=6), per training row and
# one test row per window with 1 on the chunk median’s weekday and 0 elsewhere.
REGRESSOR_TIME_ENCODING = "scalar"
TSMN_MODE = "linear"
TSMN_DEGREE = 3
TSMN_PERIOD = None
UNIFIED_WINDOW = 4
LOG_PREFIX = "[run_experiment]"
# (with QUICK=True, SEEDS / QUA_METHODS / TSA_METHODS are ignored)
