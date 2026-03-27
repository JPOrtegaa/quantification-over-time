"""Paths and runtime settings for the ``testing`` package."""

from pathlib import Path
import os

TESTING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTING_ROOT.parent

DATA_DIR = REPO_ROOT / "time series qua"

OUTPUT_DIR = TESTING_ROOT / "output_files"
OUTPUT_REGRESSOR_DIR = OUTPUT_DIR / "regressor"
OUTPUT_CLASSIFICATION_DIR = OUTPUT_DIR / "classification"
QUANT_RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"
CLASSIFIER_OUTPUT_DIR = TESTING_ROOT / "classifier" / "outputs"

TEST_CHUNK_LOKY_JOBS = int(os.environ.get("QUA_TEST_CHUNK_JOBS", "1"))
HF_INFERENCE_BATCH_SIZE = int(os.environ.get("QUA_HF_BATCH_SIZE", "500"))
SKLEARN_N_JOBS = int(os.environ.get("QUA_SKLEARN_N_JOBS", "1"))


def ensure_dirs():
    for d in (
        OUTPUT_DIR,
        OUTPUT_REGRESSOR_DIR,
        OUTPUT_CLASSIFICATION_DIR,
        QUANT_RESULTS_DIR,
        PLOTS_DIR,
        CLASSIFIER_OUTPUT_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()

if __name__ == "__main__":
    print(f"Testing root: {TESTING_ROOT}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
