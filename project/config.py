from pathlib import Path
import os

# Base directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directory (relative to PROJECT_ROOT)
# The original code used r'../time series qua/' which suggests it's outside the project folder
# But let's check the structure:
# /home/daniel/Documents/quantification-over-time/project
# /home/daniel/Documents/quantification-over-time/time series qua
DATA_DIR = PROJECT_ROOT.parent / "time series qua"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output_files"
OUTPUT_REGRESSOR_DIR = PROJECT_ROOT / "output_regressor"
OUTPUT_CLASSIFICATION_DIR = PROJECT_ROOT / "output_classification"
QUANT_RESULTS_DIR = PROJECT_ROOT / "quant_results"
PLOTS_DIR = PROJECT_ROOT / "plots"
README_IMPLEMENT_DIR = PROJECT_ROOT / "ReadMe_Implement"

# CPU/RAM (conservative defaults for low-memory machines).
# quantifications joblib Parallel loky: 1 = sequential; 2+ = parallel;
# -1 = all cores (only if you have RAM; each worker may duplicate the HF model).
TEST_CHUNK_LOKY_JOBS = int(os.environ.get("QUA_TEST_CHUNK_JOBS", "1"))

# HuggingFace sentiment model: texts per forward pass (smaller → lower RAM spikes).
HF_INFERENCE_BATCH_SIZE = int(os.environ.get("QUA_HF_BATCH_SIZE", "1000"))

# LinearRegression, LogisticRegression, RandomForest, etc.
SKLEARN_N_JOBS = int(os.environ.get("QUA_SKLEARN_N_JOBS", "1"))


# Ensure directories exist
def ensure_dirs():
    for d in [
        OUTPUT_DIR,
        OUTPUT_REGRESSOR_DIR,
        OUTPUT_CLASSIFICATION_DIR,
        QUANT_RESULTS_DIR,
        PLOTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    ensure_dirs()
