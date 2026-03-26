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
QUANT_RESULTS_DIR = PROJECT_ROOT / "quant_results"
PLOTS_DIR = PROJECT_ROOT / "plots"
OUTPUT_CLASSIFICATION_DIR = OUTPUT_DIR / "classification"
OUTPUT_REGRESSOR_DIR = OUTPUT_DIR / "regressor"
README_IMPLEMENT_DIR = PROJECT_ROOT / "ReadMe_Implement"

# Uso de CPU/RAM (defeitos conservadores para PCs com pouca memória).
# joblib.Parallel(..., backend="loky") em quantifications: 1 = sequencial; 2+ = paralelo;
# -1 = todos os núcleos (só se tiver RAM de sobra; cada worker pode duplicar modelo HF).
TEST_CHUNK_LOKY_JOBS = int(os.environ.get("QUA_TEST_CHUNK_JOBS", "1"))

# Tamanho do batch na inferência Hugging Face (menor -> menos pico de RAM).
HF_INFERENCE_BATCH_SIZE = int(os.environ.get("QUA_HF_BATCH_SIZE", "8"))

# LinearRegression, LogisticRegression, RandomForest, etc.
SKLEARN_N_JOBS = int(os.environ.get("QUA_SKLEARN_N_JOBS", "1"))


# Ensure directories exist
def ensure_dirs():
    for d in [OUTPUT_DIR, QUANT_RESULTS_DIR, PLOTS_DIR, OUTPUT_CLASSIFICATION_DIR, OUTPUT_REGRESSOR_DIR]:
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    ensure_dirs()
