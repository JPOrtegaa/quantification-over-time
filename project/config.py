from pathlib import Path

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
README_IMPLEMENT_DIR = PROJECT_ROOT / "ReadMe_Implement"


# Ensure directories exist
def ensure_dirs():
    for d in [OUTPUT_DIR, QUANT_RESULTS_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Dir: {DATA_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    ensure_dirs()
