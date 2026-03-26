import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

import config
# import run_experiment # Avoid running the whole experiment on import if possible


def test_config():
    print("Testing config.py...")
    print(f"DATA_DIR: {config.DATA_DIR}")
    print(f"OUTPUT_DIR: {config.OUTPUT_DIR}")
    print(f"QUANT_RESULTS_DIR: {config.QUANT_RESULTS_DIR}")

    assert config.PROJECT_ROOT.name == "project"
    assert config.DATA_DIR.name == "time series qua"
    print("Config check passed!")


def test_directory_creation():
    print("Testing directory creation...")
    config.ensure_dirs()
    assert config.OUTPUT_DIR.exists()
    assert config.QUANT_RESULTS_DIR.exists()
    assert config.PLOTS_DIR.exists()
    print("Directory creation check passed!")


if __name__ == "__main__":
    try:
        test_config()
        test_directory_creation()
        print("\nAll path verification tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        sys.exit(1)
