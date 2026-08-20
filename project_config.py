"""Shared artifact locations for the demand-simulation pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / "persona_cache"
MODEL_PATH = RESULTS_DIR / "price_model.pkl"
CALIBRATION_PATH = RESULTS_DIR / "calibrated_theta.json"


def result_path(name: str) -> Path:
    """Return a result path, creating the result directory when needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name
