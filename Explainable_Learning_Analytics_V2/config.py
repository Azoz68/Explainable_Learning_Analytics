import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("OULAD_DATA", ROOT / "data" / "anonymisedData"))
PREPROCESSED_CSV = Path(os.environ.get("OULAD_PREPROCESSED", DATA_DIR / "oulad_preprocessed_noleak.csv"))
MODELS_DIR = Path(os.environ.get("OULAD_MODELS", DATA_DIR / "models"))
RESULTS_DIR = Path(os.environ.get("OULAD_RESULTS", ROOT / "results"))

SEED = 42
CUTOFF_DAY = 30
CLASS_NAMES = ["Distinction", "Fail", "Pass", "Withdrawn"]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
