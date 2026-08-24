"""
Configuration constants and model paths for the toxicity prediction pipeline.
"""

from pathlib import Path

# Base directory (parent of pipeline folder)
BASE_DIR = Path(__file__).parent.parent

# Model artifacts
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "xgb_nrahr_model.joblib"
SCALER_PATH = MODELS_DIR / "descriptor_scaler.joblib"
THRESHOLD_PATH = MODELS_DIR / "optimal_threshold.joblib"
CONFIG_PATH = MODELS_DIR / "model_config.joblib"

# Feature engineering settings
MORGAN_RADIUS = 3
MORGAN_FP_SIZE = 2048

# Molecular descriptor columns (must match training order)
DESCRIPTOR_COLS = [
    'MolWeight', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors',
    'NumRotatableBonds', 'NumAtoms', 'NumHeavyAtoms', 'NumRings',
    'NumAromaticRings', 'FractionCSP3', 'BertzCT'
]

# Target assay information
TARGET_NAME = "NR-AhR"
TARGET_DESCRIPTION = """
The NR-AhR (Aryl Hydrocarbon Receptor) assay measures activation of the AhR pathway.
AhR is activated by planar aromatics like PAHs and dioxin-like halogenated aromatics.
Activation is linked to inflammation, immunotoxicity, and carcinogenicity.
This is considered a toxicity predictor since AhR agonism isn't a typical therapeutic mechanism.
"""

# Prediction thresholds
DEFAULT_THRESHOLD = 0.5  # Fallback if threshold file not found
