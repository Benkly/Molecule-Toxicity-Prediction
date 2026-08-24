"""
Model loading and prediction logic.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
import joblib

from .config import (
    MODEL_PATH, SCALER_PATH, THRESHOLD_PATH, CONFIG_PATH,
    DESCRIPTOR_COLS, DEFAULT_THRESHOLD
)


class ToxicityPredictor:
    """
    Wrapper for the toxicity prediction model with preprocessing.
    """
    
    def __init__(self):
        """Initialize the predictor by loading model artifacts."""
        self._model = None
        self._scaler = None
        self._threshold = None
        self._config = None
        self._loaded = False
    
    def load(self) -> None:
        """Load all model artifacts from disk."""
        if self._loaded:
            return
        
        # Load XGBoost model
        if MODEL_PATH.exists():
            self._model = joblib.load(MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        # Load scaler
        if SCALER_PATH.exists():
            self._scaler = joblib.load(SCALER_PATH)
        else:
            raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
        
        # Load threshold (use default if not found)
        if THRESHOLD_PATH.exists():
            self._threshold = joblib.load(THRESHOLD_PATH)
        else:
            self._threshold = DEFAULT_THRESHOLD
        
        # Load config (optional)
        if CONFIG_PATH.exists():
            self._config = joblib.load(CONFIG_PATH)
        
        self._loaded = True
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    @property
    def threshold(self) -> float:
        """Get the optimal classification threshold."""
        return self._threshold if self._threshold is not None else DEFAULT_THRESHOLD
    
    def preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features: scale descriptors, keep ECFP unchanged.
        
        Args:
            features: Raw feature DataFrame
            
        Returns:
            Preprocessed feature DataFrame ready for prediction
        """
        if not self._loaded:
            self.load()
        
        # Separate descriptor and ECFP columns
        desc_cols = [c for c in DESCRIPTOR_COLS if c in features.columns]
        ecfp_cols = [c for c in features.columns if c.startswith('ECFP_')]
        
        # Scale descriptors
        desc_scaled = pd.DataFrame(
            self._scaler.transform(features[desc_cols]),
            columns=desc_cols,
            index=features.index
        )
        
        # Combine scaled descriptors with unscaled ECFP
        preprocessed = pd.concat([desc_scaled, features[ecfp_cols]], axis=1)
        
        return preprocessed
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions for toxicity.
        
        Args:
            features: Raw feature DataFrame
            
        Returns:
            Array of probabilities (probability of being toxic)
        """
        if not self._loaded:
            self.load()
        
        preprocessed = self.preprocess_features(features)
        probabilities = self._model.predict_proba(preprocessed)[:, 1]
        
        return probabilities
    
    def predict(self, features: pd.DataFrame, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get binary predictions and probabilities for toxicity.
        
        Args:
            features: Raw feature DataFrame
            threshold: Custom threshold (uses optimal F2 threshold if not specified)
            
        Returns:
            Tuple of (binary predictions, probabilities)
        """
        if threshold is None:
            threshold = self.threshold
        
        probabilities = self.predict_proba(features)
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions, probabilities
    
    def get_feature_importance(self, top_n: int = 20) -> pd.Series:
        """
        Get feature importance from the model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Series of feature importances sorted by importance
        """
        if not self._loaded:
            self.load()
        
        # Get feature names
        feature_names = DESCRIPTOR_COLS + [f'ECFP_{i}' for i in range(2048)]
        
        importances = pd.Series(
            self._model.feature_importances_,
            index=feature_names
        )
        
        return importances.nlargest(top_n)


# Global predictor instance (lazy loaded)
_predictor: Optional[ToxicityPredictor] = None


def get_predictor() -> ToxicityPredictor:
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = ToxicityPredictor()
    return _predictor
