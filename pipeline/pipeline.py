"""
Main orchestrator combining all pipeline components.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .molecule_utils import validate_smiles, MoleculeValidationResult
from .feature_engineering import generate_features, get_descriptor_summary, format_descriptor_summary
from .model_inference import get_predictor


@dataclass
class PredictionResult:
    """Result of a single molecule prediction."""
    smiles: str
    is_valid: bool
    prediction: Optional[int]
    probability: Optional[float]
    descriptors: Optional[Dict[str, float]]
    error: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'smiles': self.smiles,
            'is_valid': self.is_valid,
            'prediction': self.prediction,
            'probability': self.probability,
            'descriptors': self.descriptors,
            'error': self.error
        }


def predict_and_explain(smiles: str) -> PredictionResult:
    """
    Process a single SMILES string through the full pipeline.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        PredictionResult with prediction, probability, and descriptors
    """
    # Step 1: Validate SMILES
    validation = validate_smiles(smiles)
    
    if not validation.is_valid:
        return PredictionResult(
            smiles=smiles,
            is_valid=False,
            prediction=None,
            probability=None,
            descriptors=None,
            error=validation.error_message
        )
    
    mol = validation.mol
    
    # Step 2: Generate features
    features = generate_features(mol)
    
    if features is None or features.isna().any().any():
        return PredictionResult(
            smiles=smiles,
            is_valid=False,
            prediction=None,
            probability=None,
            descriptors=None,
            error="Failed to compute molecular features"
        )
    
    # Step 3: Get prediction
    predictor = get_predictor()
    predictor.load()
    
    predictions, probabilities = predictor.predict(features)
    prediction = int(predictions[0])
    probability = float(probabilities[0])
    
    # Step 4: Get descriptor summary
    descriptors = get_descriptor_summary(mol)
    
    return PredictionResult(
        smiles=smiles,
        is_valid=True,
        prediction=prediction,
        probability=probability,
        descriptors=descriptors,
        error=None
    )


def predict_batch(smiles_list: List[str]) -> List[PredictionResult]:
    """
    Process multiple SMILES strings through the pipeline.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        List of PredictionResult objects
    """
    results = []
    
    for smiles in smiles_list:
        result = predict_and_explain(smiles)
        results.append(result)
    
    return results
