"""
Main orchestrator combining all pipeline components.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .molecule_utils import validate_smiles, MoleculeValidationResult
from .feature_engineering import generate_features, get_descriptor_summary
from .model_inference import get_predictor
from .explainer import generate_explanation, format_batch_results


@dataclass
class PredictionResult:
    """Result of a single molecule prediction."""
    smiles: str
    is_valid: bool
    prediction: Optional[int]
    probability: Optional[float]
    descriptors: Optional[Dict[str, str]]
    explanation: Optional[str]
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
        PredictionResult with prediction, probability, descriptors, and explanation
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
            explanation=None,
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
            explanation=None,
            error="Failed to compute molecular features"
        )
    
    # Step 3: Get prediction
    predictor = get_predictor()
    predictor.load()
    
    predictions, probabilities = predictor.predict(features)
    prediction = int(predictions[0])
    probability = float(probabilities[0])
    
    # Step 4: Get descriptor summary for explanation
    descriptors = get_descriptor_summary(mol)
    
    # Step 5: Generate explanation
    explanation = generate_explanation(
        smiles=smiles,
        prediction=prediction,
        probability=probability,
        descriptors=descriptors
    )
    
    return PredictionResult(
        smiles=smiles,
        is_valid=True,
        prediction=prediction,
        probability=probability,
        descriptors=descriptors,
        explanation=explanation,
        error=None
    )


def predict_batch(smiles_list: List[str], verbose: bool = False) -> List[PredictionResult]:
    """
    Process multiple SMILES strings through the pipeline.
    
    Args:
        smiles_list: List of SMILES strings
        verbose: Whether to print progress
        
    Returns:
        List of PredictionResult objects
    """
    results = []
    total = len(smiles_list)
    
    for i, smiles in enumerate(smiles_list, 1):
        if verbose:
            print(f"Processing {i}/{total}: {smiles[:50]}...")
        
        result = predict_and_explain(smiles)
        results.append(result)
    
    return results


def get_batch_summary(results: List[PredictionResult]) -> str:
    """
    Generate a summary of batch prediction results.
    
    Args:
        results: List of PredictionResult objects
        
    Returns:
        Formatted summary string
    """
    result_dicts = [r.to_dict() for r in results]
    return format_batch_results(result_dicts)
