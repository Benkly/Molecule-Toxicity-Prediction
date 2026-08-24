"""
Explanation formatting for toxicity predictions.
"""

from typing import Dict, List
from .config import TARGET_NAME, TARGET_DESCRIPTION


def generate_explanation(
    smiles: str,
    prediction: int,
    probability: float,
    descriptors: Dict[str, str]
) -> str:
    """
    Generate a human-readable explanation for a toxicity prediction.
    
    Args:
        smiles: Input SMILES string
        prediction: Binary prediction (0 = non-toxic, 1 = toxic)
        probability: Probability of toxicity
        descriptors: Dictionary of molecular descriptors
        
    Returns:
        Formatted explanation string
    """
    prediction_label = "TOXIC" if prediction == 1 else "NON-TOXIC"
    confidence = probability if prediction == 1 else (1 - probability)
    
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"TOXICITY PREDICTION REPORT - {TARGET_NAME}")
    lines.append(f"{'='*60}")
    lines.append(f"\nInput SMILES: {smiles}")
    lines.append(f"\n--- PREDICTION ---")
    lines.append(f"Result: {prediction_label}")
    lines.append(f"Toxicity Probability: {probability:.1%}")
    lines.append(f"Confidence: {confidence:.1%}")
    lines.append(f"\n--- MOLECULAR PROPERTIES ---")
    
    for name, value in descriptors.items():
        lines.append(f"  {name}: {value}")
    
    lines.append(f"\n--- ABOUT THE ASSAY ---")
    lines.append(TARGET_DESCRIPTION.strip())
    lines.append(f"\n{'='*60}")
    
    return "\n".join(lines)


def format_batch_results(results: List[dict]) -> str:
    """
    Format results for multiple molecules into a summary table.
    
    Args:
        results: List of prediction result dictionaries
        
    Returns:
        Formatted summary table string
    """
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append("BATCH PREDICTION SUMMARY")
    lines.append(f"{'='*80}")
    lines.append(f"\n{'#':<4} {'Status':<10} {'Probability':<12} {'SMILES':<50}")
    lines.append("-" * 80)
    
    for i, result in enumerate(results, 1):
        if result.get('error'):
            status = "ERROR"
            prob = "-"
            smiles = result.get('smiles', 'Unknown')
        else:
            status = "TOXIC" if result.get('prediction') == 1 else "NON-TOXIC"
            prob = f"{result.get('probability', 0):.1%}"
            smiles = result.get('smiles', '')
        
        # Truncate long SMILES
        if len(smiles) > 47:
            smiles = smiles[:47] + "..."
        
        lines.append(f"{i:<4} {status:<10} {prob:<12} {smiles:<50}")
    
    # Summary statistics
    valid_results = [r for r in results if not r.get('error')]
    toxic_count = sum(1 for r in valid_results if r.get('prediction') == 1)
    
    lines.append("-" * 80)
    lines.append(f"\nTotal molecules: {len(results)}")
    lines.append(f"Valid predictions: {len(valid_results)}")
    lines.append(f"Predicted toxic: {toxic_count}")
    lines.append(f"Predicted non-toxic: {len(valid_results) - toxic_count}")
    lines.append(f"Errors: {len(results) - len(valid_results)}")
    
    return "\n".join(lines)
