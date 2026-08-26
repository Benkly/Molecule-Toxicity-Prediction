"""
LLM-based explanation generator for AhR toxicity predictions.

Uses OpenRouter API with GPT-4o-mini to generate natural language explanations
based on SHAP values and molecular descriptors.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import shap
from openai import OpenAI
from dotenv import load_dotenv
from rdkit import Chem

from .config import BASE_DIR, DESCRIPTOR_DISPLAY_TO_INTERNAL
from .model_inference import get_predictor
from .feature_engineering import generate_features

load_dotenv()

# LLM Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
MODEL = 'openai/gpt-4o-mini'
TEMPERATURE = 0
MAX_TOKENS = 800

# Prompt path
PROMPT_PATH = BASE_DIR / 'prompts' / 'explainer_system_prompt.md'


@dataclass
class LLMExplanationResult:
    """Result of LLM explanation generation."""
    success: bool
    explanation: Optional[str]
    error_message: Optional[str]
    error_type: Optional[str]  # 'credits', 'api_error', 'config'


def _get_shap_values(
    features: pd.DataFrame,
    predictor
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute SHAP values for a single prediction.
    
    Returns:
        Tuple of (top_5_positive_shap, bottom_5_negative_shap)
    """
    # Preprocess features
    preprocessed = predictor.preprocess_features(features)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(predictor._model)
    shap_values = explainer.shap_values(preprocessed)
    
    # Get feature names
    feature_names = list(preprocessed.columns)
    
    # Create Series of SHAP values
    shap_series = pd.Series(shap_values[0], index=feature_names)
    
    # Get top 5 positive and bottom 5 negative
    top_positive = shap_series.nlargest(5).to_dict()
    bottom_negative = shap_series.nsmallest(5).to_dict()
    
    return top_positive, bottom_negative


def _get_substructure_smarts(
    mol: Chem.Mol,
    ecfp_bits: list,
    radius: int = 3
) -> Dict[str, str]:
    """
    Get SMARTS patterns for significant ECFP bits.
    
    Args:
        mol: RDKit molecule
        ecfp_bits: List of ECFP bit indices to look up
        radius: Morgan fingerprint radius
        
    Returns:
        Dictionary mapping ECFP bit names to SMARTS patterns
    """
    from rdkit.Chem import rdFingerprintGenerator
    
    # Generate fingerprint with bit info
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=2048)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    
    fp = morgan_gen.GetFingerprint(mol, additionalOutput=ao)
    bit_info = ao.GetBitInfoMap()
    
    substructures = {}
    for bit_idx in ecfp_bits:
        if bit_idx in bit_info:
            # Get atom environments for this bit
            for atom_idx, radius_used in bit_info[bit_idx]:
                try:
                    # Get the atom environment as SMARTS
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius_used, atom_idx)
                    if env:
                        atoms_in_env = set()
                        for bond_idx in env:
                            bond = mol.GetBondWithIdx(bond_idx)
                            atoms_in_env.add(bond.GetBeginAtomIdx())
                            atoms_in_env.add(bond.GetEndAtomIdx())
                        if atoms_in_env:
                            submol = Chem.PathToSubmol(mol, env)
                            smarts = Chem.MolToSmarts(submol)
                            substructures[f'ECFP_{bit_idx}'] = smarts
                            break
                except Exception:
                    continue
    
    return substructures


def _format_llm_input(
    smiles: str,
    prediction: int,
    probability: float,
    descriptors: Dict[str, float],
    top_positive_shap: Dict[str, float],
    bottom_negative_shap: Dict[str, float],
    substructure_smarts: Dict[str, str]
) -> Dict[str, Any]:
    """Format input data for the LLM."""
    
    # Format prediction
    if prediction == 1:
        ahr_prediction = {"Toxic": f"{probability:.1%}"}
    else:
        ahr_prediction = {
            "prediction": "Non-Toxic",
            "confidence": f"{(1-probability):.1%}"
        }
    
    # Convert display names to internal names using config mapping
    formatted_descriptors = {}
    for display_name, value in descriptors.items():
        internal_name = DESCRIPTOR_DISPLAY_TO_INTERNAL.get(display_name, display_name)
        formatted_descriptors[internal_name] = round(value, 4)
    
    # Identify ECFP bits in SHAP results that couldn't be mapped to substructures
    ecfp_bits_in_shap = set()
    for key in list(top_positive_shap.keys()) + list(bottom_negative_shap.keys()):
        if key.startswith('ECFP_'):
            ecfp_bits_in_shap.add(key)
    
    unmapped_bits = ecfp_bits_in_shap - set(substructure_smarts.keys())
    
    result = {
        "SMILES": smiles,
        "MolecularDescriptors": formatted_descriptors,
        "AHRToxicityPrediction": ahr_prediction,
        "Top5PositiveSHAP": {k: round(v, 4) for k, v in top_positive_shap.items()},
        "Bottom5NegativeSHAP": {k: round(v, 4) for k, v in bottom_negative_shap.items()},
        "TopSubstructureContributions": substructure_smarts
    }
    
    # Add unmapped bits flag if any exist
    if unmapped_bits:
        result["UnmappedECFPBits"] = list(unmapped_bits)
    
    return result


def _load_system_prompt() -> str:
    """Load the system prompt from file."""
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def _call_llm(system_prompt: str, user_input: Dict[str, Any]) -> str:
    """
    Call the OpenRouter API.
    
    Raises:
        ValueError: If API key not configured
        Exception: Various API errors with specific messages
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not configured in .env file")
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': json.dumps(user_input)}
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    
    return response.choices[0].message.content


def generate_llm_explanation(
    smiles: str,
    prediction: int,
    probability: float,
    descriptors: Dict[str, float],
    mol: Optional[Chem.Mol] = None,
    features: Optional[pd.DataFrame] = None
) -> LLMExplanationResult:
    """
    Generate an LLM-based explanation for a toxicity prediction.
    
    Args:
        smiles: Input SMILES string
        prediction: Binary prediction (0=non-toxic, 1=toxic)
        probability: Probability of toxicity
        descriptors: Dictionary of molecular descriptors (display names)
        mol: Optional pre-computed RDKit molecule object
        features: Optional pre-computed feature DataFrame
        
    Returns:
        LLMExplanationResult with explanation or error
    """
    # Check API key
    if not OPENROUTER_API_KEY:
        return LLMExplanationResult(
            success=False,
            explanation=None,
            error_message="API key not configured. Please add OPENROUTER_API_KEY to .env file.",
            error_type='config'
        )
    
    try:
        # Use provided mol or parse from SMILES
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return LLMExplanationResult(
                    success=False,
                    explanation=None,
                    error_message="Failed to parse molecule structure.",
                    error_type='api_error'
                )
        
        # Use provided features or generate them
        if features is None:
            features = generate_features(mol)
            if features is None:
                return LLMExplanationResult(
                    success=False,
                    explanation=None,
                    error_message="Failed to compute molecular features.",
                    error_type='api_error'
                )
        
        # Get predictor and compute SHAP
        predictor = get_predictor()
        predictor.load()
        
        top_positive, bottom_negative = _get_shap_values(features, predictor)
        
        # Get ECFP bits from SHAP results
        ecfp_bits = []
        for key in list(top_positive.keys()) + list(bottom_negative.keys()):
            if key.startswith('ECFP_'):
                try:
                    bit_idx = int(key.split('_')[1])
                    ecfp_bits.append(bit_idx)
                except (IndexError, ValueError):
                    continue
        
        # Get substructure SMARTS
        substructures = _get_substructure_smarts(mol, ecfp_bits)
        
        # Format LLM input
        llm_input = _format_llm_input(
            smiles=smiles,
            prediction=prediction,
            probability=probability,
            descriptors=descriptors,
            top_positive_shap=top_positive,
            bottom_negative_shap=bottom_negative,
            substructure_smarts=substructures
        )
        
        # Load prompt and call LLM
        system_prompt = _load_system_prompt()
        explanation = _call_llm(system_prompt, llm_input)
        
        return LLMExplanationResult(
            success=True,
            explanation=explanation,
            error_message=None,
            error_type=None
        )
        
    except Exception as e:
        error_str = str(e).lower()
        
        # Detect specific error types
        if 'insufficient' in error_str or 'credit' in error_str or 'quota' in error_str:
            return LLMExplanationResult(
                success=False,
                explanation=None,
                error_message="API credits exhausted. Please add credits to your OpenRouter account.",
                error_type='credits'
            )
        elif 'rate' in error_str and 'limit' in error_str:
            return LLMExplanationResult(
                success=False,
                explanation=None,
                error_message="API rate limit reached. Please try again in a moment.",
                error_type='api_error'
            )
        elif 'timeout' in error_str or 'connection' in error_str:
            return LLMExplanationResult(
                success=False,
                explanation=None,
                error_message="Could not connect to explanation service. Please check your internet connection.",
                error_type='api_error'
            )
        else:
            return LLMExplanationResult(
                success=False,
                explanation=None,
                error_message=f"An error occurred while generating explanation: {str(e)[:100]}",
                error_type='api_error'
            )
