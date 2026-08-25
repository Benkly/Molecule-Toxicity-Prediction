"""
Molecular descriptor calculation and ECFP fingerprint generation.
"""

from typing import Dict, Optional, List
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors, rdFingerprintGenerator

from .config import MORGAN_RADIUS, MORGAN_FP_SIZE, DESCRIPTOR_COLS


# Initialize Morgan fingerprint generator (module-level for reuse)
_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, 
    fpSize=MORGAN_FP_SIZE
)


def calculate_molecular_descriptors(mol: Chem.Mol) -> Optional[Dict[str, float]]:
    """
    Calculate molecular descriptors for a single molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary of descriptor names to values, or None if calculation fails
    """
    if mol is None:
        return None
    
    try:
        descriptors = {
            'MolWeight': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAtoms': mol.GetNumAtoms(),
            'NumHeavyAtoms': mol.GetNumHeavyAtoms(),
            'NumRings': Descriptors.RingCount(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol),
            'BertzCT': GraphDescriptors.BertzCT(mol)
        }
        return descriptors
    except Exception:
        return None


def compute_ecfp(mol: Chem.Mol) -> np.ndarray:
    """
    Compute ECFP (Extended-Connectivity Fingerprint) count vector for a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        NumPy array of fingerprint counts
    """
    if mol is None:
        return np.zeros(MORGAN_FP_SIZE)
    
    return np.array(_morgan_gen.GetCountFingerprintAsNumPy(mol))


def generate_features(mol: Chem.Mol) -> Optional[pd.DataFrame]:
    """
    Generate complete feature vector for a single molecule.
    Combines molecular descriptors with ECFP fingerprints.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        DataFrame with single row containing all features, or None if failed
    """
    # Calculate descriptors
    descriptors = calculate_molecular_descriptors(mol)
    if descriptors is None:
        return None
    
    # Generate ECFP fingerprint
    ecfp = compute_ecfp(mol)
    
    # Create descriptor DataFrame
    desc_df = pd.DataFrame([descriptors])
    
    # Create ECFP DataFrame
    ecfp_df = pd.DataFrame(
        [ecfp],
        columns=[f'ECFP_{i}' for i in range(MORGAN_FP_SIZE)]
    )
    
    # Combine features (descriptors first, then ECFP)
    features = pd.concat([desc_df[DESCRIPTOR_COLS], ecfp_df], axis=1)
    
    return features


def generate_features_batch(mols: List[Chem.Mol]) -> pd.DataFrame:
    """
    Generate feature matrix for multiple molecules.
    
    Args:
        mols: List of RDKit molecule objects
        
    Returns:
        DataFrame with features for all molecules (rows may have NaN for failed molecules)
    """
    all_features = []
    
    for mol in mols:
        features = generate_features(mol)
        if features is not None:
            all_features.append(features)
        else:
            # Create a row of NaNs for failed molecules
            nan_row = pd.DataFrame(
                [[np.nan] * (len(DESCRIPTOR_COLS) + MORGAN_FP_SIZE)],
                columns=DESCRIPTOR_COLS + [f'ECFP_{i}' for i in range(MORGAN_FP_SIZE)]
            )
            all_features.append(nan_row)
    
    if not all_features:
        return pd.DataFrame()
    
    return pd.concat(all_features, ignore_index=True)


def get_descriptor_summary(mol: Chem.Mol) -> Dict[str, float]:
    """
    Get molecular descriptors as numeric values.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary with descriptor names and float values
    """
    descriptors = calculate_molecular_descriptors(mol)
    if descriptors is None:
        return {}
    
    summary = {
        'Molecular Weight': float(descriptors['MolWeight']),
        'LogP': float(descriptors['LogP']),
        'TPSA': float(descriptors['TPSA']),
        'Hydrogen Bond Donors': float(descriptors['NumHDonors']),
        'Hydrogen Bond Acceptors': float(descriptors['NumHAcceptors']),
        'Rotatable Bonds': float(descriptors['NumRotatableBonds']),
        'Total Atoms': float(descriptors['NumAtoms']),
        'Heavy Atoms': float(descriptors['NumHeavyAtoms']),
        'Ring Count': float(descriptors['NumRings']),
        'Aromatic Rings': float(descriptors['NumAromaticRings']),
        'Fraction sp3 Carbons': float(descriptors['FractionCSP3']),
        'Bertz Complexity': float(descriptors['BertzCT'])
    }
    
    return summary


def format_descriptor_summary(descriptors: Dict[str, float]) -> Dict[str, str]:
    """
    Format descriptor values for human-readable display.
    
    Args:
        descriptors: Dictionary of descriptor names to float values
        
    Returns:
        Dictionary with formatted string values
    """
    if not descriptors:
        return {}
    
    return {
        'Molecular Weight': f"{descriptors['Molecular Weight']:.2f} Da",
        'LogP': f"{descriptors['LogP']:.2f}",
        'TPSA': f"{descriptors['TPSA']:.2f} Å²",
        'Hydrogen Bond Donors': str(int(descriptors['Hydrogen Bond Donors'])),
        'Hydrogen Bond Acceptors': str(int(descriptors['Hydrogen Bond Acceptors'])),
        'Rotatable Bonds': str(int(descriptors['Rotatable Bonds'])),
        'Total Atoms': str(int(descriptors['Total Atoms'])),
        'Heavy Atoms': str(int(descriptors['Heavy Atoms'])),
        'Ring Count': str(int(descriptors['Ring Count'])),
        'Aromatic Rings': str(int(descriptors['Aromatic Rings'])),
        'Fraction sp3 Carbons': f"{descriptors['Fraction sp3 Carbons']:.3f}",
        'Bertz Complexity': f"{descriptors['Bertz Complexity']:.2f}"
    }
