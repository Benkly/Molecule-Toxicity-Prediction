"""
SMILES validation and molecule conversion utilities.
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass
class MoleculeValidationResult:
    """Result of SMILES validation."""
    smiles: str
    is_valid: bool
    mol: Optional[Chem.Mol]
    error_message: Optional[str]


def validate_smiles(smiles: str) -> MoleculeValidationResult:
    """
    Validate a SMILES string and convert to RDKit molecule object.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        MoleculeValidationResult with validation status and molecule object
    """
    if not smiles or not isinstance(smiles, str):
        return MoleculeValidationResult(
            smiles=smiles,
            is_valid=False,
            mol=None,
            error_message="Empty or invalid input"
        )
    
    # Clean whitespace
    smiles = smiles.strip()
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return MoleculeValidationResult(
                smiles=smiles,
                is_valid=False,
                mol=None,
                error_message="Invalid SMILES syntax - could not parse"
            )
        
        # Additional validation: check molecule has atoms
        if mol.GetNumAtoms() == 0:
            return MoleculeValidationResult(
                smiles=smiles,
                is_valid=False,
                mol=None,
                error_message="Molecule has no atoms"
            )
        
        return MoleculeValidationResult(
            smiles=smiles,
            is_valid=True,
            mol=mol,
            error_message=None
        )
        
    except Exception as e:
        return MoleculeValidationResult(
            smiles=smiles,
            is_valid=False,
            mol=None,
            error_message=f"Error parsing SMILES: {str(e)}"
        )


def validate_smiles_batch(smiles_list: List[str]) -> Tuple[List[MoleculeValidationResult], List[MoleculeValidationResult]]:
    """
    Validate a batch of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Tuple of (valid_results, invalid_results)
    """
    valid = []
    invalid = []
    
    for smiles in smiles_list:
        result = validate_smiles(smiles)
        if result.is_valid:
            valid.append(result)
        else:
            invalid.append(result)
    
    return valid, invalid


def get_canonical_smiles(mol: Chem.Mol) -> str:
    """
    Get canonical SMILES representation of a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Canonical SMILES string
    """
    return Chem.MolToSmiles(mol, canonical=True)
