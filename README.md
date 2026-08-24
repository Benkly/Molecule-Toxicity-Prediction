# Molecular Toxicity Prediction Pipeline

A machine learning pipeline for predicting molecular toxicity from SMILES strings, targeting the **NR-AhR (Aryl Hydrocarbon Receptor)** pathway from the Tox21 dataset.

## Overview

This pipeline takes one or more SMILES strings as input and predicts whether each molecule is likely to activate the AhR pathway - a key indicator of potential toxicity linked to inflammation, immunotoxicity, and carcinogenicity.

The model uses:
- **12 molecular descriptors** (molecular weight, LogP, TPSA, H-bond donors/acceptors, etc.)
- **ECFP-6 fingerprints** (2048-bit count vectors)
- **XGBoost classifier** optimized for PR-AUC with Bayesian hyperparameter tuning

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Application (Streamlit)

Launch the interactive web interface:
```bash
streamlit run app.py
```

Features:
- Input molecules via SMILES string or CAS registry number
- Automatic CAS to SMILES conversion using PubChem
- Visual molecule structure rendering
- Toxicity prediction with confidence score
- Full molecular descriptor summary

### Command Line

**Single molecule with full report:**
```bash
python main.py "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
```

**Multiple molecules with summary table:**
```bash
python main.py --summary "CCOc1ccc2nc(S(N)(=O)=O)sc2c1" "c1ccccc1" "CC(=O)Oc1ccccc1C(=O)O"
```

**From file (one SMILES per line):**
```bash
python main.py --file molecules.txt
python main.py --file molecules.txt --summary
```

**Options:**
- `--summary, -s`: Show summary table only (for batch predictions)
- `--file, -f`: Path to file containing SMILES strings
- `--verbose, -v`: Show progress during batch processing

### Programmatic

```python
from pipeline import predict_and_explain, predict_batch, get_batch_summary

# Single molecule
result = predict_and_explain("CCOc1ccc2nc(S(N)(=O)=O)sc2c1")
print(result.prediction)     # 1 (toxic) or 0 (non-toxic)
print(result.probability)    # Probability of toxicity (0-1)
print(result.descriptors)    # Dict of molecular properties
print(result.explanation)    # Full formatted report

# Batch processing
results = predict_batch(["SMILES1", "SMILES2", "SMILES3"])
print(get_batch_summary(results))
```

## Project Structure

```
capstone/
├── app.py                       # Streamlit web application
├── main.py                      # CLI entry point
├── requirements.txt             # Python dependencies
├── .gitignore
├── .streamlit/
│   └── config.toml              # Streamlit theme configuration
├── pipeline/
│   ├── __init__.py              # Public API exports
│   ├── config.py                # Paths, constants, model settings
│   ├── molecule_utils.py        # SMILES validation & molecule conversion
│   ├── feature_engineering.py   # Molecular descriptors & ECFP fingerprints
│   ├── model_inference.py       # Model loading & prediction
│   ├── explainer.py             # Formatted output generation
│   └── pipeline.py              # Main orchestrator
├── models/
│   ├── xgb_nrahr_model.joblib   # Trained XGBoost model
│   ├── descriptor_scaler.joblib # StandardScaler for descriptors
│   ├── optimal_threshold.joblib # Classification threshold (F2-optimized)
│   └── model_config.joblib      # Feature configuration metadata
├── data/
│   └── tox21.csv                # Training dataset
└── notebooks/
    ├── molecule-toxicity-predictor.ipynb  # Development notebook
    └── ecfp-diagram.png                   # ECFP explanation diagram
```

## Model Details

### Target: NR-AhR (Aryl Hydrocarbon Receptor)

The AhR is activated by planar aromatic compounds such as PAHs and dioxin-like halogenated aromatics. Activation is associated with:
- Inflammation
- Immunotoxicity
- Carcinogenicity

### Features

**Molecular Descriptors (12):**
| Descriptor | Description |
|------------|-------------|
| MolWeight | Molecular weight (Da) |
| LogP | Lipophilicity |
| TPSA | Topological polar surface area |
| NumHDonors | Hydrogen bond donors |
| NumHAcceptors | Hydrogen bond acceptors |
| NumRotatableBonds | Rotatable bonds |
| NumAtoms | Total atom count |
| NumHeavyAtoms | Non-hydrogen atoms |
| NumRings | Ring count |
| NumAromaticRings | Aromatic ring count |
| FractionCSP3 | Fraction of sp3 carbons |
| BertzCT | Bertz complexity index |

**ECFP Fingerprints:**
- Morgan fingerprints with radius 3 (ECFP-6)
- 2048-bit count vectors

### Performance

On held-out test set:
- **ROC-AUC**: ~0.91
- **PR-AUC**: ~0.65

The model uses an F2-optimized threshold to minimize false negatives (missed toxic compounds), as this is the more costly error in toxicity screening.

## Data Source

The model is trained on the [Tox21 dataset](https://tox21.gov), a public dataset of ~7,800 compounds screened across 12 toxicological endpoints as part of a collaborative initiative between NCATS, NTP (NIEHS), EPA, and FDA.

## Disclaimer

This prediction tool is intended for screening purposes only. Predictions should not replace experimental validation. False positives and false negatives are possible.
