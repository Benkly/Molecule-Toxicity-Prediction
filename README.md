# Aryl Hydrocarbon Receptor Toxicity Prediction Web App

A machine learning web application for predicting **molecular toxicity** from SMILES strings, targeting the **NR-AhR (Aryl Hydrocarbon Receptor)** pathway from the Tox21 dataset.

## Overview

This application takes SMILES strings as input and predicts whether each molecule is likely to activate the AhR pathway - a key indicator of potential toxicity linked to inflammation, immunotoxicity, and carcinogenicity.

The model uses:
- **12 molecular descriptors** (molecular weight, LogP, TPSA, H-bond donors/acceptors, etc.)
- **ECFP-6 fingerprints** (2048-bit count vectors)
- **XGBoost classifier** optimized for PR-AUC with Bayesian hyperparameter tuning

**Refer to the `molecule-toxicity-predictor.ipynb` notebook for a detailed breakdown of the process used to create and evaluate the model.**

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

3. Configure API key for AI explanations (optional):

Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

The AI interpretation feature requires an [OpenRouter](https://openrouter.ai) API key. The app will function without it, but the "Generate Interpretation" button will be disabled.

## Usage

Launch the web application:
```bash
streamlit run app.py
```

### Single Molecule Analysis

- Input molecules via SMILES string or CAS registry number
- Automatic CAS to SMILES conversion using PubChem
- Visual molecule structure rendering
- Toxicity prediction with confidence score
- Full molecular descriptor summary
- **AI-powered interpretation**: Generate natural language explanations of predictions using GPT-4o-mini, based on SHAP feature attributions

### Batch Upload

- Upload CSV files (with a `SMILES` column) or TXT files (one SMILES per line)
- Process multiple molecules at once with progress tracking
- View results in an interactive table
- Download results as CSV with all predictions and descriptors

*Note: AI interpretation is available for single molecules only, not batch uploads.*

## Project Structure

```
capstone/
├── app.py                             # Streamlit web application
├── requirements.txt                   # Python dependencies
├── molecule-toxicity-predictor.ipynb  # Development notebook
├── ecfp-diagram.png                   # ECFP explanation diagram
├── .gitignore
├── .env                               # API key configuration (not in repo)
├── .streamlit/
│   └── config.toml                    # Streamlit theme configuration
├── pipeline/
│   ├── __init__.py                    # Public API exports
│   ├── config.py                      # Paths, constants, model settings
│   ├── molecule_utils.py              # SMILES validation & molecule conversion
│   ├── feature_engineering.py         # Molecular descriptors & ECFP fingerprints
│   ├── model_inference.py             # Model loading & prediction
│   ├── llm_explainer.py               # GenAI-powered SHAP-based explanations
│   └── pipeline.py                    # Main orchestrator
├── prompts/
│   └── explainer_system_prompt.md     # LLM system prompt for interpretations
├── models/
│   ├── xgb_nrahr_model.joblib         # Trained XGBoost model
│   ├── descriptor_scaler.joblib       # StandardScaler for descriptors
│   ├── optimal_threshold.joblib       # Classification threshold (F2-optimized)
│   └── model_config.joblib            # Feature configuration metadata
└── data/
    └── tox21.csv                      # Training dataset
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

## AI Interpretation

The web application includes an optional AI-powered interpretation feature that generates natural language explanations of predictions.

**How it works:**
1. SHAP (SHapley Additive exPlanations) values are computed for each prediction using the XGBoost model
2. The top contributing features (both positive and negative) are identified
3. ECFP fingerprint bits are mapped back to substructure SMARTS patterns where possible
4. This data is sent to GPT-4o-mini via OpenRouter with a carefully crafted system prompt
5. The LLM generates a human-readable explanation that describes *what the model weighted*, not causal claims about toxicity

**Anti-hallucination measures:**
- The prompt explicitly forbids claims about known toxicophores or biochemical mechanisms
- All statements are framed as model correlations, not chemical facts
- ECFP bit numbers are translated to chemical language (never shown raw)
- When a substructure cannot be identified from an ECFP bit, the explanation explicitly flags this as "an unidentified structural feature" rather than guessing
- A self-verification checklist ensures output quality

**Cost:**
- Approximately $0.003-0.005 per explanation using GPT-4o-mini

## Data Source

The model is trained on the [Tox21 dataset](https://tox21.gov), a public dataset of ~7,800 compounds screened across 12 toxicological endpoints as part of a collaborative initiative between NCATS, NTP (NIEHS), EPA, and FDA.

## Disclaimer

This prediction tool is intended for screening purposes only. Predictions should not replace experimental validation. False positives and false negatives are possible.
