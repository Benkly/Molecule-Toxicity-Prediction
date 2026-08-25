"""
Test script for the AhR Toxicity Explainer prompt.

Tests 10 molecules spanning diverse chemical space.
Estimated cost: ~$0.02-0.04 with GPT-4o-mini

Usage:
    python tests/test_explainer_prompt.py
"""

import os
import sys
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
MODEL = 'openai/gpt-4o-mini'
TEMPERATURE = 0
MAX_TOKENS = 800

PROMPT_PATH = Path(__file__).parent.parent / 'prompts' / 'explainer_system_prompt.md'


@dataclass
class ValidationResult:
    name: str
    passed: bool
    message: str
    severity: str


@dataclass 
class TestCase:
    name: str
    input_data: dict
    expected_prediction: str
    description: str


# Forbidden phrases (hallucination indicators)
FORBIDDEN_PHRASES = [
    r'\bis toxic because\b',
    r'\bcauses? toxicity\b',
    r'\bis known to\b',
    r'\bknown toxicophore\b',
    r'\bdefinitely\b',
    r'\bcertainly\b',
    r'ECFP_\d{3,4}',  # Raw bit numbers should be translated
    r'\[#\d+\]',  # Raw SMARTS
]

# Required hedging phrases
HEDGING_PHRASES = [
    r'\bthe model\b',
    r'\bsuggests?\b',
    r'\bassociates?\b',
    r'\bpredicts?\b',
    r'\bpush\w*\b.{0,15}\btoward\b',
]


def load_system_prompt() -> str:
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def create_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )


def call_explainer(client: OpenAI, system_prompt: str, input_data: dict) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': json.dumps(input_data)}
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    return response.choices[0].message.content


def validate_output(output: str, test_case: TestCase) -> List[ValidationResult]:
    results = []
    
    # 1. Check forbidden phrases
    forbidden_found = []
    for pattern in FORBIDDEN_PHRASES:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            forbidden_found.extend(matches)
    
    if forbidden_found:
        results.append(ValidationResult(
            "No forbidden phrases", False,
            f"Found: {forbidden_found[:3]}...",
            'critical'
        ))
    else:
        results.append(ValidationResult(
            "No forbidden phrases", True,
            "Clean",
            'info'
        ))
    
    # 2. Hedging language
    hedging_count = sum(1 for p in HEDGING_PHRASES if re.search(p, output, re.IGNORECASE))
    results.append(ValidationResult(
        "Hedging language",
        hedging_count >= 2,
        f"{hedging_count} phrases",
        'warning' if hedging_count < 2 else 'info'
    ))
    
    # 3. Word count
    word_count = len(output.split())
    results.append(ValidationResult(
        "Word count",
        word_count <= 450,
        f"{word_count} words",
        'warning' if word_count > 450 else 'info'
    ))
    
    # 4. Prediction mentioned
    pred_lower = test_case.expected_prediction.lower()
    has_pred = pred_lower in output.lower() or 'toxic' in output.lower()
    results.append(ValidationResult(
        "Prediction stated",
        has_pred,
        f"Expected: {test_case.expected_prediction}",
        'critical' if not has_pred else 'info'
    ))
    
    # 5. SHAP symbols
    shap_symbols = len(re.findall(r'[+−]{1,3}\s+\w+', output))
    results.append(ValidationResult(
        "SHAP formatting",
        shap_symbols > 0 or all(abs(v) < 0.3 for v in test_case.input_data.get('Top5PositiveSHAP', {}).values() if isinstance(v, (int, float))),
        f"{shap_symbols} items",
        'warning' if shap_symbols == 0 else 'info'
    ))
    
    return results


def get_test_cases() -> List[TestCase]:
    """10 test cases spanning diverse chemical space. Cost: ~$0.02-0.04"""
    return [
        TestCase(
            "1. PAH (polycyclic aromatic)",
            {
                "SMILES": "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",
                "MolecularDescriptors": {"MolWeight": 252.3, "LogP": 5.74, "TPSA": 0.0, "NumHDonors": 0, "NumHAcceptors": 0, "NumRotatableBonds": 0, "NumAtoms": 20, "NumHeavyAtoms": 20, "NumRings": 5, "NumAromaticRings": 5, "FractionCSP3": 0.0, "BertzCT": 1069.7},
                "AHRToxicityPrediction": {"Toxic": "98.1%"},
                "Top5PositiveSHAP": {"ECFP_1380": 0.80, "FractionCSP3": 0.80, "BertzCT": 0.50, "NumAromaticRings": 0.49, "ECFP_1855": 0.39},
                "Bottom5NegativeSHAP": {"ECFP_1344": -0.57, "TPSA": -0.50, "ECFP_267": -0.29, "ECFP_199": -0.26, "NumHAcceptors": -0.15},
                "TopSubstructureContributions": {"ECFP_1855": "[#6]:[#6](:[#6]):[#6]"}
            },
            "Toxic", "Large fused aromatic"
        ),
        TestCase(
            "2. Simple amide (paracetamol-like)",
            {
                "SMILES": "CC(=O)Nc1ccc(O)cc1",
                "MolecularDescriptors": {"MolWeight": 151.2, "LogP": 1.35, "TPSA": 49.3, "NumHDonors": 2, "NumHAcceptors": 2, "NumRotatableBonds": 1, "NumAtoms": 11, "NumHeavyAtoms": 11, "NumRings": 1, "NumAromaticRings": 1, "FractionCSP3": 0.13, "BertzCT": 253.3},
                "AHRToxicityPrediction": {"prediction": "Non-Toxic", "confidence": "92.8%"},
                "Top5PositiveSHAP": {"FractionCSP3": 0.46, "NumHDonors": 0.26, "ECFP_1152": 0.17, "TPSA": 0.17, "ECFP_1816": 0.10},
                "Bottom5NegativeSHAP": {"ECFP_1664": -0.54, "ECFP_1380": -0.50, "BertzCT": -0.48, "LogP": -0.45, "MolWeight": -0.40},
                "TopSubstructureContributions": {"ECFP_1664": "[#7]-[#6](=[#8])-[#6]"}
            },
            "Non-Toxic", "Drug-like amide"
        ),
        TestCase(
            "3. Chlorinated biphenyl",
            {
                "SMILES": "Clc1ccc(Cl)c(c1)c2ccccc2Cl",
                "MolecularDescriptors": {"MolWeight": 257.5, "LogP": 5.2, "TPSA": 0.0, "NumHDonors": 0, "NumHAcceptors": 0, "NumRotatableBonds": 1, "NumAtoms": 15, "NumHeavyAtoms": 15, "NumRings": 2, "NumAromaticRings": 2, "FractionCSP3": 0.0, "BertzCT": 485.2},
                "AHRToxicityPrediction": {"Toxic": "87.5%"},
                "Top5PositiveSHAP": {"ECFP_1892": 0.72, "LogP": 0.58, "FractionCSP3": 0.45, "ECFP_1380": 0.38, "NumAromaticRings": 0.32},
                "Bottom5NegativeSHAP": {"BertzCT": -0.28, "ECFP_556": -0.22, "MolWeight": -0.15, "TPSA": -0.12, "NumRotatableBonds": -0.08},
                "TopSubstructureContributions": {"ECFP_1892": "[#6]1:[#6]:[#6](-[#17]):[#6]:[#6]:[#6]:1"}
            },
            "Toxic", "PCB-like halogenated"
        ),
        TestCase(
            "4. Marginal call (low SHAP)",
            {
                "SMILES": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
                "MolecularDescriptors": {"MolWeight": 206.3, "LogP": 3.5, "TPSA": 37.3, "NumHDonors": 1, "NumHAcceptors": 2, "NumRotatableBonds": 4, "NumAtoms": 15, "NumHeavyAtoms": 15, "NumRings": 1, "NumAromaticRings": 1, "FractionCSP3": 0.46, "BertzCT": 320.5},
                "AHRToxicityPrediction": {"prediction": "Non-Toxic", "confidence": "55%"},
                "Top5PositiveSHAP": {"ECFP_892": 0.15, "LogP": 0.12, "FractionCSP3": 0.08, "ECFP_1203": 0.05, "NumRings": 0.03},
                "Bottom5NegativeSHAP": {"TPSA": -0.18, "BertzCT": -0.14, "MolWeight": -0.09, "ECFP_445": -0.06, "NumHAcceptors": -0.04},
                "TopSubstructureContributions": {"ECFP_892": "[#6]-[#6](-[#6])-[#6]"}
            },
            "Non-Toxic", "All SHAP < 0.30"
        ),
        TestCase(
            "5. Pyridine heterocycle",
            {
                "SMILES": "Cc1ncccc1C(=O)N",
                "MolecularDescriptors": {"MolWeight": 136.2, "LogP": 0.45, "TPSA": 55.4, "NumHDonors": 1, "NumHAcceptors": 2, "NumRotatableBonds": 1, "NumAtoms": 10, "NumHeavyAtoms": 10, "NumRings": 1, "NumAromaticRings": 1, "FractionCSP3": 0.14, "BertzCT": 218.6},
                "AHRToxicityPrediction": {"prediction": "Non-Toxic", "confidence": "88.2%"},
                "Top5PositiveSHAP": {"FractionCSP3": 0.35, "ECFP_902": 0.22, "NumHDonors": 0.18, "ECFP_1380": 0.15, "NumAromaticRings": 0.12},
                "Bottom5NegativeSHAP": {"ECFP_1445": -0.62, "BertzCT": -0.45, "LogP": -0.38, "MolWeight": -0.35, "TPSA": -0.28},
                "TopSubstructureContributions": {"ECFP_1445": "[#7]:[#6]:[#6]"}
            },
            "Non-Toxic", "N-heterocycle"
        ),
        TestCase(
            "6. Steroid (saturated 3D)",
            {
                "SMILES": "CC12CCC3C(CCC4CC(O)CCC34C)C1CCC2O",
                "MolecularDescriptors": {"MolWeight": 290.4, "LogP": 3.8, "TPSA": 40.5, "NumHDonors": 2, "NumHAcceptors": 2, "NumRotatableBonds": 0, "NumAtoms": 21, "NumHeavyAtoms": 21, "NumRings": 4, "NumAromaticRings": 0, "FractionCSP3": 0.89, "BertzCT": 612.3},
                "AHRToxicityPrediction": {"prediction": "Non-Toxic", "confidence": "94.5%"},
                "Top5PositiveSHAP": {"NumRings": 0.22, "BertzCT": 0.18, "LogP": 0.15, "MolWeight": 0.12, "NumHeavyAtoms": 0.08},
                "Bottom5NegativeSHAP": {"FractionCSP3": -0.85, "NumAromaticRings": -0.72, "ECFP_1380": -0.55, "ECFP_234": -0.42, "TPSA": -0.28},
                "TopSubstructureContributions": {}
            },
            "Non-Toxic", "High CSP3, no aromaticity"
        ),
        TestCase(
            "7. Flavonoid (borderline)",
            {
                "SMILES": "O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",
                "MolecularDescriptors": {"MolWeight": 270.2, "LogP": 1.9, "TPSA": 90.9, "NumHDonors": 3, "NumHAcceptors": 5, "NumRotatableBonds": 1, "NumAtoms": 19, "NumHeavyAtoms": 19, "NumRings": 3, "NumAromaticRings": 3, "FractionCSP3": 0.0, "BertzCT": 598.4},
                "AHRToxicityPrediction": {"Toxic": "62.3%"},
                "Top5PositiveSHAP": {"FractionCSP3": 0.55, "NumAromaticRings": 0.48, "BertzCT": 0.35, "ECFP_1380": 0.32, "ECFP_788": 0.28},
                "Bottom5NegativeSHAP": {"TPSA": -0.58, "NumHDonors": -0.45, "NumHAcceptors": -0.38, "LogP": -0.32, "ECFP_1122": -0.25},
                "TopSubstructureContributions": {"ECFP_788": "[#6]=[#8]"}
            },
            "Toxic", "Competing signals"
        ),
        TestCase(
            "8. Sulfonamide",
            {
                "SMILES": "CC(=O)Nc1ccc(S(=O)(=O)N)cc1",
                "MolecularDescriptors": {"MolWeight": 214.2, "LogP": 0.12, "TPSA": 92.2, "NumHDonors": 2, "NumHAcceptors": 4, "NumRotatableBonds": 2, "NumAtoms": 14, "NumHeavyAtoms": 14, "NumRings": 1, "NumAromaticRings": 1, "FractionCSP3": 0.13, "BertzCT": 342.8},
                "AHRToxicityPrediction": {"prediction": "Non-Toxic", "confidence": "91.2%"},
                "Top5PositiveSHAP": {"FractionCSP3": 0.38, "ECFP_1380": 0.25, "NumAromaticRings": 0.18, "ECFP_1855": 0.15, "NumHDonors": 0.12},
                "Bottom5NegativeSHAP": {"ECFP_1998": -0.68, "TPSA": -0.55, "LogP": -0.48, "NumHAcceptors": -0.42, "BertzCT": -0.35},
                "TopSubstructureContributions": {"ECFP_1998": "[#16](=[#8])(=[#8])-[#7]"}
            },
            "Non-Toxic", "High polarity"
        ),
        TestCase(
            "9. Nitroaromatic",
            {
                "SMILES": "O=[N+]([O-])c1ccc2ccccc2c1",
                "MolecularDescriptors": {"MolWeight": 173.2, "LogP": 3.1, "TPSA": 52.9, "NumHDonors": 0, "NumHAcceptors": 2, "NumRotatableBonds": 1, "NumAtoms": 13, "NumHeavyAtoms": 13, "NumRings": 2, "NumAromaticRings": 2, "FractionCSP3": 0.0, "BertzCT": 398.5},
                "AHRToxicityPrediction": {"Toxic": "78.9%"},
                "Top5PositiveSHAP": {"FractionCSP3": 0.62, "ECFP_1380": 0.48, "NumAromaticRings": 0.42, "LogP": 0.35, "ECFP_2045": 0.32},
                "Bottom5NegativeSHAP": {"ECFP_1567": -0.38, "TPSA": -0.32, "NumHAcceptors": -0.25, "BertzCT": -0.18, "MolWeight": -0.12},
                "TopSubstructureContributions": {"ECFP_2045": "[#7+](=[#8])-[#8-]"}
            },
            "Toxic", "Electron-deficient aromatic"
        ),
        TestCase(
            "10. Large glycoside (Lipinski violation)",
            {
                "SMILES": "OCC1OC(Oc2ccc(cc2)C3=CC(=O)c4c(O)cc(O)cc4O3)C(O)C(O)C1O",
                "MolecularDescriptors": {"MolWeight": 432.4, "LogP": -0.5, "TPSA": 170.1, "NumHDonors": 6, "NumHAcceptors": 10, "NumRotatableBonds": 5, "NumAtoms": 30, "NumHeavyAtoms": 30, "NumRings": 4, "NumAromaticRings": 3, "FractionCSP3": 0.26, "BertzCT": 825.6},
                "AHRToxicityPrediction": {"prediction": "Non-Toxic", "confidence": "85.5%"},
                "Top5PositiveSHAP": {"NumAromaticRings": 0.42, "BertzCT": 0.35, "FractionCSP3": 0.28, "ECFP_1380": 0.22, "NumRings": 0.18},
                "Bottom5NegativeSHAP": {"TPSA": -0.88, "NumHDonors": -0.72, "NumHAcceptors": -0.65, "LogP": -0.58, "MolWeight": -0.45},
                "TopSubstructureContributions": {}
            },
            "Non-Toxic", "Violates Lipinski"
        ),
    ]


def print_results(test_case: TestCase, output: str, results: List[ValidationResult]) -> bool:
    print(f"\n{'─'*60}")
    print(f"TEST: {test_case.name}")
    print(f"Desc: {test_case.description}")
    print(f"{'─'*60}")
    
    # Truncate output for display
    display_output = output[:800] + "..." if len(output) > 800 else output
    print(f"\n{display_output}\n")
    
    print("VALIDATION:")
    all_passed = True
    for r in results:
        icon = "✓" if r.passed else "✗"
        if not r.passed:
            all_passed = False
        print(f"  {icon} {r.name}: {r.message}")
    
    return all_passed


def main():
    print("=" * 60)
    print("AhR EXPLAINER PROMPT TEST (10 molecules)")
    print("Estimated cost: ~$0.02-0.04")
    print("=" * 60)
    
    system_prompt = load_system_prompt()
    print(f"Prompt loaded: {len(system_prompt):,} chars")
    
    client = create_client()
    test_cases = get_test_cases()
    
    passed_count = 0
    failed_count = 0
    
    for i, tc in enumerate(test_cases, 1):
        print(f"\n[{i}/10] Testing: {tc.name}...")
        try:
            output = call_explainer(client, system_prompt, tc.input_data)
            results = validate_output(output, tc)
            if print_results(tc, output, results):
                passed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed_count += 1
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed_count}/10 passed, {failed_count}/10 failed")
    print("=" * 60)
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    exit(main())
