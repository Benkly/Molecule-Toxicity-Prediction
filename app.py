"""
Streamlit web application for molecular toxicity prediction.
"""

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import pubchempy as pcp
from io import BytesIO

from pipeline import predict_and_explain, validate_smiles
from pipeline.feature_engineering import get_descriptor_summary


# Page configuration
st.set_page_config(
    page_title="AHR Toxicity Predictor",
    page_icon="🧪",
    layout="centered"
)

# Custom CSS for color palette
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #202A25;
    }
    
    /* Text color */
    .stApp, .stMarkdown, p, span, label, .stTextInput label {
        color: #FAFAFA !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #2a3830;
        color: #FAFAFA;
        border: 1px solid #5F4BB6;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #5F4BB6;
        color: #FAFAFA;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background-color: #7a66c9;
        color: #FAFAFA;
    }
    
    /* Success/toxic boxes */
    .prediction-toxic {
        background-color: rgba(222, 26, 26, 0.2);
        border: 2px solid #DE1A1A;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-nontoxic {
        background-color: rgba(196, 235, 200, 0.2);
        border: 2px solid #C4EBC8;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #2a3830;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #5F4BB6;
    }
    
    /* Descriptor table */
    .descriptor-table {
        background-color: #2a3830;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Radio buttons */
    .stRadio > div {
        color: #FAFAFA;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2a3830;
        color: #FAFAFA;
        border-radius: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #5F4BB6 !important;
    }
    
    /* Divider */
    hr {
        border-color: #5F4BB6;
    }
    
    /* Error messages */
    .stAlert {
        background-color: rgba(95, 75, 182, 0.2);
        color: #FAFAFA;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2a3830;
        color: #FAFAFA !important;
    }
</style>
""", unsafe_allow_html=True)


def cas_to_smiles(cas_number: str) -> tuple[str | None, str | None]:
    """
    Convert CAS registry number to SMILES using PubChem.
    
    Returns:
        Tuple of (smiles, error_message)
    """
    try:
        # Clean input
        cas_number = cas_number.strip()
        
        # Search PubChem by CAS
        compounds = pcp.get_compounds(cas_number, 'name')
        
        if not compounds:
            return None, f"No compound found for CAS: {cas_number}"
        
        # Get canonical SMILES from first result
        smiles = compounds[0].canonical_smiles
        
        if not smiles:
            return None, f"No SMILES available for CAS: {cas_number}"
        
        return smiles, None
        
    except Exception as e:
        return None, f"Error looking up CAS: {str(e)}"


def render_molecule_image(smiles: str) -> BytesIO | None:
    """Render molecule as PNG image."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    img = Draw.MolToImage(mol, size=(400, 300))
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


# App header
st.title("🧪 AhR Toxicity Predictor")
st.markdown("""The **Aryl Hydrocarbon Receptor (AhR)** is a ligand-activated transcription factor 
            that is understood to respond to planar aromatic compounds including, but not limited to:
            
- Polycyclic aromatic hydrocarbons (PAHs)
- Dioxin-like halogenated aromatics
                    
Activation of AhR is associated known to be associated with undesired biological responses, including: inflammation, immunotoxicity and carcinogenicity. As a result, it is imperative to identify potential AhR activators **early** in the drug discovery pipeline, hence the utility of predictive tools as early warning systems.

This tool provides users with the means to recieve a data-backed prediction of AhR agonism. 

⚠️ *NOTE: this tool is for screening purposes only and does not replace experimental validation. A non-toxic prediction must still be confirmed experimentally.*""")

st.markdown("---")

# Input section
st.subheader("Input Molecule")

input_type = st.radio(
    "Input type:",
    ["SMILES String", "CAS Registry Number"],
    horizontal=True
)

smiles_input = None
error_message = None

if input_type == "SMILES String":
    smiles_input = st.text_input(
        "Enter SMILES string:",
        placeholder="e.g., CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
    )
else:
    cas_input = st.text_input(
        "Enter CAS Registry Number:",
        placeholder="e.g., 58-08-2 (caffeine)"
    )
    
    if cas_input:
        with st.spinner("Looking up CAS number..."):
            smiles_input, error_message = cas_to_smiles(cas_input)
        
        if smiles_input:
            st.success(f"Found SMILES: `{smiles_input}`")

# Predict button
predict_clicked = st.button("🔬 Predict Toxicity", use_container_width=True)

# Results section
if predict_clicked:
    if error_message:
        st.error(error_message)
    elif not smiles_input:
        st.warning("Please enter a molecule")
    else:
        # Validate SMILES
        validation = validate_smiles(smiles_input)
        
        if not validation.is_valid:
            st.error(f"Invalid SMILES: {validation.error_message}")
        else:
            with st.spinner("Analyzing molecule..."):
                result = predict_and_explain(smiles_input)
            
            if result.error:
                st.error(f"Prediction error: {result.error}")
            else:
                st.markdown("---")
                
                # Two column layout
                col1, col2 = st.columns([1, 1])
                
                # Left column: Molecule image
                with col1:
                    st.subheader("Molecular Structure")
                    img_buf = render_molecule_image(smiles_input)
                    if img_buf:
                        st.image(img_buf, use_container_width=True)
                
                # Right column: Prediction result
                with col2:
                    st.subheader("Prediction Result")
                    
                    is_toxic = result.prediction == 1
                    probability = result.probability
                    confidence = probability if is_toxic else (1 - probability)
                    
                    if is_toxic:
                        st.markdown(f"""
                        <div class="prediction-toxic">
                            <h2 style="color: #DE1A1A; margin: 0;">⚠️ TOXIC</h2>
                            <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                                Likely to activate NR-AhR pathway
                            </p>
                            <p style="font-size: 2rem; font-weight: bold; color: #DE1A1A; margin: 0.5rem 0;">
                                {probability:.1%}
                            </p>
                            <p style="color: #FAFAFA; opacity: 0.8;">Toxicity Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-nontoxic">
                            <h2 style="color: #C4EBC8; margin: 0;">✓ NON-TOXIC</h2>
                            <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                                Unlikely to activate NR-AhR pathway
                            </p>
                            <p style="font-size: 2rem; font-weight: bold; color: #C4EBC8; margin: 0.5rem 0;">
                                {confidence:.1%}
                            </p>
                            <p style="color: #FAFAFA; opacity: 0.8;">Confidence</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Molecular descriptors section
                st.markdown("---")
                st.subheader("Molecular Descriptors")
                
                descriptors = result.descriptors
                
                # Display in 3 columns
                desc_cols = st.columns(3)
                desc_items = list(descriptors.items())
                
                for i, (name, value) in enumerate(desc_items):
                    col_idx = i % 3
                    with desc_cols[col_idx]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <p style="color: #C4EBC8; font-size: 0.85rem; margin: 0;">{name}</p>
                            <p style="font-size: 1.3rem; font-weight: bold; margin: 0.25rem 0 0 0;">{value}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # About section
                st.markdown("---")
                with st.expander("ℹ️ About the NR-AhR Assay"):
                    st.markdown("""
                    **Model Details:**
                    - Trained on the Tox21 dataset (~7,800 compounds)
                    - Uses molecular descriptors + ECFP-6 fingerprints
                    - XGBoost classifier optimized to *minimise* false negatives
                    
                    ---
                    
                    **Performance (on test data):**
                    - Overall accuracy: 90.2%
                    - % Toxins Misclassified: 20.8%
                    
                    ⚠️ *REMEMBER: this tool is for screening purposes only and does not replace experimental validation.*
                    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; opacity: 0.6;'>Built with Streamlit | Data: Tox21 Dataset</p>",
    unsafe_allow_html=True
)
