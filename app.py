"""
Streamlit web application for molecular toxicity prediction.
"""

import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import pubchempy as pcp
from io import BytesIO

from pipeline import predict_and_explain, predict_batch, validate_smiles
from pipeline.feature_engineering import get_descriptor_summary, format_descriptor_summary
from pipeline.llm_explainer import generate_llm_explanation


# Page configuration
st.set_page_config(
    page_title="AHR Toxicity Predictor",
    page_icon="🧪",
    layout="wide"
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


def render_molecule_image(smiles: str, size: tuple = (400, 300)) -> BytesIO | None:
    """Render molecule as PNG image."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


# App header
st.title("🧪 ARIA | AhR Interactivity Assessment")
st.markdown("""#### A data-backed predictive tool for AhR agonism
---""")
st.markdown("""
The **Aryl Hydrocarbon Receptor (AhR)** is a ligand-activated transcription factor 
that is understood to respond to planar aromatic compounds including, but not limited to:
            
- Polycyclic aromatic hydrocarbons (PAHs)
- Dioxin-like halogenated aromatics
                    
Activation of AhR is associated known to be associated with undesired biological responses, including: inflammation, immunotoxicity and carcinogenicity. As a result, it is imperative to identify potential AhR activators **early** in the drug discovery pipeline, hence the utility of predictive tools as early warning systems.

⚠️ *NOTE: this tool is for screening purposes only and does not replace experimental validation. A non-toxic prediction must still be confirmed experimentally.*""")

st.markdown("---")

# Input section with tabs
st.subheader("Input")

input_tab, batch_tab = st.tabs(["🧪 Single Molecule", "📁 Batch Upload"])

# Initialize session state
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'prediction_smiles' not in st.session_state:
    st.session_state.prediction_smiles = None
if 'last_explanation' not in st.session_state:
    st.session_state.last_explanation = None
if 'explanation_error' not in st.session_state:
    st.session_state.explanation_error = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None

# ============ SINGLE MOLECULE TAB ============
with input_tab:
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
    predict_clicked = st.button("🔬 Predict Toxicity", use_container_width=True, key="single_predict")
    
    # Handle prediction
    if predict_clicked:
        # Clear previous results
        st.session_state.prediction_result = None
        st.session_state.prediction_smiles = None
        st.session_state.last_explanation = None
        st.session_state.explanation_error = None
        st.session_state.batch_results = None
        
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
                    # Store results in session state
                    st.session_state.prediction_result = result
                    st.session_state.prediction_smiles = smiles_input

# ============ BATCH UPLOAD TAB ============
with batch_tab:
    st.markdown("""
    Upload a CSV or TXT file containing SMILES strings to process multiple molecules at once.
    
    **Supported formats:**
    - **CSV**: Must contain a column named `SMILES` or `smiles`
    - **TXT**: One SMILES string per line
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'txt'],
        help="Upload CSV with SMILES column or TXT with one SMILES per line"
    )
    
    if uploaded_file is not None:
        # Parse the file
        smiles_list = []
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_type == 'csv':
                df = pd.read_csv(uploaded_file)
                # Find SMILES column (case-insensitive)
                smiles_col = None
                for col in df.columns:
                    if col.lower() == 'smiles':
                        smiles_col = col
                        break
                
                if smiles_col is None:
                    st.error("CSV must contain a column named 'SMILES' or 'smiles'")
                else:
                    smiles_list = df[smiles_col].dropna().tolist()
                    st.success(f"Found {len(smiles_list)} molecules in column '{smiles_col}'")
            else:  # txt
                content = uploaded_file.read().decode('utf-8')
                smiles_list = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
                st.success(f"Found {len(smiles_list)} molecules")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            smiles_list = []
        
        if smiles_list:
            # Show preview
            with st.expander("Preview (first 5 molecules)"):
                for i, smi in enumerate(smiles_list[:5], 1):
                    st.code(f"{i}. {smi}")
                if len(smiles_list) > 5:
                    st.caption(f"...and {len(smiles_list) - 5} more")
            
            # Process button
            if st.button("🔬 Process Batch", use_container_width=True, key="batch_predict"):
                # Clear single molecule results
                st.session_state.prediction_result = None
                st.session_state.prediction_smiles = None
                st.session_state.last_explanation = None
                st.session_state.explanation_error = None
                
                progress_bar = st.progress(0, text="Processing molecules...")
                results = []
                
                for i, smi in enumerate(smiles_list):
                    result = predict_and_explain(smi)
                    results.append(result)
                    progress_bar.progress((i + 1) / len(smiles_list), text=f"Processing {i + 1}/{len(smiles_list)}...")
                
                progress_bar.empty()
                st.session_state.batch_results = results
                st.rerun()

# ============ BATCH RESULTS DISPLAY ============
if st.session_state.batch_results is not None:
    results = st.session_state.batch_results
    
    st.markdown("---")
    st.subheader("Batch Results")
    
    # Build results dataframe
    rows = []
    for r in results:
        if r.error:
            rows.append({
                'SMILES': r.smiles,
                'Prediction': 'ERROR',
                'Probability': None,
                'Confidence': None,
                **{k: None for k in ['MolWeight', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 
                                      'NumRotatableBonds', 'NumRings', 'NumAromaticRings', 'FractionCSP3']},
                'Error': r.error
            })
        else:
            pred_label = 'TOXIC' if r.prediction == 1 else 'NON-TOXIC'
            confidence = r.probability if r.prediction == 1 else (1 - r.probability)
            
            # Get internal descriptor names
            desc_row = {}
            from pipeline.config import DESCRIPTOR_DISPLAY_TO_INTERNAL
            for display_name, value in r.descriptors.items():
                internal_name = DESCRIPTOR_DISPLAY_TO_INTERNAL.get(display_name, display_name)
                desc_row[internal_name] = round(value, 4)
            
            rows.append({
                'SMILES': r.smiles,
                'Prediction': pred_label,
                'Probability': round(r.probability, 4),
                'Confidence': round(confidence, 4),
                **desc_row,
                'Error': None
            })
    
    results_df = pd.DataFrame(rows)
    
    # Summary stats
    valid_count = sum(1 for r in results if not r.error)
    toxic_count = sum(1 for r in results if not r.error and r.prediction == 1)
    error_count = sum(1 for r in results if r.error)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(results))
    col2.metric("Toxic", toxic_count)
    col3.metric("Non-Toxic", valid_count - toxic_count)
    col4.metric("Errors", error_count)
    
    # Display table
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Download button
    csv_buffer = BytesIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv_buffer.getvalue(),
        file_name="toxicity_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

# Display results if available
if st.session_state.prediction_result is not None:
    result = st.session_state.prediction_result
    current_smiles = st.session_state.prediction_smiles
    
    st.markdown("---")
    
    # Two column layout
    col1, col2 = st.columns([1, 1])
    
    # Left column: Molecule image
    with col1:
        st.subheader("Molecular Structure")
        img_buf = render_molecule_image(current_smiles)
        if img_buf:
            st.image(img_buf, use_container_width=False)
    
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
    
    # Format descriptors for display
    formatted_descriptors = format_descriptor_summary(result.descriptors)
    
    # Display in 3 columns
    desc_cols = st.columns(3)
    desc_items = list(formatted_descriptors.items())
    
    for i, (name, value) in enumerate(desc_items):
        col_idx = i % 3
        with desc_cols[col_idx]:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color: #C4EBC8; font-size: 0.85rem; margin: 0;">{name}</p>
                <p style="font-size: 1.3rem; font-weight: bold; margin: 0.25rem 0 0 0;">{value}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # LLM Explanation section
    st.markdown("---")
    st.subheader("Explain this Result")
    st.markdown('>Note: generating an explanation incurs a small cost (a fraction of $0.01).')
    st.markdown("""
    Obtain:
    
    1. A detailed breakdown of the prediction based on 
    the molecular features and their contributions to the model's decision. 
    
    *Note, if the model cannot identify a specific functional group or substructure, it will provide a generic response. This may be somewhat inaccurate - **use your own chemical discretion***.
    
    2. An assessment of the molecule's bioavailability based on Lipinski and Verber's rules.
    """)
    
    # Explanation button
    col_btn, col_status = st.columns([1, 2])
    
    with col_btn:
        explain_disabled = st.session_state.last_explanation is not None
        explain_clicked = st.button(
            "Generate Interpretation",
            use_container_width=True,
            disabled=explain_disabled,
            key="explain_btn"
        )
    
    with col_status:
        if st.session_state.last_explanation is not None:
            st.success("✓ Interpretation generated")
    
    # Handle explanation generation
    if explain_clicked:
        with st.spinner("Generating AI interpretation..."):
            llm_result = generate_llm_explanation(
                smiles=current_smiles,
                prediction=result.prediction,
                probability=result.probability,
                descriptors=result.descriptors
            )
        
        if llm_result.success:
            st.session_state.last_explanation = llm_result.explanation
            st.session_state.explanation_error = None
            st.rerun()
        else:
            st.session_state.explanation_error = (llm_result.error_type, llm_result.error_message)
            st.rerun()
    
    # Display error if any
    if st.session_state.explanation_error:
        error_type, error_msg = st.session_state.explanation_error
        if error_type == 'credits':
            st.error(f"💳 {error_msg}")
        elif error_type == 'config':
            st.error(f"⚙️ {error_msg}")
        else:
            st.error(f"❌ {error_msg}")
    
    # Display explanation if available
    if st.session_state.last_explanation:
        st.markdown("---")
        st.markdown(st.session_state.last_explanation, unsafe_allow_html=True)
    
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
