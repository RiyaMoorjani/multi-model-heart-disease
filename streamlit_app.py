import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import shap
from PIL import Image

# Ensure project adapters and models are in the search path
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("./ecg_ptbxl_benchmarking-master/ecg_ptbxl_benchmarking-master/code"))
sys.path.append(os.path.abspath("./torchxrayvision-main/torchxrayvision-main"))

# Import clinical pipeline components
from database import HIPAADatabase
from ecg_adapter import load_ecg_model, load_ecg_metadata, preprocess_signal, predict_ecg
from vision_adapter import load_vision_model, preprocess_xray, predict_xray
from nlp_adapter import extract_clinical_features
from fusion import evaluate_logic_gate, assemble_fused_matrix, calculate_multimodal_risk, ALL_FEATURE_COLUMNS
from train_xgboost import run_training

# Page Configuration for Precision Clinical Workstation
st.set_page_config(
    page_title="Precision Clinical Workstation",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Light-Theme CSS styling for a realistic, non-AI clinical EHR interface
st.markdown("""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global style overrides */
    html, body, [data-testid="stAppViewContainer"], .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: #f8fafc !important;
        color: #1e293b !important;
    }
    
    /* Clean sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #cbd5e1 !important;
    }
    
    /* Headers typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #0f172a !important;
        letter-spacing: -0.01em;
    }
    
    /* Clinical Report Panels & Containers */
    .report-card {
        padding: 20px;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        background-color: #ffffff;
        margin-bottom: 16px;
        font-family: 'Inter', sans-serif;
    }
    
    /* Clinical Alerts */
    .clinical-alert {
        padding: 16px;
        border-radius: 4px;
        margin-bottom: 20px;
        font-family: 'Inter', sans-serif;
        font-size: 13.5px;
        line-height: 1.6;
    }
    .alert-elevated {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .alert-low {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .alert-neutral {
        background-color: #ffffff;
        color: #334155;
        border: 1px solid #cbd5e1;
    }
    
    /* Clinical Badges */
    .clinical-badge {
        padding: 10px;
        border-radius: 4px;
        text-align: center;
        margin-bottom: 8px;
        font-family: 'Inter', sans-serif;
    }
    .badge-present {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    .badge-absent {
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    /* Standardizing Form Input Widgets */
    div[data-baseweb="input"], div[data-baseweb="select"], div[data-baseweb="textarea"] {
        border-radius: 4px !important;
    }
    
    /* Stylizing buttons to look professional and grid-aligned */
    div.stButton > button:first-child {
        background-color: #0c4a6e !important; /* Sky 900 for high-contrast actions */
        color: #ffffff !important;
        border: 1px solid #0c4a6e !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
        font-size: 13.5px !important;
        padding: 6px 16px !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #0369a1 !important; /* Sky 700 */
        border-color: #0369a1 !important;
    }
    
    /* Tab formatting */
    button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: #64748b !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0c4a6e !important;
        font-weight: 600 !important;
    }
    
    /* Vitals Metric Value definitions */
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #0f172a;
        font-family: monospace;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INITIALIZATION -----------------
if 'tabular_risk' not in st.session_state:
    st.session_state.tabular_risk = None
if 'tabular_prediction' not in st.session_state:
    st.session_state.tabular_prediction = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = None
if 'ecg_probs' not in st.session_state:
    st.session_state.ecg_probs = None
if 'cxr_probs' not in st.session_state:
    st.session_state.cxr_probs = None
if 'nlp_features' not in st.session_state:
    st.session_state.nlp_features = None
if 'triage_activated' not in st.session_state:
    st.session_state.triage_activated = False

# Initialize database
db = HIPAADatabase()

# Initialize patient session state
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = "John Doe"
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = "EHR-2026-90412"

# ----------------- SIDEBAR CONSOLE CONTROLS -----------------
st.sidebar.markdown("<div style='font-size: 14px; font-weight: 700; color: #0c4a6e; margin-bottom: 8px;'>CLINICAL CONSOLE CONTROL</div>", unsafe_allow_html=True)

st.sidebar.markdown("**Clinician Credentials**")
doctor_name = st.sidebar.text_input("Name:", value="Dr. Sarah Carter")
doctor_npi = st.sidebar.text_input("NPI Number:", value="1982736450")
department = st.sidebar.selectbox("Department:", ["Cardiology", "Emergency Medicine", "Internal Medicine"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Patient Demographics**")
patient_name = st.sidebar.text_input("Full Name:", value=st.session_state.patient_name)
patient_id = st.sidebar.text_input("EHR ID Number:", value=st.session_state.patient_id)
age = st.sidebar.number_input("Age (Years):", min_value=1, max_value=120, value=58)
gender = st.sidebar.selectbox("Gender / Sex:", ["Male", "Female"])

# Sync demographic inputs to session state
st.session_state.patient_name = patient_name
st.session_state.patient_id = patient_id

st.sidebar.markdown("---")
st.sidebar.markdown("**Hugging Face Integration**")
hf_token = st.sidebar.text_input("HF API Token:", type="password", help="Needed to run microsoft/Phi-3-mini-4k-instruct for explanations")


st.sidebar.markdown("---")
st.sidebar.markdown("**Diagnostic Gatekeeper Status**")
if st.session_state.triage_activated:
    st.sidebar.markdown("Fold 2 Diagnostics: <span style='color: #991b1b; font-weight: 600;'>ACTIVE</span>", unsafe_allow_html=True)
    st.sidebar.caption("High-compute clinical diagnostics (ECG, Chest X-Ray, BioBERT Notes) are currently active.")
else:
    st.sidebar.markdown("Fold 2 Diagnostics: <span style='color: #475569; font-weight: 600;'>STANDBY</span>", unsafe_allow_html=True)
    st.sidebar.caption("Engages automatically if Tabular Risk >= 15% or scans/notes are uploaded.")

# Load pre-trained models safely
with st.spinner("Loading clinical models..."):
    # XGBoost model
    xgb_path = "./Heart_disease_report_AI-main/heart_model1.pkl"
    tabular_model = None
    if os.path.exists(xgb_path):
        with open(xgb_path, "rb") as f:
            tabular_model = pickle.load(f)
            
    # ECG models
    ecg_model = load_ecg_model()
    scaler, mlb, diag_map = load_ecg_metadata()
    
    # CXR models
    vision_model = load_vision_model()

# ----------------- MAIN INTERFACE -----------------
st.title("Precision Clinical Workstation")
st.markdown("<div style='font-size: 13px; color: #64748b; margin-top: -12px; margin-bottom: 20px;'>EHR Integration Gateway & Multi-Modal Clinical Decision Support System</div>", unsafe_allow_html=True)

# Dynamic EHR Patient Identification Header
triage_status_text = "Pending Assessment"
triage_color = "#475569" # Slate-600
triage_bg = "#f8fafc"
triage_border = "#cbd5e1"

if st.session_state.tabular_risk is not None:
    if st.session_state.triage_activated:
        triage_status_text = "Fold 2 Active (Elevated Risk)"
        triage_color = "#991b1b" # Red-800
        triage_bg = "#fef2f2"
        triage_border = "#fecaca"
    else:
        triage_status_text = "Fold 1 Triaged (Low Risk)"
        triage_color = "#166534" # Green-800
        triage_bg = "#f0fdf4"
        triage_border = "#bbf7d0"

st.markdown(f"""
<div style="background-color: #ffffff; padding: 14px 18px; border: 1px solid #cbd5e1; border-radius: 4px; margin-bottom: 24px; font-family: 'Inter', sans-serif; line-height: 1.5;">
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e2e8f0; padding-bottom: 10px; margin-bottom: 10px;">
        <div>
            <span style="font-weight: 700; color: #0f172a; font-size: 14px; letter-spacing: 0.05em; text-transform: uppercase;">Clinical Electronic Health Record (EHR)</span>
            <span style="color: #64748b; font-size: 12px; margin-left: 8px;">| Active Session Context</span>
        </div>
        <div>
            <span style="background-color: {triage_bg}; color: {triage_color}; border: 1px solid {triage_border}; font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 3px; letter-spacing: 0.05em; text-transform: uppercase;">
                {triage_status_text}
            </span>
        </div>
    </div>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; font-size: 13px;">
        <div><span style="color: #64748b;">Patient Name:</span> <strong style="color: #0f172a;">{patient_name}</strong></div>
        <div><span style="color: #64748b;">Patient ID / MRN:</span> <strong style="color: #0f172a; font-family: monospace;">{patient_id}</strong></div>
        <div><span style="color: #64748b;">Age / Sex:</span> <strong style="color: #0f172a;">{age} yrs / {gender}</strong></div>
        <div><span style="color: #64748b;">Encounter Facility:</span> <strong style="color: #0f172a;">Metro Health Medical Center</strong></div>
        <div><span style="color: #64748b;">Attending Clinician:</span> <strong style="color: #0f172a;">{doctor_name} (NPI: {doctor_npi})</strong></div>
        <div><span style="color: #64748b;">Clinical Department:</span> <strong style="color: #0f172a;">{department}</strong></div>
        <div><span style="color: #64748b;">Ingestion Gateway:</span> <strong style="color: #0f172a;">Fold 1 Tabular</strong></div>
        <div><span style="color: #64748b;">System Version:</span> <strong style="color: #0f172a; font-family: monospace;">OPCW-v2.6.2</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "Tabular Screening (Fold 1)", 
    "Advanced Diagnostics (Fold 2)", 
    "Multimodal Data Fusion", 
    "EHR Archive & Auditing", 
    "Calibration & Training"
])

# ----------------- TAB 1: TABULAR TRIAGE (FOLD 1) -----------------
with tabs[0]:
    st.markdown("### Primary Vitals & Screening Parameters")
    st.write("Input patient physiological metrics to evaluate baseline cardiovascular risk via the XGBoost inference engine.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Clinical Vitals**")
        # Demographics are read globally from the active Patient Session context in the sidebar
        
        # Mapping values to original numeric representations for XGBoost
        cp_type = st.selectbox(
            "Chest Pain Type (Anginal History):", 
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-Anginal Pain",
                3: "Asymptomatic"
            }[x]
        )
        
        resting_bp = st.number_input("Resting Systolic Blood Pressure (mm Hg):", min_value=50, max_value=250, value=120)
        cholesterol = st.number_input("Serum Cholesterol Level (mg/dl):", min_value=100, max_value=600, value=240)
        
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl:",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        
    with col2:
        st.markdown("**Stress & Electrocardiographic Vitals**")
        restecg = st.selectbox(
            "Resting ECG Findings:",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Wave Abnormality",
                2: "Left Ventricular Hypertrophy"
            }[x]
        )
        
        max_hr = st.number_input("Maximum Heart Rate Achieved (bpm):", min_value=60, max_value=220, value=160)
        
        exang = st.selectbox(
            "Exercise-Induced Angina:",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        
        oldpeak = st.number_input("ST Segment Depression Induced by Exercise (oldpeak):", min_value=0.0, max_value=10.0, value=1.0)
        
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment:",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Upsloping",
                1: "Flat",
                2: "Downsloping"
            }[x]
        )
        
        ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0-3):", min_value=0, max_value=3, value=0)
        
        thal = st.selectbox(
            "Thalassemia Type (Stress Perfusion Scan):",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Normal",
                1: "Fixed Defect",
                2: "Reversible Defect",
                3: "Unspecified"
            }[x]
        )
        
    st.markdown("---")
    
    if st.button("Execute Primary Risk Triage", type="primary"):
        if tabular_model is None:
            st.error("XGBoost model file (heart_model1.pkl) is missing. Please run model calibration in Tab 5 first.")
        else:
            gender_numeric = 1 if gender == "Male" else 0
            
            # Format inputs as 2D numpy array
            input_vector = np.array([[
                float(age), float(gender_numeric), float(cp_type), float(resting_bp),
                float(cholesterol), float(fbs), float(restecg), float(max_hr),
                float(exang), float(oldpeak), float(slope), float(ca), float(thal)
            ]])
            
            # Run prediction probability
            probs = tabular_model.predict_proba(input_vector)[0]
            pred_label = int(tabular_model.predict(input_vector)[0])
            risk_prob = float(probs[1])
            
            # Save predictions to session state
            st.session_state.tabular_risk = risk_prob
            st.session_state.tabular_prediction = pred_label
            st.session_state.input_data = input_vector
            
            # Generate SHAP explanations
            try:
                explainer = shap.Explainer(tabular_model)
                shap_vals = explainer(input_vector)
                st.session_state.shap_values = shap_vals
            except Exception as e1:
                print(f"[SHAP ERROR] shap.Explainer failed: {e1}")
                try:
                    explainer = shap.TreeExplainer(tabular_model)
                    shap_vals = explainer(input_vector)
                    st.session_state.shap_values = shap_vals
                    print("[SHAP] Successfully calculated using TreeExplainer fallback.")
                except Exception as e2:
                    print(f"[SHAP ERROR] shap.TreeExplainer fallback failed: {e2}")
                
            # Evaluate logic gate
            st.session_state.triage_activated = evaluate_logic_gate(
                risk_prob, 
                st.session_state.ecg_probs is not None,
                st.session_state.cxr_probs is not None,
                st.session_state.nlp_features is not None
            )
            st.rerun()
 
    # Display results if present in state
    if st.session_state.tabular_risk is not None:
        risk_percentage = st.session_state.tabular_risk * 100
        
        st.markdown("### Triage Outcome Summary")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            risk_color = "#991b1b" if st.session_state.tabular_risk >= 0.15 else "#166534"
            card_border = "#fecaca" if st.session_state.tabular_risk >= 0.15 else "#bbf7d0"
            card_bg = "#fef2f2" if st.session_state.tabular_risk >= 0.15 else "#f0fdf4"
            
            st.markdown(f"""
            <div style="background-color: {card_bg}; padding: 20px; border: 1px solid {card_border}; border-radius: 4px; margin-bottom: 16px; font-family: 'Inter', sans-serif;">
                <div class="metric-label">XGBoost Risk Score</div>
                <div class="metric-value" style="color: {risk_color}; font-family: monospace;">{risk_percentage:.1f}%</div>
                <div class="metric-label" style="margin-top:16px;">Triage Classification Target</div>
                <div style="font-size:18px; font-weight: 600; color: #0f172a; margin-top: 4px;">{'Elevated Cardiovascular Risk' if st.session_state.tabular_prediction == 1 else 'Low Cardiovascular Risk'}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.tabular_risk < 0.15:
                st.markdown("""
                <div class="clinical-alert alert-low">
                    <strong>Gatekeeper Clearance:</strong> Patient baseline risk falls below the 15% clinical threshold. 
                    Fold 2 diagnostic resources are locked to conserve compute resources unless overridden by diagnostic uploads in Tab 2.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="clinical-alert alert-elevated">
                    <strong>Gatekeeper Triage Warning:</strong> Patient baseline risk exceeds the 15% clinical threshold. 
                    Fold 2 high-compute diagnostic streams (ECG, Vision, NLP) have been engaged.
                </div>
                """, unsafe_allow_html=True)
            
        with res_col2:
            feature_display_names = [
                'Age', 'Gender', 'Chest Pain', 'Resting BP', 'Cholesterol', 'Fasting BS',
                'Rest ECG', 'Max HR', 'Ex Angina', 'ST Depression', 'ST Slope', 'Fluoroscopy', 'Thalassemia'
            ]
            
            contributions = None
            chart_title = "Local SHAP Feature Contribution"
            is_shap = False
            
            if st.session_state.shap_values is not None:
                try:
                    contributions = st.session_state.shap_values.values[0]
                    chart_title = "Local SHAP Feature Contribution"
                    is_shap = True
                except Exception as e:
                    print(f"Error extracting SHAP values: {e}")
                    
            if contributions is None and tabular_model is not None:
                try:
                    # Fallback to model feature importances
                    contributions = tabular_model.feature_importances_
                    chart_title = "Global Feature Importance (XGBoost Fallback)"
                    is_shap = False
                except Exception as e:
                    print(f"Error extracting feature importances: {e}")
            
            st.write(f"**{chart_title}**")
            
            if contributions is not None and len(contributions) == len(feature_display_names):
                fig, ax = plt.subplots(figsize=(6, 3.4), facecolor='none')
                ax.set_facecolor('none')
                
                sorted_idx = np.argsort(np.abs(contributions))
                y_pos = np.arange(len(feature_display_names))
                
                # Clinical red for positive/high impact, blue/navy for global importance
                if is_shap:
                    colors = ['#991b1b' if c > 0 else '#0369a1' for c in contributions[sorted_idx]]
                    xlabel = "SHAP Contribution (Model Impact Value)"
                else:
                    colors = ['#0c4a6e' for _ in contributions] # Deep blue for feature importances
                    xlabel = "Relative Feature Importance Score"
                    
                ax.barh(y_pos, contributions[sorted_idx], color=colors, height=0.55)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([feature_display_names[i] for i in sorted_idx], fontsize=8, color='#334155', fontfamily='sans-serif')
                ax.set_xlabel(xlabel, fontsize=8, color='#475569', fontfamily='sans-serif')
                
                # Styling axes
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#cbd5e1')
                ax.spines['bottom'].set_color('#cbd5e1')
                ax.tick_params(axis='both', colors='#64748b', labelsize=8)
                ax.grid(axis='x', linestyle=':', alpha=0.5, color='#cbd5e1')
                
                plt.tight_layout()
                st.pyplot(fig, transparent=True)
                plt.close(fig)
            else:
                st.write("Feature contribution values unavailable. Please calibrate the model in Tab 5.")
                
        # LLM Clinical Reasoning Report
        st.markdown("### Algorithmic Reasoning & Explanation")
        
        # Local Clinical Rules Synthesis Generator
        contributions = st.session_state.shap_values.values[0] if st.session_state.shap_values is not None else np.zeros(13)
        inputs = st.session_state.input_data
        
        # Parse inputs
        age_val = int(inputs[0, 0])
        gender_str = "Male" if int(inputs[0, 1]) == 1 else "Female"
        cp_type = int(inputs[0, 2])
        resting_bp = int(inputs[0, 3])
        cholesterol = int(inputs[0, 4])
        fbs = int(inputs[0, 5])
        max_hr = int(inputs[0, 7])
        exang = int(inputs[0, 8])
        oldpeak = inputs[0, 9]
        ca = int(inputs[0, 11])
        
        cp_names = {0: "typical angina symptoms", 1: "atypical angina symptoms", 2: "non-anginal chest pain", 3: "asymptomatic coronary status"}
        cp_desc = cp_names.get(cp_type, "atypical cardiovascular symptoms")
        
        # Sort contributions to find top features
        feature_display_names = [
            'Age', 'Gender', 'Chest Pain Type', 'Resting Blood Pressure', 'Serum Cholesterol', 'Fasting Blood Sugar',
            'Resting ECG Results', 'Maximum Heart Rate', 'Exercise Induced Angina', 'ST segment depression', 'Peak ST segment slope', 'Fluoroscopy vessels', 'Thalassemia type'
        ]
        abs_contributions = np.abs(contributions)
        sorted_indices = np.argsort(abs_contributions)[::-1]
        
        elevation_factors = []
        reduction_factors = []
        
        for idx in sorted_indices:
            feat_name = feature_display_names[idx]
            val = contributions[idx]
            if val > 0.01: # Significant risk contributor
                elevation_factors.append(feat_name.lower())
            elif val < -0.01: # Significant safety contributor
                reduction_factors.append(feat_name.lower())
                
        # Narrative construction
        if risk_percentage >= 15.0:
            severity = "elevated"
            action = "Requires further clinical investigation. Fold 2 diagnostic gate has been engaged to analyze high-compute unstructured streams."
        else:
            severity = "low"
            action = "Indicates standard clinical management and continued monitoring. Advanced diagnostic gates remain in standby."
            
        synthesis = f"Cardiovascular clinical assessment for a {age_val}-year-old {gender_str} patient indicates an {severity} risk index of **{risk_percentage:.1f}%**. "
        synthesis += f"The patient presents with {cp_desc}. "
        
        if oldpeak > 1.5 or ca > 0 or resting_bp > 135:
            synthesis += "Primary clinical observations show "
            obs = []
            if ca > 0:
                obs.append(f"{ca} major vessels colored by fluoroscopy (suggesting potential coronary calcification)")
            if oldpeak > 1.5:
                obs.append(f"ST-segment depression of {oldpeak:.1f}mm (signifying exercise-induced myocardial ischemia)")
            if resting_bp > 135:
                obs.append(f"elevated resting systolic blood pressure of {resting_bp} mm Hg")
            synthesis += ", ".join(obs) + ". "
        
        if elevation_factors:
            synthesis += f"Machine learning feature contributions identify the primary drivers of elevated cardiovascular risk as **{', '.join(elevation_factors[:3])}**. "
        
        if reduction_factors:
            synthesis += f"Conversely, protective or risk-mitigating physiological metrics include **{', '.join(reduction_factors[:2])}**. "
            
        synthesis += f"<br/><br/><strong>Clinical Recommendation:</strong> {action}"
        
        if hf_token:
            with st.spinner("Requesting clinical synthesis engine..."):
                try:
                    import requests
                    
                    top_features = []
                    for idx in sorted_indices[:5]:
                        sign = "Elevated" if contributions[idx] > 0 else "Lowered"
                        top_features.append(f"{feature_display_names[idx]} ({sign} risk by {abs_contributions[idx]:.3f})")
                        
                    prompt = (
                        f"You are a clinical cardiology expert. Explain the clinical relevance of these features for a cardiology patient who has a risk score of {risk_percentage:.1f}%: "
                        f"{', '.join(top_features)}. Provide a concise, professional medical explanation."
                    )
                    
                    # Try Gemma first, then Qwen, and finally Phi-3 on the serverless API
                    models_to_try = [
                        "google/gemma-2-2b-it",
                        "Qwen/Qwen2.5-7B-Instruct",
                        "microsoft/Phi-3-mini-4k-instruct"
                    ]
                    
                    bot_response = None
                    last_err = None
                    active_model = None
                    
                    from huggingface_hub import InferenceClient
                    client = InferenceClient(token=hf_token, timeout=15)
                    
                    for model_id in models_to_try:
                        try:
                            # Format prompt with standard turn tags if Gemma is used
                            formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n" if "gemma" in model_id else prompt
                            
                            response_text = client.text_generation(
                                formatted_prompt,
                                model=model_id,
                                max_new_tokens=250,
                                temperature=0.2
                            )
                            if response_text:
                                bot_response = response_text.strip()
                                if bot_response.startswith(prompt):
                                    bot_response = bot_response[len(prompt):].strip()
                                active_model = model_id.split("/")[-1]
                                break
                        except Exception as e:
                            last_err = f"{model_id.split('/')[-1]}: {type(e).__name__}: {str(e)}"
                    
                    if not bot_response:
                        raise RuntimeError(last_err or "No response from inference models")
                    
                    st.markdown(f"""
                    <div style="background-color: #ffffff; padding: 16px; border: 1px solid #cbd5e1; border-radius: 4px; font-family: 'Inter', sans-serif;">
                        <h4 style="margin-top: 0; font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">AI-Generated Consultation Report ({active_model})</h4>
                        <div style="font-size: 13.5px; color: #334155; line-height: 1.6;">{bot_response}</div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as le:
                    import traceback
                    traceback.print_exc()
                    
                    err_details = f"{type(le).__name__}: {str(le)}".strip()
                    if not err_details or err_details == "Exception:":
                        err_details = "API connection failed"
                        
                    st.markdown(f"""
                    <div style="background-color: #fef2f2; padding: 12px; border: 1px solid #fecaca; border-radius: 4px; color: #991b1b; font-size: 13px; margin-bottom: 16px; font-family: 'Inter', sans-serif;">
                        <strong>Network Advisory:</strong> Hugging Face API is currently unreachable ({err_details}). 
                        Falling back to local offline clinical rules synthesis engine.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="background-color: #ffffff; padding: 16px; border: 1px solid #cbd5e1; border-radius: 4px; font-family: 'Inter', sans-serif;">
                        <h4 style="margin-top: 0; font-size: 11px; color: #0c4a6e; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">Workstation Synthesis Report (Rule-Based Engine - Fallback)</h4>
                        <div style="font-size: 13.5px; color: #334155; line-height: 1.6; margin-bottom: 16px;">{synthesis}</div>
                        <div style="font-size: 11px; color: #64748b; border-top: 1px solid #cbd5e1; padding-top: 8px;">
                            <strong>Note:</strong> Verify network connection or DNS settings to restore online LLM API access.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #ffffff; padding: 16px; border: 1px solid #cbd5e1; border-radius: 4px; font-family: 'Inter', sans-serif;">
                <h4 style="margin-top: 0; font-size: 11px; color: #0c4a6e; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;">Workstation Synthesis Report (Rule-Based Engine)</h4>
                <div style="font-size: 13.5px; color: #334155; line-height: 1.6; margin-bottom: 16px;">{synthesis}</div>
                <div style="font-size: 11px; color: #64748b; border-top: 1px solid #cbd5e1; padding-top: 8px;">
                    <strong>Note:</strong> Input a Hugging Face API Token in the control panel to generate an LLM-synthesized narrative via online LLM.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ----------------- TAB 2: DEEP DIAGNOSTICS (FOLD 2) -----------------
with tabs[1]:
    st.markdown("### High-Compute Diagnostic Gateway")
    st.write("Process unstructured diagnostic streams including 12-lead ECG signals, Chest X-rays, and clinical summaries.")
    
    # Gate alert
    if not st.session_state.triage_activated:
        st.markdown("""
        <div class="clinical-alert alert-neutral">
            <strong>System Gate Status: Standby.</strong> Patient tabular screening risk is below the 15% threshold. 
            High-compute models are inactive. You can manually ingest ECG waveforms, chest X-rays, or clinical notes below to engage Fold 2 components.
        </div>
        """, unsafe_allow_html=True)
        
    diag_col1, diag_col2 = st.columns(2)
    
    with diag_col1:
        st.markdown("#### ECG Waveform Processing (PTB-XL)")
        ecg_file = st.file_uploader("Upload 12-lead ECG (.npy or .csv):", type=["npy", "csv"])
        use_mock_ecg = st.button("Load Calibration ECG Waveform")
        
        # Load or generate ECG signal
        ecg_signal = None
        if ecg_file is not None:
            if ecg_file.name.endswith(".npy"):
                ecg_signal = np.load(ecg_file)
            else:
                ecg_signal = pd.read_csv(ecg_file).values
        elif use_mock_ecg:
            # Generate a realistic 12-lead ECG signal (1000 timesteps, 12 channels)
            time_steps = np.linspace(0, 10, 1000)
            ecg_signal = np.zeros((1000, 12))
            for ch in range(12):
                p_wave = 0.1 * np.sin(2 * np.pi * 1.2 * time_steps)
                qrs = 1.1 * np.exp(-((time_steps % 0.8 - 0.25) / 0.025) ** 2) * (1 - 1.8 * (ch % 2))
                t_wave = 0.2 * np.exp(-((time_steps % 0.8 - 0.5) / 0.06) ** 2)
                ecg_signal[:, ch] = p_wave + qrs + t_wave
                
        if ecg_signal is not None:
            # Verify shape
            if ecg_signal.shape != (1000, 12):
                if ecg_signal.shape == (12, 1000):
                    ecg_signal = ecg_signal.T
                else:
                    st.error(f"ECG array size mismatch: {ecg_signal.shape}. Expected (1000, 12).")
                    ecg_signal = None
            
            if ecg_signal is not None:
                st.success("ECG waveform loaded successfully.")
                
                # Preprocess & Predict
                tensor_signal = preprocess_signal(ecg_signal, scaler)
                ecg_probs = predict_ecg(ecg_model, tensor_signal, mlb, diag_map)
                st.session_state.ecg_probs = ecg_probs
                st.session_state.triage_activated = True
                
                # Plot leads with realistic pink grid-paper aesthetic
                fig, axes = plt.subplots(4, 1, figsize=(6, 4.0), sharex=True, facecolor='none')
                leads_to_plot = [0, 1, 6, 11] # I, II, V1, V6
                lead_names = ["Lead I", "Lead II", "Lead V1", "Lead V6"]
                
                for idx, lead_idx in enumerate(leads_to_plot):
                    axes[idx].set_facecolor('#fffafb') # Faint pink background matching real ECG grid paper
                    axes[idx].plot(ecg_signal[:400, lead_idx], color="#1e293b", linewidth=1.1)
                    axes[idx].set_ylabel(lead_names[idx], fontsize=8, color='#334155', fontfamily='sans-serif')
                    axes[idx].grid(True, which='both', linestyle="-", alpha=0.5, color='#fecaca')
                    axes[idx].spines['top'].set_visible(False)
                    axes[idx].spines['right'].set_visible(False)
                    axes[idx].spines['left'].set_color('#fca5a5')
                    axes[idx].spines['bottom'].set_color('#fca5a5')
                    axes[idx].tick_params(axis='both', colors='#64748b', labelsize=7)
                
                axes[-1].set_xlabel("Time (Samples, 100Hz Sampling)", fontsize=8, color='#475569', fontfamily='sans-serif')
                plt.tight_layout()
                st.pyplot(fig, transparent=True)
                plt.close(fig)
                
                st.markdown("**Arrhythmia & Infarction Diagnostic Probabilities**")
                for c, prob in ecg_probs.items():
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 8px; font-family: 'Inter', sans-serif;">
                        <div style="font-weight: 600; font-size: 12px; width: 60px; color: #1e293b;">{c}</div>
                        <div style="flex-grow: 1; background-color: #e2e8f0; height: 6px; border-radius: 3px; margin: 0 12px;">
                            <div style="background-color: #0c4a6e; height: 100%; width: {prob*100}%; border-radius: 3px;"></div>
                        </div>
                        <div style="font-family: monospace; font-size: 11px; width: 50px; text-align: right; color: #475569;">{prob*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
    with diag_col2:
        st.markdown("#### Radiographic Vision Processing (TorchXRayVision)")
        cxr_file = st.file_uploader("Upload Chest X-ray image:", type=["png", "jpg", "jpeg"])
        use_mock_cxr = st.button("Load Calibration Chest X-Ray")
        
        cxr_img = None
        if cxr_file is not None:
            cxr_img = Image.open(cxr_file)
        elif use_mock_cxr:
            # Generate a realistic grayscale mock X-Ray image representing lungs
            x = np.linspace(-1, 1, 224)
            y = np.linspace(-1, 1, 224)
            X, Y = np.meshgrid(x, y)
            left_lung = np.exp(-((X + 0.4)**2 / 0.12 + (Y - 0.05)**2 / 0.35))
            right_lung = np.exp(-((X - 0.4)**2 / 0.12 + (Y - 0.05)**2 / 0.35))
            lungs = (left_lung + right_lung) * 0.75
            heart = np.exp(-((X + 0.12)**2 / 0.09 + (Y + 0.25)**2 / 0.12)) * 0.90
            img_data = (1.0 - (lungs + heart)) * 240.0
            img_data = np.clip(img_data, 0, 255).astype(np.uint8)
            cxr_img = Image.fromarray(img_data).convert("RGB")
            
        if cxr_img is not None:
            st.success("Chest X-Ray scan loaded successfully.")
            st.image(cxr_img, caption="Ingested Chest Radiograph", width=180)
            
            # Predict
            tensor_cxr = preprocess_xray(cxr_img)
            cxr_probs = predict_xray(vision_model, tensor_cxr)
            st.session_state.cxr_probs = cxr_probs
            st.session_state.triage_activated = True
            
            st.markdown("**Radiographic Pathology Confidences**")
            for c, prob in cxr_probs.items():
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 8px; font-family: 'Inter', sans-serif;">
                    <div style="font-weight: 600; font-size: 12px; width: 100px; color: #1e293b;">{c}</div>
                    <div style="flex-grow: 1; background-color: #e2e8f0; height: 6px; border-radius: 3px; margin: 0 12px;">
                        <div style="background-color: #0c4a6e; height: 100%; width: {prob*100}%; border-radius: 3px;"></div>
                    </div>
                    <div style="font-family: monospace; font-size: 11px; width: 50px; text-align: right; color: #475569;">{prob*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Clinical Notes Entity Extraction (BioBERT)")
    st.write("Parse unstructured clinical transcripts to extract clinical symptom vectors.")
    
    # Pre-defined sample notes
    sample_notes_dropdown = st.selectbox(
        "Select Clinical Note Templates:",
        options=[
            "Custom Transcript Input",
            "Cardiovascular Angina Case Study",
            "Exertional Dyspnea Case Study",
            "Routine Checkup Summary"
        ]
    )
    
    default_text = ""
    gender_noun = "male" if gender == "Male" else "female"
    if "Angina Case" in sample_notes_dropdown:
        default_text = f"{age}yo {gender_noun} presents with classic angina chest pain radiating down the left arm under stress. History of cigarette usage. Significant family history of premature coronary artery disease."
    elif "Dyspnea Case" in sample_notes_dropdown:
        default_text = f"Patient ({age}yo {gender_noun}) complains of exertional dyspnea and shortness of breath. Serum lipid panel shows hypercholesterolemia. No history of tobacco usage. Resting blood pressure recorded at 116/74."
    elif "Routine Checkup" in sample_notes_dropdown:
        default_text = f"Regular checkup for {age}yo {gender_noun}. Active lifestyle. Non-smoker. No reports of chest pain, dyspnea, or family history of coronary disease. Vitals within normal limits."
        
    doctor_notes = st.text_area("Clinical Text Summary:", value=default_text, height=100)
    
    if st.button("Parse Notes with BioBERT"):
        if doctor_notes.strip() == "":
            st.warning("Please enter note text.")
        else:
            with st.spinner("Encoding text with BioBERT..."):
                nlp_features = extract_clinical_features(doctor_notes)
                st.session_state.nlp_features = nlp_features
                st.session_state.triage_activated = True
                st.success("Extraction complete.")
                
    if st.session_state.nlp_features is not None:
        st.write("**Extracted Symptom Indicators**")
        badge_cols = st.columns(6)
        for i, (k, val) in enumerate(st.session_state.nlp_features.items()):
            with badge_cols[i]:
                # Custom clean clinical badges instead of generic blocks
                badge_class = "badge-present" if val >= 0.5 else "badge-absent"
                status_text = "PRESENT" if val >= 0.5 else "ABSENT"
                st.markdown(f"""
                <div class="clinical-badge {badge_class}">
                    <div style="font-weight: 600; font-size: 11px;">{k.replace('_', ' ').upper()}</div>
                    <div style="font-size: 12px; margin-top: 4px; font-family: monospace;">{status_text} ({val*100:.0f}%)</div>
                </div>
                """, unsafe_allow_html=True)

# ----------------- TAB 3: MULTI-MODAL DATA FUSION -----------------
with tabs[2]:
    st.markdown("### Intermediate Data Fusion Engine")
    st.write("Assemble clinical streams into a consolidated matrix and compute the patient's multi-modal risk score.")
    
    if st.session_state.tabular_risk is None:
        st.markdown("""
        <div class="clinical-alert alert-neutral">
            <strong>Data Ingestion Required:</strong> Please complete the initial Tabular Screening (Tab 1) to generate baseline metrics before assembling the fused patient profile matrix.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Assemble Fused Patient Profile Matrix
        fused_df = assemble_fused_matrix(
            st.session_state.input_data,
            st.session_state.tabular_risk,
            st.session_state.shap_values.values if st.session_state.shap_values is not None else None,
            st.session_state.ecg_probs,
            st.session_state.cxr_probs,
            st.session_state.nlp_features
        )
        
        st.markdown("""
        <div style="margin-top: 10px; margin-bottom: 12px;">
            <h4 style="margin: 0; font-size: 13px; font-weight: 600; text-transform: uppercase; color: #0f172a; letter-spacing: 0.05em;">Fused Multi-Modal Patient Profile Matrix (1 x 44)</h4>
            <span style="font-size: 12px; color: #64748b;">Consolidated 44-dimensional clinical vector, assembled from vitals screening, local SHAP values, ECG predictions, CXR predictions, and BioBERT text features.</span>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(fused_df.style.format("{:.4f}").highlight_max(axis=0, color="#e2e8f0"), use_container_width=True)
        
        st.markdown("---")
        
        # Compute fused risk
        fused_risk, weights = calculate_multimodal_risk(
            st.session_state.tabular_risk,
            st.session_state.ecg_probs,
            st.session_state.cxr_probs,
            st.session_state.nlp_features
        )
        
        fus_col1, fus_col2 = st.columns(2)
        
        with fus_col1:
            risk_pct = fused_risk * 100
            risk_color = "#991b1b" if fused_risk >= 0.15 else "#166534"
            card_border = "#fecaca" if fused_risk >= 0.15 else "#bbf7d0"
            card_bg = "#fef2f2" if fused_risk >= 0.15 else "#f0fdf4"
            
            st.markdown(f"""
            <div style="background-color: {card_bg}; padding: 20px; border: 1px solid {card_border}; border-radius: 4px; margin-bottom: 16px; font-family: 'Inter', sans-serif; min-height: 220px;">
                <div style="font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Precision Multi-Modal Risk Index</div>
                <div style="font-size: 36px; font-weight: 700; color: {risk_color}; margin-top: 4px; font-family: monospace;">{risk_pct:.1f}%</div>
                <div style="font-size: 11px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 16px; border-top: 1px solid #cbd5e1; padding-top: 12px;">Active Stream Weights</div>
                <div style="font-size: 12px; margin-top: 8px; line-height: 1.5; color: #334155;">
                    <strong>Tabular (XGBoost):</strong> {weights.get('tabular', 0.0)*100:.0f}% weight<br/>
                    <strong>ECG (ResNet1d):</strong> {weights.get('ecg', 0.0)*100:.0f}% weight<br/>
                    <strong>Vision (DenseNet121):</strong> {weights.get('cxr', 0.0)*100:.0f}% weight<br/>
                    <strong>NLP (BioBERT):</strong> {weights.get('nlp', 0.0)*100:.0f}% weight
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with fus_col2:
            summary_html = f"""
            <div style="background-color: #ffffff; padding: 20px; border: 1px solid #cbd5e1; border-radius: 4px; font-family: 'Inter', sans-serif; min-height: 220px;">
                <h4 style="margin-top: 0; margin-bottom: 14px; font-size: 11px; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em;">Integrated Stream Findings Summary</h4>
                <ul style="font-size: 13px; color: #334155; line-height: 1.8; margin: 0; padding-left: 20px;">
                    <li style="margin-bottom: 6px;"><strong>Tabular Screening:</strong> Baseline cardiovascular risk evaluated at <strong>{st.session_state.tabular_risk*100:.1f}%</strong>.</li>
            """
            
            if st.session_state.ecg_probs is not None:
                max_ecg = max(st.session_state.ecg_probs, key=st.session_state.ecg_probs.get)
                summary_html += f'<li style="margin-bottom: 6px;"><strong>Electrocardiogram Waveform:</strong> Lead waveform classification indicates <strong>{max_ecg}</strong> ({st.session_state.ecg_probs[max_ecg]*100:.1f}% probability).</li>'
            else:
                summary_html += '<li style="margin-bottom: 6px; color: #64748b;"><strong>Electrocardiogram Waveform:</strong> <em>No electrocardiogram waveform ingested.</em></li>'
                
            if st.session_state.cxr_probs is not None:
                max_cxr = max(st.session_state.cxr_probs, key=st.session_state.cxr_probs.get)
                summary_html += f'<li style="margin-bottom: 6px;"><strong>Radiographic Chest X-Ray:</strong> DenseNet-121 feature maps show primary pathology indicator <strong>{max_cxr}</strong> ({st.session_state.cxr_probs[max_cxr]*100:.1f}% confidence).</li>'
            else:
                summary_html += '<li style="margin-bottom: 6px; color: #64748b;"><strong>Radiographic Chest X-Ray:</strong> <em>No thoracic radiographs ingested.</em></li>'
                
            if st.session_state.nlp_features is not None:
                present_symptoms = [k.replace('_', ' ') for k, v in st.session_state.nlp_features.items() if v >= 0.5]
                symptom_str = ', '.join(present_symptoms) if present_symptoms else 'None detected'
                summary_html += f'<li style="margin-bottom: 6px;"><strong>BioBERT Entity Extraction:</strong> Extracted symptoms: <em>{symptom_str}</em>.</li>'
            else:
                summary_html += '<li style="margin-bottom: 6px; color: #64748b;"><strong>BioBERT Entity Extraction:</strong> <em>No clinical notes processed.</em></li>'
                
            summary_html += """
                </ul>
            </div>
            """
            st.markdown(summary_html, unsafe_allow_html=True)
                
        # Write to EHR Database controls
        st.markdown("---")
        st.markdown("#### EHR Encrypted Records Committal")
        st.write("Commit the current active patient case study to the secure SQLite archive. Demographics and clinical findings are encrypted locally via AES-256 before disk storage.")
        
        # Display summary of what is being committed
        st.markdown(f"""
        <div style="background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 12px 16px; margin-bottom: 16px; font-size: 13px; color: #334155; font-family: 'Inter', sans-serif;">
            <strong>Target Record Details:</strong><br/>
            Name: {patient_name} | ID: {patient_id} | Attending NPI: {doctor_npi} | Department: {department}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Commit to Secure Archive"):
            demographics = {
                "name": patient_name,
                "age": int(age),
                "gender": gender,
                "department": department
            }
            clinical_data = {
                "fused_risk": fused_risk,
                "tabular_risk": st.session_state.tabular_risk,
                "ecg_max_finding": max(st.session_state.ecg_probs, key=st.session_state.ecg_probs.get) if st.session_state.ecg_probs else None,
                "cxr_max_pathology": max(st.session_state.cxr_probs, key=st.session_state.cxr_probs.get) if st.session_state.cxr_probs else None,
                "symptoms_extracted": [k for k, v in st.session_state.nlp_features.items() if v >= 0.5] if st.session_state.nlp_features else []
            }
            
            db.save_patient_record(doctor_npi, patient_id, demographics, clinical_data)
            st.success(f"Record successfully encrypted and committed to SQLite ehr_database.db.")

# ----------------- TAB 4: SECURE EHR DATABASE -----------------
with tabs[3]:
    st.markdown("### Patient EHR Archive & HIPAA Auditing")
    st.write("Demonstration of database encryption and clinical compliance tracking.")
    
    st.markdown("**Decryption Authorization Credentials**")
    col_auth1, col_auth2 = st.columns(2)
    with col_auth1:
        auth_doctor_npi = st.text_input("Clinician NPI Verification Code:", value="")
    
    # Authorize if NPI matches
    authorized = len(auth_doctor_npi.strip()) > 0
    
    st.markdown("---")
    
    db_col1, db_col2 = st.columns(2)
    
    with db_col1:
        st.markdown("**Encrypted Database View (Ciphertext)**")
        st.caption("How data exists on disk (fulfilling HIPAA compliance requirements).")
        raw_rows = db.get_raw_encrypted_records()
        if raw_rows:
            st.table(pd.DataFrame(raw_rows))
        else:
            st.info("Database is currently empty.")
            
    with db_col2:
        st.markdown("**Decrypted Database View (Authorized Session)**")
        st.caption("Dynamic decryption runs only upon input of clinician verification credentials.")
        if authorized:
            decrypted_rows = db.get_all_records_decrypted(auth_doctor_npi)
            if decrypted_rows:
                formatted_rows = []
                for row in decrypted_rows:
                    formatted_rows.append({
                        "Patient ID": row["patient_id"],
                        "Name": row["demographics"].get("name", "Unknown"),
                        "Age": row["demographics"].get("age", "-"),
                        "Gender": row["demographics"].get("gender", "-"),
                        "Fused Risk": f"{row['clinical_data'].get('fused_risk', 0.0)*100:.1f}%",
                        "Department": row["demographics"].get("department", "-"),
                        "Timestamp": row["timestamp"][:19].replace('T', ' ')
                    })
                st.table(pd.DataFrame(formatted_rows))
            else:
                st.info("No records to decrypt.")
        else:
            st.markdown("""
            <div class="clinical-alert alert-neutral">
                <strong>Access Restricted:</strong> Input clinician NPI verification code above to authorize decryption of active record archives. All access attempts are recorded in compliance logs.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Compliance Logs (Audit Trail)**")
    st.caption("Immutable logs recording all record creations and database queries.")
    logs = db.get_compliance_logs()
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True)
    else:
        st.info("No audit logs available.")

# ----------------- TAB 5: MODEL TRAINING CONSOLE -----------------
with tabs[4]:
    st.markdown("### Diagnostic Model Calibration & Re-indexing")
    st.write("Retrain and calibrate the Fold 1 tabular XGBoost model using the local heart disease cohort data.")
    
    st.markdown("**Training Hyperparameters**")
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        lr_input = st.slider("Learning Rate:", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        max_depth_input = st.slider("Max Tree Depth:", min_value=2, max_value=12, value=5)
    with t_col2:
        n_est_input = st.slider("Number of Trees:", min_value=50, max_value=500, value=200, step=50)
        subsample_input = st.slider("Subsample Ratio:", min_value=0.2, max_value=1.0, value=0.4, step=0.1)
        
    st.markdown("---")
    
    if st.button("Run XGBoost Model Training"):
        with st.spinner("Ingesting cohort data and calibrating model..."):
            try:
                metrics = run_training()
                
                st.success("Calibration complete. heart_model1.pkl successfully updated!")
                
                res_t1, res_t2, res_t3 = st.columns(3)
                with res_t1:
                    st.metric("Training Set Accuracy", f"{metrics['train_accuracy']*100:.2f}%")
                with res_t2:
                    st.metric("Test Set Accuracy", f"{metrics['test_accuracy']*100:.2f}%")
                with res_t3:
                    st.metric("5-Fold Cross-Validation Accuracy", f"{metrics['cv_accuracy']*100:.2f}%")
                    
                st.caption(f"Training executed on {metrics['num_train_samples']} samples, evaluated on {metrics['num_test_samples']} test samples.")
                
                # Reload model
                with open(xgb_path, "rb") as f:
                    tabular_model = pickle.load(f)
            except Exception as e:
                st.error(f"Training failed: {e}")
