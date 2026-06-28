import numpy as np
import pandas as pd

# Feature Names mapping for the Fused Patient Profile Matrix
TABULAR_FEATURE_NAMES = [
    "Age", "Gender", "ChestPainType", "RestingBP", "Cholesterol", 
    "FastingBS", "RestECG", "MaxHR", "ExAngina", "STDepression", 
    "Slope", "NumVessels", "Thalassemia"
]

SHAP_FEATURE_NAMES = [f"SHAP_{f}" for f in TABULAR_FEATURE_NAMES]

ECG_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
VISION_CLASSES = ["Cardiomegaly", "Edema", "Effusion", "Pneumonia", "Atelectasis", "Infiltration"]
NLP_CONCEPTS = ["chest_pain", "shortness_of_breath", "high_cholesterol", "hypertension", "smoking_status", "family_history"]

ALL_FEATURE_COLUMNS = (
    TABULAR_FEATURE_NAMES + 
    ["TabularRisk"] + 
    SHAP_FEATURE_NAMES + 
    [f"ECG_{c}" for c in ECG_CLASSES] + 
    [f"CXR_{c}" for c in VISION_CLASSES] + 
    [f"NLP_{c}" for c in NLP_CONCEPTS]
)

def evaluate_logic_gate(tabular_risk: float, has_ecg: bool, has_cxr: bool, has_notes: bool, threshold: float = 0.15) -> bool:
    """
    Logic Gate Triage: Returns True if Fold 1 risk exceeds the threshold,
    or if any advanced diagnostic files (ECG, X-Ray, notes) are provided.
    """
    if tabular_risk >= threshold:
        return True
    if has_ecg or has_cxr or has_notes:
        return True
    return False

def assemble_fused_matrix(
    tabular_features: np.ndarray, 
    tabular_risk: float, 
    shap_values: np.ndarray, 
    ecg_probs: dict = None, 
    cxr_probs: dict = None, 
    nlp_features: dict = None
) -> pd.DataFrame:
    """
    Assemble the Fused Patient Profile Matrix by concatenating features side-by-side.
    Creates a single-row DataFrame with all 44 multi-modal features.
    """
    row_data = {}
    
    # 1. Add raw tabular features (13)
    for i, fname in enumerate(TABULAR_FEATURE_NAMES):
        row_data[fname] = float(tabular_features[0, i])
        
    # 2. Add baseline risk probability (1)
    row_data["TabularRisk"] = float(tabular_risk)
    
    # 3. Add SHAP values (13)
    for i, fname in enumerate(SHAP_FEATURE_NAMES):
        row_data[fname] = float(shap_values[0, i]) if shap_values is not None else 0.0
        
    # 4. Add ECG features (5)
    for c in ECG_CLASSES:
        row_data[f"ECG_{c}"] = float(ecg_probs[c]) if ecg_probs is not None else 0.0
        
    # 5. Add Vision features (6)
    for c in VISION_CLASSES:
        row_data[f"CXR_{c}"] = float(cxr_probs[c]) if cxr_probs is not None else 0.0
        
    # 6. Add NLP features (6)
    for c in NLP_CONCEPTS:
        row_data[f"NLP_{c}"] = float(nlp_features[c]) if nlp_features is not None else 0.0
        
    df = pd.DataFrame([row_data], columns=ALL_FEATURE_COLUMNS)
    return df

def calculate_multimodal_risk(
    tabular_risk: float, 
    ecg_probs: dict = None, 
    cxr_probs: dict = None, 
    nlp_features: dict = None
) -> tuple:
    """
    Calculates a consolidated Multi-Modal Risk Index using a dynamic weighted ensemble.
    Adjusts weights based on which diagnostics are uploaded.
    """
    # Base risks from each stream
    streams_present = {"tabular": True}
    
    # 1. Tabular Risk
    tabular_score = tabular_risk
    
    # 2. ECG Risk: maximum score of abnormal superclasses (MI, STTC, CD, HYP)
    ecg_score = 0.0
    if ecg_probs is not None:
        abnormal_ecg_probs = [ecg_probs[c] for c in ["MI", "STTC", "CD", "HYP"]]
        ecg_score = max(abnormal_ecg_probs) if abnormal_ecg_probs else 0.0
        streams_present["ecg"] = True
        
    # 3. Vision Risk: maximum of Cardiomegaly (heart structural) and Edema (cardiovascular stress)
    cxr_score = 0.0
    if cxr_probs is not None:
        cxr_score = max(cxr_probs.get("Cardiomegaly", 0.0), cxr_probs.get("Edema", 0.0))
        streams_present["cxr"] = True
        
    # 4. NLP Risk: combine chest pain, shortness of breath, family history, and smoking status
    nlp_score = 0.0
    if nlp_features is not None:
        # Weighted symptom indicator
        nlp_score = (
            nlp_features.get("chest_pain", 0.0) * 0.4 +
            nlp_features.get("shortness_of_breath", 0.0) * 0.3 +
            nlp_features.get("family_history", 0.0) * 0.2 +
            nlp_features.get("smoking_status", 0.0) * 0.1
        )
        streams_present["nlp"] = True

    # Dynamic Weighting allocation
    # Default weights: Tabular = 40%, ECG = 30%, CXR = 20%, NLP = 10%
    base_weights = {
        "tabular": 0.4,
        "ecg": 0.3,
        "cxr": 0.2,
        "nlp": 0.1
    }
    
    # Filter active weights
    active_weights = {k: base_weights[k] for k in streams_present.keys()}
    weight_sum = sum(active_weights.values())
    
    # Re-normalize weights to sum to 1.0
    normalized_weights = {k: v / weight_sum for k, v in active_weights.items()}
    
    # Compute combined risk
    fused_risk = normalized_weights.get("tabular", 0.0) * tabular_score
    if "ecg" in streams_present:
        fused_risk += normalized_weights["ecg"] * ecg_score
    if "cxr" in streams_present:
        fused_risk += normalized_weights["cxr"] * cxr_score
    if "nlp" in streams_present:
        fused_risk += normalized_weights["nlp"] * nlp_score
        
    return float(fused_risk), normalized_weights
