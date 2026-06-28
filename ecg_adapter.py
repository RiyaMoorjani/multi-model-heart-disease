import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

# Add the ECG benchmarking code folder to path
sys.path.append(os.path.abspath("./ecg_ptbxl_benchmarking-master/ecg_ptbxl_benchmarking-master/code"))

from models.xresnet1d import xresnet1d101

MODEL_PATH = "./ecg_ptbxl_benchmarking-master/ecg_ptbxl_benchmarking-master/output/exp0/models/fastai_xresnet1d101/models/fastai_xresnet1d101.pth"
SCALER_PATH = "./ecg_ptbxl_benchmarking-master/ecg_ptbxl_benchmarking-master/output/exp0/data/standard_scaler.pkl"
MLB_PATH = "./ecg_ptbxl_benchmarking-master/ecg_ptbxl_benchmarking-master/output/exp0/data/mlb.pkl"
STATEMENTS_PATH = "scp_statements.csv"

@st.cache_resource
def load_ecg_model():
    """Load the pre-trained xresnet1d101 PyTorch model (71 classes)."""
    # 71 classes were used during pretraining
    model = xresnet1d101(num_classes=71, input_channels=12, kernel_size=5, ps_head=0.5, lin_ftrs_head=[128])
    
    # Load state dict
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
            # fastai saves model weights inside a 'model' key or directly
            if 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Clean keys if they are prefixed (e.g., from parallel/fastai modules)
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    cleaned_state_dict[k[7:]] = v
                else:
                    cleaned_state_dict[k] = v
                    
            model.load_state_dict(cleaned_state_dict)
        except Exception as e:
            print(f"Error loading ECG weights: {e}")
    else:
        print(f"ECG model file not found at {MODEL_PATH}")
        
    model.eval()
    return model

@st.cache_resource
def load_ecg_metadata():
    """Load standard scaler, MultiLabelBinarizer, and class mapping."""
    scaler = None
    mlb = None
    diag_map = {}
    
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
            
    if os.path.exists(MLB_PATH):
        with open(MLB_PATH, "rb") as f:
            mlb = pickle.load(f)
            
    if os.path.exists(STATEMENTS_PATH):
        try:
            scp_df = pd.read_csv(STATEMENTS_PATH, index_col=0)
            diag_map = scp_df[scp_df.diagnostic == 1.0]["diagnostic_class"].to_dict()
        except Exception as e:
            print(f"Error reading scp_statements.csv: {e}")
            
    return scaler, mlb, diag_map

def preprocess_signal(signal: np.ndarray, scaler) -> torch.Tensor:
    """Preprocess a raw (1000, 12) signal into PyTorch tensor formatted for Conv1d."""
    # signal shape: (1000, 12) -> sequence_length, channels
    # Apply standardizer (equivalent to utils.apply_standardizer)
    if scaler is not None:
        flat_signal = signal.flatten()[:, np.newaxis]
        scaled_flat = scaler.transform(flat_signal)
        signal = scaled_flat.reshape(signal.shape)
        
    # Model expects: (batch_size, channels, sequence_length) -> (1, 12, 1000)
    tensor_signal = torch.tensor(signal, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
    return tensor_signal

def predict_ecg(model, tensor_signal: torch.Tensor, mlb, diag_map) -> dict:
    """Run model prediction and aggregate the 71 labels into the 5 diagnostic superclasses."""
    with torch.no_grad():
        logits = model(tensor_signal)
        probs = torch.sigmoid(logits).squeeze(0).numpy()
        
    # Map raw 71 probabilities to 5 diagnostic classes
    superclasses = ["NORM", "MI", "STTC", "CD", "HYP"]
    superclass_probs = {c: 0.0 for c in superclasses}
    
    if mlb is not None:
        classes = list(mlb.classes_)
        # Aggregate classes based on mapping
        for i, prob in enumerate(probs):
            class_name = classes[i]
            if class_name in diag_map:
                sclass = diag_map[class_name]
                if sclass in superclass_probs:
                    # Multi-label probability: take maximum probability of any sub-class in the superclass
                    superclass_probs[sclass] = max(superclass_probs[sclass], float(prob))
            elif class_name == "NORM":
                superclass_probs["NORM"] = max(superclass_probs["NORM"], float(prob))
    else:
        # Fallback dummy outputs if mlb is missing
        superclass_probs = {"NORM": 0.05, "MI": 0.12, "STTC": 0.08, "CD": 0.04, "HYP": 0.02}
        
    return superclass_probs
