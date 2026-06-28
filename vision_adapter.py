import os
import sys
import numpy as np
import torch
import torchvision
import streamlit as st
from PIL import Image

# Add torchxrayvision path
sys.path.append(os.path.abspath("./torchxrayvision-main/torchxrayvision-main"))

import torchxrayvision as xrv

@st.cache_resource
def load_vision_model():
    """Load the pre-trained TorchXRayVision DenseNet121 model (cached)."""
    # Using 'densenet121-res224-all' which is trained on NIH, PC, CheX, MIMIC, etc.
    model = xrv.models.get_model("densenet121-res224-all")
    model.eval()
    return model

def preprocess_xray(image: Image.Image) -> torch.Tensor:
    """Load, convert to grayscale, normalize to [-1024, 1024] and resize to 224x224."""
    # Convert PIL Image to numpy array
    img_np = np.array(image.convert("RGB"))
    
    # Convert to grayscale by taking the first channel if it's 3-channel
    if len(img_np.shape) > 2:
        img_np = img_np[:, :, 0]
        
    # Scale to [-1024, 1024] assuming original is 8-bit [0, 255]
    # (equivalent to xrv.utils.normalize(img, 255))
    img_normalized = (2.0 * (img_np.astype(np.float32) / 255.0) - 1.0) * 1024.0
    
    # Add channel dimension: (1, H, W)
    img_normalized = img_normalized[None, :, :]
    
    # Apply torchxrayvision center crop and resize
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    
    img_processed = transform(img_normalized)
    
    # Convert to torch tensor: (1, 1, 224, 224)
    tensor_img = torch.from_numpy(img_processed).unsqueeze(0)
    return tensor_img

def predict_xray(model, tensor_img: torch.Tensor) -> dict:
    """Run chest X-ray inference and output probabilities for target conditions."""
    with torch.no_grad():
        preds = model(tensor_img).cpu().squeeze(0).numpy()
        
    # Zip default pathologies with predicted scores
    pathologies = xrv.datasets.default_pathologies
    results = dict(zip(pathologies, [float(p) for p in preds]))
    
    # Keep only primary heart/lung pathologies for clinical reports
    selected_pathologies = ["Cardiomegaly", "Edema", "Effusion", "Pneumonia", "Atelectasis", "Infiltration"]
    filtered_results = {p: results.get(p, 0.0) for p in selected_pathologies}
    
    return filtered_results
