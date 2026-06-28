---
title: Precision Clinical Workstation
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: streamlit_app.py
pinned: false
---

# 🩺 Precision Clinical Workstation

A multi-fold, multi-modal clinical diagnostic workstation for cardiovascular risk triage and assessment. This application integrates tabular clinical metrics with deep signal classifiers (ECG), computer vision (CXR), and biomedical NLP (clinical notes) using a secure HIPAA-compliant architecture.

## 🚀 Key Features

*   **Fold 1 Tabular Triage**: High-efficiency XGBoost screening using patient demographics and baseline bio-markers.
*   **Fold 2 Deep Gateway**: Automatically activates deep-learning diagnostic networks for patients with elevated tabular risk ($\ge 15\%$):
    *   **12-Lead ECG Classification**: PyTorch 1D ResNet (xresnet1d101) classifying 71 arrhythmias.
    *   **Chest X-Ray Pathology**: TorchXRayVision (DenseNet-121) mapping lung/heart structural indicators.
    *   **Clinical Notes Parser**: BioBERT NLP extracting qualitative symptom concepts.
*   **Secure HIPAA Archiving**: AES-256 symmetric ciphertext databases logs with audit trail compliance.
*   **AI Clinical Explanations**: Serverless online LLM querying (Gemma, Qwen, Phi-3) for cardiological diagnostics.

## 🛠️ Local Installation & Running

1.  Clone the repository:
    `ash
    git clone https://github.com/RiyaMoorjani/multi-model-heart-disease.git
    cd multi-model-heart-disease
    `
2.  Install dependencies:
    `ash
    pip install -r requirements.txt
    `
3.  Run the Streamlit application:
    `ash
    streamlit run streamlit_app.py
    `

## 🔒 Clinician Verification
*   **Default Clinician NPI**: 1982736450 (used in Tab 4 to decrypt secure patient record ciphertext).
