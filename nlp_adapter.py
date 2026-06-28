import re
import numpy as np
import streamlit as st

# We wrap the imports so that if transformers is not yet installed or loading fails,
# the app can still initialize the fallback engine.
try:
    import torch
    from transformers import BertTokenizer, BertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

@st.cache_resource
def load_biobert_model():
    """Load pre-trained BioBERT model and tokenizer from Hugging Face (cached)."""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    try:
        tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        model = BertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"BioBERT load failed, fallback to rule-based engine: {e}")
        return None, None

def get_sentence_embedding(text: str, tokenizer, model) -> np.ndarray:
    """Extract mean-pooled sentence embedding from BioBERT."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pool token embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()
    return embeddings

# Clinical concepts and their synonyms for keyword/regex fallback matching
CLINICAL_KEYWORDS = {
    "chest_pain": [
        r"chest pain", r"angina", r"sternal pain", r"sternum pressure", r"chest pressure", 
        r"cardiac pain", r"anginal symptoms", r"chest discomfort"
    ],
    "shortness_of_breath": [
        r"shortness of breath", r"dyspnea", r"sob", r"breathless", r"breathing difficulty", 
        r"labored breathing", r"gasping"
    ],
    "high_cholesterol": [
        r"high cholesterol", r"hypercholesterolemia", r"cholesterol level elevated", 
        r"hyperlipidemia", r"elevated ldl", r"lipid panel high"
    ],
    "hypertension": [
        r"hypertension", r"high blood pressure", r"hbp", r"elevated bp", r"hypertensive", 
        r"systolic over 140", r"blood pressure high"
    ],
    "smoking_status": [
        r"smoking", r"smoker", r"smokes cigarettes", r"tobacco use", r"nicotine user", 
        r"cigar smoking", r"smoke pack"
    ],
    "family_history": [
        r"family history of", r"father had heart attack", r"mother had coronary", r"genetic risk",
        r"heart disease in family", r"familial history"
    ]
}

def extract_clinical_features(text: str) -> dict:
    """Extract symptoms and conditions using BioBERT or the regex fallback engine."""
    tokenizer, model = load_biobert_model()
    
    extracted_features = {
        "chest_pain": 0.0,
        "shortness_of_breath": 0.0,
        "high_cholesterol": 0.0,
        "hypertension": 0.0,
        "smoking_status": 0.0,
        "family_history": 0.0
    }
    
    text_lower = text.lower()
    
    # 1. First run the regex/keyword matching engine (highly reliable and interpretable)
    for concept, patterns in CLINICAL_KEYWORDS.items():
        matched = False
        for pattern in patterns:
            if re.search(pattern, text_lower):
                matched = True
                break
        extracted_features[concept] = 1.0 if matched else 0.0
        
    # 2. If BioBERT is available, refine/augment the confidence scores using semantic matching
    if tokenizer is not None and model is not None:
        try:
            # Generate embedding of the doctor note
            doc_emb = get_sentence_embedding(text, tokenizer, model)
            
            # Semantic definition vectors (simple keyword list representation)
            for concept, keywords in CLINICAL_KEYWORDS.items():
                if extracted_features[concept] == 1.0:
                    continue  # Already confidently detected by regex
                
                # Check semantic similarity with concept queries
                similarities = []
                for kw in keywords[:3]:  # Check top 3 synonyms
                    kw_emb = get_sentence_embedding(kw, tokenizer, model)
                    dot_product = np.dot(doc_emb, kw_emb)
                    norm_doc = np.linalg.norm(doc_emb)
                    norm_kw = np.linalg.norm(kw_emb)
                    cos_sim = dot_product / (norm_doc * norm_kw + 1e-8)
                    similarities.append(cos_sim)
                
                max_sim = max(similarities)
                # If similarity exceeds threshold, count as high-probability symptom
                if max_sim > 0.85:
                    extracted_features[concept] = float(max_sim)
        except Exception as e:
            print(f"Error in BioBERT semantic extraction: {e}")
            
    return extracted_features
