import unittest
import os
import numpy as np
from PIL import Image
import pickle

class TestModelAdapters(unittest.TestCase):
    def test_xgboost_loading(self):
        xgb_path = "./Heart_disease_report_AI-main/heart_model1.pkl"
        self.assertTrue(os.path.exists(xgb_path))
        with open(xgb_path, "rb") as f:
            model = pickle.load(f)
        # Mock tabular vector (13 features)
        input_vector = np.array([[58.0, 1.0, 2.0, 120.0, 240.0, 0.0, 0.0, 160.0, 0.0, 1.0, 1.0, 0.0, 2.0]])
        probs = model.predict_proba(input_vector)[0]
        self.assertEqual(len(probs), 2)
        self.assertTrue(0.0 <= probs[1] <= 1.0)

    def test_ecg_adapter_loading(self):
        try:
            import torch
            from ecg_adapter import load_ecg_model, load_ecg_metadata, preprocess_signal, predict_ecg
            
            model = load_ecg_model()
            scaler, mlb, diag_map = load_ecg_metadata()
            
            # Generate a dummy ECG signal (1000 timesteps, 12 channels)
            dummy_signal = np.random.randn(1000, 12)
            tensor_signal = preprocess_signal(dummy_signal, scaler)
            
            # Check dimensions (batch_size, channels, sequence_length) -> (1, 12, 1000)
            self.assertEqual(tensor_signal.shape, (1, 12, 1000))
            
            probs = predict_ecg(model, tensor_signal, mlb, diag_map)
            self.assertEqual(len(probs), 5)
            for c in ["NORM", "MI", "STTC", "CD", "HYP"]:
                self.assertIn(c, probs)
                self.assertTrue(0.0 <= probs[c] <= 1.0)
        except ImportError:
            self.skipTest("PyTorch not installed yet, skipping ECG adapter test.")

    def test_vision_adapter_loading(self):
        try:
            import torch
            from vision_adapter import load_vision_model, preprocess_xray, predict_xray
            
            model = load_vision_model()
            
            # Generate a dummy grayscale PIL image (224x224)
            dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            tensor_cxr = preprocess_xray(dummy_img)
            
            # Check shape (1, 1, 224, 224)
            self.assertEqual(tensor_cxr.shape, (1, 1, 224, 224))
            
            # Values should be scaled to [-1024, 1024]
            self.assertTrue(tensor_cxr.min() >= -1025.0)
            self.assertTrue(tensor_cxr.max() <= 1025.0)
            
            probs = predict_xray(model, tensor_cxr)
            self.assertEqual(len(probs), 6)
            for c in ["Cardiomegaly", "Edema", "Effusion", "Pneumonia"]:
                self.assertIn(c, probs)
                self.assertTrue(0.0 <= probs[c] <= 1.0)
        except ImportError:
            self.skipTest("PyTorch not installed yet, skipping Vision adapter test.")

    def test_nlp_adapter_extraction(self):
        from nlp_adapter import extract_clinical_features
        text = "Patient reports acute chest pain and shortness of breath. History of smoking cigarettes."
        features = extract_clinical_features(text)
        self.assertEqual(features["chest_pain"], 1.0)
        self.assertEqual(features["shortness_of_breath"], 1.0)
        self.assertEqual(features["smoking_status"], 1.0)
        self.assertEqual(features["hypertension"], 0.0)

if __name__ == "__main__":
    unittest.main()
