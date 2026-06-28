import unittest
import numpy as np
import pandas as pd
from fusion import assemble_fused_matrix, calculate_multimodal_risk, ALL_FEATURE_COLUMNS

class TestFeatureFusion(unittest.TestCase):
    def test_matrix_assembly(self):
        # 13 tabular features
        tabular_features = np.array([[58.0, 1.0, 2.0, 120.0, 240.0, 0.0, 0.0, 160.0, 0.0, 1.0, 1.0, 0.0, 2.0]])
        tabular_risk = 0.18
        # 13 SHAP values
        shap_values = np.array([[-0.05, 0.02, 0.1, -0.01, 0.04, 0.0, 0.0, -0.08, 0.0, 0.05, 0.03, 0.0, 0.02]])
        
        ecg_probs = {"NORM": 0.05, "MI": 0.35, "STTC": 0.15, "CD": 0.05, "HYP": 0.02}
        cxr_probs = {"Cardiomegaly": 0.22, "Edema": 0.18, "Effusion": 0.05, "Pneumonia": 0.02, "Atelectasis": 0.01, "Infiltration": 0.03}
        nlp_features = {"chest_pain": 1.0, "shortness_of_breath": 1.0, "high_cholesterol": 1.0, "hypertension": 0.0, "smoking_status": 1.0, "family_history": 1.0}
        
        fused_df = assemble_fused_matrix(
            tabular_features,
            tabular_risk,
            shap_values,
            ecg_probs,
            cxr_probs,
            nlp_features
        )
        
        # Verify shape (1 row, 44 columns)
        self.assertEqual(fused_df.shape, (1, 44))
        
        # Verify columns
        self.assertEqual(list(fused_df.columns), ALL_FEATURE_COLUMNS)
        
        # Verify specific values
        self.assertEqual(fused_df.loc[0, "TabularRisk"], 0.18)
        self.assertNotIn("EC_NORM", fused_df.columns)
        self.assertEqual(fused_df.loc[0, "ECG_NORM"], 0.05)
        self.assertEqual(fused_df.loc[0, "CXR_Cardiomegaly"], 0.22)
        self.assertEqual(fused_df.loc[0, "NLP_chest_pain"], 1.0)

    def test_dynamic_risk_calculation(self):
        # 1. Test when only tabular stream is present
        risk_tab, weights_tab = calculate_multimodal_risk(0.10)
        self.assertAlmostEqual(risk_tab, 0.10)
        self.assertEqual(list(weights_tab.keys()), ["tabular"])
        
        # 2. Test when all streams are present
        ecg_p = {"NORM": 0.05, "MI": 0.50, "STTC": 0.15, "CD": 0.05, "HYP": 0.02}
        cxr_p = {"Cardiomegaly": 0.40, "Edema": 0.18, "Effusion": 0.05, "Pneumonia": 0.02, "Atelectasis": 0.01, "Infiltration": 0.03}
        nlp_f = {"chest_pain": 1.0, "shortness_of_breath": 0.0, "high_cholesterol": 1.0, "hypertension": 0.0, "smoking_status": 1.0, "family_history": 1.0}
        
        fused_risk, weights = calculate_multimodal_risk(0.20, ecg_p, cxr_p, nlp_f)
        
        # Expected scores:
        # Tabular = 0.20
        # ECG = max(0.50, 0.15, 0.05, 0.02) = 0.50
        # CXR = max(0.40, 0.18) = 0.40
        # NLP = 1.0 * 0.4 + 0.0 * 0.3 + 1.0 * 0.2 + 1.0 * 0.1 = 0.70
        # Weights: Tabular = 40%, ECG = 30%, CXR = 20%, NLP = 10%
        # Expected combined: 0.20 * 0.4 + 0.50 * 0.3 + 0.40 * 0.2 + 0.70 * 0.1 = 0.08 + 0.15 + 0.08 + 0.07 = 0.38
        self.assertAlmostEqual(fused_risk, 0.38)
        self.assertAlmostEqual(weights["tabular"], 0.4)
        self.assertAlmostEqual(weights["ecg"], 0.3)

if __name__ == "__main__":
    unittest.main()
