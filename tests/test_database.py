import unittest
import os
import shutil
import sqlite3
from database import HIPAADatabase

class TestHIPAADatabase(unittest.TestCase):
    def setUp(self):
        # Use a temporary key and db for tests
        self.db = HIPAADatabase()

    def test_encryption_decryption(self):
        test_str = "Secret Patient Data 123"
        encrypted = self.db.encrypt_data(test_str)
        self.assertNotEqual(test_str, encrypted)
        
        decrypted = self.db.decrypt_data(encrypted)
        self.assertEqual(test_str, decrypted)

    def test_record_insertion_and_audit_logging(self):
        doc_id = "DOC-TEST-001"
        pat_id = "PAT-TEST-002"
        demo = {"age": 45, "gender": "Female"}
        clinical = {"fused_risk": 0.28, "tabular_risk": 0.12}
        
        # Save record
        self.db.save_patient_record(doc_id, pat_id, demo, clinical)
        
        # Retrieve decrypted
        records = self.db.get_all_records_decrypted(doc_id)
        self.assertTrue(len(records) >= 1)
        
        latest_record = records[-1]
        self.assertEqual(latest_record["patient_id"], pat_id)
        self.assertEqual(latest_record["demographics"]["age"], 45)
        self.assertEqual(latest_record["clinical_data"]["fused_risk"], 0.28)
        
        # Verify compliance log
        logs = self.db.get_compliance_logs()
        self.assertTrue(len(logs) >= 2) # At least RECORD_CREATION and DATA_READ_ACCESS logs
        
        actions = [log["action"] for log in logs]
        self.assertIn("RECORD_CREATION", actions)
        self.assertIn("DATA_READ_ACCESS", actions)

if __name__ == "__main__":
    unittest.main()
