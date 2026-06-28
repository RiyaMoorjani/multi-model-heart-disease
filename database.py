import os
import sqlite3
import json
from datetime import datetime
from cryptography.fernet import Fernet

KEY_FILE = "secret.key"
DB_FILE = "ehr_database.db"

def load_or_generate_key():
    """Load the existing encryption key or generate a new one if not present."""
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as kf:
            return kf.read()
    else:
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as kf:
            kf.write(key)
        return key

class HIPAADatabase:
    def __init__(self):
        self.key = load_or_generate_key()
        self.cipher = Fernet(self.key)
        self.init_db()

    def init_db(self):
        """Initialize the patient records table and compliance audit log table."""
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            # Encrypted patients table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id_enc TEXT NOT NULL,
                    demographics_enc TEXT NOT NULL,
                    clinical_data_enc TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            # Compliance audit log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS compliance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()

    def encrypt_data(self, data_str: str) -> str:
        """Encrypt plain text to secure ciphertext."""
        return self.cipher.encrypt(data_str.encode()).decode()

    def decrypt_data(self, ciphertext: str) -> str:
        """Decrypt ciphertext back to plain text."""
        return self.cipher.decrypt(ciphertext.encode()).decode()

    def save_patient_record(self, doctor_id: str, patient_id: str, demographics: dict, clinical_data: dict):
        """Encrypt and save a patient clinical record, adding an entry to the compliance log."""
        timestamp = datetime.now().isoformat()
        
        # Serialize and encrypt
        demographics_enc = self.encrypt_data(json.dumps(demographics))
        clinical_data_enc = self.encrypt_data(json.dumps(clinical_data))
        patient_id_enc = self.encrypt_data(patient_id)
        
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO patients (patient_id_enc, demographics_enc, clinical_data_enc, timestamp)
                VALUES (?, ?, ?, ?)
            """, (patient_id_enc, demographics_enc, clinical_data_enc, timestamp))
            
            # Log compliance action
            log_details = f"Patient ID: {patient_id} record saved"
            cursor.execute("""
                INSERT INTO compliance_logs (user_id, action, details, timestamp)
                VALUES (?, ?, ?, ?)
            """, (doctor_id, "RECORD_CREATION", log_details, timestamp))
            
            conn.commit()

    def get_all_records_decrypted(self, doctor_id: str) -> list:
        """Retrieve and decrypt all records. Logs this access event for audit compliance."""
        timestamp = datetime.now().isoformat()
        records = []
        
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, patient_id_enc, demographics_enc, clinical_data_enc, timestamp FROM patients")
            rows = cursor.fetchall()
            
            for row in rows:
                try:
                    patient_id = self.decrypt_data(row[1])
                    demographics = json.loads(self.decrypt_data(row[2]))
                    clinical_data = json.loads(self.decrypt_data(row[3]))
                    records.append({
                        "id": row[0],
                        "patient_id": patient_id,
                        "demographics": demographics,
                        "clinical_data": clinical_data,
                        "timestamp": row[4]
                    })
                except Exception as e:
                    # Decryption failed (wrong key or corrupted data)
                    records.append({
                        "id": row[0],
                        "patient_id": "[DECRYPTION_FAILED]",
                        "demographics": {},
                        "clinical_data": {},
                        "timestamp": row[4]
                    })
            
            # Log compliance access audit
            log_details = f"Read request for all patient records"
            cursor.execute("""
                INSERT INTO compliance_logs (user_id, action, details, timestamp)
                VALUES (?, ?, ?, ?)
            """, (doctor_id, "DATA_READ_ACCESS", log_details, timestamp))
            conn.commit()
            
        return records

    def get_raw_encrypted_records(self) -> list:
        """Return the raw database rows (un-decrypted) to demonstrate secure database storage."""
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, patient_id_enc, demographics_enc, clinical_data_enc, timestamp FROM patients")
            rows = cursor.fetchall()
            return [{
                "id": r[0],
                "patient_id_enc": r[1][:25] + "...",
                "demographics_enc": r[2][:25] + "...",
                "clinical_data_enc": r[3][:25] + "...",
                "timestamp": r[4]
            } for r in rows]

    def get_compliance_logs(self) -> list:
        """Fetch audit log history."""
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, user_id, action, details, timestamp FROM compliance_logs ORDER BY id DESC")
            rows = cursor.fetchall()
            return [{
                "id": r[0],
                "user_id": r[1],
                "action": r[2],
                "details": r[3],
                "timestamp": r[4]
            } for r in rows]
