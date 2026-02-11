-- ============================================================================
-- SNOWFLAKE PATIENT RISK PREDICTION SYSTEM
-- Schema Definition: Core Data Tables and Configuration Tables
-- ============================================================================

-- ============================================================================
-- 1. CORE DATA TABLES
-- ============================================================================

-- Master Patient Index
CREATE OR REPLACE TABLE patients (
    patient_id STRING PRIMARY KEY,
    dob DATE,
    gender STRING,
    current_zip STRING,
    insurance_status STRING,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Clinical Encounters
CREATE OR REPLACE TABLE encounters (
    encounter_id STRING PRIMARY KEY,
    patient_id STRING NOT NULL,
    admit_date TIMESTAMP,
    discharge_date TIMESTAMP,
    facility_name STRING,
    service_line STRING,  -- e.g., 'Emergency', 'Psychiatry', 'Addiction Medicine'
    provider_id STRING,
    discharge_disposition STRING,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- Diagnoses
CREATE OR REPLACE TABLE diagnoses (
    diagnosis_id STRING PRIMARY KEY,
    encounter_id STRING NOT NULL,
    patient_id STRING NOT NULL,
    icd10_code STRING,
    diagnosis_description STRING,
    diagnosis_date DATE,
    is_primary BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    FOREIGN KEY (encounter_id) REFERENCES encounters(encounter_id),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- Medications
CREATE OR REPLACE TABLE medications (
    medication_id STRING PRIMARY KEY,
    patient_id STRING NOT NULL,
    medication_name STRING,
    prescription_date DATE,
    refill_date DATE,
    adherence_flag BOOLEAN,  -- True if patient is adherent
    status STRING,  -- 'active', 'discontinued'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- Labs and Vitals
CREATE OR REPLACE TABLE labs_vitals (
    lab_id STRING PRIMARY KEY,
    patient_id STRING NOT NULL,
    encounter_id STRING,
    lab_type STRING,  -- 'BP', 'A1C', 'Creatinine', etc.
    lab_value FLOAT,
    lab_unit STRING,
    lab_date DATE,
    is_abnormal BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (encounter_id) REFERENCES encounters(encounter_id)
);

-- Social Determinants of Health Flags
CREATE OR REPLACE TABLE social_flags (
    flag_id STRING PRIMARY KEY,
    patient_id STRING NOT NULL,
    flag_type STRING,  -- 'homelessness', 'food_insecurity', 'housing_unstable'
    flag_date DATE,
    severity_score FLOAT,  -- 0-1 scale
    source STRING,  -- 'EHR', 'social_work', 'external'
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- Unstructured Clinical Notes (Source for RAG)
CREATE OR REPLACE TABLE clinical_notes (
    note_id STRING PRIMARY KEY,
    patient_id STRING NOT NULL,
    encounter_id STRING,
    note_date TIMESTAMP,
    note_type STRING,  -- 'Social Work', 'Discharge Summary', 'Case Management'
    author_role STRING,
    note_text STRING,  -- Full free text
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (encounter_id) REFERENCES encounters(encounter_id)
);

-- ============================================================================
-- 2. CONFIGURATION TABLES (Rules as Data)
-- ============================================================================

-- High-Risk Zip Codes
CREATE OR REPLACE TABLE config_risk_zipcodes (
    zipcode STRING PRIMARY KEY,
    risk_weight FLOAT,  -- 0-1 scale
    description STRING,
    effective_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Homeless Facilities (Shelters, Outreach Centers)
CREATE OR REPLACE TABLE config_homeless_facilities (
    facility_name STRING PRIMARY KEY,
    is_shelter BOOLEAN DEFAULT FALSE,
    facility_type STRING,  -- 'shelter', 'outreach', 'transitional_housing'
    effective_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- High-Risk Service Lines
CREATE OR REPLACE TABLE config_highrisk_service_lines (
    service_line STRING PRIMARY KEY,
    risk_flag BOOLEAN DEFAULT TRUE,
    description STRING,
    effective_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- High-Risk Document Types
CREATE OR REPLACE TABLE config_highrisk_doc_types (
    doc_type STRING PRIMARY KEY,
    risk_weight FLOAT,  -- 0-1 scale
    description STRING,
    effective_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- ============================================================================
-- 3. ML OUTPUT TABLES
-- ============================================================================

-- Patient Risk Scores (ML Model Output)
CREATE OR REPLACE TABLE patient_risk_scores (
    patient_id STRING PRIMARY KEY,
    risk_score FLOAT,  -- 0-1 scale
    risk_category STRING,  -- 'LOW_RISK', 'MED_RISK', 'HIGH_RISK'
    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    model_version STRING,
    top_factors ARRAY,  -- JSON array of top contributing features
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- RAG Summaries (Generated Text Summaries)
CREATE OR REPLACE TABLE rag_summaries (
    summary_id STRING PRIMARY KEY,
    patient_id STRING NOT NULL,
    summary_text STRING,  -- Generated summary from RAG
    source_note_ids ARRAY,  -- Array of note_ids used
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    model_version STRING,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- ============================================================================
-- 4. INDEXES FOR PERFORMANCE
-- ============================================================================

-- Index on patient_id for all tables (implicit due to FOREIGN KEY)
-- Snowflake automatically optimizes via micro-partitions

-- ============================================================================
-- 5. COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE patients IS 'Master patient index containing demographic information';
COMMENT ON TABLE encounters IS 'Clinical encounter records including facility and service line';
COMMENT ON TABLE diagnoses IS 'Diagnosis codes (ICD-10) linked to encounters';
COMMENT ON TABLE medications IS 'Medication history including adherence flags';
COMMENT ON TABLE labs_vitals IS 'Laboratory results and vital signs';
COMMENT ON TABLE social_flags IS 'SDOH flags: homelessness, food insecurity, etc.';
COMMENT ON TABLE clinical_notes IS 'Unstructured clinical notes for RAG processing';
COMMENT ON TABLE config_risk_zipcodes IS 'Configuration: high-risk zip codes';
COMMENT ON TABLE config_homeless_facilities IS 'Configuration: known homeless shelters and facilities';
COMMENT ON TABLE config_highrisk_service_lines IS 'Configuration: high-risk service lines (e.g., Addiction Medicine)';
COMMENT ON TABLE config_highrisk_doc_types IS 'Configuration: high-risk document types';
COMMENT ON TABLE patient_risk_scores IS 'ML model output: risk scores and categories';
COMMENT ON TABLE rag_summaries IS 'RAG output: generated text summaries with source citations';

-- ============================================================================
-- END OF SCHEMA DEFINITION
-- ============================================================================
