-- ============================================================================
-- SNOWFLAKE PATIENT RISK PREDICTION SYSTEM
-- Feature Engineering Views and Stored Procedures
-- ============================================================================

-- ============================================================================
-- 1. FEATURE ENGINEERING VIEWS
-- ============================================================================

-- Aggregate patient demographics and clinical history
CREATE OR REPLACE VIEW vw_patient_features AS
SELECT 
    p.patient_id,
    p.dob,
    DATEDIFF(year, p.dob, CURRENT_DATE()) AS age,
    p.gender,
    p.current_zip,
    p.insurance_status,
    
    -- Risk zip code flag
    CASE WHEN rz.zipcode IS NOT NULL THEN 1 ELSE 0 END AS is_high_risk_zip,
    COALESCE(rz.risk_weight, 0) AS zip_risk_weight,
    
    -- Encounter statistics (last 12 months)
    COUNT(DISTINCT CASE WHEN e.admit_date >= DATEADD(month, -12, CURRENT_DATE()) 
        THEN e.encounter_id END) AS encounters_12mo,
    COUNT(DISTINCT CASE WHEN e.admit_date >= DATEADD(month, -6, CURRENT_DATE()) 
        THEN e.encounter_id END) AS encounters_6mo,
    COUNT(DISTINCT CASE WHEN e.admit_date >= DATEADD(month, -3, CURRENT_DATE()) 
        THEN e.encounter_id END) AS encounters_3mo,
    
    -- Emergency visits
    COUNT(DISTINCT CASE WHEN e.service_line = 'Emergency' 
        AND e.admit_date >= DATEADD(month, -12, CURRENT_DATE()) 
        THEN e.encounter_id END) AS ed_visits_12mo,
    
    -- High-risk service line visits
    COUNT(DISTINCT CASE WHEN hsl.service_line IS NOT NULL 
        AND e.admit_date >= DATEADD(month, -12, CURRENT_DATE()) 
        THEN e.encounter_id END) AS highrisk_visits_12mo,
    
    -- Homeless facility visits
    COUNT(DISTINCT CASE WHEN hf.facility_name IS NOT NULL 
        AND e.admit_date >= DATEADD(month, -12, CURRENT_DATE()) 
        THEN e.encounter_id END) AS homeless_facility_visits_12mo,
    
    -- Average length of stay
    AVG(CASE WHEN e.discharge_date IS NOT NULL 
        AND e.admit_date >= DATEADD(month, -12, CURRENT_DATE())
        THEN DATEDIFF(day, e.admit_date, e.discharge_date) END) AS avg_los_days_12mo,
    
    -- Days since last encounter
    DATEDIFF(day, MAX(e.admit_date), CURRENT_DATE()) AS days_since_last_encounter,
    
    -- Social determinants flags
    MAX(CASE WHEN sf.flag_type = 'homelessness' AND sf.is_active THEN 1 ELSE 0 END) AS has_homelessness_flag,
    MAX(CASE WHEN sf.flag_type = 'food_insecurity' AND sf.is_active THEN 1 ELSE 0 END) AS has_food_insecurity_flag,
    MAX(CASE WHEN sf.flag_type = 'housing_unstable' AND sf.is_active THEN 1 ELSE 0 END) AS has_housing_unstable_flag,
    AVG(CASE WHEN sf.is_active THEN sf.severity_score END) AS avg_sdoh_severity,
    
    -- Diagnosis counts
    COUNT(DISTINCT d.icd10_code) AS unique_diagnoses_count,
    
    -- Medication adherence
    AVG(CASE WHEN m.status = 'active' THEN CAST(m.adherence_flag AS INT) END) AS medication_adherence_rate,
    COUNT(DISTINCT CASE WHEN m.status = 'active' THEN m.medication_id END) AS active_medications_count,
    
    -- Lab abnormalities
    COUNT(CASE WHEN lv.is_abnormal THEN 1 END) AS abnormal_labs_count_12mo,
    
    -- Clinical notes count
    COUNT(DISTINCT cn.note_id) AS clinical_notes_count_12mo,
    COUNT(DISTINCT CASE WHEN hdt.doc_type IS NOT NULL THEN cn.note_id END) AS highrisk_notes_count_12mo
    
FROM patients p
LEFT JOIN config_risk_zipcodes rz ON p.current_zip = rz.zipcode
LEFT JOIN encounters e ON p.patient_id = e.patient_id
LEFT JOIN config_highrisk_service_lines hsl ON e.service_line = hsl.service_line
LEFT JOIN config_homeless_facilities hf ON e.facility_name = hf.facility_name
LEFT JOIN social_flags sf ON p.patient_id = sf.patient_id
LEFT JOIN diagnoses d ON p.patient_id = d.patient_id 
    AND d.diagnosis_date >= DATEADD(month, -12, CURRENT_DATE())
LEFT JOIN medications m ON p.patient_id = m.patient_id
LEFT JOIN labs_vitals lv ON p.patient_id = lv.patient_id 
    AND lv.lab_date >= DATEADD(month, -12, CURRENT_DATE())
LEFT JOIN clinical_notes cn ON p.patient_id = cn.patient_id 
    AND cn.note_date >= DATEADD(month, -12, CURRENT_DATE())
LEFT JOIN config_highrisk_doc_types hdt ON cn.note_type = hdt.doc_type
GROUP BY 
    p.patient_id, p.dob, p.gender, p.current_zip, p.insurance_status,
    rz.zipcode, rz.risk_weight;

-- ============================================================================
-- 2. TRAINING DATA VIEW FOR ML MODEL
-- ============================================================================

CREATE OR REPLACE VIEW vw_ml_training_data AS
SELECT 
    pf.*,
    prs.risk_score AS historical_risk_score,
    prs.risk_category AS historical_risk_category,
    -- Label: 1 if patient had homeless shelter visit in next 90 days, 0 otherwise
    CASE WHEN EXISTS (
        SELECT 1 FROM encounters e2
        JOIN config_homeless_facilities hf2 ON e2.facility_name = hf2.facility_name
        WHERE e2.patient_id = pf.patient_id
        AND e2.admit_date BETWEEN CURRENT_DATE() AND DATEADD(day, 90, CURRENT_DATE())
    ) THEN 1 ELSE 0 END AS label_homeless_risk_90d
FROM vw_patient_features pf
LEFT JOIN patient_risk_scores prs ON pf.patient_id = prs.patient_id;

-- ============================================================================
-- 3. STORED PROCEDURES
-- ============================================================================

-- Procedure to calculate and update risk scores
CREATE OR REPLACE PROCEDURE sp_calculate_risk_scores()
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    -- This procedure will be called by Python after ML model scoring
    -- Placeholder for now - actual ML scoring happens in Python
    RETURN 'Risk score calculation triggered. Execute ML pipeline externally.';
END;
$$;

-- Procedure to refresh materialized views
CREATE OR REPLACE PROCEDURE sp_refresh_feature_views()
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    -- Refresh views (in Snowflake, views are computed on-demand)
    -- This procedure documents the refresh pattern
    RETURN 'Feature views refreshed successfully.';
END;
$$;

-- ============================================================================
-- 4. HELPER FUNCTIONS
-- ============================================================================

-- Function to categorize risk scores
CREATE OR REPLACE FUNCTION fn_categorize_risk_score(risk_score FLOAT)
RETURNS STRING
AS
$$
    CASE 
        WHEN risk_score < 0.3 THEN 'LOW_RISK'
        WHEN risk_score < 0.7 THEN 'MED_RISK'
        ELSE 'HIGH_RISK'
    END
$$;

-- ============================================================================
-- 5. COMMENTS
-- ============================================================================

COMMENT ON VIEW vw_patient_features IS 'Aggregated patient features for ML model input';
COMMENT ON VIEW vw_ml_training_data IS 'Training dataset with labels for ML model development';
COMMENT ON PROCEDURE sp_calculate_risk_scores IS 'Trigger ML risk score calculation pipeline';
COMMENT ON PROCEDURE sp_refresh_feature_views IS 'Refresh feature engineering views';
COMMENT ON FUNCTION fn_categorize_risk_score IS 'Convert numeric risk score to risk category';

-- ============================================================================
-- END OF FEATURE ENGINEERING
-- ============================================================================
