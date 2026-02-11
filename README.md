# Snowflake Patient Risk Prediction System

> Clinical risk prediction system combining Snowflake ML and RAG for homeless/at-risk patient identification

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Snowflake](https://img.shields.io/badge/Snowflake-ML-29B5E8.svg)](https://snowflake.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This system leverages **Snowflake ML** and **Retrieval-Augmented Generation (RAG)** to predict patient risk for homelessness and identify at-risk individuals in clinical settings. It combines structured clinical data with unstructured clinical notes to provide comprehensive risk assessments.

### Key Features

- **Machine Learning Risk Scoring**: Random Forest classifier trained on 20+ clinical and social determinants
- **RAG-Based Clinical Summaries**: AI-generated patient context from clinical notes using vector search
- **Snowflake Integration**: Native Snowflake ML for scalable analytics
- **Feature Engineering**: Automated feature extraction from EHR data
- **Configuration-Driven Rules**: Flexible risk rules stored as data tables

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Snowflake Database                      │
├──────────────────┬──────────────────┬──────────────────────┤
│  Core Tables     │  Config Tables   │  ML Output Tables    │
│  - patients      │  - risk_zipcodes │  - patient_risk_     │
│  - encounters    │  - homeless_     │    scores            │
│  - diagnoses     │    facilities    │  - rag_summaries     │
│  - medications   │  - highrisk_     │                      │
│  - labs_vitals   │    service_lines │                      │
│  - social_flags  │                  │                      │
│  - clinical_notes│                  │                      │
└──────────────────┴──────────────────┴──────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Feature Engineering                        │
│              (SQL Views + Stored Procedures)                 │
│  - vw_patient_features                                       │
│  - vw_ml_training_data                                       │
└─────────────────────────────────────────────────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │         Python ML Pipeline           │
        │  - Training & Scoring                │
        │  - Feature Preparation               │
        │  - Model Persistence                 │
        └──────────────────────────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │         RAG System                   │
        │  - Vector DB (ChromaDB)              │
        │  - Embedding (SentenceTransformers)  │
        │  - Summary Generation (Snowflake     │
        │    Cortex LLM)                       │
        └──────────────────────────────────────┘
```

## Project Structure

```
snowflake-patient-risk/
├── sql/
│   ├── 01_schema.sql              # Database schema
│   └── 02_feature_engineering.sql # Feature views & procedures
├── src/
│   ├── ml_pipeline.py             # ML training & scoring
│   └── rag_system.py              # RAG implementation
├── requirements.txt               # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.9+
- Snowflake account with Snowpark and Cortex enabled
- Access to clinical EHR data (synthetic or real)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/i3xpl0it/snowflake-patient-risk.git
cd snowflake-patient-risk
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Snowflake connection**
```bash
export SNOWFLAKE_ACCOUNT="your_account"
export SNOWFLAKE_USER="your_user"
export SNOWFLAKE_PASSWORD="your_password"
export SNOWFLAKE_WAREHOUSE="your_warehouse"
export SNOWFLAKE_DATABASE="your_database"
export SNOWFLAKE_SCHEMA="PUBLIC"
```

4. **Initialize Snowflake schema**
```bash
snowsql -f sql/01_schema.sql
snowsql -f sql/02_feature_engineering.sql
```

## Usage

### Training the ML Model

```python
from src.ml_pipeline import PatientRiskMLPipeline, create_snowflake_session

# Create Snowflake session
session = create_snowflake_session()

# Initialize pipeline
pipeline = PatientRiskMLPipeline(session)

# Load and prepare data
df = pipeline.load_training_data()
X, y = pipeline.prepare_features(df)

# Train model
metrics = pipeline.train_model(X, y)
print(f"Model AUC: {metrics['auc_score']:.4f}")

# Score patients
scores = pipeline.score_patients()
pipeline.save_scores_to_snowflake(scores, model_version="v1.0")

# Save model
pipeline.save_model("models/patient_risk_model.pkl")
session.close()
```

### Generating RAG Summaries

```python
from src.rag_system import ClinicalNotesRAG
from src.ml_pipeline import create_snowflake_session

# Create session
session = create_snowflake_session()

# Initialize RAG system
rag = ClinicalNotesRAG(session)
rag.initialize_vector_store()

# Embed clinical notes
notes_df = rag.load_clinical_notes()
rag.embed_notes(notes_df)

# Generate summary for a specific patient
summary = rag.generate_patient_summary(patient_id="P123456")
print(summary['summary'])

# Batch generate for all patients
summaries = rag.batch_generate_summaries()
session.close()
```

## Data Model

### Core Tables

- **patients**: Master patient index with demographics
- **encounters**: Clinical encounters (ED visits, admissions)
- **diagnoses**: ICD-10 diagnosis codes
- **medications**: Prescription history and adherence
- **labs_vitals**: Laboratory results and vital signs
- **social_flags**: SDOH flags (homelessness, food insecurity)
- **clinical_notes**: Unstructured text notes for RAG

### Configuration Tables (Rules as Data)

- **config_risk_zipcodes**: High-risk geographic areas
- **config_homeless_facilities**: Known shelters and outreach centers
- **config_highrisk_service_lines**: High-risk clinical service lines
- **config_highrisk_doc_types**: Document types indicating risk

### Output Tables

- **patient_risk_scores**: ML model risk predictions
- **rag_summaries**: AI-generated patient summaries

## Features

The ML model uses 20+ features including:

- **Demographics**: Age, gender, insurance status
- **Geographic Risk**: High-risk zip code indicator
- **Utilization**: ED visits, encounter frequency, length of stay
- **Clinical**: Diagnosis counts, medication adherence, abnormal labs
- **SDOH**: Homelessness flags, food insecurity, housing instability
- **Service Line**: Visits to addiction medicine, psychiatry, etc.

## Model Performance

Typical performance metrics (with representative data):
- **AUC-ROC**: 0.78-0.85
- **Precision**: 0.65-0.75 (at 0.5 threshold)
- **Recall**: 0.70-0.80

## Contributing

Contributions are welcome! This is a reusable framework that can be adapted for various clinical risk prediction use cases.

### To Contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - feel free to use this for your own projects!

## Acknowledgments

- Built with Snowflake ML and Cortex
- Vector search powered by ChromaDB
- Embeddings by SentenceTransformers

## Contact

For questions or collaboration:
- GitHub: [@i3xpl0it](https://github.com/i3xpl0it)
- Repository: [snowflake-patient-risk](https://github.com/i3xpl0it/snowflake-patient-risk)

---

**Note**: This system is for research and demonstration purposes. Always comply with HIPAA and local healthcare data regulations when working with real patient data.
