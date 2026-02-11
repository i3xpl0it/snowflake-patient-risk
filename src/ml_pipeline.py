"""ML Pipeline for Patient Risk Prediction.

This module handles training and scoring of the ML model for predicting
homeless/at-risk patient identification using Snowflake ML.
"""

import os
import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from snowflake.snowpark import Session
from snowflake.ml.modeling.ensemble import RandomForestClassifier
from snowflake.ml.modeling.linear_model import LogisticRegression
from snowflake.ml.modeling.preprocessing import StandardScaler
from snowflake.ml.modeling.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatientRiskMLPipeline:
    """ML Pipeline for patient risk prediction."""
    
    def __init__(self, snowflake_session: Session):
        """Initialize the ML pipeline.
        
        Args:
            snowflake_session: Active Snowflake session
        """
        self.session = snowflake_session
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'label_homeless_risk_90d'
        
    def load_training_data(self) -> pd.DataFrame:
        """Load training data from Snowflake view.
        
        Returns:
            DataFrame with training features and labels
        """
        logger.info("Loading training data from vw_ml_training_data")
        
        query = """
        SELECT *
        FROM vw_ml_training_data
        WHERE label_homeless_risk_90d IS NOT NULL
        """
        
        df = self.session.sql(query).to_pandas()
        logger.info(f"Loaded {len(df)} training samples")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (features, labels)
        """
        # Define feature columns (exclude ID and label columns)
        exclude_cols = [
            'patient_id', 'dob', 'current_zip', 'gender', 'insurance_status',
            'label_homeless_risk_90d', 'historical_risk_category'
        ]
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.feature_columns].fillna(0)
        y = df[self.target_column]
        
        logger.info(f"Prepared {len(self.feature_columns)} features")
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train the risk prediction model.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Training Random Forest model")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Define model with hyperparameter grid
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced'
        )
        
        # Train model
        self.model = rf_model.fit(X_scaled, y)
        
        # Get predictions and probabilities
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y, y_proba)
        
        metrics = {
            'auc_score': auc_score,
            'n_samples': len(X),
            'n_features': len(self.feature_columns),
            'positive_rate': y.mean()
        }
        
        logger.info(f"Model trained with AUC: {auc_score:.4f}")
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model.
        
        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importances = self.model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_imp
    
    def score_patients(self, patient_ids: List[str] = None) -> pd.DataFrame:
        """Score patients for risk prediction.
        
        Args:
            patient_ids: Optional list of patient IDs to score. If None, score all.
            
        Returns:
            DataFrame with patient_id and risk scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        logger.info("Scoring patients")
        
        # Load data to score
        if patient_ids:
            patient_list = "','".join(patient_ids)
            where_clause = f"WHERE patient_id IN ('{patient_list}')"
        else:
            where_clause = ""
        
        query = f"""
        SELECT *
        FROM vw_patient_features
        {where_clause}
        """
        
        df = self.session.sql(query).to_pandas()
        
        # Prepare features
        X = df[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        risk_scores = self.model.predict_proba(X_scaled)[:, 1]
        
        # Create results dataframe
        results = pd.DataFrame({
            'patient_id': df['patient_id'],
            'risk_score': risk_scores,
            'risk_category': pd.cut(risk_scores, 
                                   bins=[0, 0.3, 0.7, 1.0],
                                   labels=['LOW_RISK', 'MED_RISK', 'HIGH_RISK'])
        })
        
        logger.info(f"Scored {len(results)} patients")
        return results
    
    def save_scores_to_snowflake(self, scores: pd.DataFrame, model_version: str):
        """Save risk scores to Snowflake table.
        
        Args:
            scores: DataFrame with risk scores
            model_version: Model version identifier
        """
        logger.info("Saving scores to Snowflake")
        
        # Add metadata
        scores['model_version'] = model_version
        scores['prediction_date'] = pd.Timestamp.now()
        
        # Create Snowpark DataFrame and write to table
        snow_df = self.session.create_dataframe(scores)
        
        # Merge into patient_risk_scores table
        snow_df.write.mode('overwrite').save_as_table(
            'patient_risk_scores',
            mode='overwrite'
        )
        
        logger.info(f"Saved {len(scores)} risk scores to patient_risk_scores table")
    
    def save_model(self, filepath: str):
        """Save trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        
        logger.info(f"Model loaded from {filepath}")


def create_snowflake_session() -> Session:
    """Create Snowflake session from environment variables.
    
    Returns:
        Active Snowflake session
    """
    connection_parameters = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "role": os.getenv("SNOWFLAKE_ROLE", "SYSADMIN"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
    }
    
    return Session.builder.configs(connection_parameters).create()


if __name__ == "__main__":
    # Example usage
    session = create_snowflake_session()
    
    # Initialize pipeline
    pipeline = PatientRiskMLPipeline(session)
    
    # Load and prepare data
    df = pipeline.load_training_data()
    X, y = pipeline.prepare_features(df)
    
    # Train model
    metrics = pipeline.train_model(X, y)
    print(f"Training metrics: {metrics}")
    
    # Get feature importance
    feature_imp = pipeline.get_feature_importance()
    print("\nTop 10 Important Features:")
    print(feature_imp.head(10))
    
    # Score all patients
    scores = pipeline.score_patients()
    
    # Save scores to Snowflake
    pipeline.save_scores_to_snowflake(scores, model_version="v1.0")
    
    # Save model
    pipeline.save_model("models/patient_risk_model.pkl")
    
    session.close()
