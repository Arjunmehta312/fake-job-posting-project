#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deployment module for the fake job posting detection project.
Handles model serving, prediction, and API endpoints.
"""

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import logging
from pathlib import Path
from src.text_processing.preprocess import preprocess_text
from src.features.engineering import engineer_features

logger = logging.getLogger(__name__)

class JobPostingPredictor:
    """
    Class for making predictions on new job postings.
    """
    
    def __init__(self, model_path='outputs/models/best_model.joblib'):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path (str): Path to the trained model
        """
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    
    def preprocess_input(self, job_posting):
        """
        Preprocess a single job posting.
        
        Args:
            job_posting (dict): Job posting data
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        # Convert to DataFrame
        df = pd.DataFrame([job_posting])
        
        # Preprocess text
        df_processed, _ = preprocess_text(df, df)
        
        # Engineer features
        df_engineered, _ = engineer_features(df_processed, df_processed)
        
        return df_engineered
    
    def predict(self, job_posting):
        """
        Make prediction for a job posting.
        
        Args:
            job_posting (dict): Job posting data
            
        Returns:
            tuple: Prediction (0 or 1) and probability
        """
        # Preprocess input
        X = self.preprocess_input(job_posting)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0][1]
        
        return prediction, probability

def create_streamlit_app():
    """
    Create a Streamlit app for job posting prediction.
    """
    st.title('Fake Job Posting Detector')
    st.write('Enter job posting details to check if it might be fraudulent.')
    
    # Input fields
    title = st.text_input('Job Title')
    company_profile = st.text_area('Company Profile')
    description = st.text_area('Job Description')
    requirements = st.text_area('Requirements')
    benefits = st.text_area('Benefits')
    employment_type = st.selectbox('Employment Type', ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Other'])
    required_experience = st.selectbox('Required Experience', ['Entry level', 'Mid-Senior level', 'Senior level', 'Executive', 'Not Applicable'])
    required_education = st.selectbox('Required Education', ['High School', 'Bachelor', 'Master', 'PhD', 'Not Applicable'])
    industry = st.text_input('Industry')
    function = st.text_input('Function')
    location = st.text_input('Location')
    telecommuting = st.checkbox('Telecommuting')
    has_company_logo = st.checkbox('Has Company Logo')
    has_questions = st.checkbox('Has Questions')
    
    if st.button('Predict'):
        # Create job posting dictionary
        job_posting = {
            'title': title,
            'company_profile': company_profile,
            'description': description,
            'requirements': requirements,
            'benefits': benefits,
            'employment_type': employment_type,
            'required_experience': required_experience,
            'required_education': required_education,
            'industry': industry,
            'function': function,
            'location': location,
            'telecommuting': int(telecommuting),
            'has_company_logo': int(has_company_logo),
            'has_questions': int(has_questions)
        }
        
        # Make prediction
        predictor = JobPostingPredictor()
        prediction, probability = predictor.predict(job_posting)
        
        # Display results
        if prediction == 1:
            st.error(f'This job posting might be fraudulent (Probability: {probability:.2%})')
        else:
            st.success(f'This job posting appears to be legitimate (Probability: {1-probability:.2%})')

def deploy_model(model):
    """
    Deploy the best model for prediction.
    
    Args:
        model: Trained model to deploy
    """
    # Save the best model
    model_dir = Path('outputs/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / 'best_model.joblib'
    joblib.dump(model, model_path)
    logger.info(f"Saved best model to {model_path}")
    
    # Create Streamlit app
    app_path = Path('src/deployment/app.py')
    with open(app_path, 'w') as f:
        f.write("""
import streamlit as st
from deploy import create_streamlit_app

if __name__ == '__main__':
    create_streamlit_app()
        """)
    
    logger.info(f"Created Streamlit app at {app_path}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the deployment pipeline
    from src.data.load_data import load_and_preprocess_data
    from src.text_processing.preprocess import preprocess_text
    from src.features.engineering import engineer_features
    from src.models.train import train_models
    from src.evaluation.evaluate import evaluate_models
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    X_train_processed, X_test_processed = preprocess_text(X_train, X_test)
    X_train_engineered, X_test_engineered = engineer_features(X_train_processed, X_test_processed)
    trained_models = train_models(X_train_engineered, y_train)
    best_model = evaluate_models(trained_models, X_test_engineered, y_test)
    deploy_model(best_model) 