#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model deployment module for the fake job posting detection project.
Handles model deployment and serving predictions.
"""

import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import streamlit as st
from src.text_processing.preprocess import clean_text
from src.features.engineering import engineer_features

logger = logging.getLogger(__name__)

def deploy_model(model):
    """
    Deploy the trained model for serving predictions.
    
    Args:
        model: Trained model to deploy
    """
    logger.info("Deploying model...")
    
    # Create deployment directory
    deploy_dir = Path('outputs/models')
    deploy_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = deploy_dir / 'best_model.joblib'
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Create Streamlit app
    create_streamlit_app(model)

def create_streamlit_app(model):
    """
    Create a Streamlit app for model serving.
    
    Args:
        model: Trained model to use for predictions
    """
    st.title('Fake Job Posting Detection')
    st.write('Enter job posting details to check if it\'s legitimate or fraudulent.')
    
    # Input fields
    title = st.text_input('Job Title')
    company_profile = st.text_area('Company Profile')
    description = st.text_area('Job Description')
    requirements = st.text_area('Requirements')
    benefits = st.text_area('Benefits')
    
    # Additional fields
    location = st.text_input('Location')
    department = st.text_input('Department')
    salary_range = st.text_input('Salary Range')
    employment_type = st.selectbox(
        'Employment Type',
        ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Other']
    )
    required_experience = st.selectbox(
        'Required Experience',
        ['Entry level', 'Mid-Senior level', 'Senior level', 'Executive level', 'Not specified']
    )
    required_education = st.selectbox(
        'Required Education',
        ['High School', 'Bachelor\'s', 'Master\'s', 'PhD', 'Not specified']
    )
    industry = st.text_input('Industry')
    function = st.text_input('Function')
    
    # Boolean fields
    telecommuting = st.checkbox('Telecommuting')
    has_company_logo = st.checkbox('Has Company Logo')
    has_questions = st.checkbox('Has Questions')
    
    if st.button('Predict'):
        # Create input data
        input_data = pd.DataFrame({
            'title': [title],
            'company_profile': [company_profile],
            'description': [description],
            'requirements': [requirements],
            'benefits': [benefits],
            'location': [location],
            'department': [department],
            'salary_range': [salary_range],
            'employment_type': [employment_type],
            'required_experience': [required_experience],
            'required_education': [required_education],
            'industry': [industry],
            'function': [function],
            'telecommuting': [int(telecommuting)],
            'has_company_logo': [int(has_company_logo)],
            'has_questions': [int(has_questions)]
        })
        
        # Preprocess input
        input_data = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Display result
        if prediction == 1:
            st.error('⚠️ This job posting appears to be fraudulent!')
        else:
            st.success('✅ This job posting appears to be legitimate!')
        
        st.write(f'Confidence: {probability:.2%}')

def preprocess_input(input_data):
    """
    Preprocess input data for prediction.
    
    Args:
        input_data (pd.DataFrame): Raw input data
        
    Returns:
        pd.DataFrame: Preprocessed input data
    """
    # Clean text fields
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_columns:
        input_data[col] = input_data[col].apply(clean_text)
    
    # Engineer features
    input_data, _ = engineer_features(input_data, input_data)
    
    return input_data

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