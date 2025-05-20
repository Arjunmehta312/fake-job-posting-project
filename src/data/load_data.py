#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading and preprocessing module for the fake job posting detection project.
Handles data loading, cleaning, and initial preprocessing steps.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

def load_data(file_path='dataset/fake_job_postings.csv'):
    """
    Load the fake job postings dataset.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded dataset with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def clean_data(df):
    """
    Clean the dataset by handling missing values and inconsistencies.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_columns:
        df_clean[col] = df_clean[col].fillna('')
    
    # Fill missing values in categorical columns
    categorical_columns = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    for col in categorical_columns:
        df_clean[col] = df_clean[col].fillna('Unknown')
    
    # Clean salary range
    df_clean['salary_range'] = df_clean['salary_range'].fillna('Unknown')
    
    # Clean location
    df_clean['location'] = df_clean['location'].fillna('Unknown')
    
    # Convert boolean columns
    boolean_columns = ['telecommuting', 'has_company_logo', 'has_questions']
    for col in boolean_columns:
        df_clean[col] = df_clean[col].fillna(0).astype(int)
    
    logger.info("Data cleaning completed")
    return df_clean

def preprocess_data(df):
    """
    Preprocess the cleaned dataset for modeling.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        
    Returns:
        tuple: Preprocessed features and target
    """
    # Select features for initial modeling
    feature_columns = [
        'title', 'company_profile', 'description', 'requirements', 'benefits',
        'employment_type', 'required_experience', 'required_education',
        'industry', 'function', 'location', 'telecommuting',
        'has_company_logo', 'has_questions'
    ]
    
    X = df[feature_columns]
    y = df['fraudulent']
    
    # Encode categorical variables
    categorical_columns = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    label_encoders = {}
    
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])
    
    logger.info("Data preprocessing completed")
    return X, y

def load_and_preprocess_data(test_size=0.2, random_state=42):
    """
    Complete data loading and preprocessing pipeline.
    
    Args:
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Load data
    df = load_data()
    
    # Clean data
    df_clean = clean_data(df)
    
    # Preprocess data
    X, y = preprocess_data(df_clean)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Data split into train ({len(X_train)} samples) and test ({len(X_test)} samples) sets")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the pipeline
    X_train, X_test, y_train, y_test = load_and_preprocess_data() 