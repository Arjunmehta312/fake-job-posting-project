#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading and preprocessing module for the fake job posting detection project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    """
    Load and preprocess the fake job posting dataset.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    logger.info("Loading and preprocessing data...")
    
    # Load the dataset
    try:
        df = pd.read_csv('dataset/fake_job_postings.csv')
    except FileNotFoundError:
        logger.error("Dataset not found. Please ensure 'fake_job_postings.csv' is in the dataset directory.")
        raise
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Encode categorical variables
    df = encode_categorical_variables(df)
    
    # Split features and target
    X = df.drop('fraudulent', axis=1)
    y = df['fraudulent']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Data loaded successfully. Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    # Fill missing text columns with empty string
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_columns:
        df[col] = df[col].fillna('')
    
    # Fill missing categorical columns with 'Unknown'
    categorical_columns = ['location', 'department', 'salary_range', 'employment_type', 'required_experience', 'required_education', 'industry', 'function']
    for col in categorical_columns:
        df[col] = df[col].fillna('Unknown')
    
    return df

def encode_categorical_variables(df):
    """
    Encode categorical variables in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical variables
    """
    # Initialize label encoder
    le = LabelEncoder()
    
    # Encode categorical columns
    categorical_columns = ['location', 'department', 'salary_range', 'employment_type', 
                         'required_experience', 'required_education', 'industry', 'function']
    
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the data loading pipeline
    X_train, X_test, y_train, y_test = load_and_preprocess_data() 