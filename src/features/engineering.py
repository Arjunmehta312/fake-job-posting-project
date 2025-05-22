#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering module for the fake job posting detection project.
Handles feature creation, transformation, and selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import logging

logger = logging.getLogger(__name__)

def engineer_features(X_train, X_test):
    """
    Engineer features for both training and test sets.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: Engineered training and test features
    """
    logger.info("Engineering features...")
    
    # Create interaction features (optimized)
    X_train = create_interaction_features(X_train)
    X_test = create_interaction_features(X_test)
    
    # Scale numerical features
    X_train, X_test = scale_numerical_features(X_train, X_test)
    
    # Select best features
    X_train, X_test = select_best_features(X_train, X_test, X_train['fraudulent'])
    
    logger.info(f"Feature engineering completed. Final feature dimensions - Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test

def create_interaction_features(df):
    """
    Create interaction features between numerical columns (optimized version).
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with interaction features
    """
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Create a new DataFrame for interactions
    interaction_df = pd.DataFrame(index=df.index)
    
    # Create interactions for top 5 numerical columns only
    top_cols = numerical_cols[:5]
    
    for i in range(len(top_cols)):
        for j in range(i + 1, len(top_cols)):
            col1 = top_cols[i]
            col2 = top_cols[j]
            
            # Create multiplication interaction
            interaction_df[f'{col1}_{col2}_mul'] = df[col1] * df[col2]
            
            # Create division interaction (avoid division by zero)
            interaction_df[f'{col1}_{col2}_div'] = df[col1] / (df[col2] + 1e-6)
    
    # Concatenate original DataFrame with interactions
    return pd.concat([df, interaction_df], axis=1)

def scale_numerical_features(X_train, X_test):
    """
    Scale numerical features using StandardScaler.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: Scaled training and test features
    """
    # Get numerical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Scale training data
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    
    # Scale test data using training scaler
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return X_train, X_test

def select_best_features(X_train, X_test, y_train, k=100):
    """
    Select best features using SelectKBest with f_classif.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training target
        k (int): Number of features to select
        
    Returns:
        tuple: Selected training and test features
    """
    # Initialize feature selector
    selector = SelectKBest(score_func=f_classif, k=k)
    
    # Fit and transform training data
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    # Transform test data
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    # Create DataFrames with selected features
    X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)
    
    return X_train_selected, X_test_selected

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the feature engineering pipeline
    from src.data.load_data import load_and_preprocess_data
    from src.text_processing.preprocess import preprocess_text
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    X_train_processed, X_test_processed = preprocess_text(X_train, X_test)
    X_train_engineered, X_test_engineered = engineer_features(X_train_processed, X_test_processed) 