#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model training module for the fake job posting detection project.
Handles model training, hyperparameter tuning, and model selection.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def create_model_directory():
    """Create directory for saving models."""
    model_dir = Path('outputs/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir

def get_model_params():
    """
    Define model parameters for grid search.
    
    Returns:
        dict: Model parameters for grid search
    """
    return {
        'logistic_regression': {
            'model': LogisticRegression(random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'class_weight': ['balanced', None],
                'max_iter': [1000]
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': ['balanced', 'balanced_subsample']
            }
        },
        'xgboost': {
            'model': XGBClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'scale_pos_weight': [1, 3, 5]
            }
        }
    }

def get_scoring_metrics():
    """
    Define scoring metrics for model evaluation.
    
    Returns:
        dict: Scoring metrics
    """
    return {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }

def handle_class_imbalance(X, y):
    """
    Handle class imbalance using SMOTE.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        
    Returns:
        tuple: Balanced features and target
    """
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced

def train_model(X, y, model_name, model_params):
    """
    Train a single model with hyperparameter tuning.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        model_name (str): Name of the model
        model_params (dict): Model parameters for grid search
        
    Returns:
        tuple: Best model and its parameters
    """
    logger.info(f"Training {model_name}")
    
    # Create cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create grid search
    grid_search = GridSearchCV(
        estimator=model_params['model'],
        param_grid=model_params['params'],
        cv=cv,
        scoring=get_scoring_metrics(),
        refit='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X, y)
    
    logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
    logger.info(f"Best F1 score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def save_model(model, model_name, model_dir):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        model_name (str): Name of the model
        model_dir (Path): Directory to save the model
    """
    model_path = model_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Saved {model_name} to {model_path}")

def train_models(X_train, y_train):
    """
    Train multiple models and select the best one.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        dict: Dictionary of trained models
    """
    logger.info("Starting model training")
    
    # Create model directory
    model_dir = create_model_directory()
    
    # Handle class imbalance
    X_balanced, y_balanced = handle_class_imbalance(X_train, y_train)
    
    # Get model parameters
    model_params = get_model_params()
    
    # Train models
    trained_models = {}
    for model_name, params in model_params.items():
        model, best_params = train_model(X_balanced, y_balanced, model_name, params)
        trained_models[model_name] = {
            'model': model,
            'params': best_params
        }
        
        # Save model
        save_model(model, model_name, model_dir)
    
    logger.info("Model training completed")
    return trained_models

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the model training pipeline
    from src.data.load_data import load_and_preprocess_data
    from src.text_processing.preprocess import preprocess_text
    from src.features.engineering import engineer_features
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    X_train_processed, X_test_processed = preprocess_text(X_train, X_test)
    X_train_engineered, X_test_engineered = engineer_features(X_train_processed, X_test_processed)
    trained_models = train_models(X_train_engineered, y_train) 