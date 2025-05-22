#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model evaluation module for the fake job posting detection project.
Handles model evaluation, metrics calculation, and model selection.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def evaluate_models(models, X_test, y_test):
    """
    Evaluate multiple models and select the best one.
    
    Args:
        models (dict): Dictionary of trained models
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        object: Best performing model
    """
    logger.info("Evaluating models...")
    
    # Create directory for evaluation results
    eval_dir = Path('outputs/plots')
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each model
    results = {}
    for model_name, model_info in models.items():
        model = model_info['model']
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        results[model_name] = metrics
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, model_name, eval_dir)
        
        # Plot ROC curve
        plot_roc_curve(y_test, y_pred_proba, model_name, eval_dir)
        
        logger.info(f"{model_name} metrics:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Select best model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = models[best_model_name]['model']
    
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Best F1 score: {results[best_model_name]['f1']:.4f}")
    
    return best_model

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate various evaluation metrics.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        y_pred_proba (array-like): Predicted probabilities
        
    Returns:
        dict: Dictionary of metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        model_name (str): Name of the model
        save_dir (Path): Directory to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_dir / f'confusion_matrix_{model_name}.png')
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, model_name, save_dir):
    """
    Plot and save ROC curve.
    
    Args:
        y_true (array-like): True labels
        y_pred_proba (array-like): Predicted probabilities
        model_name (str): Name of the model
        save_dir (Path): Directory to save the plot
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(save_dir / f'roc_curve_{model_name}.png')
    plt.close()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the evaluation pipeline
    from src.data.load_data import load_and_preprocess_data
    from src.text_processing.preprocess import preprocess_text
    from src.features.engineering import engineer_features
    from src.models.train import train_models
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    X_train_processed, X_test_processed = preprocess_text(X_train, X_test)
    X_train_engineered, X_test_engineered = engineer_features(X_train_processed, X_test_processed)
    trained_models = train_models(X_train_engineered, y_train)
    best_model = evaluate_models(trained_models, X_test_engineered, y_test) 