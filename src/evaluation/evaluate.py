#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model evaluation module for the fake job posting detection project.
Handles model evaluation, metrics calculation, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import shap
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def create_evaluation_directory():
    """Create directory for evaluation results."""
    eval_dir = Path('outputs/evaluation')
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir

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
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'average_precision': average_precision_score(y_true, y_pred_proba)
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, output_dir, model_name):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        output_dir (Path): Directory to save plot
        model_name (str): Name of the model
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(output_dir / f'{model_name}_confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, output_dir, model_name):
    """
    Plot and save ROC curve.
    
    Args:
        y_true (array-like): True labels
        y_pred_proba (array-like): Predicted probabilities
        output_dir (Path): Directory to save plot
        model_name (str): Name of the model
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(output_dir / f'{model_name}_roc_curve.png')
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_proba, output_dir, model_name):
    """
    Plot and save precision-recall curve.
    
    Args:
        y_true (array-like): True labels
        y_pred_proba (array-like): Predicted probabilities
        output_dir (Path): Directory to save plot
        model_name (str): Name of the model
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.savefig(output_dir / f'{model_name}_pr_curve.png')
    plt.close()

def plot_feature_importance(model, X, output_dir, model_name):
    """
    Plot and save feature importance.
    
    Args:
        model: Trained model
        X (pd.DataFrame): Features
        output_dir (Path): Directory to save plot
        model_name (str): Name of the model
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(X.shape[1]), importances[indices])
        plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_feature_importance.png')
        plt.close()

def plot_shap_values(model, X, output_dir, model_name):
    """
    Plot and save SHAP values.
    
    Args:
        model: Trained model
        X (pd.DataFrame): Features
        output_dir (Path): Directory to save plot
        model_name (str): Name of the model
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, show=False)
        plt.title(f'SHAP Values - {model_name}')
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_shap_values.png')
        plt.close()
    except Exception as e:
        logger.warning(f"Could not generate SHAP values for {model_name}: {str(e)}")

def evaluate_models(models, X_test, y_test):
    """
    Evaluate multiple models and select the best one.
    
    Args:
        models (dict): Dictionary of trained models
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        tuple: Best model and its metrics
    """
    logger.info("Starting model evaluation")
    
    # Create evaluation directory
    eval_dir = create_evaluation_directory()
    
    # Evaluate each model
    results = {}
    best_f1 = 0
    best_model = None
    
    for model_name, model_info in models.items():
        model = model_info['model']
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        results[model_name] = metrics
        
        # Generate plots
        plot_confusion_matrix(y_test, y_pred, eval_dir, model_name)
        plot_roc_curve(y_test, y_pred_proba, eval_dir, model_name)
        plot_precision_recall_curve(y_test, y_pred_proba, eval_dir, model_name)
        plot_feature_importance(model, X_test, eval_dir, model_name)
        plot_shap_values(model, X_test, eval_dir, model_name)
        
        # Save classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        pd.DataFrame(report).to_csv(eval_dir / f'{model_name}_classification_report.csv')
        
        # Update best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model = model
    
    # Save all metrics
    pd.DataFrame(results).to_csv(eval_dir / 'model_metrics.csv')
    
    logger.info("Model evaluation completed")
    return best_model

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