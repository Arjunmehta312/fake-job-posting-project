#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to run the complete fake job posting detection pipeline.
This script orchestrates the entire process from data loading to model deployment.
"""

import os
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create necessary directories
def create_directories():
    """Create necessary directories for the project."""
    directories = [
        'src/data',
        'src/eda',
        'src/text_processing',
        'src/features',
        'src/models',
        'src/evaluation',
        'src/deployment',
        'outputs/plots',
        'outputs/models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def main():
    """Main function to run the complete pipeline."""
    start_time = datetime.now()
    logger.info("Starting fake job posting detection pipeline")
    
    # Create project structure
    create_directories()
    
    try:
        # Import pipeline components
        from src.data.load_data import load_and_preprocess_data
        from src.eda.analyze_data import perform_eda
        from src.text_processing.preprocess import preprocess_text
        from src.features.engineering import engineer_features
        from src.models.train import train_models
        from src.evaluation.evaluate import evaluate_models
        from src.deployment.deploy import deploy_model
        
        # 1. Load and preprocess data
        logger.info("Loading and preprocessing data...")
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        
        # 2. Perform EDA
        logger.info("Performing exploratory data analysis...")
        perform_eda(X_train, y_train)
        
        # 3. Preprocess text data
        logger.info("Preprocessing text data...")
        X_train_processed, X_test_processed = preprocess_text(X_train, X_test)
        
        # 4. Engineer features
        logger.info("Engineering features...")
        X_train_features, X_test_features = engineer_features(X_train_processed, X_test_processed)
        
        # 5. Train models
        logger.info("Training models...")
        models = train_models(X_train_features, y_train)
        
        # 6. Evaluate models
        logger.info("Evaluating models...")
        best_model = evaluate_models(models, X_test_features, y_test)
        
        # 7. Deploy model
        logger.info("Deploying best model...")
        deploy_model(best_model)
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Pipeline completed successfully in {duration}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 