#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering module for the fake job posting detection project.
Handles feature creation, selection, and transformation.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import logging
from pathlib import Path
from transformers import BertTokenizer, BertModel
import torch

logger = logging.getLogger(__name__)

def create_text_length_features(X):
    """
    Create features based on text lengths.
    
    Args:
        X (pd.DataFrame): Input features
        
    Returns:
        pd.DataFrame: Features with text length metrics
    """
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    
    for col in text_columns:
        # Character length
        X[f'{col}_char_length'] = X[col].str.len()
        
        # Word count
        X[f'{col}_word_count'] = X[col].str.split().str.len()
        
        # Average word length
        X[f'{col}_avg_word_length'] = X[col].str.split().apply(
            lambda x: np.mean([len(word) for word in x]) if x else 0
        )
    
    return X

def create_entity_features(X):
    """
    Create features based on extracted entities.
    
    Args:
        X (pd.DataFrame): Input features
        
    Returns:
        pd.DataFrame: Features with entity metrics
    """
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    
    for col in text_columns:
        # Entity count
        X[f'{col}_entity_count'] = X[f'{col}_entities'].str.len()
        
        # Key phrase count
        X[f'{col}_key_phrase_count'] = X[f'{col}_key_phrases'].str.len()
    
    return X

def create_tfidf_features(X_train, X_test, text_columns, max_features=5000):
    """
    Create TF-IDF features for text columns.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        text_columns (list): List of text columns to process
        max_features (int): Maximum number of features to extract
        
    Returns:
        tuple: TF-IDF features for training and test sets
    """
    vectorizers = {}
    tfidf_features_train = {}
    tfidf_features_test = {}
    
    for col in text_columns:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Fit and transform training data
        tfidf_train = vectorizer.fit_transform(X_train[col])
        tfidf_features_train[col] = tfidf_train
        
        # Transform test data
        tfidf_test = vectorizer.transform(X_test[col])
        tfidf_features_test[col] = tfidf_test
        
        vectorizers[col] = vectorizer
    
    return tfidf_features_train, tfidf_features_test, vectorizers

def create_bert_features(X_train, X_test, text_columns, max_length=128):
    """
    Create BERT embeddings for text columns.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        text_columns (list): List of text columns to process
        max_length (int): Maximum sequence length for BERT
        
    Returns:
        tuple: BERT embeddings for training and test sets
    """
    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    
    bert_features_train = {}
    bert_features_test = {}
    
    for col in text_columns:
        # Process training data
        train_embeddings = []
        for text in X_train[col]:
            inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            train_embeddings.append(embeddings[0])
        
        # Process test data
        test_embeddings = []
        for text in X_test[col]:
            inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            test_embeddings.append(embeddings[0])
        
        bert_features_train[col] = np.array(train_embeddings)
        bert_features_test[col] = np.array(test_embeddings)
    
    return bert_features_train, bert_features_test

def select_features(X, y, k=100):
    """
    Select the most important features using ANOVA F-value.
    
    Args:
        X (pd.DataFrame): Input features
        y (pd.Series): Target variable
        k (int): Number of features to select
        
    Returns:
        tuple: Selected features and feature selector
    """
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    return X_selected, selector

def engineer_features(X_train, X_test):
    """
    Complete feature engineering pipeline.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: Engineered features for training and test sets
    """
    logger.info("Starting feature engineering")
    
    # Create text length features
    X_train = create_text_length_features(X_train)
    X_test = create_text_length_features(X_test)
    
    # Create entity features
    X_train = create_entity_features(X_train)
    X_test = create_entity_features(X_test)
    
    # Create TF-IDF features
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    tfidf_features_train, tfidf_features_test, vectorizers = create_tfidf_features(
        X_train, X_test, text_columns
    )
    
    # Create BERT features
    bert_features_train, bert_features_test = create_bert_features(
        X_train, X_test, text_columns
    )
    
    # Combine all features
    X_train_engineered = pd.concat([
        X_train,
        pd.DataFrame(tfidf_features_train['description'].toarray()),
        pd.DataFrame(bert_features_train['description'])
    ], axis=1)
    
    X_test_engineered = pd.concat([
        X_test,
        pd.DataFrame(tfidf_features_test['description'].toarray()),
        pd.DataFrame(bert_features_test['description'])
    ], axis=1)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = X_train_engineered.select_dtypes(include=[np.number]).columns
    X_train_engineered[numerical_columns] = scaler.fit_transform(X_train_engineered[numerical_columns])
    X_test_engineered[numerical_columns] = scaler.transform(X_test_engineered[numerical_columns])
    
    logger.info("Feature engineering completed")
    return X_train_engineered, X_test_engineered

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