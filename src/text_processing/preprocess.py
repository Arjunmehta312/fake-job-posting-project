#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text preprocessing module for the fake job posting detection project.
Handles text cleaning, normalization, and feature extraction from text data.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def clean_text(text):
    """
    Clean and normalize text data.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def preprocess_text(X_train, X_test):
    """
    Preprocess text data for both training and test sets.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: Processed training and test features
    """
    logger.info("Preprocessing text data...")
    
    # Text columns to process
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    
    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    
    # Process each text column
    for col in text_columns:
        # Clean text
        X_train[col] = X_train[col].apply(clean_text)
        X_test[col] = X_test[col].apply(clean_text)
        
        # Create TF-IDF features
        tfidf_features = tfidf.fit_transform(X_train[col])
        tfidf_features_test = tfidf.transform(X_test[col])
        
        # Convert to DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=[f'{col}_tfidf_{i}' for i in range(tfidf_features.shape[1])]
        )
        tfidf_df_test = pd.DataFrame(
            tfidf_features_test.toarray(),
            columns=[f'{col}_tfidf_{i}' for i in range(tfidf_features_test.shape[1])]
        )
        
        # Add TF-IDF features to original data
        X_train = pd.concat([X_train, tfidf_df], axis=1)
        X_test = pd.concat([X_test, tfidf_df_test], axis=1)
        
        # Drop original text column
        X_train = X_train.drop(col, axis=1)
        X_test = X_test.drop(col, axis=1)
    
    logger.info(f"Text preprocessing completed. New feature dimensions - Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test

def extract_text_features(text):
    """
    Extract additional features from text.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary of text features
    """
    if not isinstance(text, str):
        return {
            'word_count': 0,
            'char_count': 0,
            'avg_word_length': 0,
            'stopword_count': 0
        }
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Calculate features
    word_count = len(tokens)
    char_count = len(text)
    avg_word_length = char_count / word_count if word_count > 0 else 0
    stopword_count = sum(1 for word in tokens if word in stop_words)
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'avg_word_length': avg_word_length,
        'stopword_count': stopword_count
    }

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the text preprocessing pipeline
    from src.data.load_data import load_and_preprocess_data
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    X_train_processed, X_test_processed = preprocess_text(X_train, X_test) 