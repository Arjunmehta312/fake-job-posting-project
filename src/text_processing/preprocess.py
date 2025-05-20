#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text processing module for the fake job posting detection project.
Handles text cleaning, preprocessing, and feature extraction from job descriptions.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from pathlib import Path

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

logger = logging.getLogger(__name__)

def clean_text(text):
    """
    Clean text by removing special characters, extra whitespace, etc.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
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

def remove_stopwords(text):
    """
    Remove stopwords from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text with stopwords removed
    """
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    """
    Lemmatize text using NLTK's WordNetLemmatizer.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Lemmatized text
    """
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def extract_entities(text):
    """
    Extract named entities from text using spaCy.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of extracted entities
    """
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

def extract_key_phrases(text):
    """
    Extract key phrases from text using spaCy.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of extracted key phrases
    """
    doc = nlp(text)
    key_phrases = []
    
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) > 1:  # Only keep multi-word phrases
            key_phrases.append(chunk.text)
    
    return key_phrases

def create_tfidf_features(texts, max_features=5000):
    """
    Create TF-IDF features from text data.
    
    Args:
        texts (list): List of text documents
        max_features (int): Maximum number of features to extract
        
    Returns:
        scipy.sparse.csr.csr_matrix: TF-IDF features
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    return vectorizer.fit_transform(texts)

def preprocess_text(X_train, X_test):
    """
    Preprocess text data for both training and test sets.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: Preprocessed training and test features
    """
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    
    # Process training data
    X_train_processed = X_train.copy()
    for col in text_columns:
        # Clean text
        X_train_processed[col] = X_train_processed[col].apply(clean_text)
        
        # Remove stopwords
        X_train_processed[col] = X_train_processed[col].apply(remove_stopwords)
        
        # Lemmatize
        X_train_processed[col] = X_train_processed[col].apply(lemmatize_text)
        
        # Extract entities and key phrases
        X_train_processed[f'{col}_entities'] = X_train_processed[col].apply(extract_entities)
        X_train_processed[f'{col}_key_phrases'] = X_train_processed[col].apply(extract_key_phrases)
    
    # Process test data
    X_test_processed = X_test.copy()
    for col in text_columns:
        # Clean text
        X_test_processed[col] = X_test_processed[col].apply(clean_text)
        
        # Remove stopwords
        X_test_processed[col] = X_test_processed[col].apply(remove_stopwords)
        
        # Lemmatize
        X_test_processed[col] = X_test_processed[col].apply(lemmatize_text)
        
        # Extract entities and key phrases
        X_test_processed[f'{col}_entities'] = X_test_processed[col].apply(extract_entities)
        X_test_processed[f'{col}_key_phrases'] = X_test_processed[col].apply(extract_key_phrases)
    
    logger.info("Text preprocessing completed")
    return X_train_processed, X_test_processed

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the text processing pipeline
    from src.data.load_data import load_and_preprocess_data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    X_train_processed, X_test_processed = preprocess_text(X_train, X_test) 