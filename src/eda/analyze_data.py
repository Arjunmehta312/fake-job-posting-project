#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Exploratory Data Analysis module for the fake job posting detection project.
Generates various visualizations and statistical analyses of the dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def create_output_dir():
    """Create output directory for plots if it doesn't exist."""
    output_dir = Path('outputs/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def plot_class_distribution(y, output_dir):
    """
    Plot the distribution of fraudulent vs legitimate job postings.
    
    Args:
        y (pd.Series): Target variable
        output_dir (Path): Directory to save plots
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y)
    plt.title('Distribution of Job Postings (Fraudulent vs Legitimate)')
    plt.xlabel('Fraudulent')
    plt.ylabel('Count')
    plt.savefig(output_dir / 'class_distribution.png')
    plt.close()
    
    # Calculate percentages
    total = len(y)
    fraud_percentage = (y == 1).sum() / total * 100
    logger.info(f"Fraudulent job postings: {fraud_percentage:.2f}%")

def plot_categorical_distributions(X, output_dir):
    """
    Plot distributions of categorical variables.
    
    Args:
        X (pd.DataFrame): Features
        output_dir (Path): Directory to save plots
    """
    categorical_columns = ['employment_type', 'required_experience', 'required_education', 'industry', 'function']
    
    for col in categorical_columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(y=X[col])
        plt.title(f'Distribution of {col}')
        plt.xlabel('Count')
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(output_dir / f'{col}_distribution.png')
        plt.close()

def plot_correlation_heatmap(X, output_dir):
    """
    Plot correlation heatmap of numerical features.
    
    Args:
        X (pd.DataFrame): Features
        output_dir (Path): Directory to save plots
    """
    numerical_columns = ['telecommuting', 'has_company_logo', 'has_questions']
    corr = X[numerical_columns].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png')
    plt.close()

def generate_wordclouds(X, output_dir):
    """
    Generate word clouds for text columns.
    
    Args:
        X (pd.DataFrame): Features
        output_dir (Path): Directory to save plots
    """
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    
    for col in text_columns:
        text = ' '.join(X[col].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {col}')
        plt.savefig(output_dir / f'{col}_wordcloud.png')
        plt.close()

def plot_text_length_distributions(X, output_dir):
    """
    Plot distributions of text lengths for text columns.
    
    Args:
        X (pd.DataFrame): Features
        output_dir (Path): Directory to save plots
    """
    text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    
    for col in text_columns:
        lengths = X[col].str.len()
        
        plt.figure(figsize=(12, 6))
        sns.histplot(lengths, bins=50)
        plt.title(f'Distribution of {col} Lengths')
        plt.xlabel('Length (characters)')
        plt.ylabel('Count')
        plt.savefig(output_dir / f'{col}_length_distribution.png')
        plt.close()

def perform_eda(X, y):
    """
    Perform comprehensive exploratory data analysis.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
    """
    output_dir = create_output_dir()
    logger.info("Starting exploratory data analysis")
    
    # Plot class distribution
    plot_class_distribution(y, output_dir)
    
    # Plot categorical distributions
    plot_categorical_distributions(X, output_dir)
    
    # Plot correlation heatmap
    plot_correlation_heatmap(X, output_dir)
    
    # Generate word clouds
    generate_wordclouds(X, output_dir)
    
    # Plot text length distributions
    plot_text_length_distributions(X, output_dir)
    
    # Generate summary statistics
    summary_stats = X.describe()
    summary_stats.to_csv(output_dir / 'summary_statistics.csv')
    
    logger.info("Exploratory data analysis completed")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the EDA pipeline
    from src.data.load_data import load_and_preprocess_data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    perform_eda(X_train, y_train) 