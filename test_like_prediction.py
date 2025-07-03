#!/usr/bin/env python3
"""
Test script for Like Prediction (CTR) using Stack Overflow votes data
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import os
from prediction_models import CTRPredictor

def parse_xml(path):
    """Parse XML file and return DataFrame"""
    tree = ET.parse(path)
    root = tree.getroot()
    return pd.DataFrame([row.attrib for row in root])

def load_data(base_path='data'):
    """Load XML data files"""
    print("=== Loading Data Files ===")
    
    # Load required files
    df_posts = parse_xml(os.path.join(base_path, 'Posts.xml'))
    df_users = parse_xml(os.path.join(base_path, 'Users.xml'))
    df_votes = parse_xml(os.path.join(base_path, 'Votes.xml'))
    
    print(f"Loaded:")
    print(f"  - {len(df_posts)} posts")
    print(f"  - {len(df_users)} users")
    print(f"  - {len(df_votes)} votes")
    
    return df_posts, df_users, df_votes

def test_like_prediction():
    """Test like prediction functionality"""
    print("=== Testing Like Prediction ===")
    
    # Load data
    df_posts, df_users, df_votes = load_data()
    
    # Initialize predictor
    predictor = CTRPredictor()
    
    # Test different models
    models = ['xgboost', 'random_forest', 'logistic_regression']
    
    for model_type in models:
        print(f"\n{'='*50}")
        print(f"Testing {model_type.upper()} model")
        print(f"{'='*50}")
        
        try:
            # Train model
            results = predictor.train_like_prediction_model(df_votes, df_posts, df_users, model_type=model_type)
            
            # Print results
            print(f"\nResults for {model_type}:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  Precision: {results['precision']:.4f}")
            print(f"  Recall: {results['recall']:.4f}")
            print(f"  F1-Score: {results['f1']:.4f}")
            print(f"  AUC: {results['auc']:.4f}")
            
            # Show top features
            if results['feature_importance'] is not None:
                print(f"\nTop 5 features for {model_type}:")
                print(results['feature_importance'].head())
            
        except Exception as e:
            print(f"Error with {model_type}: {e}")
            continue

def analyze_vote_data():
    """Analyze vote data distribution"""
    print("=== Vote Data Analysis ===")
    
    # Load votes data
    df_votes = parse_xml('data/Votes.xml')
    
    # Analyze vote types
    print("Vote type distribution:")
    vote_counts = df_votes['VoteTypeId'].value_counts().sort_index()
    for vote_type, count in vote_counts.items():
        print(f"  Type {vote_type}: {count:,} votes")
    
    # Analyze upvotes vs downvotes
    upvotes = len(df_votes[df_votes['VoteTypeId'] == '2'])
    downvotes = len(df_votes[df_votes['VoteTypeId'] == '3'])
    total_votes = upvotes + downvotes
    
    print(f"\nLike prediction dataset:")
    print(f"  Upvotes (2): {upvotes:,}")
    print(f"  Downvotes (3): {downvotes:,}")
    print(f"  Total: {total_votes:,}")
    print(f"  Like rate: {upvotes/total_votes:.3f}")

if __name__ == "__main__":
    # Analyze data first
    analyze_vote_data()
    
    # Test like prediction
    test_like_prediction() 