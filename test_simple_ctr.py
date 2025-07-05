#!/usr/bin/env python3
"""
Simple CTR Model Test
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def test_simple_ctr():
    """Simple CTR model test"""
    print("=== Simple CTR Model Test ===")
    
    # Load CTR samples
    df_ctr = pd.read_csv('user_post_click_samples.csv')
    print(f"Loaded {len(df_ctr)} CTR samples")
    print(f"Columns: {df_ctr.columns.tolist()}")
    
    # Sample for quick test
    if len(df_ctr) > 10000:
        df_ctr = df_ctr.sample(n=10000, random_state=42)
        print(f"Sampled to {len(df_ctr)} samples")
    
    # Create simple features from existing data
    print("Creating simple features...")
    
    # Use existing features if available, otherwise create dummy features
    feature_cols = []
    
    if 'interest_score' in df_ctr.columns:
        feature_cols.append('interest_score')
        print("Using interest_score feature")
    else:
        df_ctr['interest_score'] = np.random.randn(len(df_ctr))
        feature_cols.append('interest_score')
        print("Created dummy interest_score feature")
    
    # Add more dummy features for testing
    df_ctr['feature1'] = np.random.randn(len(df_ctr))
    df_ctr['feature2'] = np.random.randn(len(df_ctr))
    df_ctr['feature3'] = np.random.randn(len(df_ctr))
    feature_cols.extend(['feature1', 'feature2', 'feature3'])
    
    # Prepare data
    X = df_ctr[feature_cols]
    y = df_ctr['is_click']
    
    print(f"Feature columns: {feature_cols}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Positive class ratio: {y.mean():.3f}")
    
    # Train simple model
    print("Training logistic regression...")
    model = LogisticRegression(random_state=42, max_iter=100)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Save model
    import pickle
    import os
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/ctr_simple_test.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    print("=== Simple Test Complete ===")
    return True

if __name__ == "__main__":
    test_simple_ctr() 