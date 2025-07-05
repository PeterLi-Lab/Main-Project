#!/usr/bin/env python3
"""
Quick test for CTR model training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

def test_ctr_training():
    """Quick test of CTR model training"""
    print("=== Quick CTR Model Test ===")
    
    # Load CTR samples
    try:
        df_ctr = pd.read_csv('user_post_click_samples.csv')
        print(f"Loaded {len(df_ctr)} CTR samples")
        
        # Sample for quick test
        if len(df_ctr) > 10000:
            df_ctr = df_ctr.sample(n=10000, random_state=42)
            print(f"Sampled to {len(df_ctr)} samples for quick test")
        
        # Create simple features
        print("Creating simple features...")
        df_ctr['feature1'] = np.random.randn(len(df_ctr))  # Dummy feature
        df_ctr['feature2'] = np.random.randn(len(df_ctr))  # Dummy feature
        df_ctr['feature3'] = np.random.randn(len(df_ctr))  # Dummy feature
        
        # Prepare data
        X = df_ctr[['feature1', 'feature2', 'feature3']]
        y = df_ctr['is_click']
        
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
        
        print("=== Quick Test Complete ===")
        return True
        
    except Exception as e:
        print(f"Error in quick test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ctr_training() 