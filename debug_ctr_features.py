#!/usr/bin/env python3
"""
Debug CTR Model Features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def debug_ctr_features():
    """Debug CTR model features and data"""
    print("=== Debugging CTR Model Features ===")
    
    # Load CTR samples
    df_ctr = pd.read_csv('user_post_click_samples.csv')
    print(f"Loaded {len(df_ctr)} CTR samples")
    
    # Sample for debugging
    if len(df_ctr) > 10000:
        df_ctr = df_ctr.sample(n=10000, random_state=42)
        print(f"Sampled to {len(df_ctr)} samples")
    
    # Check target distribution
    print(f"\n=== Target Distribution ===")
    print(f"is_click distribution: {df_ctr['is_click'].value_counts().to_dict()}")
    print(f"Positive ratio: {df_ctr['is_click'].mean():.3f}")
    
    # Check feature distributions
    print(f"\n=== Feature Distributions ===")
    feature_cols = ['interest_score', 'user_post_count', 'user_account_age_days', 
                   'post_age_days', 'post_title_length', 'post_tag_count']
    
    for col in feature_cols:
        if col in df_ctr.columns:
            print(f"{col}:")
            print(f"  Mean: {df_ctr[col].mean():.3f}")
            print(f"  Std: {df_ctr[col].std():.3f}")
            print(f"  Min: {df_ctr[col].min():.3f}")
            print(f"  Max: {df_ctr[col].max():.3f}")
            print(f"  Non-zero: {(df_ctr[col] != 0).sum()}/{len(df_ctr)}")
        else:
            print(f"{col}: NOT FOUND")
    
    # Check for data leakage
    print(f"\n=== Data Leakage Check ===")
    # Check if user_id and post_id combinations are unique
    unique_combinations = df_ctr[['user_id', 'post_id']].drop_duplicates().shape[0]
    total_combinations = len(df_ctr)
    print(f"Unique user-post combinations: {unique_combinations}/{total_combinations}")
    print(f"Duplication ratio: {1 - unique_combinations/total_combinations:.3f}")
    
    # Check correlation between features and target
    print(f"\n=== Feature-Target Correlations ===")
    for col in feature_cols:
        if col in df_ctr.columns:
            corr = df_ctr[col].corr(df_ctr['is_click'])
            print(f"{col} correlation with is_click: {corr:.3f}")
    
    # Check if all features are zero
    print(f"\n=== Zero Feature Check ===")
    for col in feature_cols:
        if col in df_ctr.columns:
            zero_ratio = (df_ctr[col] == 0).mean()
            print(f"{col} zero ratio: {zero_ratio:.3f}")
    
    # Check train/test split
    print(f"\n=== Train/Test Split Check ===")
    X = df_ctr[feature_cols].fillna(0)
    y = df_ctr['is_click']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set positive ratio: {y_train.mean():.3f}")
    print(f"Test set positive ratio: {y_test.mean():.3f}")
    
    # Check if features are all zero in train/test
    for col in feature_cols:
        if col in df_ctr.columns:
            train_zero = (X_train[col] == 0).mean()
            test_zero = (X_test[col] == 0).mean()
            print(f"{col} - Train zero ratio: {train_zero:.3f}, Test zero ratio: {test_zero:.3f}")
    
    return df_ctr

if __name__ == "__main__":
    debug_ctr_features() 