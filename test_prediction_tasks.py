#!/usr/bin/env python3
"""
Test script for the four prediction tasks:
1. CTR Prediction (Classification)
2. Retention Prediction (Classification) 
3. Retention Duration Estimation (Regression)
4. Uplift Modeling (Treatment Effect Estimation)
"""

import os
import sys
import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor
from prediction_models import CTRPredictor, RetentionPredictor, RetentionDurationPredictor, UpliftModeling

def test_prediction_tasks():
    """Test all four prediction tasks"""
    print("=== Testing Four Prediction Tasks ===")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df_posts, df_users, df_tags, df_votes, df_badges = preprocessor.load_data()
    
    if df_posts is None:
        print("Failed to load data. Please check your data files.")
        return False
    
    # Run preprocessing
    df_combined = preprocessor.preprocess_all(include_normalization=True)
    
    if df_combined is None:
        print("Failed to preprocess data.")
        return False
    
    print(f"Data loaded successfully. Shape: {df_combined.shape}")
    
    # Test 1: CTR Prediction
    print("\n2. Testing CTR Prediction...")
    try:
        ctr_predictor = CTRPredictor()
        ctr_results = ctr_predictor.train_ctr_model(
            df_combined, 
            target_col='ctr_proxy_normalized', 
            model_type='xgboost'
        )
        if ctr_results:
            print(f"✓ CTR Prediction successful - Accuracy: {ctr_results.get('accuracy', 'N/A')}")
        else:
            print("✗ CTR Prediction failed")
    except Exception as e:
        print(f"✗ CTR Prediction error: {e}")
    
    # Test 2: Retention Prediction
    print("\n3. Testing Retention Prediction...")
    try:
        retention_predictor = RetentionPredictor()
        retention_results = retention_predictor.train_retention_model(
            df_combined, 
            target_col='is_retained', 
            model_type='xgboost'
        )
        if retention_results:
            print(f"✓ Retention Prediction successful - Accuracy: {retention_results.get('accuracy', 'N/A')}")
        else:
            print("✗ Retention Prediction failed")
    except Exception as e:
        print(f"✗ Retention Prediction error: {e}")
    
    # Test 3: Duration Prediction
    print("\n4. Testing Retention Duration Prediction...")
    try:
        duration_predictor = RetentionDurationPredictor()
        duration_results = duration_predictor.train_duration_model(
            df_combined, 
            target_col='days_to_next_action', 
            model_type='xgboost'
        )
        if duration_results:
            print(f"✓ Duration Prediction successful - R²: {duration_results.get('r2', 'N/A'):.3f}")
        else:
            print("✗ Duration Prediction failed")
    except Exception as e:
        print(f"✗ Duration Prediction error: {e}")
    
    # Test 4: Uplift Modeling
    print("\n5. Testing Uplift Modeling...")
    try:
        uplift_model = UpliftModeling()
        uplift_results = uplift_model.train_uplift_models(
            df_combined, 
            model_type='xgboost'
        )
        if uplift_results:
            print(f"✓ Uplift Modeling successful - Control MSE: {uplift_results.get('control_mse', 'N/A'):.4f}")
        else:
            print("✗ Uplift Modeling failed")
    except Exception as e:
        print(f"✗ Uplift Modeling error: {e}")
    
    print("\n=== Testing Completed ===")
    return True

def test_individual_tasks():
    """Test individual tasks with different model types"""
    print("\n=== Testing Individual Tasks with Different Models ===")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    df_posts, df_users, df_tags, df_votes, df_badges = preprocessor.load_data()
    df_combined = preprocessor.preprocess_all(include_normalization=True)
    
    if df_combined is None:
        print("Failed to preprocess data.")
        return False
    
    # Test different model types for each task
    model_types = ['xgboost', 'lightgbm', 'random_forest']
    
    for model_type in model_types:
        print(f"\n--- Testing with {model_type.upper()} ---")
        
        # CTR Prediction
        try:
            ctr_predictor = CTRPredictor()
            ctr_results = ctr_predictor.train_ctr_model(
                df_combined, target_col='ctr_proxy_normalized', model_type=model_type
            )
            if ctr_results:
                print(f"CTR ({model_type}): ✓")
        except Exception as e:
            print(f"CTR ({model_type}): ✗ - {e}")
        
        # Retention Prediction
        try:
            retention_predictor = RetentionPredictor()
            retention_results = retention_predictor.train_retention_model(
                df_combined, target_col='is_retained', model_type=model_type
            )
            if retention_results:
                print(f"Retention ({model_type}): ✓")
        except Exception as e:
            print(f"Retention ({model_type}): ✗ - {e}")
        
        # Duration Prediction
        try:
            duration_predictor = RetentionDurationPredictor()
            duration_results = duration_predictor.train_duration_model(
                df_combined, target_col='days_to_next_action', model_type=model_type
            )
            if duration_results:
                print(f"Duration ({model_type}): ✓")
        except Exception as e:
            print(f"Duration ({model_type}): ✗ - {e}")
        
        # Uplift Modeling
        try:
            uplift_model = UpliftModeling()
            uplift_results = uplift_model.train_uplift_models(
                df_combined, model_type=model_type
            )
            if uplift_results:
                print(f"Uplift ({model_type}): ✓")
        except Exception as e:
            print(f"Uplift ({model_type}): ✗ - {e}")
    
    return True

if __name__ == "__main__":
    print("Starting prediction tasks testing...")
    
    # Test basic functionality
    success = test_prediction_tasks()
    
    if success:
        # Test with different model types
        test_individual_tasks()
        
        print("\n=== All Tests Completed ===")
        print("You can now run individual tasks using:")
        print("  python main.py --mode ctr")
        print("  python main.py --mode retention")
        print("  python main.py --mode duration")
        print("  python main.py --mode uplift")
        print("  python main.py --mode all")
    else:
        print("Testing failed. Please check your data and dependencies.") 