#!/usr/bin/env python3
"""
Test script for 7-day retention prediction with progress bars
"""

import pandas as pd
import numpy as np
from prediction_models import RetentionPredictor
from data_preprocessing import DataPreprocessor
from tqdm import tqdm
import time

def test_7day_retention_prediction():
    """Test 7-day retention prediction functionality with progress tracking"""
    print("=== Testing 7-Day Retention Prediction ===")
    
    # Load and preprocess data
    print("Loading data...")
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    
    if preprocessor.df_combined is None or len(preprocessor.df_combined) == 0:
        print("No data available. Please ensure data files are present.")
        return
    
    print(f"Data loaded: {len(preprocessor.df_combined)} records")
    
    # Create basic features if not present
    if 'user_post_count' not in preprocessor.df_combined.columns:
        print("Creating basic features...")
        with tqdm(total=4, desc="Feature Engineering") as pbar:
            preprocessor.create_derived_variables()
            pbar.update(1)
            
            preprocessor.create_badge_features()
            pbar.update(1)
            
            preprocessor.create_user_influence_features()
            pbar.update(1)
            
            preprocessor.create_categorical_variables()
            pbar.update(1)
    
    # Test 7-day retention prediction
    print("\n=== Testing 7-Day Retention Prediction ===")
    
    # Generate retention samples
    print("Generating 7-day retention samples...")
    retention_samples = preprocessor.create_7day_retention_samples()
    
    # Test with different models
    models_to_test = ['xgboost', 'random_forest', 'logistic_regression']
    
    for model_type in tqdm(models_to_test, desc="Training Models"):
        try:
            print(f"\n--- Testing {model_type.upper()} for 7-day retention ---")
            
            # Create retention predictor with preprocessor
            retention_predictor = RetentionPredictor(preprocessor=preprocessor)
            
            # Train 7-day retention model
            result = retention_predictor.train_retention_model(
                preprocessor.df_combined, 
                model_type=model_type,
                retention_window_days=7
            )
            
            if result:
                print(f"✓ {model_type.upper()} 7-day retention model trained successfully")
                print(f"  Accuracy: {result['accuracy']:.4f}")
                print(f"  Model type: {type(result['model']).__name__}")
                
                # Test prediction on a sample
                sample_data = preprocessor.df_combined.head(10)
                predictions = retention_predictor.predict_retention(sample_data, retention_window_days=7)
                
                if predictions is not None:
                    print(f"  Sample predictions: {predictions[:5]}")  # Show first 5 predictions
                    print(f"  Retention rate in sample: {predictions.mean():.3f}")
                else:
                    print("  ✗ Prediction failed")
            else:
                print(f"✗ {model_type.upper()} training failed")
                
        except Exception as e:
            print(f"✗ Error with {model_type}: {e}")
    
    print("\n=== 7-Day Retention Prediction Test Complete ===")

def test_retention_comparison():
    """Compare 7-day vs 30-day retention prediction with progress tracking"""
    print("\n=== Comparing 7-Day vs 30-Day Retention ===")
    
    # Load data
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    
    if preprocessor.df_combined is None:
        print("No data available for comparison")
        return
    
    # Create features with progress tracking
    print("Creating features for comparison...")
    with tqdm(total=4, desc="Feature Creation") as pbar:
        if 'user_post_count' not in preprocessor.df_combined.columns:
            preprocessor.create_derived_variables()
        pbar.update(1)
        
        if 'total_badges' not in preprocessor.df_combined.columns:
            preprocessor.create_badge_features()
        pbar.update(1)
        
        if 'total_influence_score' not in preprocessor.df_combined.columns:
            preprocessor.create_user_influence_features()
        pbar.update(1)
        
        if 'influence_level' not in preprocessor.df_combined.columns:
            preprocessor.create_categorical_variables()
        pbar.update(1)
    
    # Generate retention samples for 7-day prediction
    print("Generating 7-day retention samples...")
    retention_samples = preprocessor.create_7day_retention_samples()
    
    # Test both retention windows with XGBoost
    try:
        print("Training 30-day retention model...")
        with tqdm(total=1, desc="30-Day Model") as pbar:
            retention_predictor_30d = RetentionPredictor()
            result_30d = retention_predictor_30d.train_retention_model(
                preprocessor.df_combined, 
                model_type='xgboost',
                retention_window_days=30
            )
            pbar.update(1)
        
        print("Training 7-day retention model...")
        with tqdm(total=1, desc="7-Day Model") as pbar:
            retention_predictor_7d = RetentionPredictor(preprocessor=preprocessor)
            result_7d = retention_predictor_7d.train_retention_model(
                preprocessor.df_combined, 
                model_type='xgboost',
                retention_window_days=7
            )
            pbar.update(1)
        
        if result_30d and result_7d:
            print("\n=== Retention Model Comparison ===")
            print(f"30-Day Retention - Accuracy: {result_30d['accuracy']:.4f}")
            print(f"7-Day Retention  - Accuracy: {result_7d['accuracy']:.4f}")
            
            # Compare retention rates
            if 'is_retained' in preprocessor.df_combined.columns:
                retention_30d_rate = preprocessor.df_combined['is_retained'].mean()
                print(f"30-Day Retention Rate in Data: {retention_30d_rate:.3f}")
            
            if 'is_retained_7d' in preprocessor.df_combined.columns:
                retention_7d_rate = preprocessor.df_combined['is_retained_7d'].mean()
                print(f"7-Day Retention Rate in Data: {retention_7d_rate:.3f}")
            
            print("\n✓ Retention comparison completed successfully")
        else:
            print("✗ One or both models failed to train")
            
    except Exception as e:
        print(f"✗ Error in retention comparison: {e}")

if __name__ == "__main__":
    print("7-Day Retention Prediction Test (Full Version)")
    print("=" * 50)
    
    # Test basic functionality
    test_7day_retention_prediction()
    
    # Test comparison
    test_retention_comparison()
    
    print("\nTest completed!") 