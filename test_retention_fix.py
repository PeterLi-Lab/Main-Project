#!/usr/bin/env python3
"""
Quick test to verify retention label fix
"""

import pandas as pd
import numpy as np
from prediction_models import RetentionPredictor
from data_preprocessing import DataPreprocessor

def test_retention_labels():
    """Test if retention labels are created correctly"""
    print("=== Testing Retention Label Creation ===")
    
    # Load data
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    
    if preprocessor.df_combined is None:
        print("No data available")
        return
    
    print(f"Data loaded: {len(preprocessor.df_combined)} records")
    
    # Create basic features
    if 'user_post_count' not in preprocessor.df_combined.columns:
        preprocessor.create_derived_variables()
    if 'total_badges' not in preprocessor.df_combined.columns:
        preprocessor.create_badge_features()
    
    # Test retention predictor
    retention_predictor = RetentionPredictor()
    
    # Test 7-day retention
    print("\n--- Testing 7-Day Retention Labels ---")
    df_7d, features_7d, encoders_7d, target_7d = retention_predictor.prepare_retention_features(
        preprocessor.df_combined, retention_window_days=7
    )
    
    retention_rate_7d = df_7d[target_7d].mean()
    print(f"7-Day retention rate: {retention_rate_7d:.3f}")
    print(f"7-Day retention distribution: {df_7d[target_7d].value_counts().to_dict()}")
    
    # Test 30-day retention
    print("\n--- Testing 30-Day Retention Labels ---")
    df_30d, features_30d, encoders_30d, target_30d = retention_predictor.prepare_retention_features(
        preprocessor.df_combined, retention_window_days=30
    )
    
    retention_rate_30d = df_30d[target_30d].mean()
    print(f"30-Day retention rate: {retention_rate_30d:.3f}")
    print(f"30-Day retention distribution: {df_30d[target_30d].value_counts().to_dict()}")
    
    # Test model training if retention rates are reasonable
    if 0.1 < retention_rate_7d < 0.9:
        print("\n--- Testing 7-Day Model Training ---")
        try:
            result = retention_predictor.train_7day_retention_model(
                preprocessor.df_combined, model_type='random_forest'
            )
            if result:
                print(f"✓ 7-Day model trained successfully")
                print(f"  Accuracy: {result['accuracy']:.4f}")
            else:
                print("✗ 7-Day model training failed")
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print(f"✗ 7-Day retention rate {retention_rate_7d:.3f} is not reasonable (should be between 0.1 and 0.9)")
    
    if 0.1 < retention_rate_30d < 0.9:
        print("\n--- Testing 30-Day Model Training ---")
        try:
            result = retention_predictor.train_retention_model(
                preprocessor.df_combined, model_type='random_forest'
            )
            if result:
                print(f"✓ 30-Day model trained successfully")
                print(f"  Accuracy: {result['accuracy']:.4f}")
            else:
                print("✗ 30-Day model training failed")
        except Exception as e:
            print(f"✗ Error: {e}")
    else:
        print(f"✗ 30-Day retention rate {retention_rate_30d:.3f} is not reasonable (should be between 0.1 and 0.9)")

if __name__ == "__main__":
    test_retention_labels()
    print("\nTest completed!") 