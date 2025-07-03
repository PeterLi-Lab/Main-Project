#!/usr/bin/env python3
"""
Debug script for retention label creation
"""

import pandas as pd
import numpy as np
from prediction_models import RetentionPredictor

def debug_retention_labels():
    """Debug retention label creation"""
    print("=== Debugging Retention Label Creation ===")
    
    # Create a simple test dataset
    test_data = pd.DataFrame({
        'user_post_count': [1, 5, 10, 15, 20],
        'total_badges': [0, 2, 5, 8, 12],
        'Score': [10, 50, 100, 150, 200],
        'ViewCount': [100, 500, 1000, 1500, 2000]
    })
    
    print("Test data:")
    print(test_data)
    
    # Test retention predictor
    retention_predictor = RetentionPredictor()
    
    # Test 7-day retention
    print("\n--- Testing 7-Day Retention ---")
    try:
        df_7d, features_7d, encoders_7d, target_7d = retention_predictor.prepare_retention_features(
            test_data, retention_window_days=7
        )
        
        print(f"Target column: {target_7d}")
        print(f"Retention labels: {df_7d[target_7d].tolist()}")
        print(f"Retention rate: {df_7d[target_7d].mean():.3f}")
        print(f"Features created: {len(features_7d)}")
        
    except Exception as e:
        print(f"Error in 7-day retention: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 30-day retention
    print("\n--- Testing 30-Day Retention ---")
    try:
        df_30d, features_30d, encoders_30d, target_30d = retention_predictor.prepare_retention_features(
            test_data, retention_window_days=30
        )
        
        print(f"Target column: {target_30d}")
        print(f"Retention labels: {df_30d[target_30d].tolist()}")
        print(f"Retention rate: {df_30d[target_30d].mean():.3f}")
        print(f"Features created: {len(features_30d)}")
        
    except Exception as e:
        print(f"Error in 30-day retention: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_retention_labels()
    print("\nDebug completed!") 