#!/usr/bin/env python3
"""
Simplified test script for 7-day retention prediction
"""

import pandas as pd
import numpy as np
from prediction_models import RetentionPredictor
from data_preprocessing import DataPreprocessor

def test_7day_retention_simple():
    """Test 7-day retention prediction with limited data"""
    print("=== Testing 7-Day Retention Prediction (Simplified) ===")
    
    # Load and preprocess data
    print("Loading data...")
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    
    if preprocessor.df_combined is None or len(preprocessor.df_combined) == 0:
        print("No data available. Please ensure data files are present.")
        return
    
    print(f"Data loaded: {len(preprocessor.df_combined)} records")
    
    # Use only a subset of data for faster testing
    sample_size = min(1000, len(preprocessor.df_combined))
    df_sample = preprocessor.df_combined.sample(n=sample_size, random_state=42).copy()
    print(f"Using sample of {len(df_sample)} records for testing")
    
    # Create basic features if not present
    if 'user_post_count' not in df_sample.columns:
        print("Creating basic features...")
        # Create minimal features for testing
        df_sample['user_post_count'] = np.random.randint(1, 10, len(df_sample))
        df_sample['post_age_days'] = np.random.randint(1, 365, len(df_sample))
        df_sample['total_badges'] = np.random.randint(0, 20, len(df_sample))
        df_sample['Score'] = np.random.randint(0, 100, len(df_sample))
        df_sample['ViewCount'] = np.random.randint(0, 1000, len(df_sample))
        df_sample['AnswerCount'] = np.random.randint(0, 10, len(df_sample))
        df_sample['CommentCount'] = np.random.randint(0, 20, len(df_sample))
        
        # Create mock influence features
        df_sample['total_influence_score'] = np.random.uniform(0, 100, len(df_sample))
        df_sample['high_quality_influence'] = np.random.uniform(0, 50, len(df_sample))
        df_sample['influence_domains_count'] = np.random.randint(1, 5, len(df_sample))
        df_sample['gold_badges'] = np.random.randint(0, 5, len(df_sample))
        df_sample['silver_badges'] = np.random.randint(0, 10, len(df_sample))
        df_sample['bronze_badges'] = np.random.randint(0, 20, len(df_sample))
        df_sample['vote_ratio'] = np.random.uniform(0, 1, len(df_sample))
        
        # Create mock categorical features
        df_sample['influence_level'] = np.random.choice(['Low', 'Medium', 'High'], len(df_sample))
        df_sample['multi_domain_influence'] = np.random.choice(['Single', 'Multi'], len(df_sample))
        df_sample['badge_level'] = np.random.choice(['Bronze', 'Silver', 'Gold'], len(df_sample))
        df_sample['badge_quality_level'] = np.random.choice(['Basic', 'Advanced', 'Expert'], len(df_sample))
        
        # Create mock retention target (7-day)
        df_sample['recent_badges_7d'] = np.random.randint(0, 3, len(df_sample))
        df_sample['is_retained_7d'] = (df_sample['recent_badges_7d'] > 0).astype(int)
        
        print("Mock features created successfully")
    
    # Test 7-day retention prediction with Random Forest only
    print("\n=== Testing 7-Day Retention Prediction ===")
    retention_predictor = RetentionPredictor()
    
    try:
        print("Training Random Forest model for 7-day retention...")
        
        # Train 7-day retention model
        result = retention_predictor.train_7day_retention_model(
            df_sample, 
            model_type='random_forest'
        )
        
        if result:
            print(f"✓ Random Forest 7-day retention model trained successfully")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Model type: {type(result['model']).__name__}")
            
            # Test prediction on a small sample
            test_sample = df_sample.head(5)
            predictions, probabilities = retention_predictor.predict_7day_retention(test_sample)
            
            if predictions is not None:
                print(f"  Sample predictions: {predictions}")
                print(f"  Retention rate in sample: {predictions.mean():.3f}")
                print("✓ Prediction test successful")
            else:
                print("  ✗ Prediction failed")
        else:
            print("✗ Random Forest training failed")
            
    except Exception as e:
        print(f"✗ Error with Random Forest: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 7-Day Retention Prediction Test Complete ===")

if __name__ == "__main__":
    print("Simplified 7-Day Retention Prediction Test")
    print("=" * 50)
    
    # Test basic functionality
    test_7day_retention_simple()
    
    print("\nTest completed!") 