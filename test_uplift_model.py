#!/usr/bin/env python3
"""
Test UpliftModeling Class
"""

import numpy as np
import pandas as pd
from prediction_models import UpliftModeling

def test_uplift_model():
    """Test the UpliftModeling class"""
    print("=== Testing UpliftModeling Class ===")
    
    # Create test data
    np.random.seed(42)
    n_samples = 500
    
    # Generate features
    user_reputation = np.random.exponential(1000, n_samples)
    post_score = np.random.normal(5, 3, n_samples)
    time_diff_hours = np.random.exponential(24, n_samples)
    
    # Create treatment
    treatment = (time_diff_hours <= 12).astype(int)
    
    # Generate click behavior
    base_prob = 0.3
    feature_effect = (post_score / 10) * 0.1 + (user_reputation / 10000) * 0.05
    treatment_effect = treatment * 0.15
    click_prob = np.clip(base_prob + feature_effect + treatment_effect, 0, 1)
    is_click = np.random.binomial(1, click_prob)
    
    # Create DataFrame
    df = pd.DataFrame({
        'treatment': treatment,
        'is_click': is_click,
        'Score': post_score,
        'Reputation': user_reputation,
        'time_diff_hours': time_diff_hours,
        'ViewCount': np.random.exponential(100, n_samples),
        'AnswerCount': np.random.poisson(2, n_samples),
        'CommentCount': np.random.poisson(3, n_samples),
        'Views': np.random.exponential(5000, n_samples),
        'UpVotes': np.random.poisson(user_reputation / 100),
        'DownVotes': np.random.poisson(5),
        'user_total_likes': np.random.poisson(user_reputation / 100),
        'post_total_likes': np.random.poisson(np.abs(post_score) * 2),
        'is_early_vote': (time_diff_hours <= 24).astype(int),
        'is_very_early_vote': (time_diff_hours <= 1).astype(int),
        'is_late_vote': (time_diff_hours > 168).astype(int)
    })
    
    print(f"Created {len(df)} samples")
    print(f"Treatment group: {df['treatment'].sum()}")
    print(f"Control group: {(df['treatment'] == 0).sum()}")
    print(f"Overall click rate: {df['is_click'].mean():.3f}")
    
    # Test UpliftModeling class
    print(f"\nTesting UpliftModeling class...")
    uplift_model = UpliftModeling()
    
    try:
        # Test feature preparation
        print("1. Testing feature preparation...")
        df_prepared, features = uplift_model.prepare_uplift_features(df)
        print(f"   ✓ Prepared {len(features)} features")
        print(f"   Features: {features}")
        
        # Test model training
        print("2. Testing model training...")
        result = uplift_model.train_uplift_models(df, model_type='xgboost')
        
        if result:
            print(f"   ✓ Model training successful")
            print(f"   Treatment click rate: {result['treatment_click_rate']:.3f}")
            print(f"   Control click rate: {result['control_click_rate']:.3f}")
            print(f"   Uplift: {result['uplift']:.3f}")
            print(f"   Uplift percentage: {result['uplift']/result['control_click_rate']*100:.1f}%")
            
            # Test prediction with properly prepared data
            print("3. Testing prediction...")
            new_data = df.head(10).copy()
            # Prepare new data with same features
            new_data_prepared, _ = uplift_model.prepare_uplift_features(new_data)
            uplift_scores, treatment_probs, control_probs = uplift_model.predict_uplift(new_data_prepared)
            
            print(f"   ✓ Prediction successful")
            print(f"   Uplift scores - Mean: {np.mean(uplift_scores):.3f}, Std: {np.std(uplift_scores):.3f}")
            print(f"   Positive uplift: {(uplift_scores > 0).mean()*100:.1f}%")
            
            # Test feature importance
            if hasattr(uplift_model.treatment_model, 'feature_importances_'):
                print("4. Testing feature importance...")
                feature_importance = pd.DataFrame({
                    'feature': uplift_model.feature_names,
                    'importance': uplift_model.treatment_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print("   Top 5 most important features:")
                for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
                    print(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")
            
            return result
        else:
            print("   ✗ Model training failed")
            return None
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_uplift_model()
    if result:
        print(f"\n=== Test Successful ===")
        print(f"Uplift achieved: {result['uplift']:.3f}")
        print(f"Uplift percentage: {result['uplift']/result['control_click_rate']*100:.1f}%")
    else:
        print(f"\n=== Test Failed ===") 