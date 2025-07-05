#!/usr/bin/env python3
"""
Quick Uplift Model Test
"""

import numpy as np
import pandas as pd
from prediction_models import UpliftModeling

def quick_uplift_test():
    """Quick test of uplift model"""
    print("=== Quick Uplift Model Test ===")
    
    # Create simple test data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate simple features
    user_reputation = np.random.exponential(1000, n_samples)
    post_score = np.random.normal(5, 3, n_samples)
    time_diff_hours = np.random.exponential(24, n_samples)
    
    # Create treatment (recommendation vs search)
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
    
    # Calculate baseline
    treatment_rate = df[df['treatment'] == 1]['is_click'].mean()
    control_rate = df[df['treatment'] == 0]['is_click'].mean()
    baseline_uplift = treatment_rate - control_rate
    
    print(f"\nBaseline Analysis:")
    print(f"Treatment click rate: {treatment_rate:.3f}")
    print(f"Control click rate: {control_rate:.3f}")
    print(f"Baseline uplift: {baseline_uplift:.3f}")
    print(f"Uplift percentage: {baseline_uplift/control_rate*100:.1f}%")
    
    # Train uplift model
    print(f"\nTraining XGBoost uplift model...")
    uplift_model = UpliftModeling()
    
    try:
        result = uplift_model.train_uplift_models(df, model_type='xgboost')
        if result:
            print(f"✓ Model trained successfully")
            print(f"  Treatment click rate: {result['treatment_click_rate']:.3f}")
            print(f"  Control click rate: {result['control_click_rate']:.3f}")
            print(f"  Model uplift: {result['uplift']:.3f}")
            print(f"  Uplift percentage: {result['uplift']/result['control_click_rate']*100:.1f}%")
            
            # Test prediction
            print(f"\nTesting prediction on new data...")
            new_data = df.head(50).copy()
            uplift_scores, treatment_probs, control_probs = uplift_model.predict_uplift(new_data)
            
            print(f"Predicted uplift scores:")
            print(f"  Mean: {np.mean(uplift_scores):.3f}")
            print(f"  Std: {np.std(uplift_scores):.3f}")
            print(f"  Min: {np.min(uplift_scores):.3f}")
            print(f"  Max: {np.max(uplift_scores):.3f}")
            print(f"  Positive uplift: {(uplift_scores > 0).mean()*100:.1f}%")
            
            return result
        else:
            print("✗ Model training failed")
            return None
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = quick_uplift_test()
    print("\n=== Test Complete ===") 