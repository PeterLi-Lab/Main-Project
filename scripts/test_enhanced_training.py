#!/usr/bin/env python3
"""
Test Enhanced Uplift Model Training Script
This script tests the enhanced uplift training with proper data leakage prevention.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhanced_uplift_training import EnhancedUpliftTraining

def create_clean_test_data():
    """Create clean test data without data leakage"""
    print("=== Creating Clean Test Data ===")
    
    np.random.seed(42)
    n_samples = 5000  # Smaller dataset for testing
    
    # Create features that don't leak information about the target
    df = pd.DataFrame({
        # User features
        'user_post_count': np.random.poisson(5, n_samples),
        'user_reputation': np.random.exponential(100, n_samples),
        'user_activity_level': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),
        
        # Post features
        'post_title_length': np.random.normal(50, 20, n_samples),
        'post_tag_count': np.random.poisson(3, n_samples),
        'post_length': np.random.exponential(200, n_samples),
        'post_score': np.random.poisson(5, n_samples),
        'post_view_count': np.random.exponential(500, n_samples),
        
        # Content features
        'content_quality_score': np.random.beta(3, 7, n_samples),
        'content_complexity': np.random.normal(5, 2, n_samples),
        'ai_keyword_count': np.random.poisson(2, n_samples),
        
        # Interaction features (before treatment)
        'user_ai_interest_score': np.random.beta(1, 3, n_samples),
        'user_previous_clicks': np.random.poisson(3, n_samples),
        'user_engagement_level': np.random.beta(2, 5, n_samples),
        
        # Treatment assignment (independent of features)
        'treatment_ai_content': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    # Create response variable with realistic treatment effect
    base_response_prob = 0.15
    
    # Add feature effects (but not direct leakage)
    response_prob = base_response_prob + \
                   df['user_ai_interest_score'] * 0.1 + \
                   df['content_quality_score'] * 0.05 + \
                   df['user_engagement_level'] * 0.08
    
    # Add treatment effect
    treatment_effect = 0.06  # Realistic treatment effect
    response_prob += df['treatment_ai_content'] * treatment_effect
    
    # Add some interaction effects
    response_prob += df['treatment_ai_content'] * df['user_ai_interest_score'] * 0.03
    
    # Ensure probabilities are between 0 and 1
    response_prob = np.clip(response_prob, 0, 1)
    
    # Generate response
    df['response'] = np.random.binomial(1, response_prob, n_samples)
    
    print(f"Created test data with shape: {df.shape}")
    print(f"Treatment distribution: {df['treatment_ai_content'].value_counts(normalize=True).to_dict()}")
    print(f"Response distribution: {df['response'].value_counts(normalize=True).to_dict()}")
    
    # Calculate actual uplift
    treatment_response_rate = df[df['treatment_ai_content'] == 1]['response'].mean()
    control_response_rate = df[df['treatment_ai_content'] == 0]['response'].mean()
    actual_uplift = treatment_response_rate - control_response_rate
    print(f"Actual uplift: {actual_uplift:.4f}")
    
    return df

def test_enhanced_training():
    """Test the enhanced training pipeline"""
    print("=== Testing Enhanced Uplift Training ===")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create clean test data
    df = create_clean_test_data()
    
    # Prepare features
    exclude_cols = ['treatment_ai_content', 'response']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].fillna(0)
    y = df['response']
    t = df['treatment_ai_content']
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nData split:")
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Initialize enhanced trainer
    trainer = EnhancedUpliftTraining()
    
    # Test with default models first (faster)
    print("\n" + "="*50)
    print("TESTING WITH DEFAULT MODELS")
    print("="*50)
    
    results = trainer.train_models_with_monitoring(
        X_train, y_train, t_train, X_test, y_test, t_test, 
        use_automated_tuning=False  # Use default models for testing
    )
    
    # Print results
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  AUC: {result['auc']:.4f}")
        print(f"  Uplift Score: {result['uplift_score']:.4f}")
        print(f"  Qini Score: {result['qini_score']:.4f}")
    
    # Check for reasonable performance
    best_model_name = max(results.keys(), key=lambda x: results[x]['uplift_score'])
    best_result = results[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best uplift score: {best_result['uplift_score']:.4f}")
    
    # Validate that performance is reasonable
    if best_result['auc'] > 0.95:
        print("⚠️  WARNING: AUC too high, possible data leakage!")
    elif best_result['auc'] < 0.5:
        print("⚠️  WARNING: AUC too low, model not learning!")
    else:
        print("✅ AUC in reasonable range")
    
    if abs(best_result['uplift_score']) > 0.3:
        print("⚠️  WARNING: Uplift score too high, possible data leakage!")
    else:
        print("✅ Uplift score in reasonable range")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

def main():
    """Main execution function"""
    try:
        results = test_enhanced_training()
        print("\n✅ Test completed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
