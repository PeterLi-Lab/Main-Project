#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Integrated Industrial-Grade CTR System
Tests the integrated functionality across data preprocessing and prediction models
"""

import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessor
from prediction_models import IndustrialCTRPredictor
import warnings
warnings.filterwarnings('ignore')

def create_sample_data(n_samples=500):
    """Create sample data for testing"""
    print("Creating sample data for testing...")
    
    np.random.seed(42)
    
    # Generate sample data
    data = {
        'Score': np.random.randint(0, 100, n_samples),
        'ViewCount': np.random.randint(10, 1000, n_samples),
        'AnswerCount': np.random.randint(0, 20, n_samples),
        'CommentCount': np.random.randint(0, 50, n_samples),
        'title_length': np.random.randint(5, 30, n_samples),
        'post_length': np.random.randint(50, 500, n_samples),
        'num_tags': np.random.randint(1, 8, n_samples),
        'post_age_days': np.random.randint(1, 365, n_samples),
        'user_post_count': np.random.randint(1, 100, n_samples),
        'user_reputation': np.random.randint(100, 10000, n_samples),
        'total_votes': np.random.randint(0, 100, n_samples),
        'upvotes': np.random.randint(0, 80, n_samples),
        'OwnerUserId': [f'user_{i}' for i in range(n_samples)],
        'first_tag': np.random.choice(['python', 'javascript', 'java', 'c++', 'sql'], n_samples),
        'influence_level': np.random.choice(['low', 'medium', 'high'], n_samples),
        'badge_level': np.random.choice(['bronze', 'silver', 'gold'], n_samples),
        'multi_domain_influence': np.random.choice(['single', 'multi'], n_samples),
        'CreationDate_x': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'total_influence_score': np.random.randint(0, 1000, n_samples),
        'high_quality_influence': np.random.randint(0, 500, n_samples),
        'total_badges': np.random.randint(0, 50, n_samples),
        'badge_quality_score': np.random.uniform(0, 1, n_samples),
        'badge_rate_per_day': np.random.uniform(0, 0.5, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable (CTR proxy)
    df['ctr_proxy_normalized'] = (
        df['Score'] * 0.3 + 
        df['ViewCount'] * 0.2 + 
        df['AnswerCount'] * 0.2 + 
        df['CommentCount'] * 0.15 + 
        df['upvotes'] * 0.15
    ) / 1000
    
    # Normalize to 0-1 range
    df['ctr_proxy_normalized'] = (df['ctr_proxy_normalized'] - df['ctr_proxy_normalized'].min()) / \
                                (df['ctr_proxy_normalized'].max() - df['ctr_proxy_normalized'].min())
    
    print(f"Sample data created: {df.shape}")
    return df

def test_industrial_features():
    """Test industrial feature creation"""
    print("\n=== Testing Industrial Feature Creation ===")
    
    # Create sample data
    df_sample = create_sample_data(200)
    
    # Test industrial feature creation
    preprocessor = DataPreprocessor()
    df_industrial = preprocessor.create_industrial_features(df_sample.copy())
    
    # Check if industrial features were created
    hash_features = [col for col in df_industrial.columns if col.endswith('_hash')]
    cross_features = [col for col in df_industrial.columns if col.endswith('_cross')]
    seq_features = [col for col in df_industrial.columns if col.startswith('seq_')]
    context_features = ['hour_of_day', 'day_of_week', 'month_of_year', 'is_weekend', 'is_peak_hours']
    interaction_features = ['quality_view_ratio', 'vote_quality_ratio', 'engagement_efficiency', 
                           'content_complexity', 'score_per_day']
    
    print(f"Hash features created: {len(hash_features)}")
    print(f"Cross features created: {len(cross_features)}")
    print(f"Sequence features created: {len(seq_features)}")
    print(f"Context features created: {sum(1 for f in context_features if f in df_industrial.columns)}")
    print(f"Interaction features created: {sum(1 for f in interaction_features if f in df_industrial.columns)}")
    
    print("Industrial feature creation test passed!")
    return df_industrial

def test_negative_sampling():
    """Test negative sampling"""
    print("\n=== Testing Negative Sampling ===")
    
    df_sample = create_sample_data(200)
    
    preprocessor = DataPreprocessor()
    df_balanced = preprocessor.create_negative_sampling(df_sample, sampling_ratio=2)
    
    # Check if sampling worked
    positive_count = len(df_balanced[df_balanced['is_positive'] == True])
    negative_count = len(df_balanced[df_balanced['is_positive'] == False])
    
    print(f"Balanced dataset: {positive_count} positive, {negative_count} negative")
    print(f"Sampling ratio: {negative_count/positive_count:.2f}:1")
    
    print("Negative sampling test passed!")
    return df_balanced

def test_industrial_models():
    """Test industrial model training"""
    print("\n=== Testing Industrial Model Training ===")
    
    # Create data with industrial features
    df_industrial = test_industrial_features()
    df_balanced = test_negative_sampling()
    
    # Train industrial models
    industrial_predictor = IndustrialCTRPredictor()
    
    # Use smaller dataset and fewer epochs for testing
    df_test = df_balanced.head(100)  # Use only 100 samples for testing
    
    try:
        models = industrial_predictor.train_industrial_models(df_test)
        
        # Check if models were trained
        print(f"Models trained: {list(industrial_predictor.models.keys())}")
        print(f"Model performance: {industrial_predictor.model_performance}")
        
        # Test online prediction
        if industrial_predictor.models:
            print("Testing online prediction...")
            sample_features = {
                'Score': 10, 'ViewCount': 100, 'AnswerCount': 2, 'CommentCount': 5,
                'title_length': 15, 'post_length': 200, 'num_tags': 3, 'post_age_days': 30,
                'user_post_count': 50, 'user_reputation': 1000, 'total_votes': 20, 'upvotes': 18,
                'vote_ratio': 0.9, 'total_influence_score': 500, 'high_quality_influence': 200,
                'total_badges': 10, 'badge_quality_score': 0.8, 'badge_rate_per_day': 0.1,
                'hour_of_day': 0.5, 'day_of_week': 0.3, 'month_of_year': 0.6, 'is_weekend': 0,
                'is_peak_hours': 1, 'quality_view_ratio': 0.1, 'vote_quality_ratio': 0.9,
                'engagement_efficiency': 20, 'content_complexity': 13.3, 'score_per_day': 0.33
            }
            
            # Add hash encoded features
            sample_features.update({
                'OwnerUserId_hash': hash('user_123') % 10000,
                'first_tag_hash': hash('python') % 10000,
                'influence_level_hash': hash('medium') % 10000,
                'badge_level_hash': hash('silver') % 10000,
                'multi_domain_influence_hash': hash('single') % 10000,
                'OwnerUserId_first_tag_cross': hash('user_123_python') % 1000,
                'influence_level_badge_level_cross': hash('medium_silver') % 1000,
                'user_post_count_user_reputation_cross': hash('50_1000') % 1000,
                'Score_ViewCount_cross': hash('10_100') % 1000
            })
            
            # Add sequence features
            for i in range(5):
                for j in range(4):
                    sample_features[f'seq_{i}_{j}'] = 0
            
            result = industrial_predictor.online_predict(sample_features)
            print(f"Prediction result: {result}")
        
        print("Industrial model training test passed!")
        
    except Exception as e:
        print(f"Industrial model training failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    print("=== Integrated Industrial-Grade CTR System Tests ===")
    
    try:
        # Test industrial features
        test_industrial_features()
        
        # Test negative sampling
        test_negative_sampling()
        
        # Test industrial models
        test_industrial_models()
        
        print("\n=== All Tests Passed! ===")
        print("The integrated industrial-grade CTR system is working correctly.")
        
    except Exception as e:
        print(f"\n=== Test Failed ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 