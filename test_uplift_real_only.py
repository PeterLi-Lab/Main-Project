#!/usr/bin/env python3
"""
Test Uplift Model with Real Data Only
Focuses only on the uplift modeling part using real Stack Overflow data
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import DataPreprocessor
from prediction_models import UpliftModeling

def test_uplift_real_data():
    """Test uplift model with real Stack Overflow data only"""
    print("=== Testing Uplift Model with Real Data ===")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(base_path='data')
    
    # Load data
    print("Loading data...")
    df_posts, df_users, df_tags, df_votes, df_badges = preprocessor.load_data()
    
    if df_posts is None:
        print("Error: Could not load data files. Please ensure XML files are in the 'data' directory.")
        return
    
    # Create uplift samples with interest-tag-based negative sampling
    print("\nCreating uplift samples with interest-tag-based negative sampling...")
    uplift_data = preprocessor.create_uplift_samples()
    
    if uplift_data is None or len(uplift_data) == 0:
        print("Error: No uplift data created.")
        return
    
    print(f"Uplift dataset shape: {uplift_data.shape}")
    print(f"Treatment distribution: {uplift_data['treatment'].value_counts().to_dict()}")
    print(f"Click rate: {uplift_data['is_click'].mean():.3f}")
    
    # Check for class imbalance
    click_rate = uplift_data['is_click'].mean()
    print(f"\nClass Balance Analysis:")
    print(f"Click rate: {click_rate:.3f}")
    print(f"Non-click rate: {1-click_rate:.3f}")
    
    if click_rate > 0.95:
        print("WARNING: Very high click rate - model may struggle to learn")
    elif click_rate < 0.05:
        print("WARNING: Very low click rate - model may struggle to learn")
    else:
        print("Click rate looks reasonable for modeling")
    
    # Initialize uplift model
    print("\nInitializing Uplift Model...")
    uplift_model = UpliftModeling()
    
    # Prepare features
    print("Preparing features...")
    feature_columns = [
        'Score', 'ViewCount', 'AnswerCount', 'CommentCount', 
        'Reputation', 'Views', 'UpVotes', 'DownVotes',
        'user_total_likes', 'post_total_likes',
        'time_diff_hours', 'is_early_vote', 'is_very_early_vote', 'is_late_vote'
    ]
    
    # Filter out rows with missing features
    available_features = [col for col in feature_columns if col in uplift_data.columns]
    print(f"Available features: {available_features}")
    
    # Remove rows with missing values in key features
    uplift_clean = uplift_data.dropna(subset=available_features + ['treatment', 'is_click'])
    print(f"Data after removing missing values: {uplift_clean.shape}")
    
    if len(uplift_clean) == 0:
        print("Error: No data left after removing missing values")
        return
    
    # Train uplift model
    print("\nTraining Uplift Model...")
    try:
        results = uplift_model.train_uplift_models(
            uplift_clean,
            model_type='xgboost'
        )
        
        if results:
            print("\n=== Uplift Model Results ===")
            print(f"Treatment group click rate: {results.get('treatment_click_rate', 'N/A')}")
            print(f"Control group click rate: {results.get('control_click_rate', 'N/A')}")
            print(f"Uplift (Treatment - Control): {results.get('uplift', 'N/A')}")
            
            # Show uplift distribution if available
            uplift_val = results.get('uplift', None)
            if uplift_val is not None:
                if uplift_val > 0:
                    print("Positive uplift: 用户推荐后点击率提升")
                elif uplift_val < 0:
                    print("Negative uplift: 推荐反而降低点击率")
                else:
                    print("No uplift: 推荐无明显效果")
            
        else:
            print("Error: Model training failed")
            
    except Exception as e:
        print(f"Error training uplift model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_uplift_real_data() 