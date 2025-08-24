#!/usr/bin/env python3
"""
Enhanced Uplift Model Training Script
This script runs the complete enhanced uplift training pipeline with automated tuning,
custom scorers, and comprehensive monitoring.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhanced_uplift_training import EnhancedUpliftTraining

def load_and_prepare_data():
    """Load and prepare data for training"""
    print("=== Data Loading and Preparation ===")
    
    # Try to load existing processed data
    data_files = [
        'uplift_model_data.csv',
        'uplift_dataset.csv',
        'user_post_click_samples.csv'
    ]
    
    df = None
    for file in data_files:
        try:
            print(f"Attempting to load {file}...")
            df = pd.read_csv(file)
            print(f"Successfully loaded {file} with shape: {df.shape}")
            break
        except FileNotFoundError:
            print(f"{file} not found, trying next file...")
            continue
    
    if df is None:
        print("No existing data files found. Creating sample data for demonstration...")
        # Create comprehensive sample data for demonstration
        np.random.seed(42)
        n_samples = 15000
        
        # Create realistic features
        df = pd.DataFrame({
            'user_post_count': np.random.poisson(5, n_samples),
            'post_title_length': np.random.normal(50, 20, n_samples),
            'post_tag_count': np.random.poisson(3, n_samples),
            'interest_score': np.random.beta(2, 5, n_samples),
            'user_post_interaction': np.random.poisson(10, n_samples),
            'user_ai_interest_score': np.random.beta(1, 3, n_samples),
            'user_ai_interest_weighted': np.random.beta(1, 4, n_samples),
            'user_ai_interactions': np.random.poisson(2, n_samples),
            'content_quality_score': np.random.beta(3, 7, n_samples),
            'content_complexity': np.random.normal(5, 2, n_samples),
            'user_activity_level': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.6, 0.3, 0.1]),
            'user_reputation_level': np.random.choice(['Beginner', 'Intermediate', 'Expert'], n_samples, p=[0.5, 0.3, 0.2]),
            'user_reputation': np.random.exponential(100, n_samples),
            'Score': np.random.poisson(5, n_samples),
            'ViewCount': np.random.exponential(500, n_samples),
            'AnswerCount': np.random.poisson(2, n_samples),
            'CommentCount': np.random.poisson(3, n_samples),
            'title_length': np.random.normal(60, 25, n_samples),
            'post_length': np.random.exponential(200, n_samples),
            'num_tags': np.random.poisson(3, n_samples),
            'total_votes': np.random.poisson(8, n_samples),
            'upvotes': np.random.poisson(6, n_samples),
            'user_post_tag_overlap': np.random.beta(2, 8, n_samples),
            'user_previous_ai_click_rate': np.random.beta(1, 4, n_samples),
            'ai_interest_x_treatment': np.random.beta(1, 3, n_samples),
            'treatment_ai_content': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'treatment_web_development': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'treatment_mobile_development': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'treatment_database': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        })
        
        # Create realistic response variable with treatment effect
        base_response_prob = 0.15
        treatment_effect = 0.08
        
        # Add treatment effect to response probability
        response_prob = base_response_prob + df['treatment_ai_content'] * treatment_effect
        response_prob += df['user_ai_interest_score'] * 0.05  # User interest effect
        response_prob += df['content_quality_score'] * 0.03   # Content quality effect
        
        # Ensure probabilities are between 0 and 1
        response_prob = np.clip(response_prob, 0, 1)
        
        df['response'] = np.random.binomial(1, response_prob, n_samples)
        
        print(f"Created sample data with shape: {df.shape}")
        print(f"Treatment distribution: {df['treatment_ai_content'].value_counts(normalize=True).to_dict()}")
        print(f"Response distribution: {df['response'].value_counts(normalize=True).to_dict()}")
    
    return df

def prepare_features(df):
    """Prepare features for training"""
    print("\n=== Feature Preparation ===")
    
    # Identify feature columns
    exclude_cols = ['treatment_ai_content', 'response', 'user_id', 'post_id']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Additional feature filtering to prevent data leakage
    leakage_suspicious_features = [
        'response', 'is_click', 'clicked', 'engagement', 'conversion',
        'treatment', 'treatment_effect', 'uplift', 'qini',
        'user_response', 'post_response', 'interaction_response'
    ]
    
    # Remove suspicious features
    feature_cols = [col for col in feature_cols if not any(leak in col.lower() for leak in leakage_suspicious_features)]
    
    print(f"Total features after filtering: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols[:10]}...")  # Show first 10 features
    
    # Handle categorical features
    categorical_features = df[feature_cols].select_dtypes(include=['object']).columns
    if len(categorical_features) > 0:
        print(f"Categorical features found: {list(categorical_features)}")
        # Encode categorical features
        for col in categorical_features:
            df[col] = pd.Categorical(df[col]).codes
    
    # Prepare feature matrix
    X = df[feature_cols].fillna(0)
    y = df['response']
    t = df['treatment_ai_content']
    
    # Check for potential data leakage
    print("\n=== Data Leakage Check ===")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts(normalize=True).to_dict()}")
    print(f"Treatment distribution: {t.value_counts(normalize=True).to_dict()}")
    
    # Check correlation between features and target
    correlations = []
    for col in X.columns:
        corr = abs(X[col].corr(y))
        if corr > 0.8:  # High correlation threshold
            correlations.append((col, corr))
    
    if correlations:
        print(f"WARNING: High correlation features found: {correlations[:5]}")
        # Remove highly correlated features
        high_corr_features = [col for col, corr in correlations if corr > 0.9]
        if high_corr_features:
            print(f"Removing {len(high_corr_features)} highly correlated features")
            X = X.drop(columns=high_corr_features)
            feature_cols = [col for col in feature_cols if col not in high_corr_features]
    
    print(f"Final feature matrix shape: {X.shape}")
    
    return X, y, t, feature_cols

def run_enhanced_training():
    """Run the complete enhanced training pipeline"""
    print("=== Enhanced Uplift Model Training Pipeline ===")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load and prepare data
    df = load_and_prepare_data()
    X, y, t, feature_cols = prepare_features(df)
    
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
    
    # Train models with enhanced monitoring
    print("\n" + "="*50)
    print("STARTING ENHANCED MODEL TRAINING")
    print("="*50)
    
    results = trainer.train_models_with_monitoring(
        X_train, y_train, t_train, X_test, y_test, t_test, 
        use_automated_tuning=True
    )
    
    # Analyze and visualize performance
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS AND VISUALIZATION")
    print("="*50)
    
    performance_df = trainer.performance_analysis_and_visualization()
    
    # Save results
    print("\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)
    
    trainer.save_enhanced_results()
    
    # Final summary
    print("\n" + "="*50)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*50)
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['uplift_score'])
    best_result = results[best_model_name]
    
    print(f"Best model: {best_model_name}")
    print(f"Best uplift score: {best_result['uplift_score']:.4f}")
    print(f"Best Qini score: {best_result['qini_score']:.4f}")
    print(f"Best AUC: {best_result['auc']:.4f}")
    print(f"Best accuracy: {best_result['accuracy']:.4f}")
    
    if trainer.tuning_results:
        print(f"\nHyperparameter tuning completed for {len(trainer.tuning_results)} models")
        for model_name, tuning_result in trainer.tuning_results.items():
            print(f"  {model_name}: {tuning_result['tuning_time']:.2f}s tuning time")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results, performance_df

def main():
    """Main execution function"""
    try:
        results, performance_df = run_enhanced_training()
        print("\n✅ Enhanced training completed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)



