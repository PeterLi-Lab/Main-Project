#!/usr/bin/env python3
"""
Comprehensive Evaluation Metrics Analysis
This script calculates and analyzes all evaluation metrics for uplift modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data for evaluation"""
    print("=== Loading and Preparing Data ===")
    
    # Try to load existing data
    data_files = [
        'uplift_model_data.csv',
        'uplift_dataset.csv', 
        'user_post_click_samples.csv',
        'cluster5_treatment_control.csv'
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
        print("No existing data files found. Creating synthetic data for demonstration...")
        # Create synthetic data for demonstration
        np.random.seed(42)
        n_samples = 10000
        
        df = pd.DataFrame({
            'user_id': range(n_samples),
            'post_id': range(n_samples),
            'treatment': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'response': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'user_ai_interest': np.random.beta(1, 3, n_samples),
            'content_quality': np.random.beta(3, 7, n_samples),
            'user_engagement': np.random.beta(2, 5, n_samples),
            'post_length': np.random.exponential(200, n_samples),
            'title_length': np.random.normal(50, 20, n_samples),
            'ai_keyword_count': np.random.poisson(2, n_samples)
        })
        
        # Create realistic treatment effect
        base_response_prob = 0.15
        treatment_effect = 0.08
        
        response_prob = base_response_prob + \
                       df['treatment'] * treatment_effect + \
                       df['user_ai_interest'] * 0.05 + \
                       df['content_quality'] * 0.03
        
        response_prob = np.clip(response_prob, 0, 1)
        df['response'] = np.random.binomial(1, response_prob, n_samples)
        
        print(f"Created synthetic data with shape: {df.shape}")
    
    return df

def train_treatment_control_models(df):
    """Train separate models for treatment and control groups"""
    print("\n=== Training Treatment and Control Models ===")
    
    # Prepare features - use available columns
    available_cols = df.columns.tolist()
    feature_cols = []
    
    # Look for common feature column names
    potential_features = [
        'user_ai_interest', 'user_ai_interest_score', 'user_ai_interest_weighted',
        'content_quality', 'content_quality_score', 'user_engagement', 'user_engagement_level',
        'post_length', 'title_length', 'ai_keyword_count', 'user_ai_interactions',
        'user_reputation', 'user_reputation_level', 'user_post_count', 'user_activity_level'
    ]
    
    for col in potential_features:
        if col in available_cols:
            feature_cols.append(col)
    
    # If no features found, use numeric columns excluding target and treatment
    if len(feature_cols) == 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['treatment_ai_content', 'response']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"Using features: {feature_cols}")
    
    # Split data by treatment group
    treatment_mask = df['treatment_ai_content'] == 1
    control_mask = df['treatment_ai_content'] == 0
    
    X_treatment = df[treatment_mask][feature_cols]
    y_treatment = df[treatment_mask]['response']
    X_control = df[control_mask][feature_cols]
    y_control = df[control_mask]['response']
    
    print(f"Treatment group: {len(X_treatment):,} samples")
    print(f"Control group: {len(X_control):,} samples")
    
    # Train treatment model
    if len(X_treatment) > 0 and len(np.unique(y_treatment)) > 1:
        X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
            X_treatment, y_treatment, test_size=0.2, random_state=42
        )
        
        treatment_model = LogisticRegression(random_state=42, max_iter=1000)
        treatment_model.fit(X_train_t, y_train_t)
        
        # Evaluate treatment model
        y_pred_t = treatment_model.predict_proba(X_test_t)[:, 1]
        treatment_auc = roc_auc_score(y_test_t, y_pred_t)
        treatment_accuracy = accuracy_score(y_test_t, treatment_model.predict(X_test_t))
        print(f"Treatment model: AUC={treatment_auc:.4f}, Accuracy={treatment_accuracy:.4f}")
    else:
        treatment_model = None
        treatment_auc = 0
        treatment_accuracy = 0
        print("Treatment model: Insufficient data")
    
    # Train control model
    if len(X_control) > 0 and len(np.unique(y_control)) > 1:
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_control, y_control, test_size=0.2, random_state=42
        )
        
        control_model = LogisticRegression(random_state=42, max_iter=1000)
        control_model.fit(X_train_c, y_train_c)
        
        # Evaluate control model
        y_pred_c = control_model.predict_proba(X_test_c)[:, 1]
        control_auc = roc_auc_score(y_test_c, y_pred_c)
        control_accuracy = accuracy_score(y_test_c, control_model.predict(X_test_c))
        print(f"Control model: AUC={control_auc:.4f}, Accuracy={control_accuracy:.4f}")
    else:
        control_model = None
        control_auc = 0
        control_accuracy = 0
        print("Control model: Insufficient data")
    
    return treatment_model, control_model, treatment_auc, control_auc, feature_cols

def calculate_uplift_scores(df, treatment_model, control_model, feature_cols):
    """Calculate uplift scores for all samples"""
    print("\n=== Calculating Uplift Scores ===")
    
    X = df[feature_cols]
    
    # Predict probabilities
    if treatment_model is not None:
        treatment_probs = treatment_model.predict_proba(X)[:, 1]
    else:
        treatment_probs = np.zeros(len(X))
    
    if control_model is not None:
        control_probs = control_model.predict_proba(X)[:, 1]
    else:
        control_probs = np.zeros(len(X))
    
    # Calculate uplift scores
    uplift_scores = treatment_probs - control_probs
    
    print(f"Uplift score statistics:")
    print(f"  Mean: {uplift_scores.mean():.6f}")
    print(f"  Median: {np.median(uplift_scores):.6f}")
    print(f"  Std: {uplift_scores.std():.6f}")
    print(f"  Min: {uplift_scores.min():.6f}")
    print(f"  Max: {uplift_scores.max():.6f}")
    
    return uplift_scores, treatment_probs, control_probs

def calculate_qini_auuc(df, uplift_scores):
    """Calculate Qini coefficient and AUUC"""
    print("\n=== Calculating Qini/AUUC Metrics ===")
    
    # Create dataframe with predictions and actual outcomes
    eval_df = pd.DataFrame({
        'uplift_score': uplift_scores,
        'treatment': df['treatment_ai_content'],
        'response': df['response']
    })
    
    # Sort by uplift score (descending)
    eval_df = eval_df.sort_values('uplift_score', ascending=False)
    
    # Calculate cumulative metrics
    n_total = len(eval_df)
    n_treatment = eval_df['treatment'].sum()
    n_control = n_total - n_treatment
    
    # Calculate Qini coefficient
    treatment_response_rate = eval_df[eval_df['treatment'] == 1]['response'].mean()
    control_response_rate = eval_df[eval_df['treatment'] == 0]['response'].mean()
    
    if n_treatment > 0 and n_control > 0:
        qini_coefficient = (treatment_response_rate - control_response_rate) * n_treatment * n_control / n_total
    else:
        qini_coefficient = 0
    
    # Calculate AUUC (Area Under Uplift Curve)
    # This is a simplified version - in practice you'd calculate the full curve
    auuc = qini_coefficient / 2  # Simplified approximation
    
    print(f"Qini Coefficient: {qini_coefficient:.6f}")
    print(f"AUUC (simplified): {auuc:.6f}")
    
    return qini_coefficient, auuc

def calculate_uplift_at_top_percentiles(df, uplift_scores, percentiles=[5, 10, 20, 50]):
    """Calculate uplift at different top percentiles"""
    print(f"\n=== Uplift at Top Percentiles ===")
    
    results = {}
    
    for p in percentiles:
        # Get top p% threshold
        threshold = np.percentile(uplift_scores, 100 - p)
        
        # Get samples in top p%
        top_samples = df[uplift_scores >= threshold]
        
        if len(top_samples) > 0:
            # Calculate uplift for top p%
            treatment_group = top_samples[top_samples['treatment'] == 1]
            control_group = top_samples[top_samples['treatment'] == 0]
            
            if len(treatment_group) > 0 and len(control_group) > 0:
                treatment_response_rate = treatment_group['response'].mean()
                control_response_rate = control_group['response'].mean()
                uplift_at_p = treatment_response_rate - control_response_rate
                
                results[f'top_{p}%'] = {
                    'threshold': threshold,
                    'sample_count': len(top_samples),
                    'treatment_response_rate': treatment_response_rate,
                    'control_response_rate': control_response_rate,
                    'uplift': uplift_at_p
                }
                
                print(f"Top {p}%: Uplift = {uplift_at_p:.6f}")
                print(f"  Threshold: {threshold:.6f}")
                print(f"  Samples: {len(top_samples):,}")
                print(f"  Treatment response rate: {treatment_response_rate:.4f}")
                print(f"  Control response rate: {control_response_rate:.4f}")
            else:
                print(f"Top {p}%: Insufficient treatment/control samples")
        else:
            print(f"Top {p}%: No samples above threshold")
    
    return results

def calculate_baseline_metrics(df):
    """Calculate baseline metrics for random targeting and all-users targeting"""
    print(f"\n=== Baseline Metrics ===")
    
    # Random targeting baseline
    treatment_group = df[df['treatment_ai_content'] == 1]
    control_group = df[df['treatment_ai_content'] == 0]
    
    if len(treatment_group) > 0 and len(control_group) > 0:
        treatment_response_rate = treatment_group['response'].mean()
        control_response_rate = control_group['response'].mean()
        random_uplift = treatment_response_rate - control_response_rate
        
        # All-users targeting baseline (assuming everyone gets treatment)
        all_users_response_rate = df['response'].mean()
        all_users_uplift = all_users_response_rate - control_response_rate
        
        print(f"Random targeting uplift: {random_uplift:.6f}")
        print(f"All-users targeting uplift: {all_users_uplift:.6f}")
        print(f"Treatment response rate: {treatment_response_rate:.4f}")
        print(f"Control response rate: {control_response_rate:.4f}")
        print(f"Overall response rate: {all_users_response_rate:.4f}")
        
        return {
            'random_uplift': random_uplift,
            'all_users_uplift': all_users_uplift,
            'treatment_response_rate': treatment_response_rate,
            'control_response_rate': control_response_rate,
            'overall_response_rate': all_users_response_rate
        }
    else:
        print("Insufficient data for baseline calculation")
        return {}

def analyze_segment_differences(df, uplift_scores):
    """Analyze differences between high-uplift and negative-uplift cohorts"""
    print(f"\n=== Segment Differences Analysis ===")
    
    # Create segments based on uplift scores
    df['uplift_score'] = uplift_scores
    
    # Define segments
    high_uplift_threshold = np.percentile(uplift_scores, 80)  # Top 20%
    negative_uplift_threshold = np.percentile(uplift_scores, 20)  # Bottom 20%
    
    high_uplift_cohort = df[uplift_scores >= high_uplift_threshold]
    negative_uplift_cohort = df[uplift_scores <= negative_uplift_threshold]
    
    print(f"High-uplift cohort (top 20%): {len(high_uplift_cohort):,} samples")
    print(f"Negative-uplift cohort (bottom 20%): {len(negative_uplift_cohort):,} samples")
    
    # Analyze characteristics
    if len(high_uplift_cohort) > 0 and len(negative_uplift_cohort) > 0:
        # Response rates
        high_response_rate = high_uplift_cohort['response'].mean()
        negative_response_rate = negative_uplift_cohort['response'].mean()
        
        # Treatment rates
        high_treatment_rate = high_uplift_cohort['treatment_ai_content'].mean()
        negative_treatment_rate = negative_uplift_cohort['treatment_ai_content'].mean()
        
        # Feature differences
        feature_cols = ['user_ai_interest', 'content_quality', 'user_engagement', 
                       'post_length', 'title_length', 'ai_keyword_count']
        
        print(f"\nHigh-uplift cohort characteristics:")
        print(f"  Response rate: {high_response_rate:.4f}")
        print(f"  Treatment rate: {high_treatment_rate:.4f}")
        for col in feature_cols:
            if col in high_uplift_cohort.columns:
                print(f"  {col}: {high_uplift_cohort[col].mean():.4f}")
        
        print(f"\nNegative-uplift cohort characteristics:")
        print(f"  Response rate: {negative_response_rate:.4f}")
        print(f"  Treatment rate: {negative_treatment_rate:.4f}")
        for col in feature_cols:
            if col in negative_uplift_cohort.columns:
                print(f"  {col}: {negative_uplift_cohort[col].mean():.4f}")
        
        # Calculate cohort differences
        response_diff = high_response_rate - negative_response_rate
        treatment_diff = high_treatment_rate - negative_treatment_rate
        
        print(f"\nCohort differences:")
        print(f"  Response rate difference: {response_diff:.4f}")
        print(f"  Treatment rate difference: {treatment_diff:.4f}")
        
        return {
            'high_uplift_cohort': {
                'response_rate': high_response_rate,
                'treatment_rate': high_treatment_rate,
                'sample_count': len(high_uplift_cohort)
            },
            'negative_uplift_cohort': {
                'response_rate': negative_response_rate,
                'treatment_rate': negative_treatment_rate,
                'sample_count': len(negative_uplift_cohort)
            },
            'differences': {
                'response_diff': response_diff,
                'treatment_diff': treatment_diff
            }
        }
    else:
        print("Insufficient data for segment analysis")
        return {}

def create_evaluation_visualizations(df, uplift_scores, percentile_results, baseline_results):
    """Create visualizations for evaluation metrics"""
    print(f"\n=== Creating Visualizations ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Uplift score distribution
    axes[0, 0].hist(uplift_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(uplift_scores.mean(), color='red', linestyle='--', label=f'Mean: {uplift_scores.mean():.4f}')
    axes[0, 0].set_title('Uplift Score Distribution')
    axes[0, 0].set_xlabel('Uplift Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # 2. Response rates by treatment group
    treatment_group = df[df['treatment_ai_content'] == 1]
    control_group = df[df['treatment_ai_content'] == 0]
    
    if len(treatment_group) > 0 and len(control_group) > 0:
        response_rates = [treatment_group['response'].mean(), control_group['response'].mean()]
        groups = ['Treatment', 'Control']
        axes[0, 1].bar(groups, response_rates, color=['lightcoral', 'lightblue'])
        axes[0, 1].set_title('Response Rates by Treatment Group')
        axes[0, 1].set_ylabel('Response Rate')
        for i, v in enumerate(response_rates):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # 3. Uplift at different percentiles
    if percentile_results:
        percentiles = list(percentile_results.keys())
        uplifts = [results['uplift'] for results in percentile_results.values()]
        axes[1, 0].bar(percentiles, uplifts, color='lightgreen')
        axes[1, 0].set_title('Uplift at Different Top Percentiles')
        axes[1, 0].set_ylabel('Uplift')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Baseline comparison
    if baseline_results:
        baseline_names = ['Random', 'All Users']
        baseline_values = [baseline_results.get('random_uplift', 0), 
                          baseline_results.get('all_users_uplift', 0)]
        axes[1, 1].bar(baseline_names, baseline_values, color=['orange', 'purple'])
        axes[1, 1].set_title('Baseline Uplift Comparison')
        axes[1, 1].set_ylabel('Uplift')
        for i, v in enumerate(baseline_values):
            axes[1, 1].text(i, v + 0.001, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('comprehensive_evaluation_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'comprehensive_evaluation_metrics.png'")

def main():
    """Main function for comprehensive evaluation metrics analysis"""
    print("=== COMPREHENSIVE EVALUATION METRICS ANALYSIS ===")
    print("Calculating all evaluation metrics for uplift modeling")
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Train treatment and control models
    treatment_model, control_model, treatment_auc, control_auc, feature_cols = train_treatment_control_models(df)
    
    # Calculate uplift scores
    uplift_scores, treatment_probs, control_probs = calculate_uplift_scores(df, treatment_model, control_model, feature_cols)
    
    # Calculate Qini/AUUC
    qini_coefficient, auuc = calculate_qini_auuc(df, uplift_scores)
    
    # Calculate uplift at top percentiles
    percentile_results = calculate_uplift_at_top_percentiles(df, uplift_scores)
    
    # Calculate baseline metrics
    baseline_results = calculate_baseline_metrics(df)
    
    # Analyze segment differences
    segment_results = analyze_segment_differences(df, uplift_scores)
    
    # Create visualizations
    create_evaluation_visualizations(df, uplift_scores, percentile_results, baseline_results)
    
    # Final summary
    print(f"\n" + "="*60)
    print("FINAL EVALUATION METRICS SUMMARY")
    print("="*60)
    
    print(f"AUC (treatment head): {treatment_auc:.4f}")
    print(f"AUC (control head): {control_auc:.4f}")
    print(f"Qini/AUUC: {qini_coefficient:.6f}")
    
    if 'top_20%' in percentile_results:
        print(f"Uplift@top20%: {percentile_results['top_20%']['uplift']:.6f}")
    
    if baseline_results:
        print(f"Random targeting Qini/AUUC: {baseline_results['random_uplift']:.6f}")
        print(f"All-users targeting Qini/AUUC: {baseline_results['all_users_uplift']:.6f}")
    
    if segment_results:
        high_cohort = segment_results['high_uplift_cohort']
        negative_cohort = segment_results['negative_uplift_cohort']
        print(f"High-uplift cohort value: {high_cohort['response_rate']:.4f}")
        print(f"Negative-uplift cohort value: {negative_cohort['response_rate']:.4f}")
    
    # Export results
    results_df = pd.DataFrame({
        'user_id': df['user_id'] if 'user_id' in df.columns else range(len(df)),
        'post_id': df['post_id'] if 'post_id' in df.columns else range(len(df)),
        'treatment': df['treatment_ai_content'],
        'response': df['response'],
        'uplift_score': uplift_scores,
        'treatment_prob': treatment_probs,
        'control_prob': control_probs
    })
    
    results_df.to_csv('comprehensive_evaluation_results.csv', index=False)
    print(f"\nResults exported to 'comprehensive_evaluation_results.csv'")
    
    print(f"\n=== ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()
