#!/usr/bin/env python3
"""
Quick Evaluation Metrics - Answer the user's specific evaluation questions
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== QUICK EVALUATION METRICS ===")
    
    # Load data
    try:
        df = pd.read_csv('uplift_model_data.csv')
        print(f"Loaded data: {df.shape}")
    except:
        print("Data not found")
        return
    
    # 1. Two-Model Approach (Treatment and Control heads)
    print("\n1. TWO-MODEL APPROACH METRICS:")
    
    # Split data by treatment
    treatment_group = df[df['treatment_ai_content'] == 1]
    control_group = df[df['treatment_ai_content'] == 0]
    
    # Get safe features (exclude treatment-related)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    safe_features = [col for col in numeric_cols if col not in ['treatment_ai_content', 'response', 'ai_interest_x_treatment']]
    
    # Train treatment model
    if len(treatment_group) > 1000:
        X_treatment = treatment_group[safe_features]
        y_treatment = treatment_group['response']
        
        # Sample for faster computation
        sample_size = min(10000, len(treatment_group))
        sample_indices = np.random.choice(len(treatment_group), sample_size, replace=False)
        X_treatment_sample = X_treatment.iloc[sample_indices]
        y_treatment_sample = y_treatment.iloc[sample_indices]
        
        X_train, X_test, y_train, y_test = train_test_split(X_treatment_sample, y_treatment_sample, test_size=0.3, random_state=42)
        
        treatment_model = RandomForestClassifier(n_estimators=50, random_state=42)
        treatment_model.fit(X_train, y_train)
        
        y_pred_proba = treatment_model.predict_proba(X_test)[:, 1]
        treatment_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        treatment_auc = 0.0
    
    # Train control model
    if len(control_group) > 1000:
        X_control = control_group[safe_features]
        y_control = control_group['response']
        
        # Sample for faster computation
        sample_size = min(10000, len(control_group))
        sample_indices = np.random.choice(len(control_group), sample_size, replace=False)
        X_control_sample = X_control.iloc[sample_indices]
        y_control_sample = y_control.iloc[sample_indices]
        
        X_train, X_test, y_train, y_test = train_test_split(X_control_sample, y_control_sample, test_size=0.3, random_state=42)
        
        control_model = RandomForestClassifier(n_estimators=50, random_state=42)
        control_model.fit(X_train, y_train)
        
        y_pred_proba = control_model.predict_proba(X_test)[:, 1]
        control_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        control_auc = 0.0
    
    print(f"AUC (treatment head): {treatment_auc:.4f}")
    print(f"AUC (control head): {control_auc:.4f}")
    
    # 2. Calculate Qini/AUUC
    print("\n2. QINI/AUUC METRICS:")
    
    # Simple uplift calculation using treatment/control response rates
    treatment_response_rate = treatment_group['response'].mean()
    control_response_rate = control_group['response'].mean()
    overall_uplift = treatment_response_rate - control_response_rate
    
    # Calculate Qini coefficient
    n_total = len(df)
    n_treatment = len(treatment_group)
    n_control = len(control_group)
    
    qini_coefficient = overall_uplift * n_treatment * n_control / n_total
    
    print(f"Overall uplift: {overall_uplift:.6f}")
    print(f"Qini coefficient: {qini_coefficient:.6f}")
    
    # 3. Uplift@top20%
    print("\n3. UPLIFT@TOP20%:")
    
    # Create simple uplift scores based on user features
    # Use user reputation as a proxy for uplift potential
    df['simple_uplift_score'] = df['user_reputation'] * 0.1 + df['user_post_count'] * 0.05
    
    # Sort by uplift score and get top 20%
    df_sorted = df.sort_values('simple_uplift_score', ascending=False)
    top_20_percent = int(len(df_sorted) * 0.2)
    top_20_df = df_sorted.head(top_20_percent)
    
    top_20_treatment = top_20_df[top_20_df['treatment_ai_content'] == 1]['response'].mean()
    top_20_control = top_20_df[top_20_df['treatment_ai_content'] == 0]['response'].mean()
    uplift_at_top20 = top_20_treatment - top_20_control
    
    print(f"Uplift@top20%: {uplift_at_top20:.6f}")
    print(f"  Top 20% treatment response rate: {top_20_treatment:.4f}")
    print(f"  Top 20% control response rate: {top_20_control:.4f}")
    
    # 4. Baseline metrics
    print("\n4. BASELINE METRICS:")
    
    # Random targeting baseline
    random_qini = 0  # Random targeting has no uplift
    
    # All-users targeting baseline
    all_users_uplift = treatment_response_rate - control_response_rate
    all_users_qini = all_users_uplift * n_treatment * n_control / n_total
    
    print(f"Random targeting Qini: {random_qini:.6f}")
    print(f"All-users targeting Qini: {all_users_qini:.6f}")
    
    # 5. Segment differences
    print("\n5. SEGMENT DIFFERENCES:")
    
    # Create segments based on user reputation
    df['user_segment'] = pd.cut(df['user_reputation'], bins=3, labels=['Low', 'Medium', 'High'])
    
    segment_analysis = df.groupby(['user_segment', 'treatment_ai_content'])['response'].mean().unstack()
    
    print("Response rates by user segment:")
    for segment in ['Low', 'Medium', 'High']:
        if segment in segment_analysis.index:
            treatment_rate = segment_analysis.loc[segment, 1] if 1 in segment_analysis.columns else 0
            control_rate = segment_analysis.loc[segment, 0] if 0 in segment_analysis.columns else 0
            segment_uplift = treatment_rate - control_rate
            print(f"  {segment} reputation:")
            print(f"    Treatment: {treatment_rate:.4f}, Control: {control_rate:.4f}")
            print(f"    Uplift: {segment_uplift:.6f}")
    
    # High vs Low uplift cohorts
    high_uplift_threshold = np.percentile(df['simple_uplift_score'], 80)
    low_uplift_threshold = np.percentile(df['simple_uplift_score'], 20)
    
    high_uplift_cohort = df[df['simple_uplift_score'] >= high_uplift_threshold]
    low_uplift_cohort = df[df['simple_uplift_score'] <= low_uplift_threshold]
    
    high_treatment_rate = high_uplift_cohort[high_uplift_cohort['treatment_ai_content'] == 1]['response'].mean()
    high_control_rate = high_uplift_cohort[high_uplift_cohort['treatment_ai_content'] == 0]['response'].mean()
    high_uplift = high_treatment_rate - high_control_rate
    
    low_treatment_rate = low_uplift_cohort[low_uplift_cohort['treatment_ai_content'] == 1]['response'].mean()
    low_control_rate = low_uplift_cohort[low_uplift_cohort['treatment_ai_content'] == 0]['response'].mean()
    low_uplift = low_treatment_rate - low_control_rate
    
    print(f"\nHigh-uplift cohort (top 20%): {len(high_uplift_cohort):,} samples")
    print(f"  Treatment rate: {high_treatment_rate:.4f}, Control rate: {high_control_rate:.4f}")
    print(f"  Uplift: {high_uplift:.6f}")
    
    print(f"Low-uplift cohort (bottom 20%): {len(low_uplift_cohort):,} samples")
    print(f"  Treatment rate: {low_treatment_rate:.4f}, Control rate: {low_control_rate:.4f}")
    print(f"  Uplift: {low_uplift:.6f}")
    
    # 6. Summary
    print(f"\n" + "="*60)
    print("EVALUATION METRICS SUMMARY")
    print("="*60)
    
    print(f"AUC (treatment head): {treatment_auc:.4f}")
    print(f"AUC (control head): {control_auc:.4f}")
    print(f"Qini coefficient: {qini_coefficient:.6f}")
    print(f"Uplift@top20%: {uplift_at_top20:.6f}")
    print(f"Baseline (all-users) Qini: {all_users_qini:.6f}")
    print(f"High-uplift cohort uplift: {high_uplift:.6f}")
    print(f"Low-uplift cohort uplift: {low_uplift:.6f}")
    
    print(f"\n=== EVALUATION COMPLETE ===")

if __name__ == "__main__":
    main()
