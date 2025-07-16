import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def deep_feature_analysis_english():
    """Deep analysis of features that may cause problems"""
    print("=== Deep Feature Analysis ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data size: {len(df):,}")
    
    # 1. Detailed analysis of high correlation features
    print("\n=== 1. High Correlation Features Detailed Analysis ===")
    
    feature_cols = [col for col in df.columns if col not in ['treatment_ai_content', 'response']]
    numeric_features = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
    
    # Calculate correlations
    treatment_correlations = []
    response_correlations = []
    
    for col in numeric_features:
        treatment_corr = abs(df[col].corr(df['treatment_ai_content']))
        response_corr = abs(df[col].corr(df['response']))
        treatment_correlations.append((col, treatment_corr))
        response_correlations.append((col, response_corr))
    
    treatment_correlations.sort(key=lambda x: x[1], reverse=True)
    response_correlations.sort(key=lambda x: x[1], reverse=True)
    
    # Analyze top 10 high correlation features
    high_corr_features = [col for col, corr in treatment_correlations[:10]]
    
    print("Top 10 features with highest treatment correlation:")
    for i, (col, corr) in enumerate(treatment_correlations[:10]):
        print(f"{i+1}. {col}: {corr:.4f}")
        
        # Analyze distribution of each feature
        treatment_group = df[df['treatment_ai_content'] == 1][col]
        control_group = df[df['treatment_ai_content'] == 0][col]
        
        print(f"   Treatment group: mean={treatment_group.mean():.4f}, std={treatment_group.std():.4f}")
        print(f"   Control group: mean={control_group.mean():.4f}, std={control_group.std():.4f}")
        print(f"   Difference: {treatment_group.mean() - control_group.mean():.4f}")
        print()
    
    # 2. Check if features directly contain treatment information
    print("\n=== 2. Treatment Information Leakage Check ===")
    
    # Check ai_interest_x_treatment feature
    if 'ai_interest_x_treatment' in df.columns:
        print("Analyzing ai_interest_x_treatment feature:")
        ai_interest = df['user_ai_interest_score']
        treatment = df['treatment_ai_content']
        interaction = df['ai_interest_x_treatment']
        
        print(f"  user_ai_interest_score range: [{ai_interest.min():.4f}, {ai_interest.max():.4f}]")
        print(f"  treatment values: {treatment.unique()}")
        print(f"  ai_interest_x_treatment range: [{interaction.min():.4f}, {interaction.max():.4f}]")
        
        # Verify interaction feature calculation
        expected_interaction = ai_interest * treatment
        is_correct = np.allclose(interaction, expected_interaction, rtol=1e-10)
        print(f"  Interaction feature calculation correct: {is_correct}")
        
        if is_correct:
            print("  ⚠️  This feature directly contains treatment information, causing data leakage!")
    
    # 3. Analyze user AI-related features
    print("\n=== 3. User AI-Related Features Analysis ===")
    
    ai_related_features = [col for col in numeric_features if 'ai' in col.lower()]
    print(f"AI-related features: {ai_related_features}")
    
    for col in ai_related_features:
        print(f"\nAnalyzing {col}:")
        
        # Analyze by treatment group
        treatment_group = df[df['treatment_ai_content'] == 1][col]
        control_group = df[df['treatment_ai_content'] == 0][col]
        
        print(f"  Treatment group statistics:")
        print(f"    Mean: {treatment_group.mean():.4f}")
        print(f"    Median: {treatment_group.median():.4f}")
        print(f"    Std: {treatment_group.std():.4f}")
        print(f"    Min: {treatment_group.min():.4f}")
        print(f"    Max: {treatment_group.max():.4f}")
        
        print(f"  Control group statistics:")
        print(f"    Mean: {control_group.mean():.4f}")
        print(f"    Median: {control_group.median():.4f}")
        print(f"    Std: {control_group.std():.4f}")
        print(f"    Min: {control_group.min():.4f}")
        print(f"    Max: {control_group.max():.4f}")
        
        # Check for significant distribution differences
        mean_diff = treatment_group.mean() - control_group.mean()
        print(f"  Mean difference: {mean_diff:.4f}")
        
        if abs(mean_diff) > 0.1:
            print(f"  ⚠️  This feature has significant differences between treatment and control groups, possible data leakage")
    
    # 4. Check feature temporal order
    print("\n=== 4. Feature Temporal Order Check ===")
    
    # Check for features based on future information
    future_features = []
    for col in numeric_features:
        if 'previous' in col.lower() or 'history' in col.lower():
            future_features.append(col)
    
    print(f"Features containing historical information: {future_features}")
    
    for col in future_features:
        print(f"\nAnalyzing {col}:")
        
        # Check if this feature contains information after treatment
        treatment_group = df[df['treatment_ai_content'] == 1][col]
        control_group = df[df['treatment_ai_content'] == 0][col]
        
        # If treatment group has significantly higher feature values, possible data leakage
        mean_diff = treatment_group.mean() - control_group.mean()
        print(f"  Treatment vs Control mean difference: {mean_diff:.4f}")
        
        if mean_diff > 0.1:
            print(f"  ⚠️  This feature may contain information after treatment")
    
    # 5. Check feature multicollinearity
    print("\n=== 5. Multicollinearity Check ===")
    
    # Select top correlation features
    top_features = [col for col, corr in treatment_correlations[:15]]
    
    # Calculate correlation matrix between features
    feature_matrix = df[top_features].fillna(0)
    correlation_matrix = feature_matrix.corr()
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            corr = abs(correlation_matrix.iloc[i, j])
            if corr > 0.8:
                high_corr_pairs.append((top_features[i], top_features[j], corr))
    
    print(f"Highly correlated feature pairs (|correlation| > 0.8):")
    for feat1, feat2, corr in high_corr_pairs:
        print(f"  {feat1} <-> {feat2}: {corr:.4f}")
    
    # 6. Check feature interpretability
    print("\n=== 6. Feature Interpretability Check ===")
    
    # Check for overly complex features
    complex_features = []
    for col in numeric_features:
        # Check feature value distribution
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.9:  # If almost every value is different
            complex_features.append((col, unique_ratio))
    
    print(f"Overly complex features (unique value ratio > 90%):")
    for col, ratio in complex_features:
        print(f"  {col}: {ratio:.2%}")
    
    # 7. Check feature data quality issues
    print("\n=== 7. Data Quality Issues Check ===")
    
    # Check outliers
    outlier_features = []
    for col in numeric_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_ratio = outliers / len(df)
        if outlier_ratio > 0.1:  # If outliers exceed 10%
            outlier_features.append((col, outlier_ratio))
    
    print(f"Features with high outlier ratio (>10%):")
    for col, ratio in outlier_features:
        print(f"  {col}: {ratio:.2%}")
    
    # 8. Recommendations and conclusions
    print("\n=== 8. Recommendations and Conclusions ===")
    
    issues = []
    
    # Check data leakage
    if 'ai_interest_x_treatment' in df.columns:
        issues.append("ai_interest_x_treatment feature directly contains treatment information")
    
    high_corr_treatment = [col for col, corr in treatment_correlations if corr > 0.5]
    if len(high_corr_treatment) > 0:
        issues.append(f"Found {len(high_corr_treatment)} features highly correlated with treatment")
    
    if len(high_corr_pairs) > 0:
        issues.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs, multicollinearity exists")
    
    if len(complex_features) > 0:
        issues.append(f"Found {len(complex_features)} overly complex features")
    
    if len(outlier_features) > 0:
        issues.append(f"Found {len(outlier_features)} features with high outlier ratios")
    
    if issues:
        print("⚠️  Found the following issues:")
        for issue in issues:
            print(f"  - {issue}")
        
        print("\nRecommended solutions:")
        print("  1. Remove ai_interest_x_treatment feature")
        print("  2. Remove user AI features highly correlated with treatment")
        print("  3. Handle multicollinearity issues")
        print("  4. Handle outliers")
    else:
        print("✅ No obvious data quality issues found")
    
    return {
        'high_corr_features': high_corr_features,
        'high_corr_pairs': high_corr_pairs,
        'complex_features': complex_features,
        'outlier_features': outlier_features,
        'issues': issues
    }

if __name__ == "__main__":
    results = deep_feature_analysis_english() 