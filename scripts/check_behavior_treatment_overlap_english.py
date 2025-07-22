import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def check_behavior_treatment_overlap():
    """Check for overlap between behavior features and treatment assignment"""
    print("=== Behavior-Treatment Overlap Analysis ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Check data structure
    print(f"Data columns: {list(df.columns)}")
    print(f"Data shape: {df.shape}")
    
    # Check for treatment and response columns
    if 'treatment_ai_content' not in df.columns:
        print("❌ No treatment_ai_content column found")
        return None
    
    if 'response' not in df.columns:
        print("❌ No response column found")
        return None
    
    print(f"Treatment distribution:")
    treatment_dist = df['treatment_ai_content'].value_counts(normalize=True)
    for value, ratio in treatment_dist.items():
        print(f"  {value}: {ratio:.2%}")
    
    print(f"\nResponse distribution:")
    response_dist = df['response'].value_counts(normalize=True)
    for value, ratio in response_dist.items():
        print(f"  {value}: {ratio:.2%}")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['treatment_ai_content', 'response', 'user_id', 'post_id']]
    print(f"\nNumber of features: {len(feature_cols)}")
    
    # Check feature types
    numeric_features = []
    categorical_features = []
    
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            numeric_features.append(col)
        else:
            categorical_features.append(col)
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # 1. Check feature correlation with treatment
    print("\n=== 1. Feature-Treatment Correlation Analysis ===")
    
    treatment_correlations = []
    for col in numeric_features:
        corr = abs(df[col].corr(df['treatment_ai_content']))
        treatment_correlations.append((col, corr))
    
    treatment_correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 15 features with highest correlation to treatment:")
    for col, corr in treatment_correlations[:15]:
        print(f"  {col}: {corr:.4f}")
    
    # 2. Check for highly correlated features with treatment
    high_corr_with_treatment = [col for col, corr in treatment_correlations if corr > 0.1]
    print(f"\nFeatures with correlation > 0.1 to treatment: {len(high_corr_with_treatment)}")
    
    if high_corr_with_treatment:
        print("⚠️  Potential data leakage detected - features highly correlated with treatment")
        print("Features that may cause issues:")
        for col in high_corr_with_treatment[:10]:
            print(f"  - {col}")
    
    # 3. Check feature distributions by treatment group
    print("\n=== 2. Feature Distribution by Treatment Group ===")
    
    for col in numeric_features[:10]:  # Only check first 10 features
        print(f"\nAnalyzing {col}:")
        
        # Check distribution by treatment
        treatment_0_mean = df[df['treatment_ai_content'] == 0][col].mean()
        treatment_1_mean = df[df['treatment_ai_content'] == 1][col].mean()
        treatment_0_std = df[df['treatment_ai_content'] == 0][col].std()
        treatment_1_std = df[df['treatment_ai_content'] == 1][col].std()
        
        print(f"  Treatment=0: Mean={treatment_0_mean:.4f}, Std={treatment_0_std:.4f}")
        print(f"  Treatment=1: Mean={treatment_1_mean:.4f}, Std={treatment_1_std:.4f}")
        print(f"  Difference: {abs(treatment_1_mean - treatment_0_mean):.4f}")
        
        # Check for significant differences
        if abs(treatment_1_mean - treatment_0_mean) > 0.1:
            print(f"  ⚠️  Large difference detected - potential leakage")
    
    # 4. Check for perfect predictors of treatment
    print("\n=== 3. Perfect Treatment Predictors Check ===")
    
    perfect_predictors = []
    for col in numeric_features:
        # Check if feature perfectly predicts treatment
        unique_values = df[col].nunique()
        if unique_values == 1:
            continue
        
        # Check if treatment is perfectly predicted by this feature
        treatment_by_feature = df.groupby(col)['treatment_ai_content'].agg(['mean', 'std'])
        
        # If any group has all 0s or all 1s for treatment
        if (treatment_by_feature['std'] == 0).any():
            perfect_predictors.append(col)
    
    if perfect_predictors:
        print(f"❌ Found {len(perfect_predictors)} perfect treatment predictors:")
        for col in perfect_predictors:
            print(f"  - {col}")
    else:
        print("✅ No perfect treatment predictors found")
    
    # 5. Check for time-based leakage
    print("\n=== 4. Time-Based Leakage Check ===")
    
    time_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    
    if time_columns:
        print(f"Time-related columns found: {time_columns}")
        
        for col in time_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    
                    # Check if treatment assignment is time-dependent
                    treatment_by_time = df.groupby(df[col].dt.date)['treatment_ai_content'].mean()
                    
                    if treatment_by_time.std() > 0.1:
                        print(f"  ⚠️  {col}: Treatment assignment varies by time")
                        print(f"     Treatment rate by date: {treatment_by_time.mean():.2%} ± {treatment_by_time.std():.2%}")
                    else:
                        print(f"  ✅ {col}: Treatment assignment is time-independent")
                        
                except:
                    print(f"  Could not process time column: {col}")
    else:
        print("No time-related columns found")
    
    # 6. Check for user-based leakage
    print("\n=== 5. User-Based Leakage Check ===")
    
    if 'user_id' in df.columns:
        # Check if treatment assignment is user-dependent
        treatment_by_user = df.groupby('user_id')['treatment_ai_content'].agg(['mean', 'std'])
        
        # Users with consistent treatment assignment
        consistent_users = (treatment_by_user['std'] == 0).sum()
        total_users = len(treatment_by_user)
        
        print(f"Users with consistent treatment: {consistent_users}/{total_users} ({consistent_users/total_users:.1%})")
        
        if consistent_users / total_users > 0.8:
            print("  ⚠️  Most users have consistent treatment assignment - potential user-based leakage")
        else:
            print("  ✅ Treatment assignment varies by user (good)")
    
    # 7. Check for post-based leakage
    print("\n=== 6. Post-Based Leakage Check ===")
    
    if 'post_id' in df.columns:
        # Check if treatment assignment is post-dependent
        treatment_by_post = df.groupby('post_id')['treatment_ai_content'].agg(['mean', 'std'])
        
        # Posts with consistent treatment assignment
        consistent_posts = (treatment_by_post['std'] == 0).sum()
        total_posts = len(treatment_by_post)
        
        print(f"Posts with consistent treatment: {consistent_posts}/{total_posts} ({consistent_posts/total_posts:.1%})")
        
        if consistent_posts / total_posts > 0.8:
            print("  ⚠️  Most posts have consistent treatment assignment - potential post-based leakage")
        else:
            print("  ✅ Treatment assignment varies by post (good)")
    
    # 8. Summary and recommendations
    print("\n=== 7. Summary and Recommendations ===")
    
    issues = []
    
    if len(high_corr_with_treatment) > 0:
        issues.append(f"Found {len(high_corr_with_treatment)} features highly correlated with treatment")
    
    if len(perfect_predictors) > 0:
        issues.append(f"Found {len(perfect_predictors)} perfect treatment predictors")
    
    if issues:
        print("❌ Found the following overlap issues:")
        for issue in issues:
            print(f"  - {issue}")
        
        print("\nRecommendations:")
        print("  1. Remove or modify features highly correlated with treatment")
        print("  2. Ensure treatment assignment is random and independent of features")
        print("  3. Check for data leakage in feature engineering process")
        print("  4. Consider using propensity score matching or other causal inference methods")
    else:
        print("✅ No obvious behavior-treatment overlap issues found")
        print("\nRecommendations:")
        print("  1. Continue with current feature set")
        print("  2. Monitor for any new features that might introduce leakage")
    
    return {
        'treatment_correlations': treatment_correlations,
        'high_corr_with_treatment': high_corr_with_treatment,
        'perfect_predictors': perfect_predictors,
        'issues': issues
    }

if __name__ == "__main__":
    results = check_behavior_treatment_overlap() 