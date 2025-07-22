import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def check_leaky_features():
    """Check for data leakage in features"""
    print("=== Data Leakage Check ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Check data structure
    print(f"Data columns: {list(df.columns)}")
    print(f"Data shape: {df.shape}")
    
    # Check for required columns
    required_cols = ['treatment_ai_content', 'response']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return None
    
    print("✅ All required columns present")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['treatment_ai_content', 'response', 'user_id', 'post_id']]
    
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
    
    # 1. Check for perfect predictors
    print("\n=== 1. Perfect Predictors Check ===")
    
    perfect_treatment_predictors = []
    perfect_response_predictors = []
    
    for col in numeric_features:
        # Check if feature perfectly predicts treatment
        treatment_by_feature = df.groupby(col)['treatment_ai_content'].agg(['mean', 'std'])
        if (treatment_by_feature['std'] == 0).any():
            perfect_treatment_predictors.append(col)
        
        # Check if feature perfectly predicts response
        response_by_feature = df.groupby(col)['response'].agg(['mean', 'std'])
        if (response_by_feature['std'] == 0).any():
            perfect_response_predictors.append(col)
    
    if perfect_treatment_predictors:
        print(f"❌ Perfect treatment predictors found:")
        for col in perfect_treatment_predictors:
            print(f"  - {col}")
    
    if perfect_response_predictors:
        print(f"❌ Perfect response predictors found:")
        for col in perfect_response_predictors:
            print(f"  - {col}")
    
    if not perfect_treatment_predictors and not perfect_response_predictors:
        print("✅ No perfect predictors found")
    
    # 2. Check for highly correlated features
    print("\n=== 2. Highly Correlated Features Check ===")
    
    # Check feature correlation with treatment
    treatment_correlations = []
    for col in numeric_features:
        corr = abs(df[col].corr(df['treatment_ai_content']))
        treatment_correlations.append((col, corr))
    
    treatment_correlations.sort(key=lambda x: x[1], reverse=True)
    
    high_corr_with_treatment = [col for col, corr in treatment_correlations if corr > 0.1]
    
    if high_corr_with_treatment:
        print(f"⚠️  Features highly correlated with treatment (correlation > 0.1):")
        for col in high_corr_with_treatment[:10]:
            corr = next(corr for feat, corr in treatment_correlations if feat == col)
            print(f"  {col}: {corr:.4f}")
    else:
        print("✅ No features highly correlated with treatment")
    
    # Check feature correlation with response
    response_correlations = []
    for col in numeric_features:
        corr = abs(df[col].corr(df['response']))
        response_correlations.append((col, corr))
    
    response_correlations.sort(key=lambda x: x[1], reverse=True)
    
    high_corr_with_response = [col for col, corr in response_correlations if corr > 0.8]
    
    if high_corr_with_response:
        print(f"\n⚠️  Features highly correlated with response (correlation > 0.8):")
        for col in high_corr_with_response[:10]:
            corr = next(corr for feat, corr in response_correlations if feat == col)
            print(f"  {col}: {corr:.4f}")
    else:
        print("✅ No features highly correlated with response")
    
    # 3. Check for time-based leakage
    print("\n=== 3. Time-Based Leakage Check ===")
    
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
    
    # 4. Check for user-based leakage
    print("\n=== 4. User-Based Leakage Check ===")
    
    if 'user_id' in df.columns:
        # Check if treatment assignment is user-dependent
        user_treatment = df.groupby('user_id')['treatment_ai_content'].agg(['mean', 'std'])
        
        # Users with consistent treatment assignment
        consistent_users = (user_treatment['std'] == 0).sum()
        total_users = len(user_treatment)
        
        print(f"Users with consistent treatment: {consistent_users}/{total_users} ({consistent_users/total_users:.1%})")
        
        if consistent_users / total_users > 0.8:
            print("  ⚠️  Most users have consistent treatment assignment - potential user-based leakage")
        else:
            print("  ✅ Treatment assignment varies by user (good)")
    
    # 5. Check for post-based leakage
    print("\n=== 5. Post-Based Leakage Check ===")
    
    if 'post_id' in df.columns:
        # Check if treatment assignment is post-dependent
        post_treatment = df.groupby('post_id')['treatment_ai_content'].agg(['mean', 'std'])
        
        # Posts with consistent treatment assignment
        consistent_posts = (post_treatment['std'] == 0).sum()
        total_posts = len(post_treatment)
        
        print(f"Posts with consistent treatment: {consistent_posts}/{total_posts} ({consistent_posts/total_posts:.1%})")
        
        if consistent_posts / total_posts > 0.8:
            print("  ⚠️  Most posts have consistent treatment assignment - potential post-based leakage")
        else:
            print("  ✅ Treatment assignment varies by post (good)")
    
    # 6. Check for suspicious feature names
    print("\n=== 6. Suspicious Feature Names Check ===")
    
    suspicious_features = []
    for col in numeric_features:
        # Check if feature name suggests it might be a leak
        suspicious_keywords = ['treatment', 'response', 'target', 'label', 'outcome', 'result']
        if any(keyword in col.lower() for keyword in suspicious_keywords):
            suspicious_features.append(col)
    
    if suspicious_features:
        print(f"⚠️  Features with suspicious names:")
        for col in suspicious_features:
            print(f"  - {col}")
    else:
        print("✅ No features with suspicious names found")
    
    # 7. Check for future information
    print("\n=== 7. Future Information Check ===")
    
    future_features = []
    for col in numeric_features:
        # Check if feature name suggests future information
        future_keywords = ['future', 'next', 'later', 'after', 'subsequent']
        if any(keyword in col.lower() for keyword in future_keywords):
            future_features.append(col)
    
    if future_features:
        print(f"⚠️  Features that might contain future information:")
        for col in future_features:
            print(f"  - {col}")
    else:
        print("✅ No features with potential future information found")
    
    # 8. Summary and recommendations
    print("\n=== 8. Summary and Recommendations ===")
    
    issues = []
    recommendations = []
    
    if len(perfect_treatment_predictors) > 0:
        issues.append(f"Found {len(perfect_treatment_predictors)} perfect treatment predictors")
        recommendations.append("Remove perfect treatment predictors")
    
    if len(perfect_response_predictors) > 0:
        issues.append(f"Found {len(perfect_response_predictors)} perfect response predictors")
        recommendations.append("Remove perfect response predictors")
    
    if len(high_corr_with_treatment) > 0:
        issues.append(f"Found {len(high_corr_with_treatment)} features highly correlated with treatment")
        recommendations.append("Review features highly correlated with treatment")
    
    if len(high_corr_with_response) > 0:
        issues.append(f"Found {len(high_corr_with_response)} features highly correlated with response")
        recommendations.append("Review features highly correlated with response")
    
    if len(suspicious_features) > 0:
        issues.append(f"Found {len(suspicious_features)} features with suspicious names")
        recommendations.append("Review features with suspicious names")
    
    if len(future_features) > 0:
        issues.append(f"Found {len(future_features)} features with potential future information")
        recommendations.append("Review features with potential future information")
    
    if issues:
        print("❌ Found the following leakage issues:")
        for issue in issues:
            print(f"  - {issue}")
        
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
    else:
        print("✅ No obvious data leakage issues found")
    
    return {
        'perfect_treatment_predictors': perfect_treatment_predictors,
        'perfect_response_predictors': perfect_response_predictors,
        'high_corr_with_treatment': high_corr_with_treatment,
        'high_corr_with_response': high_corr_with_response,
        'suspicious_features': suspicious_features,
        'future_features': future_features,
        'issues': issues,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    results = check_leaky_features() 