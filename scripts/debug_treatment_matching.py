import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def debug_treatment_matching():
    """Debug treatment matching and assignment"""
    print("=== Treatment Matching Debug ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Check data structure
    print(f"Data columns: {list(df.columns)}")
    print(f"Data shape: {df.shape}")
    
    # Check for treatment column
    if 'treatment_ai_content' not in df.columns:
        print("❌ No treatment_ai_content column found")
        return None
    
    print("✅ treatment_ai_content column found")
    
    # 1. Analyze treatment distribution
    print("\n=== 1. Treatment Distribution Analysis ===")
    
    treatment_dist = df['treatment_ai_content'].value_counts(normalize=True)
    print(f"Treatment distribution:")
    for value, ratio in treatment_dist.items():
        print(f"  {value}: {ratio:.2%}")
    
    # Check treatment balance
    treatment_counts = df['treatment_ai_content'].value_counts()
    total_samples = len(df)
    
    print(f"\nTreatment balance:")
    print(f"  Control (0): {treatment_counts[0]:,} ({treatment_counts[0]/total_samples:.1%})")
    print(f"  Treatment (1): {treatment_counts[1]:,} ({treatment_counts[1]/total_samples:.1%})")
    
    # Check if treatment is balanced
    balance_ratio = min(treatment_counts) / max(treatment_counts)
    if balance_ratio > 0.8:
        print("✅ Treatment groups are well balanced")
    elif balance_ratio > 0.5:
        print("⚠️  Treatment groups are moderately balanced")
    else:
        print("❌ Treatment groups are imbalanced")
    
    # 2. Check treatment assignment patterns
    print("\n=== 2. Treatment Assignment Patterns ===")
    
    # Check if treatment assignment is random
    if 'user_id' in df.columns:
        # Check user-level treatment consistency
        user_treatment = df.groupby('user_id')['treatment_ai_content'].agg(['mean', 'std'])
        
        # Users with consistent treatment
        consistent_users = (user_treatment['std'] == 0).sum()
        total_users = len(user_treatment)
        
        print(f"User treatment consistency:")
        print(f"  Users with consistent treatment: {consistent_users}/{total_users} ({consistent_users/total_users:.1%})")
        
        if consistent_users / total_users > 0.8:
            print("⚠️  Most users have consistent treatment assignment")
        else:
            print("✅ Treatment assignment varies by user (good)")
    
    if 'post_id' in df.columns:
        # Check post-level treatment consistency
        post_treatment = df.groupby('post_id')['treatment_ai_content'].agg(['mean', 'std'])
        
        # Posts with consistent treatment
        consistent_posts = (post_treatment['std'] == 0).sum()
        total_posts = len(post_treatment)
        
        print(f"\nPost treatment consistency:")
        print(f"  Posts with consistent treatment: {consistent_posts}/{total_posts} ({consistent_posts/total_posts:.1%})")
        
        if consistent_posts / total_posts > 0.8:
            print("⚠️  Most posts have consistent treatment assignment")
        else:
            print("✅ Treatment assignment varies by post (good)")
    
    # 3. Check for time-based treatment patterns
    print("\n=== 3. Time-Based Treatment Patterns ===")
    
    time_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    
    if time_columns:
        print(f"Time-related columns found: {time_columns}")
        
        for col in time_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    
                    # Check treatment assignment by time
                    treatment_by_time = df.groupby(df[col].dt.date)['treatment_ai_content'].mean()
                    
                    print(f"\n{col} treatment patterns:")
                    print(f"  Treatment rate by date: {treatment_by_time.mean():.2%} ± {treatment_by_time.std():.2%}")
                    
                    if treatment_by_time.std() > 0.1:
                        print("  ⚠️  Treatment assignment varies significantly by time")
                    else:
                        print("  ✅ Treatment assignment is time-independent")
                        
                except:
                    print(f"  Could not process time column: {col}")
    else:
        print("No time-related columns found")
    
    # 4. Check treatment-response relationship
    print("\n=== 4. Treatment-Response Relationship ===")
    
    if 'response' in df.columns:
        # Calculate response rates by treatment
        treatment_response = df.groupby('treatment_ai_content')['response'].agg(['mean', 'count'])
        
        print(f"Response rates by treatment:")
        print(treatment_response)
        
        # Calculate uplift
        treatment_response_rate = treatment_response.loc[1, 'mean']
        control_response_rate = treatment_response.loc[0, 'mean']
        uplift = treatment_response_rate - control_response_rate
        
        print(f"\nUplift analysis:")
        print(f"  Treatment response rate: {treatment_response_rate:.2%}")
        print(f"  Control response rate: {control_response_rate:.2%}")
        print(f"  Uplift: {uplift:.2%}")
        
        # Check if uplift is significant
        if abs(uplift) > 0.01:
            print("✅ Significant treatment effect detected")
        else:
            print("⚠️  No significant treatment effect detected")
    
    # 5. Check for confounding variables
    print("\n=== 5. Confounding Variable Check ===")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['treatment_ai_content', 'response', 'user_id', 'post_id']]
    
    # Check feature correlation with treatment
    confounding_variables = []
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            corr = abs(df[col].corr(df['treatment_ai_content']))
            if corr > 0.1:
                confounding_variables.append((col, corr))
    
    confounding_variables.sort(key=lambda x: x[1], reverse=True)
    
    if confounding_variables:
        print(f"Features highly correlated with treatment (correlation > 0.1):")
        for col, corr in confounding_variables[:10]:
            print(f"  {col}: {corr:.4f}")
    else:
        print("✅ No features highly correlated with treatment")
    
    # 6. Check for data quality issues
    print("\n=== 6. Data Quality Issues ===")
    
    issues = []
    
    # Check for missing treatment values
    missing_treatment = df['treatment_ai_content'].isnull().sum()
    if missing_treatment > 0:
        issues.append(f"Found {missing_treatment} missing treatment values")
    
    # Check for invalid treatment values
    invalid_treatment = (~df['treatment_ai_content'].isin([0, 1])).sum()
    if invalid_treatment > 0:
        issues.append(f"Found {invalid_treatment} invalid treatment values")
    
    # Check for extreme imbalance
    if balance_ratio < 0.1:
        issues.append("Extreme treatment imbalance detected")
    
    if issues:
        print("⚠️  Found the following issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ No obvious data quality issues found")
    
    # 7. Summary and recommendations
    print("\n=== 7. Summary and Recommendations ===")
    
    recommendations = []
    
    if balance_ratio < 0.5:
        recommendations.append("Consider rebalancing treatment groups")
    
    if 'user_id' in df.columns and consistent_users / total_users > 0.8:
        recommendations.append("Review user-level treatment assignment strategy")
    
    if 'post_id' in df.columns and consistent_posts / total_posts > 0.8:
        recommendations.append("Review post-level treatment assignment strategy")
    
    if len(confounding_variables) > 0:
        recommendations.append("Address confounding variables in analysis")
    
    if 'response' in df.columns and abs(uplift) < 0.01:
        recommendations.append("Consider if treatment effect is meaningful")
    
    if recommendations:
        print("Recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
    else:
        print("✅ No specific recommendations needed")
    
    return {
        'treatment_distribution': treatment_dist,
        'balance_ratio': balance_ratio,
        'confounding_variables': confounding_variables,
        'uplift': uplift if 'response' in df.columns else None,
        'issues': issues,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    results = debug_treatment_matching() 