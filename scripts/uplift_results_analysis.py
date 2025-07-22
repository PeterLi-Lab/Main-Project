import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def analyze_uplift_results():
    """Analyze uplift modeling results"""
    print("=== Uplift Results Analysis ===\n")
    
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
    
    # 1. Overall uplift analysis
    print("\n=== 1. Overall Uplift Analysis ===")
    
    # Treatment distribution
    treatment_dist = df['treatment_ai_content'].value_counts(normalize=True)
    print(f"Treatment distribution:")
    for value, ratio in treatment_dist.items():
        print(f"  {value}: {ratio:.2%}")
    
    # Response distribution
    response_dist = df['response'].value_counts(normalize=True)
    print(f"\nResponse distribution:")
    for value, ratio in response_dist.items():
        print(f"  {value}: {ratio:.2%}")
    
    # Calculate overall uplift
    treatment_response_rate = df[df['treatment_ai_content'] == 1]['response'].mean()
    control_response_rate = df[df['treatment_ai_content'] == 0]['response'].mean()
    overall_uplift = treatment_response_rate - control_response_rate
    
    print(f"\nOverall uplift analysis:")
    print(f"  Treatment response rate: {treatment_response_rate:.2%}")
    print(f"  Control response rate: {control_response_rate:.2%}")
    print(f"  Overall uplift: {overall_uplift:.2%}")
    
    # 2. Subgroup analysis
    print("\n=== 2. Subgroup Analysis ===")
    
    # Analyze by user segments if user_id exists
    if 'user_id' in df.columns:
        print("User-level analysis:")
        
        # Calculate user-level metrics
        user_metrics = df.groupby('user_id').agg({
            'treatment_ai_content': ['mean', 'count'],
            'response': ['mean', 'count']
        }).round(4)
        
        user_metrics.columns = ['treatment_rate', 'interaction_count', 'response_rate', 'response_count']
        
        # Find users with both treatment and control exposure
        mixed_users = user_metrics[
            (user_metrics['treatment_rate'] > 0) & 
            (user_metrics['treatment_rate'] < 1)
        ]
        
        print(f"Users with mixed treatment exposure: {len(mixed_users)}")
        
        if len(mixed_users) > 0:
            # Calculate user-level uplift
            user_uplift = mixed_users['response_rate'].describe()
            print(f"User-level response rate statistics:")
            print(user_uplift)
    
    # Analyze by post segments if post_id exists
    if 'post_id' in df.columns:
        print("\nPost-level analysis:")
        
        # Calculate post-level metrics
        post_metrics = df.groupby('post_id').agg({
            'treatment_ai_content': ['mean', 'count'],
            'response': ['mean', 'count']
        }).round(4)
        
        post_metrics.columns = ['treatment_rate', 'interaction_count', 'response_rate', 'response_count']
        
        # Find posts with both treatment and control exposure
        mixed_posts = post_metrics[
            (post_metrics['treatment_rate'] > 0) & 
            (post_metrics['treatment_rate'] < 1)
        ]
        
        print(f"Posts with mixed treatment exposure: {len(mixed_posts)}")
        
        if len(mixed_posts) > 0:
            # Calculate post-level uplift
            post_uplift = mixed_posts['response_rate'].describe()
            print(f"Post-level response rate statistics:")
            print(post_uplift)
    
    # 3. Feature-based analysis
    print("\n=== 3. Feature-Based Analysis ===")
    
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
    
    # Analyze feature correlations with uplift
    if numeric_features:
        print(f"\nFeature correlations with response:")
        
        response_correlations = []
        for col in numeric_features:
            corr = abs(df[col].corr(df['response']))
            response_correlations.append((col, corr))
        
        response_correlations.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 10 features correlated with response:")
        for col, corr in response_correlations[:10]:
            print(f"  {col}: {corr:.4f}")
    
    # 4. Treatment effect heterogeneity
    print("\n=== 4. Treatment Effect Heterogeneity ===")
    
    # Analyze treatment effects by feature quartiles
    if numeric_features:
        print("Treatment effects by feature quartiles:")
        
        for col in numeric_features[:5]:  # Analyze top 5 features
            print(f"\n{col}:")
            
            # Create quartiles
            df[f'{col}_quartile'] = pd.qcut(df[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            
            # Calculate uplift by quartile
            quartile_uplift = df.groupby(f'{col}_quartile').apply(
                lambda x: x[x['treatment_ai_content'] == 1]['response'].mean() - 
                         x[x['treatment_ai_content'] == 0]['response'].mean()
            )
            
            for quartile, uplift in quartile_uplift.items():
                print(f"  {quartile}: {uplift:.4f}")
    
    # 5. Statistical significance
    print("\n=== 5. Statistical Significance ===")
    
    # Perform chi-square test for treatment-response relationship
    from scipy.stats import chi2_contingency
    
    contingency_table = pd.crosstab(df['treatment_ai_content'], df['response'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"Chi-square test for treatment-response relationship:")
    print(f"  Chi-square statistic: {chi2:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Degrees of freedom: {dof}")
    
    if p_value < 0.05:
        print("  ✅ Statistically significant relationship (p < 0.05)")
    else:
        print("  ❌ No statistically significant relationship (p >= 0.05)")
    
    # 6. Effect size analysis
    print("\n=== 6. Effect Size Analysis ===")
    
    # Calculate Cohen's d for treatment effect
    treatment_group = df[df['treatment_ai_content'] == 1]['response']
    control_group = df[df['treatment_ai_content'] == 0]['response']
    
    pooled_std = np.sqrt(((treatment_group.var() * (len(treatment_group) - 1)) + 
                          (control_group.var() * (len(control_group) - 1))) / 
                         (len(treatment_group) + len(control_group) - 2))
    
    cohens_d = (treatment_group.mean() - control_group.mean()) / pooled_std
    
    print(f"Cohen's d effect size: {cohens_d:.4f}")
    
    if abs(cohens_d) < 0.2:
        print("  Effect size: Small")
    elif abs(cohens_d) < 0.5:
        print("  Effect size: Medium")
    else:
        print("  Effect size: Large")
    
    # 7. Summary and recommendations
    print("\n=== 7. Summary and Recommendations ===")
    
    insights = []
    recommendations = []
    
    # Overall uplift assessment
    if overall_uplift > 0.01:
        insights.append(f"Positive overall uplift: {overall_uplift:.2%}")
        recommendations.append("Treatment shows positive effect")
    elif overall_uplift < -0.01:
        insights.append(f"Negative overall uplift: {overall_uplift:.2%}")
        recommendations.append("Treatment shows negative effect")
    else:
        insights.append("No significant overall uplift")
        recommendations.append("Consider if treatment is necessary")
    
    # Statistical significance
    if p_value < 0.05:
        insights.append("Statistically significant treatment effect")
    else:
        insights.append("No statistically significant treatment effect")
        recommendations.append("Consider increasing sample size or treatment effect")
    
    # Effect size
    if abs(cohens_d) < 0.2:
        insights.append("Small effect size")
        recommendations.append("Consider if practical significance is sufficient")
    elif abs(cohens_d) > 0.5:
        insights.append("Large effect size")
        recommendations.append("Treatment shows strong practical effect")
    
    print("Key insights:")
    for insight in insights:
        print(f"  - {insight}")
    
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"  - {rec}")
    
    return {
        'overall_uplift': overall_uplift,
        'treatment_response_rate': treatment_response_rate,
        'control_response_rate': control_response_rate,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'insights': insights,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    results = analyze_uplift_results() 