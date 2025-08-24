#!/usr/bin/env python3
"""
Fixed Uplift Analysis - Addressing Common Uplift Modeling Issues
This script implements proper uplift modeling techniques to avoid the problems identified.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def main():
    """Fixed uplift analysis using IPTW to address selection bias"""
    
    print("=== FIXED UPLIFT ANALYSIS WITH IPTW ===")
    print("Addressing selection bias and causal inference issues")
    
    # Load data
    print("Loading data...")
    try:
        df = pd.read_csv('optimized_post_clusters.csv')
        click_data = pd.read_csv('user_post_click_samples.csv')
        click_data = click_data.rename(columns={'is_click': 'clicked'})
        print(f"Loaded data with shape: {df.shape}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    # Aggregate click data to post level
    print("Aggregating click data...")
    post_click_data = click_data.groupby('post_id').agg({
        'clicked': ['mean', 'sum', 'count'],
        'user_id': 'nunique'
    }).reset_index()
    post_click_data.columns = ['post_id', 'click_rate', 'total_clicks', 'total_interactions', 'unique_users']
    
    # Calculate proper CTR (use clicks per interaction instead of clicks per user)
    post_click_data['proper_ctr'] = post_click_data['total_clicks'] / post_click_data['total_interactions']
    
    # Merge data
    merged_data = df.merge(post_click_data, left_on='Id', right_on='post_id', how='inner')
    
    # Filter for Cluster 5 (AI content cluster)
    cluster5_data = merged_data[merged_data['cluster_id'] == 5].copy()
    print(f"Cluster 5 data: {len(cluster5_data):,} posts")
    
    if len(cluster5_data) == 0:
        print("No data found in Cluster 5. Please check the clustering results.")
        return
    
    # Create treatment/control based on AI tags
    ai_tag_keywords = [
        'artificial-intelligence', 'ai', 'machine-learning', 'ml', 'deep-learning',
        'neural-network', 'neural-networks', 'tensorflow', 'pytorch',
        'keras', 'scikit-learn', 'sklearn', 'classification', 'regression',
        'clustering', 'supervised', 'unsupervised', 'reinforcement-learning',
        'natural-language-processing', 'nlp', 'computer-vision', 'cv'
    ]
    
    def has_ai_tag(tags):
        if pd.isna(tags):
            return False
        tags_lower = str(tags).lower()
        return any(keyword in tags_lower for keyword in ai_tag_keywords)
    
    cluster5_data['has_ai_tag'] = cluster5_data['Tags'].apply(has_ai_tag)
    cluster5_data['treatment'] = cluster5_data['has_ai_tag'].map({True: 1, False: 0})
    
    # Create features for propensity score modeling
    print("Creating features for propensity score modeling...")
    
    # Content features
    cluster5_data['title_length'] = cluster5_data['Title'].fillna('').str.len()
    cluster5_data['body_length'] = cluster5_data['Body'].fillna('').str.len()
    cluster5_data['tags_count'] = cluster5_data['Tags'].fillna('').str.count(',') + 1
    
    # AI content density (how much AI content is in the post)
    ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural', 'tensorflow', 'pytorch']
    cluster5_data['ai_content_density'] = 0
    for keyword in ai_keywords:
        cluster5_data['ai_content_density'] += cluster5_data['merged_content'].str.contains(keyword, case=False, na=False).astype(int)
    
    # User engagement features (from click data)
    cluster5_data['user_engagement'] = cluster5_data['proper_ctr']
    cluster5_data['post_popularity'] = cluster5_data['unique_users']
    
    # Time features (if available)
    if 'CreationDate' in cluster5_data.columns:
        cluster5_data['CreationDate'] = pd.to_datetime(cluster5_data['CreationDate'])
        cluster5_data['day_of_week'] = cluster5_data['CreationDate'].dt.dayofweek
        cluster5_data['hour'] = cluster5_data['CreationDate'].dt.hour
    else:
        cluster5_data['day_of_week'] = 0  # default
        cluster5_data['hour'] = 12  # default
    
    # Select covariates for propensity model
    covariate_cols = [
        'title_length', 'body_length', 'tags_count', 'ai_content_density',
        'user_engagement', 'post_popularity', 'day_of_week', 'hour'
    ]
    
    # Remove rows with missing values and handle infinite values
    cluster5_data = cluster5_data.dropna(subset=covariate_cols + ['treatment', 'proper_ctr'])
    
    # Replace infinite values with large finite values
    for col in covariate_cols:
        cluster5_data[col] = cluster5_data[col].replace([np.inf, -np.inf], np.nan)
        cluster5_data[col] = cluster5_data[col].fillna(cluster5_data[col].median())
    
    if len(cluster5_data) == 0:
        print("No data remaining after removing missing values.")
        return
    
    X = cluster5_data[covariate_cols]
    y_treatment = cluster5_data['treatment']
    y_outcome = cluster5_data['proper_ctr']
    
    print(f"Final dataset: {len(cluster5_data):,} posts")
    print(f"Treatment group: {y_treatment.sum():,} posts ({y_treatment.mean()*100:.1f}%)")
    print(f"Control group: {(1-y_treatment).sum():,} posts ({(1-y_treatment).mean()*100:.1f}%)")
    
    # STEP 1: Fit propensity score model
    print("\n=== STEP 1: PROPENSITY SCORE MODELING ===")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit logistic regression for propensity scores
    prop_model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    prop_model.fit(X_scaled, y_treatment)
    
    # Get predicted propensity scores
    cluster5_data['propensity'] = prop_model.predict_proba(X_scaled)[:, 1]
    
    print(f"Propensity score - Mean: {cluster5_data['propensity'].mean():.4f}")
    print(f"Propensity score - Treatment group: {cluster5_data[cluster5_data['treatment']==1]['propensity'].mean():.4f}")
    print(f"Propensity score - Control group: {cluster5_data[cluster5_data['treatment']==0]['propensity'].mean():.4f}")
    
    # STEP 2: Calculate IPTW weights
    print("\n=== STEP 2: CALCULATING IPTW WEIGHTS ===")
    
    epsilon = 1e-6  # avoid division by zero
    cluster5_data['iptw_weight'] = np.where(
        cluster5_data['treatment'] == 1,
        1 / (cluster5_data['propensity'] + epsilon),
        1 / (1 - cluster5_data['propensity'] + epsilon)
    )
    
    # Stabilized weights (less variance)
    treat_prob = cluster5_data['treatment'].mean()
    cluster5_data['stabilized_weight'] = np.where(
        cluster5_data['treatment'] == 1,
        treat_prob / (cluster5_data['propensity'] + epsilon),
        (1 - treat_prob) / (1 - cluster5_data['propensity'] + epsilon)
    )
    
    print(f"IPTW weights - Mean: {cluster5_data['iptw_weight'].mean():.4f}")
    print(f"Stabilized weights - Mean: {cluster5_data['stabilized_weight'].mean():.4f}")
    
    # STEP 3: Weighted outcome analysis
    print("\n=== STEP 3: WEIGHTED OUTCOME ANALYSIS ===")
    
    # Calculate weighted means
    treatment_group = cluster5_data[cluster5_data['treatment'] == 1]
    control_group = cluster5_data[cluster5_data['treatment'] == 0]
    
    # Unweighted analysis
    unweighted_treatment_ctr = treatment_group['proper_ctr'].mean()
    unweighted_control_ctr = control_group['proper_ctr'].mean()
    unweighted_uplift = unweighted_treatment_ctr - unweighted_control_ctr
    
    # Weighted analysis
    weighted_treatment_ctr = np.average(treatment_group['proper_ctr'], weights=treatment_group['stabilized_weight'])
    weighted_control_ctr = np.average(control_group['proper_ctr'], weights=control_group['stabilized_weight'])
    weighted_uplift = weighted_treatment_ctr - weighted_control_ctr
    
    print(f"Unweighted Analysis:")
    print(f"  Treatment CTR: {unweighted_treatment_ctr:.4f}")
    print(f"  Control CTR: {unweighted_control_ctr:.4f}")
    print(f"  Uplift: {unweighted_uplift:.4f}")
    
    print(f"\nWeighted Analysis (IPTW):")
    print(f"  Treatment CTR: {weighted_treatment_ctr:.4f}")
    print(f"  Control CTR: {weighted_control_ctr:.4f}")
    print(f"  Uplift: {weighted_uplift:.4f}")
    
    # STEP 4: Statistical significance testing
    print("\n=== STEP 4: STATISTICAL SIGNIFICANCE TESTING ===")
    
    # Bootstrap confidence intervals for weighted uplift
    n_bootstrap = 1000
    bootstrap_uplifts = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        bootstrap_indices = np.random.choice(len(cluster5_data), len(cluster5_data), replace=True)
        bootstrap_data = cluster5_data.iloc[bootstrap_indices]
        
        # Calculate weighted uplift for bootstrap sample
        boot_treatment = bootstrap_data[bootstrap_data['treatment'] == 1]
        boot_control = bootstrap_data[bootstrap_data['treatment'] == 0]
        
        if len(boot_treatment) > 0 and len(boot_control) > 0:
            boot_treatment_ctr = np.average(boot_treatment['proper_ctr'], weights=boot_treatment['stabilized_weight'])
            boot_control_ctr = np.average(boot_control['proper_ctr'], weights=boot_control['stabilized_weight'])
            bootstrap_uplifts.append(boot_treatment_ctr - boot_control_ctr)
    
    if len(bootstrap_uplifts) > 0:
        bootstrap_uplifts = np.array(bootstrap_uplifts)
        ci_lower = np.percentile(bootstrap_uplifts, 2.5)
        ci_upper = np.percentile(bootstrap_uplifts, 97.5)
        
        print(f"Bootstrap 95% CI for uplift: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"Uplift is significant: {'Yes' if (ci_lower > 0 or ci_upper < 0) else 'No'}")
    
    # STEP 5: User segmentation analysis
    print("\n=== STEP 5: USER SEGMENTATION ANALYSIS ===")
    
    # Merge with user click data for user-level analysis
    user_post_data = click_data.merge(
        cluster5_data[['Id', 'treatment', 'proper_ctr', 'stabilized_weight']], 
        left_on='post_id', right_on='Id', how='inner'
    )
    
    # Calculate user-level features
    user_features = user_post_data.groupby('user_id').agg({
        'treatment': ['mean', 'sum', 'count'],
        'clicked': ['mean', 'sum', 'count'],
        'proper_ctr': 'mean',
        'stabilized_weight': 'mean'
    }).reset_index()
    
    user_features.columns = [
        'user_id', 'ai_tag_exposure_rate', 'ai_tag_posts_seen', 'total_posts_seen',
        'overall_click_rate', 'total_clicks_made', 'total_interactions',
        'avg_content_ctr', 'avg_weight'
    ]
    
    # Create user segments based on AI tag sensitivity
    user_features['ai_sensitivity'] = user_features['ai_tag_exposure_rate'] * user_features['overall_click_rate']
    
    # Simple segmentation (handle edge cases)
    try:
        if user_features['ai_sensitivity'].nunique() >= 4:
            user_features['segment'] = pd.qcut(user_features['ai_sensitivity'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'], duplicates='drop')
        elif user_features['ai_sensitivity'].nunique() >= 2:
            user_features['segment'] = pd.qcut(user_features['ai_sensitivity'], q=2, labels=['Low', 'High'], duplicates='drop')
        else:
            user_features['segment'] = 'All'
    except:
        user_features['segment'] = 'All'
    
    print(f"User segmentation created for {len(user_features):,} users")
    
    # Analyze segments
    for segment in user_features['segment'].unique():
        segment_data = user_features[user_features['segment'] == segment]
        print(f"\n{segment} Sensitivity Segment: {len(segment_data):,} users")
        print(f"  AI tag exposure rate: {segment_data['ai_tag_exposure_rate'].mean():.3f}")
        print(f"  Overall click rate: {segment_data['overall_click_rate'].mean():.3f}")
        print(f"  AI sensitivity score: {segment_data['ai_sensitivity'].mean():.4f}")
    
    # STEP 6: Export results
    print("\n=== STEP 6: EXPORTING RESULTS ===")
    
    # Export cluster analysis
    cluster5_data.to_csv('fixed_cluster5_analysis.csv', index=False)
    
    # Export user segments
    user_features.to_csv('fixed_user_segments.csv', index=False)
    
    # Create summary report
    with open('fixed_uplift_analysis_report.txt', 'w') as f:
        f.write("=== FIXED UPLIFT ANALYSIS REPORT ===\n\n")
        f.write(f"Dataset: {len(cluster5_data):,} posts from Cluster 5\n")
        f.write(f"Treatment group: {y_treatment.sum():,} posts ({y_treatment.mean()*100:.1f}%)\n")
        f.write(f"Control group: {(1-y_treatment).sum():,} posts ({(1-y_treatment).mean()*100:.1f}%)\n\n")
        
        f.write("=== UPLIFT RESULTS ===\n")
        f.write(f"Unweighted uplift: {unweighted_uplift:.4f}\n")
        f.write(f"Weighted uplift (IPTW): {weighted_uplift:.4f}\n")
        if len(bootstrap_uplifts) > 0:
            f.write(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n")
            f.write(f"Significant: {'Yes' if (ci_lower > 0 or ci_upper < 0) else 'No'}\n")
        
        f.write(f"\n=== USER SEGMENTATION ===\n")
        f.write(f"Total users: {len(user_features):,}\n")
        for segment in user_features['segment'].unique():
            segment_data = user_features[user_features['segment'] == segment]
            f.write(f"{segment}: {len(segment_data):,} users ({len(segment_data)/len(user_features)*100:.1f}%)\n")
    
    print("Analysis complete! Results exported to:")
    print("- fixed_cluster5_analysis.csv")
    print("- fixed_user_segments.csv") 
    print("- fixed_uplift_analysis_report.txt")
    
    # Final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"AI tag uplift effect (IPTW): {weighted_uplift:.4f}")
    if len(bootstrap_uplifts) > 0 and (ci_lower > 0 or ci_upper < 0):
        print(f"Effect is statistically significant")
    else:
        print(f"Effect is not statistically significant")
    
    if weighted_uplift > 0:
        print(f"Recommendation: AI tags have a positive effect on engagement")
    else:
        print(f"Recommendation: AI tags do not improve engagement")

if __name__ == "__main__":
    main()
