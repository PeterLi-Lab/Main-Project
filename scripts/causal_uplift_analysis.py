import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy import stats

def main():
    """Causal uplift analysis with proper CTR calculation and user segmentation validation"""
    
    print("=== CAUSAL UPLIFT ANALYSIS ===")
    print("Addressing fundamental issues with click rate distribution")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('optimized_post_clusters.csv')
    click_data = pd.read_csv('user_post_click_samples.csv')
    click_data = click_data.rename(columns={'is_click': 'clicked'})
    
    # Aggregate click data to post level with proper CTR calculation
    post_click_data = click_data.groupby('post_id').agg({
        'clicked': ['mean', 'sum', 'count']
    }).reset_index()
    post_click_data.columns = ['post_id', 'click_rate', 'total_clicks', 'unique_users']
    
    # Calculate proper CTR = total_clicks / unique_users (avoiding 1.0 issue)
    post_click_data['proper_ctr'] = post_click_data['total_clicks'] / post_click_data['unique_users']
    
    # Merge data
    merged_data = df.merge(post_click_data, left_on='Id', right_on='post_id', how='inner')
    
    # Filter for Cluster 5
    cluster5_data = merged_data[merged_data['cluster_id'] == 5].copy()
    print(f"Cluster 5 data: {len(cluster5_data):,} posts")
    
    # Analyze CTR distributions
    print(f"\n=== CTR ANALYSIS ===")
    print(f"Original click_rate - Mean: {cluster5_data['click_rate'].mean():.4f}, Median: {cluster5_data['click_rate'].median():.4f}")
    print(f"Proper CTR - Mean: {cluster5_data['proper_ctr'].mean():.4f}, Median: {cluster5_data['proper_ctr'].median():.4f}")
    print(f"Total clicks - Mean: {cluster5_data['total_clicks'].mean():.2f}, Median: {cluster5_data['total_clicks'].median():.2f}")
    print(f"Unique users - Mean: {cluster5_data['unique_users'].mean():.2f}, Median: {cluster5_data['unique_users'].median():.2f}")
    
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
    
    # Analysis
    treatment_count = cluster5_data['treatment'].sum()
    control_count = len(cluster5_data) - treatment_count
    
    print(f"\nTreatment posts (AI content + AI tag): {treatment_count:,} ({treatment_count/len(cluster5_data)*100:.1f}%)")
    print(f"Control posts (AI content + no AI tag): {control_count:,} ({control_count/len(cluster5_data)*100:.1f}%)")
    
    # STEP 1: Calculate proper CTR-based uplift
    print(f"\n=== STEP 1: PROPER CTR-BASED UPLIFT ===")
    
    treatment_group = cluster5_data[cluster5_data['treatment'] == 1]
    control_group = cluster5_data[cluster5_data['treatment'] == 0]
    
    # Observed uplift using proper CTR
    treatment_ctr = treatment_group['proper_ctr'].mean() if len(treatment_group) > 0 else 0
    control_ctr = control_group['proper_ctr'].mean() if len(control_group) > 0 else 0
    observed_uplift_ctr = treatment_ctr - control_ctr
    
    print(f"Treatment CTR: {treatment_ctr:.4f}")
    print(f"Control CTR: {control_ctr:.4f}")
    print(f"Observed uplift (CTR): {observed_uplift_ctr:.4f}")
    print(f"Uplift percentage: {observed_uplift_ctr/control_ctr*100:.2f}%" if control_ctr > 0 else "Cannot calculate percentage")
    
    # Statistical significance test
    if len(treatment_group) > 0 and len(control_group) > 0:
        t_stat, p_value = stats.ttest_ind(treatment_group['proper_ctr'], control_group['proper_ctr'])
        print(f"T-test: t={t_stat:.3f}, p={p_value:.6f}")
        print(f"Statistically significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    # STEP 2: Create user-level features by AI tag exposure
    print(f"\n=== STEP 2: USER-LEVEL FEATURES BY AI TAG EXPOSURE ===")
    
    # Merge click data with post data to get user-tag exposure
    user_post_data = click_data.merge(
        cluster5_data[['Id', 'treatment', 'proper_ctr', 'total_clicks', 'unique_users']], 
        left_on='post_id', right_on='Id', how='inner'
    )
    
    # Calculate user behavior by treatment exposure
    user_features = user_post_data.groupby('user_id').agg({
        'treatment': ['mean', 'sum', 'count'],  # mean = % of AI-tagged content seen
        'clicked': ['mean', 'sum', 'count'],    # overall click behavior
        'proper_ctr': 'mean',                   # average CTR of content seen
        'total_clicks': 'mean'                  # average total clicks of content seen
    }).reset_index()
    
    user_features.columns = [
        'user_id', 'ai_tag_exposure_rate', 'ai_tag_posts_seen', 'total_posts_seen',
        'overall_click_rate', 'total_clicks_made', 'total_interactions',
        'avg_content_ctr', 'avg_content_total_clicks'
    ]
    
    # Calculate AI-specific behavior
    ai_exposed_users = user_post_data[user_post_data['treatment'] == 1].groupby('user_id').agg({
        'clicked': ['mean', 'sum', 'count']
    }).reset_index()
    ai_exposed_users.columns = ['user_id', 'ai_click_rate', 'ai_clicks_made', 'ai_interactions']
    
    non_ai_users = user_post_data[user_post_data['treatment'] == 0].groupby('user_id').agg({
        'clicked': ['mean', 'sum', 'count']
    }).reset_index()
    non_ai_users.columns = ['user_id', 'non_ai_click_rate', 'non_ai_clicks_made', 'non_ai_interactions']
    
    # Merge all user features
    user_features = user_features.merge(ai_exposed_users, on='user_id', how='left')
    user_features = user_features.merge(non_ai_users, on='user_id', how='left')
    
    # Fill NaN values
    user_features = user_features.fillna(0)
    
    # Calculate user-specific uplift (difference in behavior between AI and non-AI content)
    user_features['user_uplift'] = user_features['ai_click_rate'] - user_features['non_ai_click_rate']
    user_features['user_uplift_ratio'] = user_features['ai_click_rate'] / (user_features['non_ai_click_rate'] + 0.001)
    
    print(f"User features calculated for {len(user_features):,} users")
    print(f"User uplift - Mean: {user_features['user_uplift'].mean():.4f}, Std: {user_features['user_uplift'].std():.4f}")
    
    # STEP 3: User segmentation based on AI tag sensitivity
    print(f"\n=== STEP 3: USER SEGMENTATION BY AI TAG SENSITIVITY ===")
    
    # Create features for segmentation
    segmentation_features = [
        'ai_tag_exposure_rate', 'overall_click_rate', 'total_interactions',
        'user_uplift', 'ai_click_rate', 'non_ai_click_rate'
    ]
    
    X_seg = user_features[segmentation_features].values
    
    # Simple clustering based on AI sensitivity
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)
    
    # Use 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    user_features['segment'] = kmeans.fit_predict(X_scaled)
    
    # Analyze segments
    print(f"\nSegment Analysis:")
    for segment in sorted(user_features['segment'].unique()):
        segment_data = user_features[user_features['segment'] == segment]
        print(f"\nSegment {segment}: {len(segment_data):,} users ({len(segment_data)/len(user_features)*100:.1f}%)")
        print(f"  AI tag exposure rate: {segment_data['ai_tag_exposure_rate'].mean():.3f}")
        print(f"  Overall click rate: {segment_data['overall_click_rate'].mean():.3f}")
        print(f"  User uplift: {segment_data['user_uplift'].mean():.4f}")
        print(f"  AI click rate: {segment_data['ai_click_rate'].mean():.3f}")
        print(f"  Non-AI click rate: {segment_data['non_ai_click_rate'].mean():.3f}")
        print(f"  Total interactions: {segment_data['total_interactions'].mean():.1f}")
    
    # STEP 4: Validate segments with uplift analysis
    print(f"\n=== STEP 4: SEGMENT UPLIFT VALIDATION ===")
    
    segment_analysis = []
    
    for segment in sorted(user_features['segment'].unique()):
        segment_users = user_features[user_features['segment'] == segment]['user_id'].tolist()
        
        # Get posts seen by this segment
        segment_posts = user_post_data[user_post_data['user_id'].isin(segment_users)]
        
        if len(segment_posts) == 0:
            continue
            
        # Calculate segment-specific uplift
        segment_treatment = segment_posts[segment_posts['treatment'] == 1]
        segment_control = segment_posts[segment_posts['treatment'] == 0]
        
        if len(segment_treatment) > 0 and len(segment_control) > 0:
            treatment_ctr = segment_treatment['clicked'].mean()
            control_ctr = segment_control['clicked'].mean()
            segment_uplift = treatment_ctr - control_ctr
            
            # Statistical test
            t_stat, p_value = stats.ttest_ind(
                segment_treatment['clicked'], 
                segment_control['clicked']
            )
            
            print(f"\nSegment {segment} Uplift Validation:")
            print(f"  Treatment CTR: {treatment_ctr:.4f} (n={len(segment_treatment):,})")
            print(f"  Control CTR: {control_ctr:.4f} (n={len(segment_control):,})")
            print(f"  Uplift: {segment_uplift:.4f}")
            print(f"  P-value: {p_value:.6f}")
            print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
            
            segment_analysis.append({
                'segment': segment,
                'users': len(segment_users),
                'treatment_ctr': treatment_ctr,
                'control_ctr': control_ctr,
                'uplift': segment_uplift,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
    
    # STEP 5: Final recommendations
    print(f"\n=== STEP 5: FINAL TREATMENT RECOMMENDATIONS ===")
    
    # Sort segments by uplift
    segment_analysis.sort(key=lambda x: x['uplift'], reverse=True)
    
    print(f"\nSegments ranked by AI tag uplift effect:")
    for i, analysis in enumerate(segment_analysis):
        print(f"\n{i+1}. Segment {analysis['segment']}")
        print(f"   Users: {analysis['users']:,}")
        print(f"   Uplift: {analysis['uplift']:.4f}")
        print(f"   Treatment CTR: {analysis['treatment_ctr']:.4f}")
        print(f"   Control CTR: {analysis['control_ctr']:.4f}")
        print(f"   Significant: {'Yes' if analysis['significant'] else 'No'}")
        
        # Recommendation
        if analysis['uplift'] > 0 and analysis['significant']:
            recommendation = "STRONGLY RECOMMEND"
        elif analysis['uplift'] > 0:
            recommendation = "RECOMMEND"
        elif analysis['uplift'] < 0 and analysis['significant']:
            recommendation = "AVOID"
        else:
            recommendation = "TEST FIRST"
        
        print(f"   Recommendation: {recommendation}")
    
    # Export results
    user_features.to_csv('causal_user_segments.csv', index=False)
    
    segment_df = pd.DataFrame(segment_analysis)
    segment_df.to_csv('causal_segment_analysis.csv', index=False)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print("Results exported to:")
    print("- causal_user_segments.csv")
    print("- causal_segment_analysis.csv")
    
    # Summary
    positive_segments = [s for s in segment_analysis if s['uplift'] > 0 and s['significant']]
    total_recommended_users = sum(s['users'] for s in positive_segments)
    
    print(f"\nSummary:")
    print(f"Total users: {len(user_features):,}")
    print(f"Users in positive uplift segments: {total_recommended_users:,} ({total_recommended_users/len(user_features)*100:.1f}%)")
    print(f"Significant positive segments: {len(positive_segments)}")

if __name__ == "__main__":
    main()







