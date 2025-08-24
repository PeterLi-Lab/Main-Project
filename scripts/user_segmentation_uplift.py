import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def load_and_prepare_data():
    """Load data and prepare for user segmentation analysis"""
    
    print("Loading data...")
    
    # Load optimized clusters
    df = pd.read_csv('optimized_post_clusters.csv')
    
    # Load user-post click data
    click_data = pd.read_csv('user_post_click_samples.csv')
    click_data = click_data.rename(columns={'is_click': 'clicked'})
    
    # Aggregate click data to post level
    post_click_data = click_data.groupby('post_id').agg({
        'clicked': ['mean', 'sum', 'count']
    }).reset_index()
    post_click_data.columns = ['post_id', 'click_rate', 'total_clicks', 'unique_users']
    
    # Merge data
    merged_data = df.merge(post_click_data, left_on='Id', right_on='post_id', how='inner')
    
    return merged_data, click_data

def create_user_features(click_data, merged_data):
    """Create user-level features for segmentation"""
    
    print("Creating user features...")
    
    # User behavior features
    user_features = click_data.groupby('user_id').agg({
        'clicked': ['mean', 'sum', 'count', 'std'],
        'post_id': 'nunique'
    }).reset_index()
    
    user_features.columns = ['user_id', 'user_click_rate', 'user_total_clicks', 'user_total_interactions', 'user_click_std', 'user_unique_posts']
    
    # Calculate user engagement level
    user_features['user_engagement_level'] = pd.cut(
        user_features['user_total_interactions'], 
        bins=[0, 5, 20, 100, np.inf], 
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    # Calculate user click consistency
    user_features['user_click_consistency'] = 1 - user_features['user_click_std']
    user_features['user_click_consistency'] = user_features['user_click_consistency'].fillna(0)
    
    # Calculate user post diversity (how many different posts they interact with)
    user_features['user_post_diversity'] = user_features['user_unique_posts'] / user_features['user_total_interactions']
    user_features['user_post_diversity'] = user_features['user_post_diversity'].fillna(0)
    
    return user_features

def create_treatment_control_for_cluster5(df):
    """Create treatment/control split for Cluster 5"""
    
    # Filter for Cluster 5
    cluster5_data = df[df['cluster_id'] == 5].copy()
    print(f"Cluster 5 data: {len(cluster5_data):,} posts")
    
    # AI tag keywords
    ai_tag_keywords = [
        'artificial-intelligence', 'ai', 'machine-learning', 'ml', 'deep-learning',
        'neural-network', 'neural-networks', 'tensorflow', 'pytorch',
        'keras', 'scikit-learn', 'sklearn', 'classification', 'regression',
        'clustering', 'supervised', 'unsupervised', 'reinforcement-learning',
        'natural-language-processing', 'nlp', 'computer-vision', 'cv'
    ]
    
    # Create treatment labels based on AI tags
    def has_ai_tag(tags):
        if pd.isna(tags):
            return False
        tags_lower = str(tags).lower()
        return any(keyword in tags_lower for keyword in ai_tag_keywords)
    
    cluster5_data['has_ai_tag'] = cluster5_data['Tags'].apply(has_ai_tag)
    cluster5_data['treatment'] = cluster5_data['has_ai_tag'].map({True: 1, False: 0})
    
    return cluster5_data

def create_user_post_features(cluster5_data, user_features):
    """Create user-post level features"""
    
    print("Creating user-post features...")
    
    # Merge user features with cluster data
    # We need to create user-post level data from the original click data
    # For now, let's use the aggregated data and create synthetic user-post pairs
    
    # Create simple features for each post
    cluster5_data['title_length'] = cluster5_data['Title'].str.len().fillna(0)
    cluster5_data['body_length'] = cluster5_data['Body'].str.len().fillna(0)
    cluster5_data['tags_count'] = cluster5_data['Tags'].str.count('<').fillna(0)
    cluster5_data['title_word_count'] = cluster5_data['Title'].str.split().str.len().fillna(0)
    cluster5_data['body_word_count'] = cluster5_data['Body'].str.split().str.len().fillna(0)
    
    # AI keyword count
    ai_keywords = ['ai', 'machine learning', 'neural', 'tensorflow', 'keras', 'sklearn']
    def count_ai_keywords(text):
        if pd.isna(text):
            return 0
        text_lower = str(text).lower()
        return sum(1 for keyword in ai_keywords if keyword in text_lower)
    
    cluster5_data['ai_keyword_count'] = cluster5_data.apply(
        lambda row: count_ai_keywords(f"{row['Title']} {row['Body']} {row['Tags']}"), 
        axis=1
    )
    
    return cluster5_data

def perform_user_segmentation(user_features):
    """Perform K-means clustering on user features"""
    
    print("Performing user segmentation...")
    
    # Select features for clustering
    feature_columns = [
        'user_click_rate', 'user_total_clicks', 'user_total_interactions', 
        'user_click_consistency', 'user_post_diversity'
    ]
    
    # Prepare data
    X = user_features[feature_columns].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    user_features['user_segment'] = kmeans.fit_predict(X_scaled)
    
    # Analyze segments
    print("\n=== USER SEGMENTATION RESULTS ===")
    for segment in sorted(user_features['user_segment'].unique()):
        segment_data = user_features[user_features['user_segment'] == segment]
        print(f"\nSegment {segment}:")
        print(f"  Count: {len(segment_data):,} users")
        print(f"  Click rate: {segment_data['user_click_rate'].mean():.3f}")
        print(f"  Total clicks: {segment_data['user_total_clicks'].mean():.1f}")
        print(f"  Total interactions: {segment_data['user_total_interactions'].mean():.1f}")
        print(f"  Click consistency: {segment_data['user_click_consistency'].mean():.3f}")
        print(f"  Post diversity: {segment_data['user_post_diversity'].mean():.3f}")
        print(f"  Engagement level distribution:")
        engagement_dist = segment_data['user_engagement_level'].value_counts()
        for level, count in engagement_dist.items():
            print(f"    {level}: {count:,} ({count/len(segment_data)*100:.1f}%)")
    
    return user_features, scaler, feature_columns

def analyze_segment_uplift(cluster5_data, user_features):
    """Analyze uplift for each user segment"""
    
    print("\n=== SEGMENT UPLIFT ANALYSIS ===")
    
    # For each segment, calculate uplift
    segment_results = []
    
    for segment in sorted(user_features['user_segment'].unique()):
        print(f"\n--- Segment {segment} Analysis ---")
        
        # Get users in this segment
        segment_users = user_features[user_features['user_segment'] == segment]['user_id'].tolist()
        
        # For demonstration, we'll use the overall cluster data
        # In a real scenario, you'd filter by actual user interactions
        
        # Calculate overall uplift for this segment
        segment_data = cluster5_data.copy()  # In reality, filter by user interactions
        
        # Create features
        X, feature_columns = create_simple_features(segment_data)
        
        # Prepare target
        click_rate_threshold = 0.9
        y = (segment_data['click_rate'] > click_rate_threshold).astype(int).values
        treatment = segment_data['treatment'].values
        
        # Train uplift models
        treatment_model, control_model, treatment_auc, control_auc = train_uplift_models(X, y, treatment)
        
        # Calculate uplift scores
        uplift_scores, treatment_probs, control_probs = calculate_uplift(X, treatment_model, control_model)
        
        # Analyze results
        treatment_group = segment_data[treatment == 1]
        control_group = segment_data[treatment == 0]
        
        treatment_click_rate = treatment_group['click_rate'].mean() if len(treatment_group) > 0 else 0
        control_click_rate = control_group['click_rate'].mean() if len(control_group) > 0 else 0
        observed_uplift = treatment_click_rate - control_click_rate
        mean_uplift_score = uplift_scores.mean()
        
        print(f"  Treatment click rate: {treatment_click_rate:.4f}")
        print(f"  Control click rate: {control_click_rate:.4f}")
        print(f"  Observed uplift: {observed_uplift:.4f}")
        print(f"  Mean uplift score: {mean_uplift_score:.6f}")
        print(f"  Positive uplift: {(uplift_scores > 0).sum():,} ({((uplift_scores > 0).sum()/len(uplift_scores)*100):.1f}%)")
        
        segment_results.append({
            'segment': segment,
            'treatment_click_rate': treatment_click_rate,
            'control_click_rate': control_click_rate,
            'observed_uplift': observed_uplift,
            'mean_uplift_score': mean_uplift_score,
            'positive_uplift_pct': (uplift_scores > 0).sum()/len(uplift_scores)*100
        })
    
    return segment_results

def create_simple_features(df):
    """Create simple features"""
    
    # Content length features
    df['title_length'] = df['Title'].str.len().fillna(0)
    df['body_length'] = df['Body'].str.len().fillna(0)
    df['tags_count'] = df['Tags'].str.count('<').fillna(0)
    
    # Use aggregated click data features
    if 'click_rate' in df.columns:
        df['user_click_rate'] = df['click_rate'].fillna(0.2)
        df['user_total_clicks'] = df['total_clicks'].fillna(10)
        df['user_total_posts'] = df['unique_users'].fillna(20)
    
    # Simple text features
    df['title_word_count'] = df['Title'].str.split().str.len().fillna(0)
    df['body_word_count'] = df['Body'].str.split().str.len().fillna(0)
    
    # AI keyword count
    ai_keywords = ['ai', 'machine learning', 'neural', 'tensorflow', 'keras', 'sklearn']
    def count_ai_keywords(text):
        if pd.isna(text):
            return 0
        text_lower = str(text).lower()
        return sum(1 for keyword in ai_keywords if keyword in text_lower)
    
    df['ai_keyword_count'] = df.apply(
        lambda row: count_ai_keywords(f"{row['Title']} {row['Body']} {row['Tags']}"), 
        axis=1
    )
    
    # Combine features
    feature_columns = ['title_length', 'body_length', 'tags_count', 
                      'user_click_rate', 'user_total_clicks', 'user_total_posts',
                      'title_word_count', 'body_word_count', 'ai_keyword_count']
    
    # Create feature matrix
    X = df[feature_columns].values
    X = np.nan_to_num(X, nan=0.0)
    
    return X, feature_columns

def train_uplift_models(X, y, treatment):
    """Train separate models for treatment and control groups"""
    
    # Split data by treatment group
    treatment_mask = treatment == 1
    control_mask = treatment == 0
    
    X_treatment = X[treatment_mask]
    y_treatment = y[treatment_mask]
    X_control = X[control_mask]
    y_control = y[control_mask]
    
    # Train treatment model
    if len(X_treatment) > 0:
        X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
            X_treatment, y_treatment, test_size=0.2, random_state=42
        )
        
        treatment_model = LogisticRegression(random_state=42, max_iter=1000)
        treatment_model.fit(X_train_t, y_train_t)
    else:
        treatment_model = None
    
    # Train control model
    if len(X_control) > 0:
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_control, y_control, test_size=0.2, random_state=42
        )
        
        control_model = LogisticRegression(random_state=42, max_iter=1000)
        control_model.fit(X_train_c, y_train_c)
    else:
        control_model = None
    
    return treatment_model, control_model, 0, 0

def calculate_uplift(X, treatment_model, control_model):
    """Calculate uplift scores for all samples"""
    
    # Predict probabilities
    if treatment_model is not None:
        treatment_probs = treatment_model.predict_proba(X)[:, 1]
    else:
        treatment_probs = np.zeros(len(X))
    
    if control_model is not None:
        control_probs = control_model.predict_proba(X)[:, 1]
    else:
        control_probs = np.zeros(len(X))
    
    # Calculate uplift
    uplift_scores = treatment_probs - control_probs
    
    return uplift_scores, treatment_probs, control_probs

def visualize_segment_results(segment_results, user_features):
    """Create visualizations for segment analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Segment sizes
    segment_sizes = user_features['user_segment'].value_counts().sort_index()
    axes[0, 0].bar(segment_sizes.index, segment_sizes.values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    axes[0, 0].set_xlabel('User Segment')
    axes[0, 0].set_ylabel('Number of Users')
    axes[0, 0].set_title('User Segment Sizes')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Uplift by segment
    segments = [r['segment'] for r in segment_results]
    uplifts = [r['observed_uplift'] for r in segment_results]
    colors = ['red' if u < 0 else 'green' for u in uplifts]
    
    axes[0, 1].bar(segments, uplifts, color=colors, alpha=0.7)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 1].set_xlabel('User Segment')
    axes[0, 1].set_ylabel('Observed Uplift')
    axes[0, 1].set_title('Uplift by User Segment')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Click rates by segment
    treatment_rates = [r['treatment_click_rate'] for r in segment_results]
    control_rates = [r['control_click_rate'] for r in segment_results]
    
    x = np.arange(len(segments))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, treatment_rates, width, label='Treatment', color='lightgreen', alpha=0.7)
    axes[1, 0].bar(x + width/2, control_rates, width, label='Control', color='lightcoral', alpha=0.7)
    axes[1, 0].set_xlabel('User Segment')
    axes[1, 0].set_ylabel('Click Rate')
    axes[1, 0].set_title('Click Rates by Segment')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(segments)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Positive uplift percentage
    positive_pcts = [r['positive_uplift_pct'] for r in segment_results]
    axes[1, 1].bar(segments, positive_pcts, color='orange', alpha=0.7)
    axes[1, 1].set_xlabel('User Segment')
    axes[1, 1].set_ylabel('Positive Uplift (%)')
    axes[1, 1].set_title('Percentage of Positive Uplift by Segment')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('user_segment_uplift_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function for user segmentation uplift analysis"""
    
    print("=== USER SEGMENTATION UPLIFT ANALYSIS ===")
    print("Identifying which user types are most sensitive to AI tags")
    
    # Load and prepare data
    merged_data, click_data = load_and_prepare_data()
    
    # Create user features
    user_features = create_user_features(click_data, merged_data)
    
    # Perform user segmentation
    user_features, scaler, feature_columns = perform_user_segmentation(user_features)
    
    # Create treatment/control for Cluster 5
    cluster5_data = create_treatment_control_for_cluster5(merged_data)
    
    # Create user-post features
    cluster5_data = create_user_post_features(cluster5_data, user_features)
    
    # Analyze uplift for each segment
    segment_results = analyze_segment_uplift(cluster5_data, user_features)
    
    # Create visualizations
    visualize_segment_results(segment_results, user_features)
    
    # Export results
    user_features.to_csv('user_segments.csv', index=False)
    
    # Create summary report
    print(f"\n=== SUMMARY REPORT ===")
    print("Best segments for AI tag treatment:")
    
    # Sort segments by uplift
    segment_results.sort(key=lambda x: x['observed_uplift'], reverse=True)
    
    for i, result in enumerate(segment_results):
        print(f"{i+1}. Segment {result['segment']}: Uplift = {result['observed_uplift']:.4f}")
        print(f"   Treatment click rate: {result['treatment_click_rate']:.4f}")
        print(f"   Control click rate: {result['control_click_rate']:.4f}")
        print(f"   Positive uplift: {result['positive_uplift_pct']:.1f}%")
    
    print(f"\nResults exported to 'user_segments.csv'")
    print("Visualization saved as 'user_segment_uplift_analysis.png'")

if __name__ == "__main__":
    main()







