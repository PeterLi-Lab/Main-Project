import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def load_and_prepare_data():
    """Load optimized clustering data and prepare for uplift analysis"""
    
    # Load optimized clusters
    print("Loading optimized clustering data...")
    df = pd.read_csv('optimized_post_clusters.csv')
    print(f"Total posts: {len(df):,}")
    
    # Load user-post click data
    print("Loading user-post click data...")
    try:
        click_data = pd.read_csv('user_post_click_samples.csv')
        print(f"Click data: {len(click_data):,} records")
        
        # Rename is_click to clicked for consistency
        click_data = click_data.rename(columns={'is_click': 'clicked'})
        
        # Aggregate click data to post level (one record per post)
        print("Aggregating click data to post level...")
        post_click_data = click_data.groupby('post_id').agg({
            'clicked': ['mean', 'sum', 'count']  # mean = click rate, sum = total clicks, count = unique users
        }).reset_index()
        post_click_data.columns = ['post_id', 'click_rate', 'total_clicks', 'unique_users']
        
        print(f"Aggregated click data: {len(post_click_data):,} posts")
        
        # Merge data
        merged_data = df.merge(post_click_data, left_on='Id', right_on='post_id', how='inner')
        print(f"Merged data: {len(merged_data):,} records")
        
    except FileNotFoundError:
        print("Warning: user_post_click_samples.csv not found. Creating synthetic click data for demonstration.")
        # Create synthetic click data for demonstration
        np.random.seed(42)
        click_data = pd.DataFrame({
            'user_id': np.random.randint(1, 1001, size=len(df)),
            'post_id': df['Id'].values,
            'clicked': np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])
        })
        
        # Aggregate click data to post level (one record per post)
        print("Aggregating click data to post level...")
        post_click_data = click_data.groupby('post_id').agg({
            'clicked': ['mean', 'sum', 'count']  # mean = click rate, sum = total clicks, count = unique users
        }).reset_index()
        post_click_data.columns = ['post_id', 'click_rate', 'total_clicks', 'unique_users']
        
        # Merge data
        merged_data = df.merge(post_click_data, left_on='Id', right_on='post_id', how='inner')
        print(f"Merged data: {len(merged_data):,} records")
    
    return merged_data

def create_treatment_control_for_cluster5(df):
    """Create treatment/control split for Cluster 5 (best AI cluster)"""
    
    # Filter for Cluster 5
    cluster5_data = df[df['cluster_id'] == 5].copy()
    print(f"\nCluster 5 data: {len(cluster5_data):,} posts")
    
    # AI tag keywords for treatment identification
    ai_tag_keywords = [
        'artificial-intelligence', 'ai', 'machine-learning', 'ml', 'deep-learning',
        'neural-network', 'neural-networks', 'tensorflow', 'pytorch',
        'keras', 'scikit-learn', 'sklearn', 'classification', 'regression',
        'clustering', 'supervised', 'unsupervised', 'reinforcement-learning',
        'natural-language-processing', 'nlp', 'computer-vision', 'cv'
    ]
    
    # Create treatment labels based on AI tags (not content)
    def has_ai_tag(tags):
        if pd.isna(tags):
            return False
        tags_lower = str(tags).lower()
        return any(keyword in tags_lower for keyword in ai_tag_keywords)
    
    cluster5_data['has_ai_tag'] = cluster5_data['Tags'].apply(has_ai_tag)
    
    # Create treatment/control groups
    # Treatment = AI content + has AI tag (manually processed)
    # Control = AI content + no AI tag (natural control group)
    cluster5_data['treatment'] = cluster5_data['has_ai_tag'].map({True: 1, False: 0})
    
    # Analysis
    treatment_count = cluster5_data['treatment'].sum()
    control_count = len(cluster5_data) - treatment_count
    
    print(f"Treatment posts (AI content + AI tag): {treatment_count:,} ({treatment_count/len(cluster5_data)*100:.1f}%)")
    print(f"Control posts (AI content + no AI tag): {control_count:,} ({control_count/len(cluster5_data)*100:.1f}%)")
    
    # Show some examples
    print(f"\nSample treatment posts (with AI tags):")
    treatment_samples = cluster5_data[cluster5_data['treatment'] == 1][['Title', 'Tags']].head(3)
    for _, row in treatment_samples.iterrows():
        title = str(row['Title']) if pd.notna(row['Title']) else "No title"
        print(f"  Title: {title[:50]}...")
        print(f"  Tags: {row['Tags']}")
        print()
    
    print(f"Sample control posts (no AI tags):")
    control_samples = cluster5_data[cluster5_data['treatment'] == 0][['Title', 'Tags']].head(3)
    for _, row in control_samples.iterrows():
        title = str(row['Title']) if pd.notna(row['Title']) else "No title"
        print(f"  Title: {title[:50]}...")
        print(f"  Tags: {row['Tags']}")
        print()
    
    return cluster5_data

def create_simple_features(df):
    """Create simple features without re-running TF-IDF"""
    
    print("Creating simple features...")
    
    # Content length features
    df['title_length'] = df['Title'].str.len().fillna(0)
    df['body_length'] = df['Body'].str.len().fillna(0)
    df['tags_count'] = df['Tags'].str.count('<').fillna(0)  # Count HTML tags as proxy for tag count
    
    # Use aggregated click data features
    if 'click_rate' in df.columns:
        print("Using aggregated click data features...")
        # click_rate is already the click rate per post
        # total_clicks is total clicks for the post
        # unique_users is number of unique users who interacted with the post
        df['user_click_rate'] = df['click_rate'].fillna(0.2)
        df['user_total_clicks'] = df['total_clicks'].fillna(10)
        df['user_total_posts'] = df['unique_users'].fillna(20)  # Using unique_users as proxy
    else:
        print("Creating synthetic user features...")
        # Create synthetic user features
        df['user_click_rate'] = np.random.uniform(0.1, 0.3, len(df))
        df['user_total_clicks'] = np.random.randint(1, 50, len(df))
        df['user_total_posts'] = np.random.randint(1, 100, len(df))
    
    # Simple text features (word counts)
    df['title_word_count'] = df['Title'].str.split().str.len().fillna(0)
    df['body_word_count'] = df['Body'].str.split().str.len().fillna(0)
    
    # AI keyword count as feature
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
    
    # Create feature matrix and handle any remaining NaN values
    X = df[feature_columns].values
    X = np.nan_to_num(X, nan=0.0)  # Replace any remaining NaN with 0
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {feature_columns}")
    
    return X, feature_columns

def train_uplift_models(X, y, treatment):
    """Train separate models for treatment and control groups"""
    
    print("Training uplift models...")
    
    # Split data by treatment group
    treatment_mask = treatment == 1
    control_mask = treatment == 0
    
    X_treatment = X[treatment_mask]
    y_treatment = y[treatment_mask]
    X_control = X[control_mask]
    y_control = y[control_mask]
    
    print(f"Treatment group: {len(X_treatment):,} samples")
    print(f"Control group: {len(X_control):,} samples")
    
    # Train treatment model
    if len(X_treatment) > 0:
        X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
            X_treatment, y_treatment, test_size=0.2, random_state=42
        )
        
        treatment_model = LogisticRegression(random_state=42, max_iter=1000)
        treatment_model.fit(X_train_t, y_train_t)
        
        # Evaluate treatment model
        y_pred_t = treatment_model.predict_proba(X_test_t)[:, 1]
        treatment_auc = roc_auc_score(y_test_t, y_pred_t)
        print(f"Treatment model AUC: {treatment_auc:.3f}")
    else:
        treatment_model = None
        treatment_auc = 0
    
    # Train control model
    if len(X_control) > 0:
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_control, y_control, test_size=0.2, random_state=42
        )
        
        control_model = LogisticRegression(random_state=42, max_iter=1000)
        control_model.fit(X_train_c, y_train_c)
        
        # Evaluate control model
        y_pred_c = control_model.predict_proba(X_test_c)[:, 1]
        control_auc = roc_auc_score(y_test_c, y_pred_c)
        print(f"Control model AUC: {control_auc:.3f}")
    else:
        control_model = None
        control_auc = 0
    
    return treatment_model, control_model, treatment_auc, control_auc

def calculate_uplift(X, treatment_model, control_model):
    """Calculate uplift scores for all samples"""
    
    print("Calculating uplift scores...")
    
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

def analyze_uplift_results(df, uplift_scores, treatment_probs, control_probs):
    """Analyze and visualize uplift results"""
    
    print("Analyzing uplift results...")
    
    # Add predictions to dataframe
    df['uplift_score'] = uplift_scores
    df['treatment_prob'] = treatment_probs
    df['control_prob'] = control_probs
    
    # Overall uplift analysis
    print("\n=== OVERALL UPLIFT ANALYSIS ===")
    print(f"Mean uplift score: {uplift_scores.mean():.4f}")
    print(f"Uplift score std: {uplift_scores.std():.4f}")
    print(f"Positive uplift: {(uplift_scores > 0).sum():,} ({((uplift_scores > 0).sum()/len(uplift_scores)*100):.1f}%)")
    print(f"Negative uplift: {(uplift_scores < 0).sum():,} ({((uplift_scores < 0).sum()/len(uplift_scores)*100):.1f}%)")
    
    # Treatment vs Control analysis
    treatment_group = df[df['treatment'] == 1]
    control_group = df[df['treatment'] == 0]
    
    if 'click_rate' in df.columns:
        print(f"\nTreatment group click rate: {treatment_group['click_rate'].mean():.4f}")
        print(f"Control group click rate: {control_group['click_rate'].mean():.4f}")
        print(f"Observed uplift: {treatment_group['click_rate'].mean() - control_group['click_rate'].mean():.4f}")
    else:
        print(f"\nTreatment group click rate: {treatment_group['clicked'].mean():.4f}")
        print(f"Control group click rate: {control_group['clicked'].mean():.4f}")
        print(f"Observed uplift: {treatment_group['clicked'].mean() - control_group['clicked'].mean():.4f}")
    
    # Uplift distribution analysis
    print("\n=== UPLIFT DISTRIBUTION ANALYSIS ===")
    uplift_quartiles = np.percentile(uplift_scores, [25, 50, 75])
    print(f"Uplift Q1: {uplift_quartiles[0]:.4f}")
    print(f"Uplift Q2 (median): {uplift_quartiles[1]:.4f}")
    print(f"Uplift Q3: {uplift_quartiles[2]:.4f}")
    
    # Create uplift segments
    df['uplift_segment'] = pd.cut(uplift_scores, 
                                 bins=[-np.inf, uplift_quartiles[0], uplift_quartiles[1], uplift_quartiles[2], np.inf],
                                 labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
    
    segment_analysis = df.groupby('uplift_segment').agg({
        'click_rate' if 'click_rate' in df.columns else 'clicked': ['count', 'mean'],
        'uplift_score': ['mean', 'std']
    }).round(4)
    
    print("\nUplift segment analysis:")
    print(segment_analysis)
    
    # Visualizations
    create_uplift_visualizations(df, uplift_scores)
    
    return df

def create_uplift_visualizations(df, uplift_scores):
    """Create visualizations for uplift analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Uplift distribution
    axes[0, 0].hist(uplift_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(uplift_scores.mean(), color='red', linestyle='--', label=f'Mean: {uplift_scores.mean():.4f}')
    axes[0, 0].set_xlabel('Uplift Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Uplift Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Treatment vs Control click rates
    if 'click_rate' in df.columns:
        treatment_click_rate = df[df['treatment'] == 1]['click_rate'].mean()
        control_click_rate = df[df['treatment'] == 0]['click_rate'].mean()
    else:
        treatment_click_rate = df[df['treatment'] == 1]['clicked'].mean()
        control_click_rate = df[df['treatment'] == 0]['clicked'].mean()
    
    axes[0, 1].bar(['Control', 'Treatment'], [control_click_rate, treatment_click_rate], 
                   color=['lightcoral', 'lightgreen'], alpha=0.7)
    axes[0, 1].set_ylabel('Click Rate')
    axes[0, 1].set_title('Click Rates: Treatment vs Control')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Uplift by segment
    if 'click_rate' in df.columns:
        segment_click_rates = df.groupby('uplift_segment')['click_rate'].mean()
    else:
        segment_click_rates = df.groupby('uplift_segment')['clicked'].mean()
    axes[1, 0].bar(segment_click_rates.index, segment_click_rates.values, 
                   color='gold', alpha=0.7)
    axes[1, 0].set_ylabel('Click Rate')
    axes[1, 0].set_title('Click Rates by Uplift Segment')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Uplift score vs actual click rate
    if 'click_rate' in df.columns:
        axes[1, 1].scatter(df['uplift_score'], df['click_rate'], alpha=0.5, s=1)
    else:
        axes[1, 1].scatter(df['uplift_score'], df['clicked'], alpha=0.5, s=1)
    axes[1, 1].set_xlabel('Uplift Score')
    axes[1, 1].set_ylabel('Click Rate')
    axes[1, 1].set_title('Uplift Score vs Actual Click Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_uplift_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function for simple uplift analysis"""
    
    print("=== SIMPLE UPLIFT ANALYSIS ===")
    print("Using Cluster 5 (highest AI density: 93.6%)")
    print("No TF-IDF re-computation - using simple features only")
    
    # Load and prepare data
    merged_data = load_and_prepare_data()
    
    # Create treatment/control for Cluster 5
    cluster5_data = create_treatment_control_for_cluster5(merged_data)
    
    # Create simple features (no TF-IDF)
    X, feature_columns = create_simple_features(cluster5_data)
    
    # Prepare target variables
    if 'click_rate' in cluster5_data.columns:
        # Convert click_rate to binary target using a fixed threshold
        # Since mean=0.932, use 0.9 as threshold to get some variation
        click_rate_threshold = 0.9
        y = (cluster5_data['click_rate'] > click_rate_threshold).astype(int).values
        print(f"Using click_rate converted to binary (fixed threshold: {click_rate_threshold:.3f})")
        print(f"Binary target distribution: {np.bincount(y)}")
        print(f"Click rate stats: min={cluster5_data['click_rate'].min():.3f}, max={cluster5_data['click_rate'].max():.3f}, mean={cluster5_data['click_rate'].mean():.3f}")
    else:
        # Fallback to binary clicked
        y = cluster5_data['clicked'].values
        print("Using binary clicked as target")
    
    treatment = cluster5_data['treatment'].values
    
    # Train uplift models
    treatment_model, control_model, treatment_auc, control_auc = train_uplift_models(X, y, treatment)
    
    # Calculate uplift scores
    uplift_scores, treatment_probs, control_probs = calculate_uplift(X, treatment_model, control_model)
    
    # Analyze results
    results_df = analyze_uplift_results(cluster5_data, uplift_scores, treatment_probs, control_probs)
    
    # Export results
    output_file = 'simple_uplift_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults exported to {output_file}")
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Key findings:")
    print(f"- Used Cluster 5 with {len(cluster5_data):,} posts")
    print(f"- Treatment group: {(treatment == 1).sum():,} posts")
    print(f"- Control group: {(treatment == 0).sum():,} posts")
    print(f"- Mean uplift score: {uplift_scores.mean():.4f}")
    print(f"- Features used: {len(feature_columns)} simple features (no TF-IDF)")

if __name__ == "__main__":
    main()
