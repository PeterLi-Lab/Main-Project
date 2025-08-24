import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def load_and_prepare_data():
    """Load optimized clustering data and prepare for uplift analysis"""
    
    # Load optimized clusters
    print("Loading optimized clustering data...")
    df = pd.read_csv('optimized_post_clusters.csv')
    
    # Load user-post click data
    print("Loading user-post click data...")
    click_data = pd.read_csv('user_post_click_samples.csv')
    
    # Rename is_click to clicked for consistency
    click_data = click_data.rename(columns={'is_click': 'clicked'})
    
    # Aggregate click data to post level (one record per post)
    print("Aggregating click data to post level...")
    post_click_data = click_data.groupby('post_id').agg({
        'clicked': ['mean', 'sum', 'count']  # mean = click rate, sum = total clicks, count = unique users
    }).reset_index()
    post_click_data.columns = ['post_id', 'click_rate', 'total_clicks', 'unique_users']
    
    # Merge data
    merged_data = df.merge(post_click_data, left_on='Id', right_on='post_id', how='inner')
    
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
        df['user_click_rate'] = df['click_rate'].fillna(0.2)
        df['user_total_clicks'] = df['total_clicks'].fillna(10)
        df['user_total_posts'] = df['unique_users'].fillna(20)
    
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
    X = np.nan_to_num(X, nan=0.0)
    
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

def analyze_uplift_distribution(df, uplift_scores):
    """Analyze uplift distribution patterns"""
    
    print("\n=== UPLIFT DISTRIBUTION ANALYSIS ===")
    
    # Basic statistics
    print(f"Uplift score statistics:")
    print(f"  Mean: {uplift_scores.mean():.6f}")
    print(f"  Median: {np.median(uplift_scores):.6f}")
    print(f"  Std: {uplift_scores.std():.6f}")
    print(f"  Min: {uplift_scores.min():.6f}")
    print(f"  Max: {uplift_scores.max():.6f}")
    
    # Distribution shape analysis
    print(f"\nDistribution shape analysis:")
    print(f"  Skewness: {pd.Series(uplift_scores).skew():.3f}")
    print(f"  Kurtosis: {pd.Series(uplift_scores).kurtosis():.3f}")
    
    # Check for bimodal distribution
    from scipy import stats
    # Try to fit a mixture of two normal distributions
    try:
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(uplift_scores.reshape(-1, 1))
        print(f"  GMM suggests {len(gmm.means_)} components")
        print(f"  Component means: {gmm.means_.flatten()}")
        print(f"  Component weights: {gmm.weights_}")
    except:
        print("  Could not fit GMM")
    
    # Percentile analysis
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentile analysis:")
    for p in percentiles:
        value = np.percentile(uplift_scores, p)
        print(f"  {p}th percentile: {value:.6f}")
    
    # Create detailed distribution plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histogram with density
    axes[0, 0].hist(uplift_scores, bins=100, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    axes[0, 0].axvline(uplift_scores.mean(), color='red', linestyle='--', label=f'Mean: {uplift_scores.mean():.6f}')
    axes[0, 0].axvline(np.median(uplift_scores), color='green', linestyle='--', label=f'Median: {np.median(uplift_scores):.6f}')
    axes[0, 0].set_xlabel('Uplift Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Uplift Score Distribution (Density)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot
    axes[0, 1].boxplot(uplift_scores)
    axes[0, 1].set_ylabel('Uplift Score')
    axes[0, 1].set_title('Uplift Score Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(uplift_scores, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Treatment vs Control uplift comparison
    treatment_uplift = uplift_scores[df['treatment'] == 1]
    control_uplift = uplift_scores[df['treatment'] == 0]
    
    axes[1, 1].hist(treatment_uplift, bins=50, alpha=0.7, label='Treatment', color='lightgreen', density=True)
    axes[1, 1].hist(control_uplift, bins=50, alpha=0.7, label='Control', color='lightcoral', density=True)
    axes[1, 1].set_xlabel('Uplift Score')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Uplift Distribution by Group')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('uplift_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def analyze_extreme_values(df, uplift_scores):
    """Analyze posts with extreme uplift values"""
    
    print("\n=== EXTREME UPLIFT VALUES ANALYSIS ===")
    
    # Add uplift scores to dataframe
    df['uplift_score'] = uplift_scores
    
    # Find extreme values (top and bottom 1%)
    top_1_percent = np.percentile(uplift_scores, 99)
    bottom_1_percent = np.percentile(uplift_scores, 1)
    
    print(f"Top 1% threshold: {top_1_percent:.6f}")
    print(f"Bottom 1% threshold: {bottom_1_percent:.6f}")
    
    # Get extreme posts
    top_posts = df[uplift_scores >= top_1_percent]
    bottom_posts = df[uplift_scores <= bottom_1_percent]
    
    print(f"\nTop 1% posts (highest uplift): {len(top_posts):,}")
    print(f"Bottom 1% posts (lowest uplift): {len(bottom_posts):,}")
    
    # Analyze characteristics of extreme posts
    print(f"\n=== TOP 1% POSTS CHARACTERISTICS ===")
    print(f"Treatment ratio: {(top_posts['treatment'] == 1).mean():.3f}")
    print(f"Average click rate: {top_posts['click_rate'].mean():.3f}")
    print(f"Average title length: {top_posts['title_length'].mean():.1f}")
    print(f"Average body length: {top_posts['body_length'].mean():.1f}")
    print(f"Average AI keyword count: {top_posts['ai_keyword_count'].mean():.1f}")
    
    print(f"\n=== BOTTOM 1% POSTS CHARACTERISTICS ===")
    print(f"Treatment ratio: {(bottom_posts['treatment'] == 1).mean():.3f}")
    print(f"Average click rate: {bottom_posts['click_rate'].mean():.3f}")
    print(f"Average title length: {bottom_posts['title_length'].mean():.1f}")
    print(f"Average body length: {bottom_posts['body_length'].mean():.1f}")
    print(f"Average AI keyword count: {bottom_posts['ai_keyword_count'].mean():.1f}")
    
    # Show sample posts
    print(f"\n=== SAMPLE TOP UPLIFT POSTS ===")
    for i, (_, row) in enumerate(top_posts.head(5).iterrows()):
        title = str(row['Title']) if pd.notna(row['Title']) else "No title"
        print(f"{i+1}. Uplift: {row['uplift_score']:.6f}, Treatment: {row['treatment']}, Click Rate: {row['click_rate']:.3f}")
        print(f"   Title: {title[:100]}...")
        print(f"   Tags: {row['Tags']}")
        print()
    
    print(f"\n=== SAMPLE BOTTOM UPLIFT POSTS ===")
    for i, (_, row) in enumerate(bottom_posts.head(5).iterrows()):
        title = str(row['Title']) if pd.notna(row['Title']) else "No title"
        print(f"{i+1}. Uplift: {row['uplift_score']:.6f}, Treatment: {row['treatment']}, Click Rate: {row['click_rate']:.3f}")
        print(f"   Title: {title[:100]}...")
        print(f"   Tags: {row['Tags']}")
        print()
    
    return top_posts, bottom_posts

def analyze_subgroup_effects(df, uplift_scores):
    """Analyze uplift effects in different subgroups"""
    
    print("\n=== SUBGROUP UPLIFT ANALYSIS ===")
    
    # Add uplift scores to dataframe
    df['uplift_score'] = uplift_scores
    
    # Create subgroups based on different criteria
    subgroups = {}
    
    # 1. By treatment group
    subgroups['Treatment Group'] = {
        'Treatment': df['treatment'] == 1,
        'Control': df['treatment'] == 0
    }
    
    # 2. By click rate level
    click_rate_median = df['click_rate'].median()
    subgroups['Click Rate Level'] = {
        'High Click Rate': df['click_rate'] > click_rate_median,
        'Low Click Rate': df['click_rate'] <= click_rate_median
    }
    
    # 3. By content length
    title_length_median = df['title_length'].median()
    subgroups['Title Length'] = {
        'Long Title': df['title_length'] > title_length_median,
        'Short Title': df['title_length'] <= title_length_median
    }
    
    # 4. By AI keyword count
    ai_keyword_median = df['ai_keyword_count'].median()
    subgroups['AI Keyword Count'] = {
        'High AI Keywords': df['ai_keyword_count'] > ai_keyword_median,
        'Low AI Keywords': df['ai_keyword_count'] <= ai_keyword_median
    }
    
    # Analyze each subgroup
    for group_name, group_dict in subgroups.items():
        print(f"\n--- {group_name} ---")
        for subgroup_name, mask in group_dict.items():
            subgroup_data = df[mask]
            if len(subgroup_data) > 0:
                avg_uplift = subgroup_data['uplift_score'].mean()
                treatment_ratio = (subgroup_data['treatment'] == 1).mean()
                avg_click_rate = subgroup_data['click_rate'].mean()
                print(f"  {subgroup_name}:")
                print(f"    Count: {len(subgroup_data):,}")
                print(f"    Avg Uplift: {avg_uplift:.6f}")
                print(f"    Treatment Ratio: {treatment_ratio:.3f}")
                print(f"    Avg Click Rate: {avg_click_rate:.3f}")
    
    return subgroups

def main():
    """Main function for detailed uplift analysis"""
    
    print("=== DETAILED UPLIFT ANALYSIS ===")
    print("Analyzing distribution patterns and extreme values")
    
    # Load and prepare data
    merged_data = load_and_prepare_data()
    
    # Create treatment/control for Cluster 5
    cluster5_data = create_treatment_control_for_cluster5(merged_data)
    
    # Create simple features
    X, feature_columns = create_simple_features(cluster5_data)
    
    # Prepare target variables
    click_rate_threshold = 0.9
    y = (cluster5_data['click_rate'] > click_rate_threshold).astype(int).values
    treatment = cluster5_data['treatment'].values
    
    # Train uplift models
    treatment_model, control_model, treatment_auc, control_auc = train_uplift_models(X, y, treatment)
    
    # Calculate uplift scores
    uplift_scores, treatment_probs, control_probs = calculate_uplift(X, treatment_model, control_model)
    
    # Analyze uplift distribution
    analyze_uplift_distribution(cluster5_data, uplift_scores)
    
    # Analyze extreme values
    top_posts, bottom_posts = analyze_extreme_values(cluster5_data, uplift_scores)
    
    # Analyze subgroup effects
    subgroups = analyze_subgroup_effects(cluster5_data, uplift_scores)
    
    # Export results
    cluster5_data['uplift_score'] = uplift_scores
    cluster5_data['treatment_prob'] = treatment_probs
    cluster5_data['control_prob'] = control_probs
    
    output_file = 'detailed_uplift_analysis.csv'
    cluster5_data.to_csv(output_file, index=False)
    print(f"\nDetailed results exported to {output_file}")
    
    print("\n=== ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()







