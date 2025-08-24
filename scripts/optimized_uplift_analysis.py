import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        # Merge data
        merged_data = df.merge(click_data, left_on='Id', right_on='post_id', how='inner')
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
        
        # Merge data
        merged_data = df.merge(click_data, left_on='Id', right_on='post_id', how='inner')
        print(f"Merged data: {len(merged_data):,} records")
    
    return merged_data

def create_treatment_control_for_cluster5(df):
    """Create treatment/control split for Cluster 5 (best AI cluster)"""
    
    # Filter for Cluster 5
    cluster5_data = df[df['cluster_id'] == 5].copy()
    print(f"\nCluster 5 data: {len(cluster5_data):,} posts")
    
    # AI keywords for treatment identification
    ai_keywords = [
        'artificial intelligence', 'ai', 'machine learning', 'ml', 'deep learning',
        'neural network', 'neural networks', 'neural', 'tensorflow', 'pytorch',
        'keras', 'scikit-learn', 'sklearn', 'classification', 'regression',
        'clustering', 'supervised', 'unsupervised', 'reinforcement learning',
        'natural language processing', 'nlp', 'computer vision', 'cv'
    ]
    
    # Create treatment labels based on AI content
    def is_ai_content(content):
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in ai_keywords)
    
    cluster5_data['is_ai_content'] = cluster5_data.apply(
        lambda row: is_ai_content(f"{row['Title']} {row['Body']} {row['Tags']}"), 
        axis=1
    )
    
    # Create treatment/control groups
    cluster5_data['treatment'] = cluster5_data['is_ai_content'].map({True: 1, False: 0})
    
    # Analysis
    treatment_count = cluster5_data['treatment'].sum()
    control_count = len(cluster5_data) - treatment_count
    
    print(f"Treatment posts: {treatment_count:,} ({treatment_count/len(cluster5_data)*100:.1f}%)")
    print(f"Control posts: {control_count:,} ({control_count/len(cluster5_data)*100:.1f}%)")
    
    return cluster5_data

def create_features(df):
    """Create features for uplift modeling"""
    
    print("Creating features...")
    
    # Text features using TF-IDF
    print("Creating TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1,2))
    tfidf_features = tfidf.fit_transform(df['merged_content'])
    
    # Content length features
    df['title_length'] = df['Title'].str.len()
    df['body_length'] = df['Body'].str.len()
    df['tags_count'] = df['Tags'].str.count('<')  # Count HTML tags as proxy for tag count
    
    # User behavior features (simplified)
    if 'user_id' in df.columns and 'clicked' in df.columns:
        print("Creating user behavior features...")
        user_features = df.groupby('user_id').agg({
            'clicked': ['mean', 'sum', 'count']
        }).reset_index()
        user_features.columns = ['user_id', 'user_click_rate', 'user_total_clicks', 'user_total_posts']
        df = df.merge(user_features, on='user_id', how='left')
    else:
        print("Creating synthetic user features...")
        # Create synthetic user features
        df['user_click_rate'] = np.random.uniform(0.1, 0.3, len(df))
        df['user_total_clicks'] = np.random.randint(1, 50, len(df))
        df['user_total_posts'] = np.random.randint(1, 100, len(df))
    
    # Combine features
    feature_columns = ['title_length', 'body_length', 'tags_count', 
                      'user_click_rate', 'user_total_clicks', 'user_total_posts']
    
    # Convert TF-IDF to dense array and combine with other features
    tfidf_dense = tfidf_features.toarray()
    other_features = df[feature_columns].values
    
    # Combine all features
    all_features = np.hstack([tfidf_dense, other_features])
    
    print(f"Feature matrix shape: {all_features.shape}")
    print(f"TF-IDF features: {tfidf_dense.shape[1]}")
    print(f"Other features: {other_features.shape[1]}")
    
    return all_features, tfidf, feature_columns

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
        'clicked': ['count', 'mean'],
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
    treatment_click_rate = df[df['treatment'] == 1]['clicked'].mean()
    control_click_rate = df[df['treatment'] == 0]['clicked'].mean()
    
    axes[0, 1].bar(['Control', 'Treatment'], [control_click_rate, treatment_click_rate], 
                   color=['lightcoral', 'lightgreen'], alpha=0.7)
    axes[0, 1].set_ylabel('Click Rate')
    axes[0, 1].set_title('Click Rates: Treatment vs Control')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Uplift by segment
    segment_click_rates = df.groupby('uplift_segment')['clicked'].mean()
    axes[1, 0].bar(segment_click_rates.index, segment_click_rates.values, 
                   color='gold', alpha=0.7)
    axes[1, 0].set_ylabel('Click Rate')
    axes[1, 0].set_title('Click Rates by Uplift Segment')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Uplift score vs actual click rate
    axes[1, 1].scatter(df['uplift_score'], df['clicked'], alpha=0.5, s=1)
    axes[1, 1].set_xlabel('Uplift Score')
    axes[1, 1].set_ylabel('Clicked (0/1)')
    axes[1, 1].set_title('Uplift Score vs Actual Click')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimized_uplift_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function for optimized uplift analysis"""
    
    print("=== OPTIMIZED UPLIFT ANALYSIS ===")
    print("Using Cluster 5 (highest AI density: 93.6%)")
    
    # Load and prepare data
    merged_data = load_and_prepare_data()
    
    # Create treatment/control for Cluster 5
    cluster5_data = create_treatment_control_for_cluster5(merged_data)
    
    # Create features
    X, tfidf_vectorizer, feature_columns = create_features(cluster5_data)
    
    # Prepare target variables
    y = cluster5_data['clicked'].values
    treatment = cluster5_data['treatment'].values
    
    # Train uplift models
    treatment_model, control_model, treatment_auc, control_auc = train_uplift_models(X, y, treatment)
    
    # Calculate uplift scores
    uplift_scores, treatment_probs, control_probs = calculate_uplift(X, treatment_model, control_model)
    
    # Analyze results
    results_df = analyze_uplift_results(cluster5_data, uplift_scores, treatment_probs, control_probs)
    
    # Export results
    output_file = 'optimized_uplift_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults exported to {output_file}")
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Key findings:")
    print(f"- Used Cluster 5 with {len(cluster5_data):,} posts")
    print(f"- Treatment group: {(treatment == 1).sum():,} posts")
    print(f"- Control group: {(treatment == 0).sum():,} posts")
    print(f"- Mean uplift score: {uplift_scores.mean():.4f}")

if __name__ == "__main__":
    main()
