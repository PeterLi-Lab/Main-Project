import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def main():
    """Final uplift analysis addressing click rate distribution issues"""
    
    print("=== FINAL UPLIFT ANALYSIS ===")
    print("Addressing click rate distribution problems")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('optimized_post_clusters.csv')
    click_data = pd.read_csv('user_post_click_samples.csv')
    click_data = click_data.rename(columns={'is_click': 'clicked'})
    
    # Aggregate click data to post level
    post_click_data = click_data.groupby('post_id').agg({
        'clicked': ['mean', 'sum', 'count']
    }).reset_index()
    post_click_data.columns = ['post_id', 'click_rate', 'total_clicks', 'unique_users']
    
    # Merge data
    merged_data = df.merge(post_click_data, left_on='Id', right_on='post_id', how='inner')
    
    # Filter for Cluster 5
    cluster5_data = merged_data[merged_data['cluster_id'] == 5].copy()
    print(f"Cluster 5 data: {len(cluster5_data):,} posts")
    
    # Analyze click rate distribution
    print(f"\nClick rate analysis:")
    print(f"Mean: {cluster5_data['click_rate'].mean():.4f}")
    print(f"Median: {cluster5_data['click_rate'].median():.4f}")
    print(f"Std: {cluster5_data['click_rate'].std():.4f}")
    print(f"Min: {cluster5_data['click_rate'].min():.4f}")
    print(f"Max: {cluster5_data['click_rate'].max():.4f}")
    
    # Check unique click rates
    unique_rates = sorted(cluster5_data['click_rate'].unique())
    print(f"Unique click rates: {len(unique_rates)}")
    print(f"Sample rates: {unique_rates[:10]}")
    
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
    
    # Create features WITHOUT data leakage
    print("\nCreating features without data leakage...")
    
    # Content features only (no click-related features)
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
    
    # Content complexity features
    cluster5_data['avg_word_length'] = cluster5_data['body_length'] / (cluster5_data['body_word_count'] + 1)
    cluster5_data['title_body_ratio'] = cluster5_data['title_length'] / (cluster5_data['body_length'] + 1)
    
    # Feature columns (NO click-related features)
    feature_columns = [
        'title_length', 'body_length', 'tags_count', 
        'title_word_count', 'body_word_count', 'ai_keyword_count',
        'avg_word_length', 'title_body_ratio'
    ]
    
    # Create feature matrix
    X = cluster5_data[feature_columns].values
    X = np.nan_to_num(X, nan=0.0)
    
    # SOLVE THE CLICK RATE PROBLEM
    print(f"\n=== SOLVING CLICK RATE DISTRIBUTION PROBLEM ===")
    
    # Method 1: Use total_clicks as target instead of click_rate
    print("Method 1: Using total_clicks as target")
    
    # Create target based on total_clicks (more variation)
    clicks_median = cluster5_data['total_clicks'].median()
    print(f"Total clicks median: {clicks_median:.1f}")
    
    y_clicks = (cluster5_data['total_clicks'] > clicks_median).astype(int).values
    treatment = cluster5_data['treatment'].values
    
    print(f"Target distribution (total_clicks): {np.bincount(y_clicks)}")
    print(f"Treatment target distribution: {np.bincount(y_clicks[treatment == 1])}")
    print(f"Control target distribution: {np.bincount(y_clicks[treatment == 0])}")
    
    # Train uplift models with total_clicks target
    print("\nTraining uplift models with total_clicks target...")
    
    # Split data by treatment group
    treatment_mask = treatment == 1
    control_mask = treatment == 0
    
    X_treatment = X[treatment_mask]
    y_treatment = y_clicks[treatment_mask]
    X_control = X[control_mask]
    y_control = y_clicks[control_mask]
    
    print(f"Treatment group: {len(X_treatment):,} samples")
    print(f"Control group: {len(X_control):,} samples")
    
    # Train treatment model
    if len(X_treatment) > 0 and len(np.unique(y_treatment)) > 1:
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
        print("Treatment model: Insufficient data or single class")
    
    # Train control model
    if len(X_control) > 0 and len(np.unique(y_control)) > 1:
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
        print("Control model: Insufficient data or single class")
    
    # Calculate uplift scores
    print("\nCalculating uplift scores...")
    
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
    
    # Analyze results
    print("\n=== UPLIFT ANALYSIS RESULTS ===")
    
    # Observed uplift (using total_clicks)
    treatment_group = cluster5_data[treatment == 1]
    control_group = cluster5_data[treatment == 0]
    
    treatment_total_clicks = treatment_group['total_clicks'].mean() if len(treatment_group) > 0 else 0
    control_total_clicks = control_group['total_clicks'].mean() if len(control_group) > 0 else 0
    observed_uplift_clicks = treatment_total_clicks - control_total_clicks
    
    treatment_click_rate = treatment_group['click_rate'].mean() if len(treatment_group) > 0 else 0
    control_click_rate = control_group['click_rate'].mean() if len(control_group) > 0 else 0
    observed_uplift_rate = treatment_click_rate - control_click_rate
    
    print(f"Treatment total clicks: {treatment_total_clicks:.2f}")
    print(f"Control total clicks: {control_total_clicks:.2f}")
    print(f"Observed uplift (total clicks): {observed_uplift_clicks:.2f}")
    print(f"Uplift percentage (total clicks): {observed_uplift_clicks/control_total_clicks*100:.2f}%")
    
    print(f"\nTreatment click rate: {treatment_click_rate:.4f}")
    print(f"Control click rate: {control_click_rate:.4f}")
    print(f"Observed uplift (click rate): {observed_uplift_rate:.4f}")
    print(f"Uplift percentage (click rate): {observed_uplift_rate/control_click_rate*100:.2f}%")
    
    # Uplift score statistics
    print(f"\nUplift score statistics:")
    print(f"Mean: {uplift_scores.mean():.6f}")
    print(f"Median: {np.median(uplift_scores):.6f}")
    print(f"Std: {uplift_scores.std():.6f}")
    print(f"Min: {uplift_scores.min():.6f}")
    print(f"Max: {uplift_scores.max():.6f}")
    
    # Positive vs negative uplift
    positive_uplift = (uplift_scores > 0).sum()
    negative_uplift = (uplift_scores < 0).sum()
    zero_uplift = (uplift_scores == 0).sum()
    
    print(f"\nUplift distribution:")
    print(f"Positive uplift: {positive_uplift:,} ({positive_uplift/len(uplift_scores)*100:.1f}%)")
    print(f"Negative uplift: {negative_uplift:,} ({negative_uplift/len(uplift_scores)*100:.1f}%)")
    print(f"Zero uplift: {zero_uplift:,} ({zero_uplift/len(uplift_scores)*100:.1f}%)")
    
    # Export results
    cluster5_data['uplift_score'] = uplift_scores
    cluster5_data['treatment_prob'] = treatment_probs
    cluster5_data['control_prob'] = control_probs
    
    output_file = 'final_uplift_results.csv'
    cluster5_data.to_csv(output_file, index=False)
    print(f"\nResults exported to {output_file}")
    
    # Show sample posts
    print(f"\n=== SAMPLE POSTS ===")
    
    # Top uplift posts
    top_uplift = cluster5_data.nlargest(3, 'uplift_score')
    print(f"\nTop 3 uplift posts:")
    for i, (_, row) in enumerate(top_uplift.iterrows()):
        title = str(row['Title']) if pd.notna(row['Title']) else "No title"
        print(f"{i+1}. Uplift: {row['uplift_score']:.6f}, Treatment: {row['treatment']}, Total Clicks: {row['total_clicks']:.1f}")
        print(f"   Title: {title[:50]}...")
        print(f"   Tags: {row['Tags']}")
        print()
    
    # Bottom uplift posts
    bottom_uplift = cluster5_data.nsmallest(3, 'uplift_score')
    print(f"\nBottom 3 uplift posts:")
    for i, (_, row) in enumerate(bottom_uplift.iterrows()):
        title = str(row['Title']) if pd.notna(row['Title']) else "No title"
        print(f"{i+1}. Uplift: {row['uplift_score']:.6f}, Treatment: {row['treatment']}, Total Clicks: {row['total_clicks']:.1f}")
        print(f"   Title: {title[:50]}...")
        print(f"   Tags: {row['Tags']}")
        print()
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print("Final uplift analysis completed with total_clicks target")

if __name__ == "__main__":
    main()







