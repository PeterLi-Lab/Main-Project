import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def main():
    """Diagnose data issues in uplift evaluation"""
    
    print("=== DIAGNOSE DATA ISSUES ===")
    
    # Load data
    print("Loading data...")
    try:
        df = pd.read_csv('optimized_post_clusters.csv')
        click_data = pd.read_csv('user_post_click_samples.csv')
        click_data = click_data.rename(columns={'is_click': 'clicked'})
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    # Check click data structure
    print("\n=== CLICK DATA ANALYSIS ===")
    print(f"Click data shape: {click_data.shape}")
    print(f"Columns: {click_data.columns.tolist()}")
    
    # Check for data leakage - clicked column
    print(f"\nClick column analysis:")
    print(f"clicked unique values: {click_data['clicked'].unique()}")
    print(f"clicked mean: {click_data['clicked'].mean():.4f}")
    
    # Check if clicked is always 1
    if click_data['clicked'].nunique() == 1:
        print("WARNING: clicked has only one unique value!")
        print(f"Value: {click_data['clicked'].iloc[0]}")
    
    # Aggregate click data to post level
    print("\n=== POST-LEVEL AGGREGATION ===")
    post_click_data = click_data.groupby('post_id').agg({
        'clicked': ['mean', 'sum', 'count'],
        'user_id': 'nunique'
    }).reset_index()
    post_click_data.columns = ['post_id', 'click_rate', 'total_clicks', 'total_interactions', 'unique_users']
    
    # Check aggregation results
    print(f"Post-level data shape: {post_click_data.shape}")
    print(f"Click rate - Mean: {post_click_data['click_rate'].mean():.4f}")
    print(f"Click rate - Median: {post_click_data['click_rate'].median():.4f}")
    print(f"Click rate - Min: {post_click_data['click_rate'].min():.4f}")
    print(f"Click rate - Max: {post_click_data['click_rate'].max():.4f}")
    print(f"Total clicks - Mean: {post_click_data['total_clicks'].mean():.2f}")
    print(f"Total interactions - Mean: {post_click_data['total_interactions'].mean():.2f}")
    print(f"Unique users - Mean: {post_click_data['unique_users'].mean():.2f}")
    
    # Check for all-1 click rates
    all_ones = (post_click_data['click_rate'] == 1.0).sum()
    print(f"Posts with click_rate = 1.0: {all_ones:,} ({all_ones/len(post_click_data)*100:.1f}%)")
    
    # Calculate proper CTR
    post_click_data['proper_ctr'] = post_click_data['total_clicks'] / post_click_data['total_interactions']
    
    print(f"\nProper CTR - Mean: {post_click_data['proper_ctr'].mean():.4f}")
    print(f"Proper CTR - Median: {post_click_data['proper_ctr'].median():.4f}")
    print(f"Proper CTR - Min: {post_click_data['proper_ctr'].min():.4f}")
    print(f"Proper CTR - Max: {post_click_data['proper_ctr'].max():.4f}")
    
    # Check for all-1 proper CTR
    all_ones_proper = (post_click_data['proper_ctr'] == 1.0).sum()
    print(f"Posts with proper_ctr = 1.0: {all_ones_proper:,} ({all_ones_proper/len(post_click_data)*100:.1f}%)")
    
    # Merge data
    merged_data = df.merge(post_click_data, left_on='Id', right_on='post_id', how='inner')
    
    # Filter for Cluster 5
    cluster5_data = merged_data[merged_data['cluster_id'] == 5].copy()
    print(f"\nCluster 5 data: {len(cluster5_data):,} posts")
    
    # Check Cluster 5 CTR distribution
    print(f"\n=== CLUSTER 5 CTR ANALYSIS ===")
    print(f"Cluster 5 click_rate - Mean: {cluster5_data['click_rate'].mean():.4f}")
    print(f"Cluster 5 click_rate - Median: {cluster5_data['click_rate'].median():.4f}")
    print(f"Cluster 5 proper_ctr - Mean: {cluster5_data['proper_ctr'].mean():.4f}")
    print(f"Cluster 5 proper_ctr - Median: {cluster5_data['proper_ctr'].median():.4f}")
    
    # Check for data leakage in features
    print(f"\n=== FEATURE LEAKAGE CHECK ===")
    
    # Create features
    cluster5_data['title_length'] = cluster5_data['Title'].fillna('').str.len()
    cluster5_data['body_length'] = cluster5_data['Body'].fillna('').str.len()
    cluster5_data['tags_count'] = cluster5_data['Tags'].fillna('').str.count(',') + 1
    
    # AI content density
    ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural', 'tensorflow', 'pytorch']
    cluster5_data['ai_content_density'] = 0
    for keyword in ai_keywords:
        cluster5_data['ai_content_density'] += cluster5_data['merged_content'].str.contains(keyword, case=False, na=False).astype(int)
    
    # User engagement features (POTENTIAL LEAKAGE)
    cluster5_data['user_engagement'] = cluster5_data['proper_ctr']  # This is the target!
    cluster5_data['post_popularity'] = cluster5_data['unique_users']
    
    # Time features
    if 'CreationDate' in cluster5_data.columns:
        cluster5_data['CreationDate'] = pd.to_datetime(cluster5_data['CreationDate'])
        cluster5_data['day_of_week'] = cluster5_data['CreationDate'].dt.dayofweek
        cluster5_data['hour'] = cluster5_data['CreationDate'].dt.hour
    else:
        cluster5_data['day_of_week'] = 0
        cluster5_data['hour'] = 12
    
    # Check correlations with target
    feature_cols = [
        'title_length', 'body_length', 'tags_count', 'ai_content_density',
        'user_engagement', 'post_popularity', 'day_of_week', 'hour'
    ]
    
    print(f"\nFeature correlations with proper_ctr:")
    for col in feature_cols:
        corr = cluster5_data[col].corr(cluster5_data['proper_ctr'])
        print(f"  {col}: {corr:.4f}")
    
    # Check if user_engagement is identical to proper_ctr
    if cluster5_data['user_engagement'].equals(cluster5_data['proper_ctr']):
        print("\nCRITICAL ISSUE: user_engagement is identical to proper_ctr (target variable)!")
        print("This is causing data leakage!")
    
    # Create treatment/control
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
    
    print(f"\n=== TREATMENT/CONTROL ANALYSIS ===")
    print(f"Treatment group: {cluster5_data['treatment'].sum():,} posts")
    print(f"Control group: {(1-cluster5_data['treatment']).sum():,} posts")
    
    # Check CTR by treatment group
    treatment_ctr = cluster5_data[cluster5_data['treatment'] == 1]['proper_ctr'].mean()
    control_ctr = cluster5_data[cluster5_data['treatment'] == 0]['proper_ctr'].mean()
    print(f"Treatment CTR: {treatment_ctr:.4f}")
    print(f"Control CTR: {control_ctr:.4f}")
    print(f"Raw uplift: {treatment_ctr - control_ctr:.4f}")
    
    # Check if CTR values are all the same
    print(f"\nCTR value distribution:")
    ctr_values = cluster5_data['proper_ctr'].value_counts().head(10)
    for value, count in ctr_values.items():
        print(f"  {value}: {count:,} posts")
    
    # Test model without leakage
    print(f"\n=== MODEL TEST WITHOUT LEAKAGE ===")
    
    # Remove leakage features
    safe_features = [
        'title_length', 'body_length', 'tags_count', 'ai_content_density',
        'post_popularity', 'day_of_week', 'hour'
    ]
    
    # Clean data
    cluster5_data = cluster5_data.dropna(subset=safe_features + ['treatment', 'proper_ctr'])
    for col in safe_features:
        cluster5_data[col] = cluster5_data[col].replace([np.inf, -np.inf], np.nan)
        cluster5_data[col] = cluster5_data[col].fillna(cluster5_data[col].median())
    
    X = cluster5_data[safe_features]
    y_treatment = cluster5_data['treatment']
    y_outcome = cluster5_data['proper_ctr']
    
    # Check if we have variation in the target
    print(f"Target (proper_ctr) statistics:")
    print(f"  Mean: {y_outcome.mean():.4f}")
    print(f"  Std: {y_outcome.std():.4f}")
    print(f"  Min: {y_outcome.min():.4f}")
    print(f"  Max: {y_outcome.max():.4f}")
    print(f"  Unique values: {y_outcome.nunique()}")
    
    if y_outcome.nunique() <= 2:
        print("WARNING: Target has very few unique values!")
        print("This will cause model issues.")
    
    # Test simple linear regression
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_outcome, test_size=0.3, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    print(f"\nSimple linear regression R²: {r2:.4f}")
    
    if r2 > 0.9:
        print("WARNING: Very high R² suggests data leakage or overfitting!")
    
    # Check feature importance
    feature_importance = pd.DataFrame({
        'feature': safe_features,
        'coefficient': model.coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print(f"\nFeature importance (coefficients):")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.4f}")
    
    # Summary of issues
    print(f"\n=== SUMMARY OF ISSUES ===")
    
    issues = []
    
    if click_data['clicked'].nunique() == 1:
        issues.append("Click data has only one unique value")
    
    if all_ones > len(post_click_data) * 0.9:
        issues.append("Most posts have click_rate = 1.0")
    
    if cluster5_data['user_engagement'].equals(cluster5_data['proper_ctr']):
        issues.append("Feature leakage: user_engagement equals target variable")
    
    if y_outcome.nunique() <= 2:
        issues.append("Target variable has insufficient variation")
    
    if r2 > 0.9:
        issues.append("Model R² too high, suggesting overfitting or leakage")
    
    if not issues:
        print("No major issues detected")
    else:
        print("Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    
    # Recommendations
    print(f"\n=== RECOMMENDATIONS ===")
    print("1. Remove user_engagement feature (it's the target variable)")
    print("2. Check if click data is realistic (not all 1s)")
    print("3. Use different target variable if CTR has no variation")
    print("4. Consider using binary outcome (high vs low engagement)")
    print("5. Add more diverse features that don't leak target information")

if __name__ == "__main__":
    main()
