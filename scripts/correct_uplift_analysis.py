import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def main():
    """Uplift analysis with correct CTR definition"""
    
    print("=== CORRECT UPLIFT ANALYSIS ===")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('optimized_post_clusters.csv')
    click_data = pd.read_csv('user_post_click_samples.csv')
    
    print(f"Posts data: {df.shape}")
    print(f"Click data: {click_data.shape}")
    
    # Calculate proper CTR
    print("\nCalculating proper CTR...")
    post_ctr = click_data.groupby('post_id').agg({
        'is_click': ['sum', 'count']
    }).reset_index()
    post_ctr.columns = ['post_id', 'total_clicks', 'total_exposures']
    post_ctr['ctr'] = post_ctr['total_clicks'] / post_ctr['total_exposures']
    
    print(f"Post-level CTR statistics:")
    print(f"  Mean CTR: {post_ctr['ctr'].mean():.4f} ({post_ctr['ctr'].mean()*100:.2f}%)")
    print(f"  Median CTR: {post_ctr['ctr'].median():.4f} ({post_ctr['ctr'].median()*100:.2f}%)")
    print(f"  Posts with CTR = 1.0: {(post_ctr['ctr'] == 1.0).sum():,} ({(post_ctr['ctr'] == 1.0).mean()*100:.1f}%)")
    
    # Filter posts with sufficient exposures
    min_exposures = 5
    sufficient_exposure_posts = post_ctr[post_ctr['total_exposures'] >= min_exposures]
    print(f"\nPosts with ≥{min_exposures} exposures:")
    print(f"  Count: {len(sufficient_exposure_posts):,} ({len(sufficient_exposure_posts)/len(post_ctr)*100:.1f}%)")
    print(f"  Mean CTR: {sufficient_exposure_posts['ctr'].mean():.4f} ({sufficient_exposure_posts['ctr'].mean()*100:.2f}%)")
    print(f"  Median CTR: {sufficient_exposure_posts['ctr'].median():.4f} ({sufficient_exposure_posts['ctr'].median()*100:.2f}%)")
    
    # Merge with cluster data
    merged_data = df.merge(sufficient_exposure_posts, left_on='Id', right_on='post_id', how='inner')
    
    # Filter for Cluster 5 (AI content cluster)
    cluster5_data = merged_data[merged_data['cluster_id'] == 5].copy()
    print(f"\nCluster 5 data with sufficient exposures: {len(cluster5_data):,} posts")
    
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
    
    print(f"\nTreatment/Control distribution:")
    print(f"  Treatment (AI tag): {cluster5_data['treatment'].sum():,} posts ({cluster5_data['treatment'].mean()*100:.1f}%)")
    print(f"  Control (no AI tag): {(1-cluster5_data['treatment']).sum():,} posts ({(1-cluster5_data['treatment']).mean()*100:.1f}%)")
    
    # Check CTR by treatment group
    treatment_ctr = cluster5_data[cluster5_data['treatment'] == 1]['ctr'].mean()
    control_ctr = cluster5_data[cluster5_data['treatment'] == 0]['ctr'].mean()
    raw_uplift = treatment_ctr - control_ctr
    
    print(f"\nRaw CTR comparison:")
    print(f"  Treatment CTR: {treatment_ctr:.4f} ({treatment_ctr*100:.2f}%)")
    print(f"  Control CTR: {control_ctr:.4f} ({control_ctr*100:.2f}%)")
    print(f"  Raw uplift: {raw_uplift:.4f} ({raw_uplift*100:.2f}%)")
    
    # Statistical significance test
    treatment_ctrs = cluster5_data[cluster5_data['treatment'] == 1]['ctr']
    control_ctrs = cluster5_data[cluster5_data['treatment'] == 0]['ctr']
    
    t_stat, p_value = stats.ttest_ind(treatment_ctrs, control_ctrs)
    print(f"  T-test: t={t_stat:.4f}, p={p_value:.4f}")
    print(f"  Statistically significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Create features for modeling
    print("\nCreating features...")
    
    # Content features
    cluster5_data['title_length'] = cluster5_data['Title'].fillna('').str.len()
    cluster5_data['body_length'] = cluster5_data['Body'].fillna('').str.len()
    cluster5_data['tags_count'] = cluster5_data['Tags'].fillna('').str.count(',') + 1
    
    # AI content density
    ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural', 'tensorflow', 'pytorch']
    cluster5_data['ai_content_density'] = 0
    for keyword in ai_keywords:
        cluster5_data['ai_content_density'] += cluster5_data['merged_content'].str.contains(keyword, case=False, na=False).astype(int)
    
    # Engagement features (NOT the target variable)
    cluster5_data['post_popularity'] = cluster5_data['total_exposures']
    cluster5_data['click_volume'] = cluster5_data['total_clicks']
    
    # Time features
    if 'CreationDate' in cluster5_data.columns:
        cluster5_data['CreationDate'] = pd.to_datetime(cluster5_data['CreationDate'])
        cluster5_data['day_of_week'] = cluster5_data['CreationDate'].dt.dayofweek
        cluster5_data['hour'] = cluster5_data['CreationDate'].dt.hour
    else:
        cluster5_data['day_of_week'] = 0
        cluster5_data['hour'] = 12
    
    # Select features (NO data leakage)
    feature_cols = [
        'title_length', 'body_length', 'tags_count', 'ai_content_density',
        'post_popularity', 'click_volume', 'day_of_week', 'hour'
    ]
    
    # Clean data
    cluster5_data = cluster5_data.dropna(subset=feature_cols + ['treatment', 'ctr'])
    for col in feature_cols:
        cluster5_data[col] = cluster5_data[col].replace([np.inf, -np.inf], np.nan)
        cluster5_data[col] = cluster5_data[col].fillna(cluster5_data[col].median())
    
    # Prepare data for modeling
    X = cluster5_data[feature_cols]
    y_treatment = cluster5_data['treatment']
    y_outcome = cluster5_data['ctr']
    
    print(f"\nFinal dataset: {len(cluster5_data):,} posts")
    print(f"Target (CTR) statistics:")
    print(f"  Mean: {y_outcome.mean():.4f} ({y_outcome.mean()*100:.2f}%)")
    print(f"  Std: {y_outcome.std():.4f}")
    print(f"  Min: {y_outcome.min():.4f} ({y_outcome.min()*100:.2f}%)")
    print(f"  Max: {y_outcome.max():.4f} ({y_outcome.max()*100:.2f}%)")
    print(f"  Unique values: {y_outcome.nunique()}")
    
    # Train Two-Model Uplift Model
    print("\n=== TRAINING TWO-MODEL UPLIFT MODEL ===")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_treatment_train, y_treatment_test, y_outcome_train, y_outcome_test = train_test_split(
        X_scaled, y_treatment, y_outcome, test_size=0.3, random_state=42, stratify=y_treatment
    )
    
    # Train treatment model
    from sklearn.linear_model import LinearRegression
    treatment_model = LinearRegression()
    treatment_indices = y_treatment_train == 1
    treatment_model.fit(X_train[treatment_indices], y_outcome_train[treatment_indices])
    
    # Train control model
    control_model = LinearRegression()
    control_indices = y_treatment_train == 0
    control_model.fit(X_train[control_indices], y_outcome_train[control_indices])
    
    # Make predictions
    treatment_preds = treatment_model.predict(X_test)
    control_preds = control_model.predict(X_test)
    
    # Calculate uplift scores
    uplift_scores = treatment_preds - control_preds
    
    # Calculate AUC metrics
    print("\n=== MODEL PERFORMANCE METRICS ===")
    
    # R² for treatment model
    treatment_indices_test = y_treatment_test == 1
    r2_treatment = r2_score(y_outcome_test[treatment_indices_test], treatment_preds[treatment_indices_test])
    
    # R² for control model
    control_indices_test = y_treatment_test == 0
    r2_control = r2_score(y_outcome_test[control_indices_test], control_preds[control_indices_test])
    
    print(f"R² (treatment head): {r2_treatment:.4f}")
    print(f"R² (control head): {r2_control:.4f}")
    
    # Check if R² is reasonable
    if r2_treatment > 0.9 or r2_control > 0.9:
        print("WARNING: Very high R² suggests data leakage or overfitting!")
    
    # Calculate Qini score
    print("\n=== QINI SCORE CALCULATION ===")
    
    # Create dataframe for Qini calculation
    df_test = pd.DataFrame({
        'y_true': y_outcome_test,
        'uplift_scores': uplift_scores,
        'treatment': y_treatment_test
    })
    
    # Sort by uplift scores
    df_test = df_test.sort_values('uplift_scores', ascending=False).reset_index(drop=True)
    
    # Create bins
    n_bins = 10
    bin_size = len(df_test) // n_bins
    df_test['bin'] = (df_test.index // bin_size).astype(int)
    df_test.loc[df_test['bin'] >= n_bins, 'bin'] = n_bins - 1
    
    # Calculate metrics per bin
    bin_metrics = []
    cumulative_treatment = 0
    cumulative_control = 0
    cumulative_treatment_sum = 0
    cumulative_control_sum = 0
    
    for bin_num in range(n_bins):
        bin_data = df_test[df_test['bin'] == bin_num]
        
        if len(bin_data) == 0:
            continue
            
        # Treatment group in this bin
        treatment_bin = bin_data[bin_data['treatment'] == 1]
        control_bin = bin_data[bin_data['treatment'] == 0]
        
        # Counts and sums
        treatment_count = len(treatment_bin)
        control_count = len(control_bin)
        treatment_sum = treatment_bin['y_true'].sum()
        control_sum = control_bin['y_true'].sum()
        
        # Cumulative counts and sums
        cumulative_treatment += treatment_count
        cumulative_control += control_count
        cumulative_treatment_sum += treatment_sum
        cumulative_control_sum += control_sum
        
        # Calculate means
        treatment_mean = treatment_sum / treatment_count if treatment_count > 0 else 0
        control_mean = control_sum / control_count if control_count > 0 else 0
        cumulative_treatment_mean = cumulative_treatment_sum / cumulative_treatment if cumulative_treatment > 0 else 0
        cumulative_control_mean = cumulative_control_sum / cumulative_control if cumulative_control > 0 else 0
        
        # Uplift
        uplift = treatment_mean - control_mean
        cumulative_uplift = cumulative_treatment_mean - cumulative_control_mean
        
        bin_metrics.append({
            'bin': bin_num,
            'treatment_count': treatment_count,
            'control_count': control_count,
            'treatment_mean': treatment_mean,
            'control_mean': control_mean,
            'uplift': uplift,
            'cumulative_treatment_mean': cumulative_treatment_mean,
            'cumulative_control_mean': cumulative_control_mean,
            'cumulative_uplift': cumulative_uplift
        })
    
    # Calculate Qini score
    qini_score = 0
    for i in range(1, len(bin_metrics)):
        x1 = i - 1
        x2 = i
        y1 = bin_metrics[i-1]['cumulative_uplift']
        y2 = bin_metrics[i]['cumulative_uplift']
        qini_score += (x2 - x1) * (y1 + y2) / 2
    
    print(f"Qini/AUUC Score: {qini_score:.4f}")
    
    # Calculate Uplift@Top20%
    print("\n=== UPLIFT@TOP20% ===")
    
    k_percent = 20
    k_count = int(len(df_test) * k_percent / 100)
    top_k = df_test.head(k_count)
    
    treatment_top_k = top_k[top_k['treatment'] == 1]
    control_top_k = top_k[top_k['treatment'] == 0]
    
    treatment_rate = treatment_top_k['y_true'].mean() if len(treatment_top_k) > 0 else 0
    control_rate = control_top_k['y_true'].mean() if len(control_top_k) > 0 else 0
    
    uplift_at_20 = treatment_rate - control_rate
    print(f"Uplift@top20%: {uplift_at_20:.4f} ({uplift_at_20*100:.2f}%)")
    print(f"Top 20% - Treatment users: {len(treatment_top_k)}, Control users: {len(control_top_k)}")
    
    # Export results
    print("\n=== EXPORTING RESULTS ===")
    
    results = {
        'metric': [
            'R2_treatment', 'R2_control', 'Qini_AUUC', 'Uplift_top20',
            'Raw_uplift', 'Treatment_CTR', 'Control_CTR', 'P_value'
        ],
        'value': [
            r2_treatment, r2_control, qini_score, uplift_at_20,
            raw_uplift, treatment_ctr, control_ctr, p_value
        ]
    }
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('correct_uplift_metrics.csv', index=False)
    
    # Create summary report
    with open('correct_uplift_report.txt', 'w') as f:
        f.write("=== CORRECT UPLIFT ANALYSIS REPORT ===\n\n")
        f.write(f"Dataset: {len(cluster5_data):,} posts from Cluster 5 (≥{min_exposures} exposures)\n")
        f.write(f"Test set: {len(y_outcome_test):,} posts\n\n")
        
        f.write("=== RAW COMPARISON ===\n")
        f.write(f"Treatment CTR: {treatment_ctr:.4f} ({treatment_ctr*100:.2f}%)\n")
        f.write(f"Control CTR: {control_ctr:.4f} ({control_ctr*100:.2f}%)\n")
        f.write(f"Raw uplift: {raw_uplift:.4f} ({raw_uplift*100:.2f}%)\n")
        f.write(f"P-value: {p_value:.4f}\n")
        f.write(f"Statistically significant: {'Yes' if p_value < 0.05 else 'No'}\n\n")
        
        f.write("=== MODEL PERFORMANCE ===\n")
        f.write(f"R² (treatment head): {r2_treatment:.4f}\n")
        f.write(f"R² (control head): {r2_control:.4f}\n")
        f.write(f"Qini/AUUC Score: {qini_score:.4f}\n")
        f.write(f"Uplift@top20%: {uplift_at_20:.4f} ({uplift_at_20*100:.2f}%)\n")
    
    print("Results exported to:")
    print("- correct_uplift_metrics.csv")
    print("- correct_uplift_report.txt")
    
    # Final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Raw Analysis:")
    print(f"  Treatment CTR: {treatment_ctr:.4f} ({treatment_ctr*100:.2f}%)")
    print(f"  Control CTR: {control_ctr:.4f} ({control_ctr*100:.2f}%)")
    print(f"  Uplift: {raw_uplift:.4f} ({raw_uplift*100:.2f}%)")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    print(f"\nModel Performance:")
    print(f"  R² (treatment): {r2_treatment:.4f}")
    print(f"  R² (control): {r2_control:.4f}")
    print(f"  Qini/AUUC: {qini_score:.4f}")
    print(f"  Uplift@top20%: {uplift_at_20:.4f} ({uplift_at_20*100:.2f}%)")

if __name__ == "__main__":
    main()
