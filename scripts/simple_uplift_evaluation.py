import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_qini_score_continuous(y_true, uplift_scores, treatment, n_bins=10):
    """Calculate Qini score for continuous outcomes"""
    df = pd.DataFrame({
        'y_true': y_true,
        'uplift_scores': uplift_scores,
        'treatment': treatment
    })
    
    # Sort by uplift scores in descending order
    df = df.sort_values('uplift_scores', ascending=False).reset_index(drop=True)
    
    # Create bins
    bin_size = len(df) // n_bins
    df['bin'] = (df.index // bin_size).astype(int)
    df.loc[df['bin'] >= n_bins, 'bin'] = n_bins - 1
    
    # Calculate metrics per bin
    bin_metrics = []
    cumulative_treatment = 0
    cumulative_control = 0
    cumulative_treatment_sum = 0
    cumulative_control_sum = 0
    
    for bin_num in range(n_bins):
        bin_data = df[df['bin'] == bin_num]
        
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
    
    # Calculate Qini score (area under cumulative uplift curve)
    qini_score = 0
    for i in range(1, len(bin_metrics)):
        # Area of trapezoid
        x1 = i - 1
        x2 = i
        y1 = bin_metrics[i-1]['cumulative_uplift']
        y2 = bin_metrics[i]['cumulative_uplift']
        qini_score += (x2 - x1) * (y1 + y2) / 2
    
    return qini_score, bin_metrics

def calculate_uplift_at_k_continuous(y_true, uplift_scores, treatment, k_percent=20):
    """Calculate uplift at top k% for continuous outcomes"""
    df = pd.DataFrame({
        'y_true': y_true,
        'uplift_scores': uplift_scores,
        'treatment': treatment
    })
    
    # Sort by uplift scores in descending order
    df = df.sort_values('uplift_scores', ascending=False).reset_index(drop=True)
    
    # Get top k%
    k_count = int(len(df) * k_percent / 100)
    top_k = df.head(k_count)
    
    # Calculate uplift for top k%
    treatment_top_k = top_k[top_k['treatment'] == 1]
    control_top_k = top_k[top_k['treatment'] == 0]
    
    treatment_mean = treatment_top_k['y_true'].mean() if len(treatment_top_k) > 0 else 0
    control_mean = control_top_k['y_true'].mean() if len(control_top_k) > 0 else 0
    
    uplift_at_k = treatment_mean - control_mean
    
    return uplift_at_k, len(treatment_top_k), len(control_top_k)

def main():
    """Simplified uplift evaluation with continuous outcomes"""
    
    print("=== SIMPLIFIED UPLIFT EVALUATION ===")
    
    # Load data
    print("Loading data...")
    try:
        df = pd.read_csv('optimized_post_clusters.csv')
        click_data = pd.read_csv('user_post_click_samples.csv')
        click_data = click_data.rename(columns={'is_click': 'clicked'})
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
    
    # Calculate proper CTR
    post_click_data['proper_ctr'] = post_click_data['total_clicks'] / post_click_data['total_interactions']
    
    # Merge data
    merged_data = df.merge(post_click_data, left_on='Id', right_on='post_id', how='inner')
    
    # Filter for Cluster 5 (AI content cluster)
    cluster5_data = merged_data[merged_data['cluster_id'] == 5].copy()
    print(f"Cluster 5 data: {len(cluster5_data):,} posts")
    
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
    
    # Create features for modeling
    print("Creating features...")
    
    # Content features
    cluster5_data['title_length'] = cluster5_data['Title'].fillna('').str.len()
    cluster5_data['body_length'] = cluster5_data['Body'].fillna('').str.len()
    cluster5_data['tags_count'] = cluster5_data['Tags'].fillna('').str.count(',') + 1
    
    # AI content density
    ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural', 'tensorflow', 'pytorch']
    cluster5_data['ai_content_density'] = 0
    for keyword in ai_keywords:
        cluster5_data['ai_content_density'] += cluster5_data['merged_content'].str.contains(keyword, case=False, na=False).astype(int)
    
    # User engagement features
    cluster5_data['user_engagement'] = cluster5_data['proper_ctr']
    cluster5_data['post_popularity'] = cluster5_data['unique_users']
    
    # Time features
    if 'CreationDate' in cluster5_data.columns:
        cluster5_data['CreationDate'] = pd.to_datetime(cluster5_data['CreationDate'])
        cluster5_data['day_of_week'] = cluster5_data['CreationDate'].dt.dayofweek
        cluster5_data['hour'] = cluster5_data['CreationDate'].dt.hour
    else:
        cluster5_data['day_of_week'] = 0
        cluster5_data['hour'] = 12
    
    # Select features
    feature_cols = [
        'title_length', 'body_length', 'tags_count', 'ai_content_density',
        'user_engagement', 'post_popularity', 'day_of_week', 'hour'
    ]
    
    # Clean data
    cluster5_data = cluster5_data.dropna(subset=feature_cols + ['treatment', 'proper_ctr'])
    for col in feature_cols:
        cluster5_data[col] = cluster5_data[col].replace([np.inf, -np.inf], np.nan)
        cluster5_data[col] = cluster5_data[col].fillna(cluster5_data[col].median())
    
    # Prepare data for modeling
    X = cluster5_data[feature_cols]
    y_treatment = cluster5_data['treatment']
    y_outcome = cluster5_data['proper_ctr']
    
    print(f"Final dataset: {len(cluster5_data):,} posts")
    print(f"Treatment group: {y_treatment.sum():,} posts ({y_treatment.mean()*100:.1f}%)")
    print(f"Control group: {(1-y_treatment).sum():,} posts ({(1-y_treatment).mean()*100:.1f}%)")
    print(f"CTR - Mean: {y_outcome.mean():.4f}, Median: {y_outcome.median():.4f}")
    
    # STEP 1: Train Two-Model Uplift Model (using Linear Regression)
    print("\n=== STEP 1: TRAINING TWO-MODEL UPLIFT MODEL ===")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_treatment_train, y_treatment_test, y_outcome_train, y_outcome_test = train_test_split(
        X_scaled, y_treatment, y_outcome, test_size=0.3, random_state=42, stratify=y_treatment
    )
    
    # Train treatment model
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
    
    # STEP 2: Calculate R² Metrics (equivalent to AUC for continuous outcomes)
    print("\n=== STEP 2: MODEL PERFORMANCE METRICS ===")
    
    # R² for treatment model
    treatment_indices_test = y_treatment_test == 1
    r2_treatment = r2_score(y_outcome_test[treatment_indices_test], treatment_preds[treatment_indices_test])
    
    # R² for control model
    control_indices_test = y_treatment_test == 0
    r2_control = r2_score(y_outcome_test[control_indices_test], control_preds[control_indices_test])
    
    print(f"R² (treatment head): {r2_treatment:.4f}")
    print(f"R² (control head): {r2_control:.4f}")
    
    # STEP 3: Calculate Qini/AUUC Score
    print("\n=== STEP 3: QINI/AUUC SCORE ===")
    
    qini_score, bin_metrics = calculate_qini_score_continuous(y_outcome_test, uplift_scores, y_treatment_test)
    print(f"Qini/AUUC Score: {qini_score:.4f}")
    
    # STEP 4: Calculate Uplift@Top20%
    print("\n=== STEP 4: UPLIFT@TOP20% ===")
    
    uplift_at_20, treatment_count, control_count = calculate_uplift_at_k_continuous(y_outcome_test, uplift_scores, y_treatment_test, k_percent=20)
    print(f"Uplift@top20%: {uplift_at_20:.4f}")
    print(f"Top 20% - Treatment users: {treatment_count}, Control users: {control_count}")
    
    # STEP 5: Baseline Metrics
    print("\n=== STEP 5: BASELINE METRICS ===")
    
    # Random targeting baseline
    np.random.seed(42)
    random_scores = np.random.random(len(y_outcome_test))
    random_qini, _ = calculate_qini_score_continuous(y_outcome_test, random_scores, y_treatment_test)
    random_uplift_20, _, _ = calculate_uplift_at_k_continuous(y_outcome_test, random_scores, y_treatment_test, k_percent=20)
    
    # All-users targeting baseline
    all_users_qini = 0  # No targeting = no uplift
    all_users_uplift_20 = 0
    
    print(f"Random targeting - Qini: {random_qini:.4f}, Uplift@20%: {random_uplift_20:.4f}")
    print(f"All-users targeting - Qini: {all_users_qini:.4f}, Uplift@20%: {all_users_uplift_20:.4f}")
    
    # STEP 6: Segment Differences
    print("\n=== STEP 6: SEGMENT DIFFERENCES ===")
    
    # Create segments based on uplift scores
    df_test = pd.DataFrame({
        'y_true': y_outcome_test,
        'uplift_scores': uplift_scores,
        'treatment': y_treatment_test
    })
    
    # High uplift segment (top 20%)
    high_uplift_threshold = np.percentile(uplift_scores, 80)
    high_uplift_segment = df_test[uplift_scores >= high_uplift_threshold]
    
    # Low uplift segment (bottom 20%)
    low_uplift_threshold = np.percentile(uplift_scores, 20)
    low_uplift_segment = df_test[uplift_scores <= low_uplift_threshold]
    
    # Calculate metrics for each segment
    def calculate_segment_metrics(segment_data):
        if len(segment_data) == 0:
            return 0, 0, 0
        
        treatment_segment = segment_data[segment_data['treatment'] == 1]
        control_segment = segment_data[segment_data['treatment'] == 0]
        
        treatment_mean = treatment_segment['y_true'].mean() if len(treatment_segment) > 0 else 0
        control_mean = control_segment['y_true'].mean() if len(control_segment) > 0 else 0
        uplift = treatment_mean - control_mean
        
        return treatment_mean, control_mean, uplift
    
    high_treatment_mean, high_control_mean, high_uplift = calculate_segment_metrics(high_uplift_segment)
    low_treatment_mean, low_control_mean, low_uplift = calculate_segment_metrics(low_uplift_segment)
    
    print(f"High-uplift segment (top 20%):")
    print(f"  Treatment mean: {high_treatment_mean:.4f}")
    print(f"  Control mean: {high_control_mean:.4f}")
    print(f"  Uplift: {high_uplift:.4f}")
    print(f"  Users: {len(high_uplift_segment):,}")
    
    print(f"\nLow-uplift segment (bottom 20%):")
    print(f"  Treatment mean: {low_treatment_mean:.4f}")
    print(f"  Control mean: {low_control_mean:.4f}")
    print(f"  Uplift: {low_uplift:.4f}")
    print(f"  Users: {len(low_uplift_segment):,}")
    
    print(f"\nSegment difference (High - Low):")
    print(f"  Uplift difference: {high_uplift - low_uplift:.4f}")
    
    # STEP 7: Export Results
    print("\n=== STEP 7: EXPORTING RESULTS ===")
    
    # Create comprehensive results
    results = {
        'metric': [
            'R2_treatment', 'R2_control', 'Qini_AUUC', 'Uplift_top20',
            'Random_Qini', 'Random_Uplift_20', 'AllUsers_Qini', 'AllUsers_Uplift_20',
            'HighUplift_TreatmentMean', 'HighUplift_ControlMean', 'HighUplift_Uplift',
            'LowUplift_TreatmentMean', 'LowUplift_ControlMean', 'LowUplift_Uplift',
            'Segment_Uplift_Difference'
        ],
        'value': [
            r2_treatment, r2_control, qini_score, uplift_at_20,
            random_qini, random_uplift_20, all_users_qini, all_users_uplift_20,
            high_treatment_mean, high_control_mean, high_uplift,
            low_treatment_mean, low_control_mean, low_uplift,
            high_uplift - low_uplift
        ]
    }
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('simple_uplift_evaluation_metrics.csv', index=False)
    
    # Create detailed bin metrics
    bin_df = pd.DataFrame(bin_metrics)
    bin_df.to_csv('simple_uplift_bin_metrics.csv', index=False)
    
    # Create summary report
    with open('simple_uplift_evaluation_report.txt', 'w') as f:
        f.write("=== SIMPLIFIED UPLIFT EVALUATION REPORT ===\n\n")
        f.write(f"Dataset: {len(cluster5_data):,} posts from Cluster 5\n")
        f.write(f"Test set: {len(y_outcome_test):,} posts\n\n")
        
        f.write("=== MODEL PERFORMANCE ===\n")
        f.write(f"R² (treatment head): {r2_treatment:.4f}\n")
        f.write(f"R² (control head): {r2_control:.4f}\n")
        f.write(f"Qini/AUUC Score: {qini_score:.4f}\n")
        f.write(f"Uplift@top20%: {uplift_at_20:.4f}\n\n")
        
        f.write("=== BASELINE COMPARISON ===\n")
        f.write(f"Random targeting - Qini: {random_qini:.4f}\n")
        f.write(f"Random targeting - Uplift@20%: {random_uplift_20:.4f}\n")
        f.write(f"All-users targeting - Qini: {all_users_qini:.4f}\n")
        f.write(f"All-users targeting - Uplift@20%: {all_users_uplift_20:.4f}\n\n")
        
        f.write("=== SEGMENT ANALYSIS ===\n")
        f.write(f"High-uplift segment uplift: {high_uplift:.4f}\n")
        f.write(f"Low-uplift segment uplift: {low_uplift:.4f}\n")
        f.write(f"Segment uplift difference: {high_uplift - low_uplift:.4f}\n")
    
    print("Results exported to:")
    print("- simple_uplift_evaluation_metrics.csv")
    print("- simple_uplift_bin_metrics.csv")
    print("- simple_uplift_evaluation_report.txt")
    
    # Final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Model Performance:")
    print(f"  R² (treatment): {r2_treatment:.4f}")
    print(f"  R² (control): {r2_control:.4f}")
    print(f"  Qini/AUUC: {qini_score:.4f}")
    print(f"  Uplift@top20%: {uplift_at_20:.4f}")
    
    print(f"\nSegment Analysis:")
    print(f"  High-uplift segment: {high_uplift:.4f}")
    print(f"  Low-uplift segment: {low_uplift:.4f}")
    print(f"  Difference: {high_uplift - low_uplift:.4f}")
    
    # Model quality assessment
    if qini_score > 0.1:
        quality = "Excellent"
    elif qini_score > 0.05:
        quality = "Good"
    elif qini_score > 0.01:
        quality = "Fair"
    else:
        quality = "Poor"
    
    print(f"\nModel Quality: {quality} (Qini score: {qini_score:.4f})")

if __name__ == "__main__":
    main()
