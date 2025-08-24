import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_qini_score(y_true, uplift_scores, treatment, n_bins=10):
    """Calculate Qini/AUUC score"""
    df = pd.DataFrame({
        'y_true': y_true,
        'uplift_scores': uplift_scores,
        'treatment': treatment
    })
    
    # Sort by uplift scores
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
    
    # Calculate Qini score
    qini_score = 0
    for i in range(1, len(bin_metrics)):
        x1 = i - 1
        x2 = i
        y1 = bin_metrics[i-1]['cumulative_uplift']
        y2 = bin_metrics[i]['cumulative_uplift']
        qini_score += (x2 - x1) * (y1 + y2) / 2
    
    return qini_score, bin_metrics

def calculate_uplift_at_k(y_true, uplift_scores, treatment, k_percent=20):
    """Calculate uplift at top k%"""
    df = pd.DataFrame({
        'y_true': y_true,
        'uplift_scores': uplift_scores,
        'treatment': treatment
    })
    
    # Sort by uplift scores
    df = df.sort_values('uplift_scores', ascending=False).reset_index(drop=True)
    
    # Get top k%
    k_count = int(len(df) * k_percent / 100)
    top_k = df.head(k_count)
    
    treatment_top_k = top_k[top_k['treatment'] == 1]
    control_top_k = top_k[top_k['treatment'] == 0]
    
    treatment_rate = treatment_top_k['y_true'].mean() if len(treatment_top_k) > 0 else 0
    control_rate = control_top_k['y_true'].mean() if len(control_top_k) > 0 else 0
    
    uplift_at_k = treatment_rate - control_rate
    
    return uplift_at_k, treatment_rate, control_rate, len(treatment_top_k), len(control_top_k)

def calculate_baseline_metrics(y_true, treatment):
    """Calculate baseline metrics for random targeting and all-users targeting"""
    
    # Random targeting baseline
    # Simulate random selection by shuffling treatment assignments
    np.random.seed(42)
    random_treatment = np.random.permutation(treatment)
    
    treatment_random = y_true[random_treatment == 1]
    control_random = y_true[random_treatment == 0]
    
    random_uplift = treatment_random.mean() - control_random.mean() if len(treatment_random) > 0 and len(control_random) > 0 else 0
    
    # All-users targeting baseline
    # This represents treating everyone vs treating no one
    overall_mean = y_true.mean()
    treatment_mean = y_true[treatment == 1].mean() if len(y_true[treatment == 1]) > 0 else 0
    control_mean = y_true[treatment == 0].mean() if len(y_true[treatment == 0]) > 0 else 0
    
    all_users_uplift = treatment_mean - control_mean
    
    return random_uplift, all_users_uplift

def segment_users_by_uplift_sensitivity(X, y_true, uplift_scores, treatment, n_segments=4):
    """Segment users based on uplift sensitivity"""
    
    # Create user features for segmentation
    user_features = pd.DataFrame(X)
    user_features['uplift_sensitivity'] = uplift_scores
    user_features['y_true'] = y_true
    user_features['treatment'] = treatment
    
    # Segment by uplift sensitivity
    try:
        user_features['segment'] = pd.qcut(user_features['uplift_sensitivity'], 
                                         q=n_segments, labels=['Low', 'Medium-Low', 'Medium-High', 'High'],
                                         duplicates='drop')
    except ValueError:
        # If not enough unique values, use fewer segments
        unique_values = user_features['uplift_sensitivity'].nunique()
        if unique_values >= 2:
            user_features['segment'] = pd.qcut(user_features['uplift_sensitivity'], 
                                             q=2, labels=['Low', 'High'],
                                             duplicates='drop')
        else:
            user_features['segment'] = 'All'
    
    # Calculate segment metrics
    segment_metrics = []
    for segment in user_features['segment'].unique():
        segment_data = user_features[user_features['segment'] == segment]
        
        treatment_segment = segment_data[segment_data['treatment'] == 1]
        control_segment = segment_data[segment_data['treatment'] == 0]
        
        treatment_mean = treatment_segment['y_true'].mean() if len(treatment_segment) > 0 else 0
        control_mean = control_segment['y_true'].mean() if len(control_segment) > 0 else 0
        segment_uplift = treatment_mean - control_mean
        
        segment_metrics.append({
            'segment': segment,
            'count': len(segment_data),
            'treatment_mean': treatment_mean,
            'control_mean': control_mean,
            'uplift': segment_uplift,
            'uplift_percentage': (segment_uplift / control_mean * 100) if control_mean > 0 else 0
        })
    
    return segment_metrics

def main():
    """Complete uplift evaluation with all requested metrics"""
    
    print("=== COMPLETE UPLIFT EVALUATION ===")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('optimized_post_clusters.csv')
    click_data = pd.read_csv('user_post_click_samples.csv')
    
    print(f"Posts data: {df.shape}")
    print(f"Click data: {click_data.shape}")
    
    # Calculate proper CTR at exposure level
    print("\n=== DATA PREPARATION ===")
    
    # Merge with cluster data
    merged_data = df.merge(click_data, left_on='Id', right_on='post_id', how='inner')
    
    # Filter for Cluster 5 (AI content cluster)
    cluster5_data = merged_data[merged_data['cluster_id'] == 5].copy()
    print(f"Cluster 5 data: {len(cluster5_data):,} exposures")
    
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
    
    # Engagement features
    cluster5_data['post_popularity'] = cluster5_data.groupby('post_id')['post_id'].transform('count')
    cluster5_data['click_volume'] = cluster5_data.groupby('post_id')['is_click'].transform('sum')
    
    # Time features
    if 'CreationDate' in cluster5_data.columns:
        cluster5_data['CreationDate'] = pd.to_datetime(cluster5_data['CreationDate'])
        cluster5_data['day_of_week'] = cluster5_data['CreationDate'].dt.dayofweek
        cluster5_data['hour'] = cluster5_data['CreationDate'].dt.hour
    else:
        cluster5_data['day_of_week'] = 0
        cluster5_data['hour'] = 12
    
    # User features
    cluster5_data['user_engagement'] = cluster5_data.groupby('user_id')['is_click'].transform('mean')
    cluster5_data['user_activity'] = cluster5_data.groupby('user_id')['user_id'].transform('count')
    
    # Select features
    feature_cols = [
        'title_length', 'body_length', 'tags_count', 'ai_content_density',
        'post_popularity', 'click_volume', 'day_of_week', 'hour',
        'user_engagement', 'user_activity'
    ]
    
    # Clean data
    cluster5_data = cluster5_data.dropna(subset=feature_cols + ['treatment', 'is_click'])
    for col in feature_cols:
        cluster5_data[col] = cluster5_data[col].replace([np.inf, -np.inf], np.nan)
        cluster5_data[col] = cluster5_data[col].fillna(cluster5_data[col].median())
    
    # Prepare data for modeling
    X = cluster5_data[feature_cols]
    y_treatment = cluster5_data['treatment']
    y_outcome = cluster5_data['is_click']  # Use binary click as outcome
    
    print(f"Final dataset: {len(cluster5_data):,} exposures")
    print(f"Treatment distribution: {y_treatment.mean()*100:.1f}% treatment, {(1-y_treatment.mean())*100:.1f}% control")
    
    # Train Two-Model Uplift Model
    print("\n=== TRAINING UPLIFT MODEL ===")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
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
    
    # Calculate evaluation metrics
    print("\n=== EVALUATION METRICS ===")
    
    # R² for treatment and control models
    treatment_indices_test = y_treatment_test == 1
    control_indices_test = y_treatment_test == 0
    
    r2_treatment = r2_score(y_outcome_test[treatment_indices_test], treatment_preds[treatment_indices_test])
    r2_control = r2_score(y_outcome_test[control_indices_test], control_preds[control_indices_test])
    
    print(f"R² (treatment head): {r2_treatment:.4f}")
    print(f"R² (control head): {r2_control:.4f}")
    
    # Qini/AUUC score
    qini_score, bin_metrics = calculate_qini_score(y_outcome_test, uplift_scores, y_treatment_test)
    print(f"Qini/AUUC Score: {qini_score:.4f}")
    
    # Uplift@top20%
    uplift_at_20, treatment_rate_20, control_rate_20, treatment_count_20, control_count_20 = calculate_uplift_at_k(
        y_outcome_test, uplift_scores, y_treatment_test, k_percent=20
    )
    print(f"Uplift@top20%: {uplift_at_20:.4f} ({uplift_at_20*100:.2f}%)")
    print(f"Top 20% - Treatment rate: {treatment_rate_20:.4f}, Control rate: {control_rate_20:.4f}")
    print(f"Top 20% - Treatment users: {treatment_count_20}, Control users: {control_count_20}")
    
    # Baseline metrics
    random_uplift, all_users_uplift = calculate_baseline_metrics(y_outcome_test, y_treatment_test)
    print(f"\nBaseline metrics:")
    print(f"Random targeting uplift: {random_uplift:.4f} ({random_uplift*100:.2f}%)")
    print(f"All-users targeting uplift: {all_users_uplift:.4f} ({all_users_uplift*100:.2f}%)")
    
    # Segment differences
    print(f"\n=== SEGMENT ANALYSIS ===")
    segment_metrics = segment_users_by_uplift_sensitivity(X_test, y_outcome_test, uplift_scores, y_treatment_test)
    
    for segment in segment_metrics:
        print(f"{segment['segment']} segment:")
        print(f"  Count: {segment['count']:,}")
        print(f"  Treatment mean: {segment['treatment_mean']:.4f}")
        print(f"  Control mean: {segment['control_mean']:.4f}")
        print(f"  Uplift: {segment['uplift']:.4f} ({segment['uplift_percentage']:.2f}%)")
    
    # Find high and low uplift segments
    if len(segment_metrics) >= 2:
        high_uplift_segment = max(segment_metrics, key=lambda x: x['uplift'])
        low_uplift_segment = min(segment_metrics, key=lambda x: x['uplift'])
        
        print(f"\nSegment differences:")
        print(f"High-uplift cohort ({high_uplift_segment['segment']}): {high_uplift_segment['uplift']:.4f} ({high_uplift_segment['uplift_percentage']:.2f}%)")
        print(f"Low-uplift cohort ({low_uplift_segment['segment']}): {low_uplift_segment['uplift']:.4f} ({low_uplift_segment['uplift_percentage']:.2f}%)")
        print(f"Difference: {high_uplift_segment['uplift'] - low_uplift_segment['uplift']:.4f}")
    
    # Export results
    print(f"\n=== EXPORTING RESULTS ===")
    
    results = {
        'metric': [
            'R2_treatment', 'R2_control', 'Qini_AUUC', 'Uplift_top20',
            'Random_targeting_uplift', 'All_users_targeting_uplift',
            'Treatment_rate_top20', 'Control_rate_top20',
            'Treatment_count_top20', 'Control_count_top20'
        ],
        'value': [
            r2_treatment, r2_control, qini_score, uplift_at_20,
            random_uplift, all_users_uplift,
            treatment_rate_20, control_rate_20,
            treatment_count_20, control_count_20
        ]
    }
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('complete_uplift_evaluation_metrics.csv', index=False)
    
    # Export segment data
    segment_df = pd.DataFrame(segment_metrics)
    segment_df.to_csv('uplift_segment_analysis.csv', index=False)
    
    # Create summary report
    with open('complete_uplift_evaluation_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== COMPLETE UPLIFT EVALUATION REPORT ===\n\n")
        f.write(f"Dataset: {len(cluster5_data):,} exposures from Cluster 5\n")
        f.write(f"Test set: {len(y_outcome_test):,} exposures\n\n")
        
        f.write("=== MODEL PERFORMANCE ===\n")
        f.write(f"R² (treatment head): {r2_treatment:.4f}\n")
        f.write(f"R² (control head): {r2_control:.4f}\n")
        f.write(f"Qini/AUUC Score: {qini_score:.4f}\n")
        f.write(f"Uplift@top20%: {uplift_at_20:.4f} ({uplift_at_20*100:.2f}%)\n\n")
        
        f.write("=== BASELINE METRICS ===\n")
        f.write(f"Random targeting uplift: {random_uplift:.4f} ({random_uplift*100:.2f}%)\n")
        f.write(f"All-users targeting uplift: {all_users_uplift:.4f} ({all_users_uplift*100:.2f}%)\n\n")
        
        f.write("=== SEGMENT ANALYSIS ===\n")
        for segment in segment_metrics:
            f.write(f"{segment['segment']} segment:\n")
            f.write(f"  Count: {segment['count']:,}\n")
            f.write(f"  Treatment mean: {segment['treatment_mean']:.4f}\n")
            f.write(f"  Control mean: {segment['control_mean']:.4f}\n")
            f.write(f"  Uplift: {segment['uplift']:.4f} ({segment['uplift_percentage']:.2f}%)\n\n")
    
    print("Results exported to:")
    print("- complete_uplift_evaluation_metrics.csv")
    print("- uplift_segment_analysis.csv")
    print("- complete_uplift_evaluation_report.txt")
    
    # Final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Model Performance:")
    print(f"  R² (treatment): {r2_treatment:.4f}")
    print(f"  R² (control): {r2_control:.4f}")
    print(f"  Qini/AUUC: {qini_score:.4f}")
    print(f"  Uplift@top20%: {uplift_at_20:.4f} ({uplift_at_20*100:.2f}%)")
    
    print(f"\nBaseline Comparison:")
    print(f"  Random targeting: {random_uplift:.4f} ({random_uplift*100:.2f}%)")
    print(f"  All-users targeting: {all_users_uplift:.4f} ({all_users_uplift*100:.2f}%)")
    
    if len(segment_metrics) >= 2:
        high_uplift = max(segment_metrics, key=lambda x: x['uplift'])
        low_uplift = min(segment_metrics, key=lambda x: x['uplift'])
        print(f"\nSegment Differences:")
        print(f"  High-uplift cohort: {high_uplift['uplift']:.4f} ({high_uplift['uplift_percentage']:.2f}%)")
        print(f"  Low-uplift cohort: {low_uplift['uplift']:.4f} ({low_uplift['uplift_percentage']:.2f}%)")

if __name__ == "__main__":
    main()
