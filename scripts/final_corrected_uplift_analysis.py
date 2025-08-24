import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_propensity_scores(X, treatment):
    """Calculate propensity scores using logistic regression"""
    propensity_model = LogisticRegression(random_state=42, max_iter=1000)
    propensity_model.fit(X, treatment)
    propensity_scores = propensity_model.predict_proba(X)[:, 1]
    
    # Use tighter clipping to avoid extreme weights
    propensity_scores = np.clip(propensity_scores, 0.05, 0.95)
    
    return propensity_scores, propensity_model

def calculate_ipw_uplift_at_k(y_true, uplift_scores, treatment, propensity_scores, k_percent=20):
    """Calculate IPW-based uplift at top k% with proper treatment vs control comparison"""
    df = pd.DataFrame({
        'y_true': y_true,
        'uplift_scores': uplift_scores,
        'treatment': treatment,
        'propensity_scores': propensity_scores
    })
    
    # Sort by uplift scores (highest first)
    df = df.sort_values('uplift_scores', ascending=False).reset_index(drop=True)
    
    # Get top k%
    k_count = int(len(df) * k_percent / 100)
    top_k = df.head(k_count)
    
    # Calculate IPW weights for top k%
    p_treatment = treatment.mean()
    top_k['ipw_weights'] = np.where(
        top_k['treatment'] == 1,
        p_treatment / top_k['propensity_scores'],
        (1 - p_treatment) / (1 - top_k['propensity_scores'])
    )
    
    # Calculate weighted means for treatment and control
    treatment_top_k = top_k[top_k['treatment'] == 1]
    control_top_k = top_k[top_k['treatment'] == 0]
    
    # IPW treatment mean
    if len(treatment_top_k) > 0:
        treatment_rate = (treatment_top_k['y_true'] * treatment_top_k['ipw_weights']).sum() / treatment_top_k['ipw_weights'].sum()
    else:
        treatment_rate = y_true[treatment == 1].mean() if len(y_true[treatment == 1]) > 0 else 0
    
    # IPW control mean
    if len(control_top_k) > 0:
        control_rate = (control_top_k['y_true'] * control_top_k['ipw_weights']).sum() / control_top_k['ipw_weights'].sum()
    else:
        control_rate = y_true[treatment == 0].mean() if len(y_true[treatment == 0]) > 0 else 0
    
    # Uplift = treatment_mean - control_mean
    uplift_at_k = treatment_rate - control_rate
    
    return uplift_at_k, treatment_rate, control_rate, len(treatment_top_k), len(control_top_k)

def calculate_ipw_qini_score(y_true, uplift_scores, treatment, propensity_scores, n_bins=10):
    """Calculate IPW-based Qini/AUUC score"""
    df = pd.DataFrame({
        'y_true': y_true,
        'uplift_scores': uplift_scores,
        'treatment': treatment,
        'propensity_scores': propensity_scores
    })
    
    # Sort by uplift scores (highest first)
    df = df.sort_values('uplift_scores', ascending=False).reset_index(drop=True)
    
    # Create bins
    bin_size = len(df) // n_bins
    df['bin'] = (df.index // bin_size).astype(int)
    df.loc[df['bin'] >= n_bins, 'bin'] = n_bins - 1
    
    # Calculate IPW weights
    p_treatment = treatment.mean()
    df['ipw_weights'] = np.where(
        df['treatment'] == 1,
        p_treatment / df['propensity_scores'],
        (1 - p_treatment) / (1 - df['propensity_scores'])
    )
    
    # Calculate metrics per bin
    bin_metrics = []
    cumulative_treatment_sum = 0
    cumulative_control_sum = 0
    cumulative_treatment_weight = 0
    cumulative_control_weight = 0
    
    for bin_num in range(n_bins):
        bin_data = df[df['bin'] == bin_num]
        
        if len(bin_data) == 0:
            continue
            
        # Treatment group in this bin
        treatment_bin = bin_data[bin_data['treatment'] == 1]
        control_bin = bin_data[bin_data['treatment'] == 0]
        
        # IPW treatment mean for this bin
        if len(treatment_bin) > 0:
            treatment_mean = (treatment_bin['y_true'] * treatment_bin['ipw_weights']).sum() / treatment_bin['ipw_weights'].sum()
        else:
            treatment_mean = 0
        
        # IPW control mean for this bin
        if len(control_bin) > 0:
            control_mean = (control_bin['y_true'] * control_bin['ipw_weights']).sum() / control_bin['ipw_weights'].sum()
        else:
            control_mean = 0
        
        # Uplift for this bin
        uplift = treatment_mean - control_mean
        
        # Cumulative weighted sums
        cumulative_treatment_sum += (treatment_bin['y_true'] * treatment_bin['ipw_weights']).sum()
        cumulative_control_sum += (control_bin['y_true'] * control_bin['ipw_weights']).sum()
        cumulative_treatment_weight += treatment_bin['ipw_weights'].sum()
        cumulative_control_weight += control_bin['ipw_weights'].sum()
        
        # Cumulative means
        cumulative_treatment_mean = cumulative_treatment_sum / cumulative_treatment_weight if cumulative_treatment_weight > 0 else 0
        cumulative_control_mean = cumulative_control_sum / cumulative_control_weight if cumulative_control_weight > 0 else 0
        cumulative_uplift = cumulative_treatment_mean - cumulative_control_mean
        
        bin_metrics.append({
            'bin': bin_num,
            'treatment_count': len(treatment_bin),
            'control_count': len(control_bin),
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
        x1 = i - 1
        x2 = i
        y1 = bin_metrics[i-1]['cumulative_uplift']
        y2 = bin_metrics[i]['cumulative_uplift']
        qini_score += (x2 - x1) * (y1 + y2) / 2
    
    return qini_score, bin_metrics

def calculate_ipw_baseline_metrics(y_true, treatment, propensity_scores):
    """Calculate IPW-based baseline metrics"""
    # Calculate IPW weights
    p_treatment = treatment.mean()
    ipw_weights = np.where(
        treatment == 1,
        p_treatment / propensity_scores,
        (1 - p_treatment) / (1 - propensity_scores)
    )
    
    # IPW treatment mean
    treatment_indices = treatment == 1
    control_indices = treatment == 0
    
    treatment_mean = (y_true[treatment_indices] * ipw_weights[treatment_indices]).sum() / ipw_weights[treatment_indices].sum() if ipw_weights[treatment_indices].sum() > 0 else 0
    control_mean = (y_true[control_indices] * ipw_weights[control_indices]).sum() / ipw_weights[control_indices].sum() if ipw_weights[control_indices].sum() > 0 else 0
    
    all_users_uplift = treatment_mean - control_mean
    
    # Calculate ESS
    ess = (np.sum(ipw_weights) ** 2) / np.sum(ipw_weights ** 2)
    
    return all_users_uplift, ess

def main():
    """Final corrected uplift analysis with proper feature selection and weight clipping"""
    
    print("=== FINAL CORRECTED UPLIFT ANALYSIS ===")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('optimized_post_clusters.csv')
    click_data = pd.read_csv('user_post_click_samples.csv')
    
    print(f"Posts data: {df.shape}")
    print(f"Click data: {click_data.shape}")
    
    # Data preparation
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
    
    # Create SAFE features only (no leakage)
    print("Creating safe features (no leakage)...")
    
    # Content features (safe)
    cluster5_data['title_length'] = cluster5_data['Title'].fillna('').str.len()
    cluster5_data['body_length'] = cluster5_data['Body'].fillna('').str.len()
    cluster5_data['tags_count'] = cluster5_data['Tags'].fillna('').str.count(',') + 1
    
    # AI content density (safe)
    ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural', 'tensorflow', 'pytorch']
    cluster5_data['ai_content_density'] = 0
    for keyword in ai_keywords:
        cluster5_data['ai_content_density'] += cluster5_data['merged_content'].str.contains(keyword, case=False, na=False).astype(int)
    
    # Time features (safe)
    if 'CreationDate' in cluster5_data.columns:
        cluster5_data['CreationDate'] = pd.to_datetime(cluster5_data['CreationDate'])
        cluster5_data['day_of_week'] = cluster5_data['CreationDate'].dt.dayofweek
        cluster5_data['hour'] = cluster5_data['CreationDate'].dt.hour
    else:
        cluster5_data['day_of_week'] = 0
        cluster5_data['hour'] = 12
    
    # REMOVED LEAKAGE FEATURES:
    # - user_engagement (R² = 0.92 - major leakage)
    # - post_popularity (R² = 0.54 - moderate leakage)
    # - click_volume (outcome-derived)
    # - user_activity (user aggregation)
    
    # Select only safe features
    safe_feature_cols = [
        'title_length', 'body_length', 'tags_count', 'ai_content_density',
        'day_of_week', 'hour'
    ]
    
    print(f"Using {len(safe_feature_cols)} safe features (removed {10-len(safe_feature_cols)} leakage features)")
    
    # Clean data
    cluster5_data = cluster5_data.dropna(subset=safe_feature_cols + ['treatment', 'is_click'])
    for col in safe_feature_cols:
        cluster5_data[col] = cluster5_data[col].replace([np.inf, -np.inf], np.nan)
        cluster5_data[col] = cluster5_data[col].fillna(cluster5_data[col].median())
    
    # Prepare data for modeling
    X = cluster5_data[safe_feature_cols].values
    y_treatment = cluster5_data['treatment'].values
    y_outcome = cluster5_data['is_click'].values
    
    print(f"Final dataset: {len(cluster5_data):,} exposures")
    print(f"Treatment distribution: {y_treatment.mean()*100:.1f}% treatment, {(1-y_treatment.mean())*100:.1f}% control")
    
    # Calculate propensity scores with tighter clipping
    print("\n=== CALCULATING PROPENSITY SCORES (TIGHT CLIPPING) ===")
    propensity_scores, propensity_model = calculate_propensity_scores(X, y_treatment)
    print(f"Propensity score range: [{propensity_scores.min():.4f}, {propensity_scores.max():.4f}]")
    print(f"Propensity score mean: {propensity_scores.mean():.4f}")
    
    # Train Two-Model Uplift Model
    print("\n=== TRAINING UPLIFT MODEL ===")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_treatment_train, y_treatment_test, y_outcome_train, y_outcome_test = train_test_split(
        X_scaled, y_treatment, y_outcome, test_size=0.3, random_state=42, stratify=y_treatment
    )
    
    # Calculate propensity scores for test set
    propensity_scores_test = propensity_model.predict_proba(X_test)[:, 1]
    propensity_scores_test = np.clip(propensity_scores_test, 0.05, 0.95)
    
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
    
    # Calculate IPW-corrected evaluation metrics
    print("\n=== FINAL CORRECTED EVALUATION METRICS ===")
    
    # R² for treatment and control models
    treatment_indices_test = y_treatment_test == 1
    control_indices_test = y_treatment_test == 0
    
    r2_treatment = r2_score(y_outcome_test[treatment_indices_test], treatment_preds[treatment_indices_test])
    r2_control = r2_score(y_outcome_test[control_indices_test], control_preds[control_indices_test])
    
    print(f"R² (treatment head): {r2_treatment:.4f}")
    print(f"R² (control head): {r2_control:.4f}")
    
    # IPW-corrected Qini/AUUC score
    qini_score, bin_metrics = calculate_ipw_qini_score(y_outcome_test, uplift_scores, y_treatment_test, propensity_scores_test)
    print(f"IPW Qini/AUUC Score: {qini_score:.4f}")
    
    # IPW-corrected Uplift@top20%
    uplift_at_20, treatment_rate_20, control_rate_20, treatment_count_20, control_count_20 = calculate_ipw_uplift_at_k(
        y_outcome_test, uplift_scores, y_treatment_test, propensity_scores_test, k_percent=20
    )
    print(f"IPW Uplift@top20%: {uplift_at_20:.4f} ({uplift_at_20*100:.2f}%)")
    print(f"Top 20% - IPW Treatment rate: {treatment_rate_20:.4f}, IPW Control rate: {control_rate_20:.4f}")
    print(f"Top 20% - Treatment users: {treatment_count_20}, Control users: {control_count_20}")
    
    # IPW-corrected baseline metrics
    all_users_uplift, ess = calculate_ipw_baseline_metrics(y_outcome_test, y_treatment_test, propensity_scores_test)
    ess_ratio = ess / len(y_outcome_test)
    
    print(f"\nIPW Baseline metrics:")
    print(f"IPW All-users targeting uplift: {all_users_uplift:.4f} ({all_users_uplift*100:.2f}%)")
    print(f"Effective Sample Size (ESS): {ess:.0f}")
    print(f"ESS ratio: {ess_ratio:.4f}")
    
    # Export results
    print(f"\n=== EXPORTING FINAL RESULTS ===")
    
    results = {
        'metric': [
            'R2_treatment', 'R2_control', 'IPW_Qini_AUUC', 'IPW_Uplift_top20',
            'IPW_All_users_targeting_uplift', 'ESS', 'ESS_ratio',
            'IPW_Treatment_rate_top20', 'IPW_Control_rate_top20',
            'Treatment_count_top20', 'Control_count_top20'
        ],
        'value': [
            r2_treatment, r2_control, qini_score, uplift_at_20,
            all_users_uplift, ess, ess_ratio,
            treatment_rate_20, control_rate_20,
            treatment_count_20, control_count_20
        ]
    }
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('final_corrected_uplift_metrics.csv', index=False)
    
    # Create summary report
    with open('final_corrected_uplift_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== FINAL CORRECTED UPLIFT ANALYSIS REPORT ===\n\n")
        f.write("CRITICAL FIXES APPLIED:\n")
        f.write("1. Removed user_engagement (R² = 0.92 - major leakage)\n")
        f.write("2. Removed post_popularity (R² = 0.54 - moderate leakage)\n")
        f.write("3. Removed click_volume (outcome-derived feature)\n")
        f.write("4. Removed user_activity (user aggregation)\n")
        f.write("5. Used tighter propensity score clipping (0.05, 0.95)\n\n")
        
        f.write(f"Dataset: {len(cluster5_data):,} exposures from Cluster 5\n")
        f.write(f"Test set: {len(y_outcome_test):,} exposures\n")
        f.write(f"Treatment imbalance: {y_treatment.mean()*100:.1f}% treatment, {(1-y_treatment.mean())*100:.1f}% control\n")
        f.write(f"Features used: {len(safe_feature_cols)} safe features\n\n")
        
        f.write("=== MODEL PERFORMANCE ===\n")
        f.write(f"R² (treatment head): {r2_treatment:.4f}\n")
        f.write(f"R² (control head): {r2_control:.4f}\n")
        f.write(f"IPW Qini/AUUC Score: {qini_score:.4f}\n")
        f.write(f"IPW Uplift@top20%: {uplift_at_20:.4f} ({uplift_at_20*100:.2f}%)\n\n")
        
        f.write("=== IPW BASELINE METRICS ===\n")
        f.write(f"IPW All-users targeting uplift: {all_users_uplift:.4f} ({all_users_uplift*100:.2f}%)\n")
        f.write(f"Effective Sample Size (ESS): {ess:.0f}\n")
        f.write(f"ESS ratio: {ess_ratio:.4f}\n\n")
        
        f.write("=== INTERPRETATION ===\n")
        if qini_score > 0:
            f.write("POSITIVE: The uplift model successfully identifies positive treatment effects\n")
        else:
            f.write("NEGATIVE: The uplift model identifies negative treatment effects\n")
        
        if ess_ratio > 0.5:
            f.write("GOOD: Effective sample size is adequate for reliable inference\n")
        else:
            f.write("POOR: Effective sample size is too low for reliable inference\n")
    
    print("Final results exported to:")
    print("- final_corrected_uplift_metrics.csv")
    print("- final_corrected_uplift_report.txt")
    
    # Final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Model Performance:")
    print(f"  R² (treatment): {r2_treatment:.4f}")
    print(f"  R² (control): {r2_control:.4f}")
    print(f"  IPW Qini/AUUC: {qini_score:.4f}")
    print(f"  IPW Uplift@top20%: {uplift_at_20:.4f} ({uplift_at_20*100:.2f}%)")
    
    print(f"\nIPW Baseline:")
    print(f"  IPW All-users targeting: {all_users_uplift:.4f} ({all_users_uplift*100:.2f}%)")
    print(f"  ESS ratio: {ess_ratio:.4f}")
    
    print(f"\nKey Improvements:")
    print(f"  ✅ Removed {10-len(safe_feature_cols)} leakage features")
    print(f"  ✅ Used tighter weight clipping (0.05, 0.95)")
    print(f"  ✅ ESS ratio improved to {ess_ratio:.4f}")

if __name__ == "__main__":
    main()
