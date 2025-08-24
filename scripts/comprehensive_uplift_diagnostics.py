import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def check_residual_confounding(X, treatment, outcome, feature_names):
    """Check for residual confounding by analyzing treatment-outcome relationships"""
    print("=== RESIDUAL CONFOUNDING ANALYSIS ===")
    
    confounding_analysis = []
    
    for i, feature_name in enumerate(feature_names):
        feature = X[:, i]
        
        # Check if feature is correlated with treatment
        treatment_corr = np.corrcoef(feature, treatment)[0, 1]
        
        # Check if feature is correlated with outcome
        outcome_corr = np.corrcoef(feature, outcome)[0, 1]
        
        # Check if feature moderates treatment effect
        # Create interaction term
        interaction = feature * treatment
        
        # Fit model with interaction
        X_with_interaction = np.column_stack([feature, treatment, interaction])
        interaction_model = LinearRegression()
        interaction_model.fit(X_with_interaction, outcome)
        
        # Extract interaction coefficient
        interaction_coef = interaction_model.coef_[2]
        
        confounding_analysis.append({
            'feature': feature_name,
            'treatment_correlation': treatment_corr,
            'outcome_correlation': outcome_corr,
            'interaction_coefficient': interaction_coef,
            'confounding_risk': abs(treatment_corr * outcome_corr),
            'moderation_risk': abs(interaction_coef)
        })
    
    confounding_df = pd.DataFrame(confounding_analysis)
    confounding_df = confounding_df.sort_values('confounding_risk', ascending=False)
    
    print("Top confounding risk features:")
    print(confounding_df.head(10))
    
    # Identify high-risk features
    high_confounding = confounding_df[confounding_df['confounding_risk'] > 0.1]
    high_moderation = confounding_df[confounding_df['moderation_risk'] > 0.05]
    
    print(f"\nHigh confounding risk features (>0.1): {len(high_confounding)}")
    print(f"High moderation risk features (>0.05): {len(high_moderation)}")
    
    return confounding_df

def check_soft_data_leakage(X, treatment, outcome, feature_names):
    """Check for soft data leakage by analyzing feature-outcome relationships"""
    print("\n=== SOFT DATA LEAKAGE ANALYSIS ===")
    
    leakage_analysis = []
    
    for i, feature_name in enumerate(feature_names):
        feature = X[:, i]
        
        # Check if feature perfectly predicts outcome
        unique_outcomes = np.unique(outcome)
        if len(unique_outcomes) == 2:  # Binary outcome
            # Calculate information value
            feature_bins = pd.qcut(feature, q=10, duplicates='drop', labels=False)
            iv_score = 0
            
            for bin_val in np.unique(feature_bins):
                if pd.isna(bin_val):
                    continue
                bin_mask = feature_bins == bin_val
                bin_outcomes = outcome[bin_mask]
                
                if len(bin_outcomes) > 0:
                    pos_rate = np.mean(bin_outcomes)
                    neg_rate = 1 - pos_rate
                    
                    if pos_rate > 0 and neg_rate > 0:
                        iv_score += (pos_rate - neg_rate) * np.log(pos_rate / neg_rate)
        
        # Check R² when using only this feature
        feature_model = LinearRegression()
        feature_model.fit(feature.reshape(-1, 1), outcome)
        feature_r2 = r2_score(outcome, feature_model.predict(feature.reshape(-1, 1)))
        
        # Check if feature is too predictive
        leakage_risk = feature_r2
        
        leakage_analysis.append({
            'feature': feature_name,
            'feature_r2': feature_r2,
            'leakage_risk': leakage_risk,
            'mean_value': np.mean(feature),
            'std_value': np.std(feature),
            'unique_values': len(np.unique(feature))
        })
    
    leakage_df = pd.DataFrame(leakage_analysis)
    leakage_df = leakage_df.sort_values('leakage_risk', ascending=False)
    
    print("Top leakage risk features:")
    print(leakage_df.head(10))
    
    # Identify high-risk features
    high_leakage = leakage_df[leakage_df['leakage_risk'] > 0.3]
    
    print(f"\nHigh leakage risk features (R² > 0.3): {len(high_leakage)}")
    
    return leakage_df

def analyze_weight_distribution(propensity_scores, treatment):
    """Analyze IPW weight distribution and calculate ESS"""
    print("\n=== IPW WEIGHT DISTRIBUTION ANALYSIS ===")
    
    # Calculate weights with different clipping levels
    p_treatment = treatment.mean()
    
    # Original weights (clipped at 1e-4)
    weights_original = np.where(
        treatment == 1,
        p_treatment / np.clip(propensity_scores, 1e-4, 1 - 1e-4),
        (1 - p_treatment) / np.clip(1 - propensity_scores, 1e-4, 1 - 1e-4)
    )
    
    # Tighter weights (clipped at 0.01)
    weights_tight = np.where(
        treatment == 1,
        p_treatment / np.clip(propensity_scores, 0.01, 0.99),
        (1 - p_treatment) / np.clip(1 - propensity_scores, 0.01, 0.99)
    )
    
    # Calculate ESS (Effective Sample Size)
    def calculate_ess(weights):
        return (np.sum(weights) ** 2) / np.sum(weights ** 2)
    
    ess_original = calculate_ess(weights_original)
    ess_tight = calculate_ess(weights_tight)
    
    # Weight statistics
    weight_stats = {
        'clipping_level': ['1e-4', '0.01'],
        'min_weight': [np.min(weights_original), np.min(weights_tight)],
        'max_weight': [np.max(weights_original), np.max(weights_tight)],
        'mean_weight': [np.mean(weights_original), np.mean(weights_tight)],
        'std_weight': [np.std(weights_original), np.std(weights_tight)],
        'ess': [ess_original, ess_tight],
        'ess_ratio': [ess_original / len(treatment), ess_tight / len(treatment)]
    }
    
    weight_df = pd.DataFrame(weight_stats)
    print("Weight distribution comparison:")
    print(weight_df)
    
    # Check for extreme weights
    extreme_weights_original = np.sum(weights_original > 10)
    extreme_weights_tight = np.sum(weights_tight > 10)
    
    print(f"\nExtreme weights (>10):")
    print(f"  Original clipping: {extreme_weights_original} ({extreme_weights_original/len(treatment)*100:.2f}%)")
    print(f"  Tight clipping: {extreme_weights_tight} ({extreme_weights_tight/len(treatment)*100:.2f}%)")
    
    return weight_df, weights_original, weights_tight

def check_temporal_leakage(X, feature_names):
    """Check for temporal leakage in features"""
    print("\n=== TEMPORAL LEAKAGE ANALYSIS ===")
    
    temporal_analysis = []
    
    # Look for time-related features
    time_features = ['day_of_week', 'hour', 'CreationDate']
    
    for feature_name in feature_names:
        if any(time_feat in feature_name.lower() for time_feat in time_features):
            temporal_analysis.append({
                'feature': feature_name,
                'temporal_risk': 'HIGH',
                'reason': 'Time-related feature'
            })
    
    # Look for aggregated features that might include future information
    aggregation_keywords = ['mean', 'sum', 'count', 'engagement', 'activity', 'popularity']
    
    for feature_name in feature_names:
        if any(keyword in feature_name.lower() for keyword in aggregation_keywords):
            temporal_analysis.append({
                'feature': feature_name,
                'temporal_risk': 'MEDIUM',
                'reason': 'Aggregated feature - check temporal scope'
            })
    
    if temporal_analysis:
        temporal_df = pd.DataFrame(temporal_analysis)
        print("Temporal leakage risk features:")
        print(temporal_df)
    else:
        print("No obvious temporal leakage risks detected")
    
    return temporal_analysis

def check_feature_engineering_leakage(X, feature_names):
    """Check for leakage in feature engineering"""
    print("\n=== FEATURE ENGINEERING LEAKAGE ANALYSIS ===")
    
    leakage_analysis = []
    
    for feature_name in feature_names:
        # Check for features that might be derived from the outcome
        if 'click' in feature_name.lower():
            leakage_analysis.append({
                'feature': feature_name,
                'leakage_type': 'OUTCOME_DERIVED',
                'risk_level': 'HIGH',
                'description': 'Feature derived from click data'
            })
        
        # Check for user-level aggregations
        if 'user_' in feature_name.lower():
            leakage_analysis.append({
                'feature': feature_name,
                'leakage_type': 'USER_AGGREGATION',
                'risk_level': 'MEDIUM',
                'description': 'User-level aggregated feature'
            })
        
        # Check for post-level aggregations
        if 'post_' in feature_name.lower() or 'popularity' in feature_name.lower():
            leakage_analysis.append({
                'feature': feature_name,
                'leakage_type': 'POST_AGGREGATION',
                'risk_level': 'MEDIUM',
                'description': 'Post-level aggregated feature'
            })
    
    if leakage_analysis:
        leakage_df = pd.DataFrame(leakage_analysis)
        print("Feature engineering leakage risks:")
        print(leakage_df)
    else:
        print("No obvious feature engineering leakage risks detected")
    
    return leakage_analysis

def compare_uplift_with_different_clipping(propensity_scores, treatment, outcome, uplift_scores):
    """Compare uplift metrics with different weight clipping levels"""
    print("\n=== UPLIFT COMPARISON WITH DIFFERENT CLIPPING ===")
    
    p_treatment = treatment.mean()
    
    # Calculate weights with different clipping levels
    clipping_levels = [1e-4, 0.01, 0.05, 0.1]
    results = []
    
    for clip_level in clipping_levels:
        clipped_propensity = np.clip(propensity_scores, clip_level, 1 - clip_level)
        
        weights = np.where(
            treatment == 1,
            p_treatment / clipped_propensity,
            (1 - p_treatment) / (1 - clipped_propensity)
        )
        
        # Calculate weighted treatment and control means
        treatment_indices = treatment == 1
        control_indices = treatment == 0
        
        treatment_mean = (outcome[treatment_indices] * weights[treatment_indices]).sum() / weights[treatment_indices].sum()
        control_mean = (outcome[control_indices] * weights[control_indices]).sum() / weights[control_indices].sum()
        
        overall_uplift = treatment_mean - control_mean
        
        # Calculate ESS
        ess = (np.sum(weights) ** 2) / np.sum(weights ** 2)
        
        results.append({
            'clipping_level': clip_level,
            'treatment_mean': treatment_mean,
            'control_mean': control_mean,
            'overall_uplift': overall_uplift,
            'ess': ess,
            'ess_ratio': ess / len(treatment)
        })
    
    results_df = pd.DataFrame(results)
    print("Uplift comparison with different clipping levels:")
    print(results_df)
    
    return results_df

def main():
    """Comprehensive diagnostic analysis for uplift modeling"""
    
    print("=== COMPREHENSIVE UPLIFT DIAGNOSTICS ===")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('optimized_post_clusters.csv')
    click_data = pd.read_csv('user_post_click_samples.csv')
    
    # Data preparation (same as in the main analysis)
    merged_data = df.merge(click_data, left_on='Id', right_on='post_id', how='inner')
    cluster5_data = merged_data[merged_data['cluster_id'] == 5].copy()
    
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
    
    # Create features (same as main analysis)
    cluster5_data['title_length'] = cluster5_data['Title'].fillna('').str.len()
    cluster5_data['body_length'] = cluster5_data['Body'].fillna('').str.len()
    cluster5_data['tags_count'] = cluster5_data['Tags'].fillna('').str.count(',') + 1
    
    ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural', 'tensorflow', 'pytorch']
    cluster5_data['ai_content_density'] = 0
    for keyword in ai_keywords:
        cluster5_data['ai_content_density'] += cluster5_data['merged_content'].str.contains(keyword, case=False, na=False).astype(int)
    
    cluster5_data['post_popularity'] = cluster5_data.groupby('post_id')['post_id'].transform('count')
    cluster5_data['click_volume'] = cluster5_data.groupby('post_id')['is_click'].transform('sum')
    
    if 'CreationDate' in cluster5_data.columns:
        cluster5_data['CreationDate'] = pd.to_datetime(cluster5_data['CreationDate'])
        cluster5_data['day_of_week'] = cluster5_data['CreationDate'].dt.dayofweek
        cluster5_data['hour'] = cluster5_data['CreationDate'].dt.hour
    else:
        cluster5_data['day_of_week'] = 0
        cluster5_data['hour'] = 12
    
    cluster5_data['user_engagement'] = cluster5_data.groupby('user_id')['is_click'].transform('mean')
    cluster5_data['user_activity'] = cluster5_data.groupby('user_id')['user_id'].transform('count')
    
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
    
    # Prepare data
    X = cluster5_data[feature_cols].values  # Convert to numpy array
    y_treatment = cluster5_data['treatment'].values
    y_outcome = cluster5_data['is_click'].values
    
    print(f"Dataset: {len(cluster5_data):,} exposures")
    print(f"Treatment distribution: {y_treatment.mean()*100:.1f}% treatment, {(1-y_treatment.mean())*100:.1f}% control")
    
    # Calculate propensity scores
    propensity_model = LogisticRegression(random_state=42, max_iter=1000)
    propensity_model.fit(X, y_treatment)
    propensity_scores = propensity_model.predict_proba(X)[:, 1]
    propensity_scores = np.clip(propensity_scores, 1e-4, 1 - 1e-4)
    
    # Run diagnostics
    confounding_df = check_residual_confounding(X, y_treatment, y_outcome, feature_cols)
    leakage_df = check_soft_data_leakage(X, y_treatment, y_outcome, feature_cols)
    weight_df, weights_original, weights_tight = analyze_weight_distribution(propensity_scores, y_treatment)
    temporal_analysis = check_temporal_leakage(X, feature_cols)
    feature_leakage = check_feature_engineering_leakage(X, feature_cols)
    
    # Train uplift model for comparison
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_treatment_train, y_treatment_test, y_outcome_train, y_outcome_test = train_test_split(
        X_scaled, y_treatment, y_outcome, test_size=0.3, random_state=42, stratify=y_treatment
    )
    
    treatment_model = LinearRegression()
    control_model = LinearRegression()
    
    treatment_indices = y_treatment_train == 1
    control_indices = y_treatment_train == 0
    
    treatment_model.fit(X_train[treatment_indices], y_outcome_train[treatment_indices])
    control_model.fit(X_train[control_indices], y_outcome_train[control_indices])
    
    treatment_preds = treatment_model.predict(X_test)
    control_preds = control_model.predict(X_test)
    uplift_scores = treatment_preds - control_preds
    
    # Compare uplift with different clipping
    propensity_scores_test = propensity_model.predict_proba(X_test)[:, 1]
    uplift_comparison = compare_uplift_with_different_clipping(
        propensity_scores_test, y_treatment_test, y_outcome_test, uplift_scores
    )
    
    # Export results
    print("\n=== EXPORTING DIAGNOSTIC RESULTS ===")
    
    confounding_df.to_csv('residual_confounding_analysis.csv', index=False)
    leakage_df.to_csv('soft_leakage_analysis.csv', index=False)
    weight_df.to_csv('weight_distribution_analysis.csv', index=False)
    uplift_comparison.to_csv('uplift_clipping_comparison.csv', index=False)
    
    # Create summary report
    with open('comprehensive_diagnostics_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== COMPREHENSIVE UPLIFT DIAGNOSTICS REPORT ===\n\n")
        
        f.write("=== RESIDUAL CONFOUNDING ===\n")
        f.write(f"High confounding risk features: {len(confounding_df[confounding_df['confounding_risk'] > 0.1])}\n")
        f.write(f"High moderation risk features: {len(confounding_df[confounding_df['moderation_risk'] > 0.05])}\n\n")
        
        f.write("=== SOFT DATA LEAKAGE ===\n")
        f.write(f"High leakage risk features (R² > 0.3): {len(leakage_df[leakage_df['leakage_risk'] > 0.3])}\n\n")
        
        f.write("=== WEIGHT DISTRIBUTION ===\n")
        f.write(f"Original ESS ratio: {weight_df.iloc[0]['ess_ratio']:.4f}\n")
        f.write(f"Tight clipping ESS ratio: {weight_df.iloc[1]['ess_ratio']:.4f}\n\n")
        
        f.write("=== UPLIFT COMPARISON ===\n")
        for _, row in uplift_comparison.iterrows():
            f.write(f"Clipping {row['clipping_level']}: Uplift = {row['overall_uplift']:.4f}, ESS ratio = {row['ess_ratio']:.4f}\n")
    
    print("Diagnostic results exported to:")
    print("- residual_confounding_analysis.csv")
    print("- soft_leakage_analysis.csv")
    print("- weight_distribution_analysis.csv")
    print("- uplift_clipping_comparison.csv")
    print("- comprehensive_diagnostics_report.txt")
    
    # Final recommendations
    print("\n=== DIAGNOSTIC RECOMMENDATIONS ===")
    
    high_confounding = len(confounding_df[confounding_df['confounding_risk'] > 0.1])
    high_leakage = len(leakage_df[leakage_df['leakage_risk'] > 0.3])
    ess_ratio = weight_df.iloc[0]['ess_ratio']
    
    if high_confounding > 0:
        print(f"⚠️  {high_confounding} features have high confounding risk - consider stratification")
    
    if high_leakage > 0:
        print(f"⚠️  {high_leakage} features have high leakage risk - review feature engineering")
    
    if ess_ratio < 0.5:
        print(f"⚠️  Low ESS ratio ({ess_ratio:.3f}) - consider tighter weight clipping")
    else:
        print(f"✅ Good ESS ratio ({ess_ratio:.3f})")

if __name__ == "__main__":
    main()
