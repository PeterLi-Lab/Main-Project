import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import beta
import warnings
warnings.filterwarnings('ignore')

def wilson_interval(click_count, exposure_count, confidence=0.95):
    """Calculate Wilson confidence interval for CTR"""
    if exposure_count == 0:
        return 0, 0
    
    p_hat = click_count / exposure_count
    z = stats.norm.ppf((1 + confidence) / 2)
    
    denominator = 1 + z**2 / exposure_count
    centre_adjusted_probability = (p_hat + z * z / (2 * exposure_count)) / denominator
    adjusted_standard_error = z * np.sqrt((p_hat * (1 - p_hat) + z * z / (4 * exposure_count)) / exposure_count) / denominator
    
    lower_bound = centre_adjusted_probability - adjusted_standard_error
    upper_bound = centre_adjusted_probability + adjusted_standard_error
    
    return max(0, lower_bound), min(1, upper_bound)

def beta_smoothing(click_count, exposure_count, alpha=1, beta_param=1):
    """Apply Beta smoothing to CTR"""
    if exposure_count == 0:
        return alpha / (alpha + beta_param)
    
    return (click_count + alpha) / (exposure_count + alpha + beta_param)

def main():
    """Corrected uplift analysis with proper CTR calculation and statistical testing"""
    
    print("=== CORRECTED UPLIFT ANALYSIS V2 ===")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('optimized_post_clusters.csv')
    click_data = pd.read_csv('user_post_click_samples.csv')
    
    print(f"Posts data: {df.shape}")
    print(f"Click data: {click_data.shape}")
    
    # Calculate proper CTR at exposure level (micro-average)
    print("\n=== EXPOSURE-LEVEL CTR CALCULATION ===")
    
    # Merge with cluster data first
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
    
    print(f"\nTreatment/Control distribution:")
    print(f"  Treatment (AI tag): {cluster5_data['treatment'].sum():,} exposures ({cluster5_data['treatment'].mean()*100:.1f}%)")
    print(f"  Control (no AI tag): {(1-cluster5_data['treatment']).sum():,} exposures ({(1-cluster5_data['treatment']).mean()*100:.1f}%)")
    
    # Calculate micro-average CTR (exposure-level)
    treatment_exposures = cluster5_data[cluster5_data['treatment'] == 1]
    control_exposures = cluster5_data[cluster5_data['treatment'] == 0]
    
    treatment_clicks = treatment_exposures['is_click'].sum()
    treatment_total = len(treatment_exposures)
    control_clicks = control_exposures['is_click'].sum()
    control_total = len(control_exposures)
    
    ctr_treatment = treatment_clicks / treatment_total if treatment_total > 0 else 0
    ctr_control = control_clicks / control_total if control_total > 0 else 0
    
    # Calculate uplift (absolute and relative)
    uplift_absolute = ctr_treatment - ctr_control
    uplift_relative = uplift_absolute / ctr_control if ctr_control > 0 else 0
    
    print(f"\n=== MICRO-AVERAGE CTR RESULTS ===")
    print(f"Treatment:")
    print(f"  Total exposures: {treatment_total:,}")
    print(f"  Total clicks: {treatment_clicks:,}")
    print(f"  CTR: {ctr_treatment:.4f} ({ctr_treatment*100:.2f}%)")
    
    print(f"\nControl:")
    print(f"  Total exposures: {control_total:,}")
    print(f"  Total clicks: {control_clicks:,}")
    print(f"  CTR: {ctr_control:.4f} ({ctr_control*100:.2f}%)")
    
    print(f"\nUplift:")
    print(f"  Absolute: {uplift_absolute:.4f} ({uplift_absolute*100:.2f}%)")
    print(f"  Relative: {uplift_relative:.4f} ({uplift_relative*100:.2f}%)")
    
    # Statistical significance test (two-proportion z-test)
    print(f"\n=== STATISTICAL SIGNIFICANCE TEST ===")
    
    # Pooled proportion
    pooled_p = (treatment_clicks + control_clicks) / (treatment_total + control_total)
    
    # Standard error
    se = np.sqrt(pooled_p * (1 - pooled_p) * (1/treatment_total + 1/control_total))
    
    # Z-statistic
    z_stat = (ctr_treatment - ctr_control) / se
    
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    print(f"Z-statistic: {z_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Statistically significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Wilson confidence intervals
    print(f"\n=== WILSON CONFIDENCE INTERVALS ===")
    
    treatment_lower, treatment_upper = wilson_interval(treatment_clicks, treatment_total)
    control_lower, control_upper = wilson_interval(control_clicks, control_total)
    
    print(f"Treatment CTR 95% CI: [{treatment_lower:.4f}, {treatment_upper:.4f}]")
    print(f"Control CTR 95% CI: [{control_lower:.4f}, {control_upper:.4f}]")
    
    # Check if intervals overlap
    intervals_overlap = not (treatment_upper < control_lower or control_upper < treatment_lower)
    print(f"Confidence intervals overlap: {'Yes' if intervals_overlap else 'No'}")
    
    # Macro-average CTR with Beta smoothing (post-level)
    print(f"\n=== MACRO-AVERAGE CTR WITH SMOOTHING ===")
    
    # Aggregate to post level
    post_level_data = cluster5_data.groupby(['post_id', 'treatment']).agg({
        'is_click': ['sum', 'count']
    }).reset_index()
    post_level_data.columns = ['post_id', 'treatment', 'clicks', 'exposures']
    
    # Apply Beta smoothing
    post_level_data['ctr_smoothed'] = post_level_data.apply(
        lambda row: beta_smoothing(row['clicks'], row['exposures']), axis=1
    )
    
    # Calculate macro-average CTR
    treatment_posts = post_level_data[post_level_data['treatment'] == 1]
    control_posts = post_level_data[post_level_data['treatment'] == 0]
    
    macro_ctr_treatment = treatment_posts['ctr_smoothed'].mean()
    macro_ctr_control = control_posts['ctr_smoothed'].mean()
    
    macro_uplift_absolute = macro_ctr_treatment - macro_ctr_control
    macro_uplift_relative = macro_uplift_absolute / macro_ctr_control if macro_ctr_control > 0 else 0
    
    print(f"Macro-average CTR (with Beta smoothing):")
    print(f"  Treatment: {macro_ctr_treatment:.4f} ({macro_ctr_treatment*100:.2f}%)")
    print(f"  Control: {macro_ctr_control:.4f} ({macro_ctr_control*100:.2f}%)")
    print(f"  Absolute uplift: {macro_uplift_absolute:.4f} ({macro_uplift_absolute*100:.2f}%)")
    print(f"  Relative uplift: {macro_uplift_relative:.4f} ({macro_uplift_relative*100:.2f}%)")
    
    # Statistical test for macro-average (t-test)
    t_stat_macro, p_value_macro = stats.ttest_ind(
        treatment_posts['ctr_smoothed'], 
        control_posts['ctr_smoothed']
    )
    
    print(f"Macro-average t-test:")
    print(f"  T-statistic: {t_stat_macro:.4f}")
    print(f"  P-value: {p_value_macro:.6f}")
    print(f"  Statistically significant: {'Yes' if p_value_macro < 0.05 else 'No'}")
    
    # Check consistency between micro and macro averages
    print(f"\n=== CONSISTENCY CHECK ===")
    micro_direction = "positive" if uplift_absolute > 0 else "negative"
    macro_direction = "positive" if macro_uplift_absolute > 0 else "negative"
    
    print(f"Micro-average direction: {micro_direction}")
    print(f"Macro-average direction: {macro_direction}")
    print(f"Directions consistent: {'Yes' if micro_direction == macro_direction else 'No'}")
    
    # Export results
    print(f"\n=== EXPORTING RESULTS ===")
    
    results = {
        'metric': [
            'Micro_CTR_Treatment', 'Micro_CTR_Control', 'Micro_Uplift_Absolute', 'Micro_Uplift_Relative',
            'Macro_CTR_Treatment', 'Macro_CTR_Control', 'Macro_Uplift_Absolute', 'Macro_Uplift_Relative',
            'Z_statistic', 'P_value_micro', 'T_statistic_macro', 'P_value_macro',
            'Treatment_Exposures', 'Control_Exposures', 'Treatment_Clicks', 'Control_Clicks'
        ],
        'value': [
            ctr_treatment, ctr_control, uplift_absolute, uplift_relative,
            macro_ctr_treatment, macro_ctr_control, macro_uplift_absolute, macro_uplift_relative,
            z_stat, p_value, t_stat_macro, p_value_macro,
            treatment_total, control_total, treatment_clicks, control_clicks
        ]
    }
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('corrected_uplift_metrics_v2.csv', index=False)
    
    # Create summary report
    with open('corrected_uplift_report_v2.txt', 'w', encoding='utf-8') as f:
        f.write("=== CORRECTED UPLIFT ANALYSIS REPORT V2 ===\n\n")
        f.write(f"Dataset: {len(cluster5_data):,} exposures from Cluster 5\n")
        f.write(f"Treatment exposures: {treatment_total:,}\n")
        f.write(f"Control exposures: {control_total:,}\n\n")
        
        f.write("=== MICRO-AVERAGE RESULTS ===\n")
        f.write(f"Treatment CTR: {ctr_treatment:.4f} ({ctr_treatment*100:.2f}%)\n")
        f.write(f"Control CTR: {ctr_control:.4f} ({ctr_control*100:.2f}%)\n")
        f.write(f"Absolute uplift: {uplift_absolute:.4f} ({uplift_absolute*100:.2f}%)\n")
        f.write(f"Relative uplift: {uplift_relative:.4f} ({uplift_relative*100:.2f}%)\n")
        f.write(f"Z-statistic: {z_stat:.4f}\n")
        f.write(f"P-value: {p_value:.6f}\n")
        f.write(f"Statistically significant: {'Yes' if p_value < 0.05 else 'No'}\n\n")
        
        f.write("=== MACRO-AVERAGE RESULTS ===\n")
        f.write(f"Treatment CTR: {macro_ctr_treatment:.4f} ({macro_ctr_treatment*100:.2f}%)\n")
        f.write(f"Control CTR: {macro_ctr_control:.4f} ({macro_ctr_control*100:.2f}%)\n")
        f.write(f"Absolute uplift: {macro_uplift_absolute:.4f} ({macro_uplift_absolute*100:.2f}%)\n")
        f.write(f"Relative uplift: {macro_uplift_relative:.4f} ({macro_uplift_relative*100:.2f}%)\n")
        f.write(f"T-statistic: {t_stat_macro:.4f}\n")
        f.write(f"P-value: {p_value_macro:.6f}\n")
        f.write(f"Statistically significant: {'Yes' if p_value_macro < 0.05 else 'No'}\n\n")
        
        f.write("=== CONSISTENCY CHECK ===\n")
        f.write(f"Micro-average direction: {micro_direction}\n")
        f.write(f"Macro-average direction: {macro_direction}\n")
        f.write(f"Directions consistent: {'Yes' if micro_direction == macro_direction else 'No'}\n")
    
    print("Results exported to:")
    print("- corrected_uplift_metrics_v2.csv")
    print("- corrected_uplift_report_v2.txt")
    
    # Final summary
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Micro-average Analysis:")
    print(f"  Treatment CTR: {ctr_treatment:.4f} ({ctr_treatment*100:.2f}%)")
    print(f"  Control CTR: {ctr_control:.4f} ({ctr_control*100:.2f}%)")
    print(f"  Uplift: {uplift_absolute:.4f} ({uplift_absolute*100:.2f}%)")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    print(f"\nMacro-average Analysis:")
    print(f"  Treatment CTR: {macro_ctr_treatment:.4f} ({macro_ctr_treatment*100:.2f}%)")
    print(f"  Control CTR: {macro_ctr_control:.4f} ({macro_ctr_control*100:.2f}%)")
    print(f"  Uplift: {macro_uplift_absolute:.4f} ({macro_uplift_absolute*100:.2f}%)")
    print(f"  Significant: {'Yes' if p_value_macro < 0.05 else 'No'}")
    
    print(f"\nConsistency: {'Yes' if micro_direction == macro_direction else 'No'}")

if __name__ == "__main__":
    main()
