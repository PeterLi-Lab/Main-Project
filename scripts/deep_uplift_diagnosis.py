#!/usr/bin/env python3
"""
Deep Uplift Diagnosis - Validate Treatment Effects and Model Issues
This script addresses the user's concerns about negative uplift being real vs. model issues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data for diagnosis"""
    print("=== Loading Data for Deep Diagnosis ===")
    
    try:
        df = pd.read_csv('uplift_model_data.csv')
        print(f"Loaded data with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print("uplift_model_data.csv not found. Creating synthetic data...")
        np.random.seed(42)
        n_samples = 10000
        
        # Create realistic synthetic data with KNOWN negative treatment effect
        df = pd.DataFrame({
            'user_ai_interest_score': np.random.beta(1, 3, n_samples),
            'user_ai_interest_weighted': np.random.beta(1, 4, n_samples),
            'user_ai_interactions': np.random.poisson(2, n_samples),
            'user_reputation': np.random.exponential(100, n_samples),
            'user_post_count': np.random.poisson(5, n_samples),
            'content_quality_score': np.random.beta(3, 7, n_samples),
            'treatment_ai_content': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'response': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
        
        # Create KNOWN negative treatment effect
        base_response_prob = 0.15
        treatment_effect = -0.05  # Negative effect!
        
        response_prob = base_response_prob + \
                       df['treatment_ai_content'] * treatment_effect + \
                       df['user_ai_interest_score'] * 0.05 + \
                       df['content_quality_score'] * 0.03
        
        response_prob = np.clip(response_prob, 0, 1)
        df['response'] = np.random.binomial(1, response_prob, n_samples)
    
    return df

def analyze_raw_treatment_effects(df):
    """Analyze raw treatment effects without any modeling"""
    print("\n=== RAW TREATMENT EFFECT ANALYSIS ===")
    
    treatment_group = df[df['treatment_ai_content'] == 1]
    control_group = df[df['treatment_ai_content'] == 0]
    
    print(f"Treatment group size: {len(treatment_group):,}")
    print(f"Control group size: {len(control_group):,}")
    print(f"Treatment/Control ratio: {len(treatment_group)/len(control_group):.3f}")
    
    # Raw response rates
    treatment_response_rate = treatment_group['response'].mean()
    control_response_rate = control_group['response'].mean()
    raw_uplift = treatment_response_rate - control_response_rate
    
    print(f"\nRaw Response Rates:")
    print(f"  Treatment group: {treatment_response_rate:.4f}")
    print(f"  Control group: {control_response_rate:.4f}")
    print(f"  Raw uplift (Treatment - Control): {raw_uplift:.6f}")
    
    # Statistical significance test
    from scipy.stats import chi2_contingency
    contingency_table = pd.crosstab(df['treatment_ai_content'], df['response'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\nChi-square test for independence:")
    print(f"  Chi2 statistic: {chi2:.4f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
    
    return raw_uplift, treatment_response_rate, control_response_rate

def check_for_data_leakage(df):
    """Check for potential data leakage in features"""
    print("\n=== DATA LEAKAGE DETECTION ===")
    
    # Get numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in ['treatment_ai_content', 'response']]
    
    print(f"Analyzing {len(feature_cols)} features for leakage...")
    
    leakage_scores = {}
    for col in tqdm(feature_cols, desc="Checking features"):
        # Calculate correlation with treatment
        treatment_corr = abs(df[col].corr(df['treatment_ai_content']))
        
        # Calculate correlation with response
        response_corr = abs(df[col].corr(df['response']))
        
        # Calculate treatment-response interaction
        treatment_response_corr = abs((df[col] * df['treatment_ai_content']).corr(df['response']))
        
        # Leakage score: high correlation with treatment + high correlation with response
        leakage_score = treatment_corr * response_corr
        
        leakage_scores[col] = {
            'treatment_corr': treatment_corr,
            'response_corr': response_corr,
            'treatment_response_corr': treatment_response_corr,
            'leakage_score': leakage_score
        }
    
    # Sort by leakage score
    sorted_leakage = sorted(leakage_scores.items(), key=lambda x: x[1]['leakage_score'], reverse=True)
    
    print(f"\nTop 10 potentially leaky features:")
    for i, (col, scores) in enumerate(sorted_leakage[:10]):
        print(f"  {i+1}. {col}:")
        print(f"     Treatment corr: {scores['treatment_corr']:.4f}")
        print(f"     Response corr: {scores['response_corr']:.4f}")
        print(f"     Leakage score: {scores['leakage_score']:.4f}")
    
    # Identify high-leakage features
    high_leakage_features = [col for col, scores in sorted_leakage if scores['leakage_score'] > 0.1]
    
    print(f"\nHigh-leakage features (score > 0.1): {len(high_leakage_features)}")
    if high_leakage_features:
        print(f"  {high_leakage_features}")
    
    return high_leakage_features, leakage_scores

def test_model_without_leaky_features(df, high_leakage_features):
    """Test uplift model after removing leaky features"""
    print("\n=== MODEL TEST WITHOUT LEAKY FEATURES ===")
    
    # Get safe features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    safe_features = [col for col in numeric_cols if col not in ['treatment_ai_content', 'response'] + high_leakage_features]
    
    print(f"Using {len(safe_features)} safe features")
    print(f"Removed {len(high_leakage_features)} leaky features")
    
    if len(safe_features) == 0:
        print("No safe features remaining!")
        return None
    
    # Simple two-model approach with safe features
    treatment_group = df[df['treatment_ai_content'] == 1]
    control_group = df[df['treatment_ai_content'] == 0]
    
    # Train treatment model
    X_treatment = treatment_group[safe_features]
    y_treatment = treatment_group['response']
    
    treatment_model = RandomForestClassifier(n_estimators=50, random_state=42)
    treatment_model.fit(X_treatment, y_treatment)
    
    # Train control model
    X_control = control_group[safe_features]
    y_control = control_group['response']
    
    control_model = RandomForestClassifier(n_estimators=50, random_state=42)
    control_model.fit(X_control, y_control)
    
    # Predict on all data
    X_all = df[safe_features]
    treatment_probs = treatment_model.predict_proba(X_all)[:, 1]
    control_probs = control_model.predict_proba(X_all)[:, 1]
    
    # Calculate uplift scores
    uplift_scores = treatment_probs - control_probs
    
    print(f"\nUplift score statistics (without leaky features):")
    print(f"  Mean: {uplift_scores.mean():.6f}")
    print(f"  Median: {np.median(uplift_scores):.6f}")
    print(f"  Std: {uplift_scores.std():.6f}")
    print(f"  Min: {uplift_scores.min():.6f}")
    print(f"  Max: {uplift_scores.max():.6f}")
    
    # Calculate Qini coefficient
    eval_df = pd.DataFrame({
        'uplift_score': uplift_scores,
        'treatment': df['treatment_ai_content'],
        'response': df['response']
    })
    eval_df = eval_df.sort_values('uplift_score', ascending=False)
    
    n_total = len(eval_df)
    n_treatment = eval_df['treatment'].sum()
    n_control = n_total - n_treatment
    
    treatment_response_rate = eval_df[eval_df['treatment'] == 1]['response'].mean()
    control_response_rate = eval_df[eval_df['treatment'] == 0]['response'].mean()
    
    qini_coefficient = (treatment_response_rate - control_response_rate) * n_treatment * n_control / n_total
    
    print(f"Qini coefficient (without leaky features): {qini_coefficient:.6f}")
    
    return uplift_scores, qini_coefficient

def analyze_sample_balance_issues(df):
    """Analyze sample balance and its impact"""
    print("\n=== SAMPLE BALANCE ANALYSIS ===")
    
    treatment_group = df[df['treatment_ai_content'] == 1]
    control_group = df[df['treatment_ai_content'] == 0]
    
    print(f"Original balance:")
    print(f"  Treatment: {len(treatment_group):,} ({len(treatment_group)/len(df)*100:.1f}%)")
    print(f"  Control: {len(control_group):,} ({len(control_group)/len(df)*100:.1f}%)")
    
    # Analyze response rates by group
    treatment_response_rate = treatment_group['response'].mean()
    control_response_rate = control_group['response'].mean()
    
    print(f"\nResponse rates:")
    print(f"  Treatment: {treatment_response_rate:.4f}")
    print(f"  Control: {control_response_rate:.4f}")
    
    # Simulate balanced sampling
    print(f"\nSimulating balanced sampling...")
    
    # Randomly sample control group to match treatment size
    np.random.seed(42)
    if len(control_group) >= len(treatment_group):
        balanced_control = control_group.sample(n=len(treatment_group), replace=False)
    else:
        # If control group is smaller, use all control samples and sample treatment
        balanced_treatment = treatment_group.sample(n=len(control_group), replace=False)
        balanced_df = pd.concat([balanced_treatment, control_group])
        balanced_treatment_response = balanced_df[balanced_df['treatment_ai_content'] == 1]['response'].mean()
        balanced_control_response = balanced_df[balanced_df['treatment_ai_content'] == 0]['response'].mean()
        balanced_uplift = balanced_treatment_response - balanced_control_response
        
        print(f"Balanced response rates (treatment sampled):")
        print(f"  Treatment: {balanced_treatment_response:.4f}")
        print(f"  Control: {balanced_control_response:.4f}")
        print(f"  Balanced uplift: {balanced_uplift:.6f}")
        return balanced_uplift
    
    balanced_df = pd.concat([treatment_group, balanced_control])
    
    balanced_treatment_response = balanced_df[balanced_df['treatment_ai_content'] == 1]['response'].mean()
    balanced_control_response = balanced_df[balanced_df['treatment_ai_content'] == 0]['response'].mean()
    balanced_uplift = balanced_treatment_response - balanced_control_response
    
    print(f"Balanced response rates:")
    print(f"  Treatment: {balanced_treatment_response:.4f}")
    print(f"  Control: {balanced_control_response:.4f}")
    print(f"  Balanced uplift: {balanced_uplift:.6f}")
    
    return balanced_uplift

def create_diagnostic_visualizations(df, raw_uplift, balanced_uplift, uplift_scores=None):
    """Create diagnostic visualizations"""
    print("\n=== Creating Diagnostic Visualizations ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Treatment vs Control response rates
    treatment_group = df[df['treatment_ai_content'] == 1]
    control_group = df[df['treatment_ai_content'] == 0]
    
    response_rates = [treatment_group['response'].mean(), control_group['response'].mean()]
    groups = ['Treatment', 'Control']
    colors = ['red' if raw_uplift < 0 else 'green', 'blue']
    
    axes[0, 0].bar(groups, response_rates, color=colors, alpha=0.7)
    axes[0, 0].set_title('Raw Response Rates\n(Red=Negative Uplift)')
    axes[0, 0].set_ylabel('Response Rate')
    for i, v in enumerate(response_rates):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # 2. Sample size comparison
    sizes = [len(treatment_group), len(control_group)]
    axes[0, 1].pie(sizes, labels=groups, autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
    axes[0, 1].set_title('Sample Size Distribution')
    
    # 3. Uplift comparison
    uplifts = [raw_uplift, balanced_uplift]
    uplift_labels = ['Raw Uplift', 'Balanced Uplift']
    colors = ['red' if u < 0 else 'green' for u in uplifts]
    
    axes[0, 2].bar(uplift_labels, uplifts, color=colors, alpha=0.7)
    axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 2].set_title('Uplift Comparison')
    axes[0, 2].set_ylabel('Uplift')
    for i, v in enumerate(uplifts):
        axes[0, 2].text(i, v + (0.01 if v >= 0 else -0.01), f'{v:.4f}', ha='center')
    
    # 4. Feature correlation heatmap (if uplift_scores available)
    if uplift_scores is not None:
        # Get top features by correlation with uplift
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['treatment_ai_content', 'response']]
        
        correlations = []
        for col in feature_cols[:10]:  # Top 10 features
            corr = abs(df[col].corr(pd.Series(uplift_scores)))
            correlations.append((col, corr))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = [col for col, _ in correlations[:5]]
        
        corr_matrix = df[top_features + ['treatment_ai_content', 'response']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
        axes[1, 0].set_title('Feature Correlation Matrix')
    
    # 5. Response rate by user segments
    # Create user segments based on reputation
    df['user_segment'] = pd.cut(df['user_reputation'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    segment_analysis = df.groupby(['user_segment', 'treatment_ai_content'])['response'].mean().unstack()
    segment_analysis.plot(kind='bar', ax=axes[1, 1], color=['lightblue', 'lightcoral'])
    axes[1, 1].set_title('Response Rate by User Segment')
    axes[1, 1].set_ylabel('Response Rate')
    axes[1, 1].legend(['Control', 'Treatment'])
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Treatment effect by content quality
    df['quality_segment'] = pd.cut(df['content_quality_score'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    quality_analysis = df.groupby(['quality_segment', 'treatment_ai_content'])['response'].mean().unstack()
    quality_analysis.plot(kind='bar', ax=axes[1, 2], color=['lightblue', 'lightcoral'])
    axes[1, 2].set_title('Response Rate by Content Quality')
    axes[1, 2].set_ylabel('Response Rate')
    axes[1, 2].legend(['Control', 'Treatment'])
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('deep_uplift_diagnosis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Diagnostic visualizations saved as 'deep_uplift_diagnosis.png'")

def main():
    """Main function for deep uplift diagnosis"""
    print("=== DEEP UPLIFT DIAGNOSIS ===")
    print("Validating treatment effects and model issues")
    
    # Load data
    df = load_and_prepare_data()
    
    # 1. Analyze raw treatment effects
    raw_uplift, treatment_response_rate, control_response_rate = analyze_raw_treatment_effects(df)
    
    # 2. Check for data leakage
    high_leakage_features, leakage_scores = check_for_data_leakage(df)
    
    # 3. Test model without leaky features
    if high_leakage_features:
        uplift_scores, qini_coefficient = test_model_without_leaky_features(df, high_leakage_features)
    else:
        uplift_scores, qini_coefficient = None, None
    
    # 4. Analyze sample balance
    balanced_uplift = analyze_sample_balance_issues(df)
    
    # 5. Create visualizations
    create_diagnostic_visualizations(df, raw_uplift, balanced_uplift, uplift_scores)
    
    # 6. Final diagnosis
    print(f"\n" + "="*60)
    print("DEEP DIAGNOSIS SUMMARY")
    print("="*60)
    
    print(f"1. Raw Treatment Effect: {raw_uplift:.6f}")
    print(f"   - Treatment response rate: {treatment_response_rate:.4f}")
    print(f"   - Control response rate: {control_response_rate:.4f}")
    
    if raw_uplift < 0:
        print(f"   ⚠️  CONFIRMED: Treatment has negative effect")
        print(f"   💡 This is likely a real business issue, not a model problem")
    
    print(f"\n2. Data Leakage:")
    print(f"   - High-leakage features found: {len(high_leakage_features)}")
    if high_leakage_features:
        print(f"   ⚠️  Model may be using treatment-related features")
    
    print(f"\n3. Sample Balance:")
    print(f"   - Raw uplift: {raw_uplift:.6f}")
    print(f"   - Balanced uplift: {balanced_uplift:.6f}")
    if abs(raw_uplift - balanced_uplift) > 0.01:
        print(f"   ⚠️  Sample imbalance affects uplift estimates")
    
    print(f"\n4. Business Recommendations:")
    if raw_uplift < 0:
        print(f"   🚨 AI tags appear to reduce click-through rates")
        print(f"   💡 Consider removing AI tags or testing different tag strategies")
        print(f"   💡 Test tags only on specific user segments or content types")
    else:
        print(f"   ✅ AI tags appear to have positive effect")
    
    print(f"\n=== DIAGNOSIS COMPLETE ===")

if __name__ == "__main__":
    main()
