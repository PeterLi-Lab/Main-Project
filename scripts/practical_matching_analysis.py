import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import mahalanobis
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

class PracticalMatchingAnalysis:
    """Practical matching analysis with better balance and visualization"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.propensity_model = None
        
    def calculate_smd(self, treatment_data, control_data, feature):
        """Calculate Standardized Mean Difference (SMD)"""
        treatment_mean = treatment_data[feature].mean()
        control_mean = control_data[feature].mean()
        treatment_std = treatment_data[feature].std()
        control_std = control_data[feature].std()
        
        # Pooled standard deviation
        pooled_std = np.sqrt((treatment_std**2 + control_std**2) / 2)
        
        if pooled_std == 0:
            return 0
        
        smd = abs(treatment_mean - control_mean) / pooled_std
        return smd
    
    def improved_propensity_matching(self, df, feature_cols, caliper=0.25):
        """Improved propensity score matching with better caliper"""
        print("Performing improved propensity score matching...")
        
        # Prepare features for propensity score model
        X = df[feature_cols].fillna(0)
        y = df['treatment_ai_content']
        
        # Fit propensity score model
        self.propensity_model = LogisticRegression(random_state=42, max_iter=1000)
        self.propensity_model.fit(X, y)
        
        # Calculate propensity scores
        propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        df['propensity_score'] = propensity_scores
        
        # Split into treatment and control
        treatment_df = df[df['treatment_ai_content'] == 1].copy()
        control_df = df[df['treatment_ai_content'] == 0].copy()
        
        print(f"Original - Treatment: {len(treatment_df):,}, Control: {len(control_df):,}")
        
        # Perform matching with multiple caliper attempts
        caliper_values = [0.25, 0.5, 1.0, 2.0]
        best_matches = None
        best_match_count = 0
        
        for caliper in caliper_values:
            matched_treatment = []
            matched_control = []
            temp_control_df = control_df.copy()
            
            for _, treatment_row in treatment_df.iterrows():
                treatment_ps = treatment_row['propensity_score']
                
                # Find control samples within caliper
                control_candidates = temp_control_df[
                    abs(temp_control_df['propensity_score'] - treatment_ps) <= caliper
                ]
                
                if len(control_candidates) > 0:
                    # Select the closest match
                    distances = abs(control_candidates['propensity_score'] - treatment_ps)
                    best_match_idx = distances.idxmin()
                    best_match = control_candidates.loc[best_match_idx]
                    
                    matched_treatment.append(treatment_row)
                    matched_control.append(best_match)
                    
                    # Remove matched control to avoid duplicates
                    temp_control_df = temp_control_df.drop(best_match_idx)
            
            if len(matched_treatment) > best_match_count:
                best_match_count = len(matched_treatment)
                best_matches = (matched_treatment, matched_control)
                print(f"Caliper {caliper}: Found {len(matched_treatment)} matches")
        
        if best_matches:
            matched_treatment, matched_control = best_matches
            matched_df = pd.concat([
                pd.DataFrame(matched_treatment),
                pd.DataFrame(matched_control)
            ], ignore_index=True)
            
            print(f"Best matching result: {len(matched_treatment)} treatment, {len(matched_control)} control")
            return matched_df
        else:
            print("No matches found with any caliper!")
            return df
    
    def nearest_neighbor_matching_improved(self, df, feature_cols, n_neighbors=5):
        """Improved nearest neighbor matching with multiple neighbors"""
        print("Performing improved nearest neighbor matching...")
        
        # Prepare features
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into treatment and control
        treatment_df = df[df['treatment_ai_content'] == 1].copy()
        control_df = df[df['treatment_ai_content'] == 0].copy()
        
        treatment_features = X_scaled[df['treatment_ai_content'] == 1]
        control_features = X_scaled[df['treatment_ai_content'] == 0]
        
        print(f"Original - Treatment: {len(treatment_df):,}, Control: {len(control_df):,}")
        
        # Fit nearest neighbor model on control group
        nn_model = NearestNeighbors(n_neighbors=min(n_neighbors, len(control_features)), algorithm='auto')
        nn_model.fit(control_features)
        
        # Find nearest neighbors for each treatment sample
        distances, indices = nn_model.kneighbors(treatment_features)
        
        # Select matched pairs with better strategy
        matched_treatment = []
        matched_control = []
        used_control_indices = set()
        
        # Sort treatment samples by their best match distance
        treatment_match_qualities = []
        for i, (treatment_idx, neighbor_indices, neighbor_distances) in enumerate(
            zip(treatment_df.index, indices, distances)
        ):
            best_distance = neighbor_distances[0]
            treatment_match_qualities.append((i, treatment_idx, best_distance))
        
        # Sort by match quality (best matches first)
        treatment_match_qualities.sort(key=lambda x: x[2])
        
        for _, treatment_idx, _ in treatment_match_qualities:
            treatment_row_idx = treatment_match_qualities.index((_, treatment_idx, _))
            neighbor_indices = indices[treatment_row_idx]
            
            for neighbor_idx in neighbor_indices:
                if neighbor_idx not in used_control_indices:
                    matched_treatment.append(treatment_df.loc[treatment_idx])
                    matched_control.append(control_df.iloc[neighbor_idx])
                    used_control_indices.add(neighbor_idx)
                    break
        
        if matched_treatment and matched_control:
            matched_df = pd.concat([
                pd.DataFrame(matched_treatment),
                pd.DataFrame(matched_control)
            ], ignore_index=True)
            
            print(f"Matched - Treatment: {len(matched_treatment):,}, Control: {len(matched_control):,}")
            return matched_df
        else:
            print("No matches found!")
            return df
    
    def evaluate_balance(self, df, feature_cols):
        """Evaluate balance between treatment and control groups"""
        print("\n=== Balance Evaluation ===")
        
        treatment_data = df[df['treatment_ai_content'] == 1]
        control_data = df[df['treatment_ai_content'] == 0]
        
        balance_metrics = {}
        
        for feature in feature_cols[:10]:  # Evaluate top 10 features
            smd = self.calculate_smd(treatment_data, control_data, feature)
            
            balance_metrics[feature] = {
                'smd': smd,
                'treatment_mean': treatment_data[feature].mean(),
                'control_mean': control_data[feature].mean(),
                'treatment_std': treatment_data[feature].std(),
                'control_std': control_data[feature].std()
            }
            
            print(f"{feature}:")
            print(f"  SMD: {smd:.4f}")
            print(f"  Treatment: mean={treatment_data[feature].mean():.4f}, std={treatment_data[feature].std():.4f}")
            print(f"  Control: mean={control_data[feature].mean():.4f}, std={control_data[feature].std():.4f}")
            print()
        
        return balance_metrics
    
    def create_balance_visualization(self, df, feature_cols, balance_metrics, method_name):
        """Create comprehensive balance visualization"""
        print(f"Creating balance visualization for {method_name}...")
        
        # Prepare data for plotting
        features = list(balance_metrics.keys())
        smd_values = [balance_metrics[f]['smd'] for f in features]
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # SMD plot
        axes[0, 0].barh(features, smd_values, color='skyblue')
        axes[0, 0].axvline(x=0.1, color='red', linestyle='--', label='SMD < 0.1 (Excellent)')
        axes[0, 0].axvline(x=0.25, color='orange', linestyle='--', label='SMD < 0.25 (Good)')
        axes[0, 0].set_xlabel('Standardized Mean Difference')
        axes[0, 0].set_title(f'Balance Assessment: {method_name}')
        axes[0, 0].legend()
        
        # Feature distribution comparison
        treatment_data = df[df['treatment_ai_content'] == 1]
        control_data = df[df['treatment_ai_content'] == 0]
        
        for i, feature in enumerate(features[:5]):
            row = i // 3 + 1
            col = i % 3
            
            axes[row, col].hist(treatment_data[feature], alpha=0.7, label='Treatment', bins=30, color='red')
            axes[row, col].hist(control_data[feature], alpha=0.7, label='Control', bins=30, color='blue')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].set_title(f'{feature} Distribution')
            axes[row, col].legend()
        
        plt.tight_layout()
        plt.savefig(f'balance_evaluation_{method_name}.png', dpi=300, bbox_inches='tight')
        print(f"Balance evaluation visualization saved as 'balance_evaluation_{method_name}.png'")
    
    def create_improved_tsne(self, df, feature_cols, method_name):
        """Create improved t-SNE visualization with proper perplexity"""
        print(f"Creating improved t-SNE visualization for {method_name}...")
        
        # Prepare features
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Adjust perplexity based on sample size
        n_samples = len(X_scaled)
        perplexity = min(30, n_samples // 4)  # Ensure perplexity < n_samples
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        treatment_mask = df['treatment_ai_content'] == 1
        plt.scatter(X_tsne[~treatment_mask, 0], X_tsne[~treatment_mask, 1], 
                   c='blue', alpha=0.6, s=20, label='Control')
        plt.scatter(X_tsne[treatment_mask, 0], X_tsne[treatment_mask, 1], 
                   c='red', alpha=0.6, s=20, label='Treatment')
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title(f'Improved t-SNE: {method_name} Matching Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f'improved_tsne_{method_name}.png', dpi=300, bbox_inches='tight')
        print(f"Improved t-SNE visualization saved as 'improved_tsne_{method_name}.png'")
        
        return X_tsne
    
    def create_uplift_analysis(self, df, method_name):
        """Create uplift analysis for matched data"""
        print(f"Creating uplift analysis for {method_name}...")
        
        if 'response' not in df.columns:
            print("No response column found, skipping uplift analysis")
            return
        
        treatment_response_rate = df[df['treatment_ai_content'] == 1]['response'].mean()
        control_response_rate = df[df['treatment_ai_content'] == 0]['response'].mean()
        uplift = treatment_response_rate - control_response_rate
        
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        uplift_samples = []
        
        for i in range(n_bootstrap):
            bootstrap_sample = df.sample(n=len(df), replace=True, random_state=i)
            treatment_rate = bootstrap_sample[bootstrap_sample['treatment_ai_content'] == 1]['response'].mean()
            control_rate = bootstrap_sample[bootstrap_sample['treatment_ai_content'] == 0]['response'].mean()
            uplift_samples.append(treatment_rate - control_rate)
        
        uplift_ci_95 = np.percentile(uplift_samples, [2.5, 97.5])
        uplift_std = np.std(uplift_samples)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(uplift_samples, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(uplift, color='red', linestyle='--', label=f'Mean: {uplift:.4f}')
        plt.axvline(uplift_ci_95[0], color='orange', linestyle=':', label=f'95% CI: [{uplift_ci_95[0]:.4f}, {uplift_ci_95[1]:.4f}]')
        plt.axvline(uplift_ci_95[1], color='orange', linestyle=':')
        plt.xlabel('Uplift')
        plt.ylabel('Frequency')
        plt.title(f'Uplift Distribution: {method_name}')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        response_rates = [treatment_response_rate, control_response_rate]
        plt.bar(['Treatment', 'Control'], response_rates, color=['red', 'blue'], alpha=0.7)
        plt.ylabel('Response Rate')
        plt.title(f'Response Rates: {method_name}')
        
        plt.subplot(2, 2, 3)
        treatment_counts = df['treatment_ai_content'].value_counts()
        plt.pie(treatment_counts.values, labels=['Control', 'Treatment'], autopct='%1.1f%%', colors=['blue', 'red'])
        plt.title(f'Sample Distribution: {method_name}')
        
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, f'Uplift: {uplift:.4f}', fontsize=12)
        plt.text(0.1, 0.7, f'95% CI: [{uplift_ci_95[0]:.4f}, {uplift_ci_95[1]:.4f}]', fontsize=12)
        plt.text(0.1, 0.6, f'Std: {uplift_std:.4f}', fontsize=12)
        plt.text(0.1, 0.5, f'Treatment Rate: {treatment_response_rate:.4f}', fontsize=12)
        plt.text(0.1, 0.4, f'Control Rate: {control_response_rate:.4f}', fontsize=12)
        plt.text(0.1, 0.3, f'Sample Size: {len(df):,}', fontsize=12)
        plt.axis('off')
        plt.title(f'Summary: {method_name}')
        
        plt.tight_layout()
        plt.savefig(f'uplift_analysis_{method_name}.png', dpi=300, bbox_inches='tight')
        print(f"Uplift analysis saved as 'uplift_analysis_{method_name}.png'")
        
        return {
            'uplift': uplift,
            'uplift_ci_95': uplift_ci_95,
            'uplift_std': uplift_std,
            'treatment_rate': treatment_response_rate,
            'control_rate': control_response_rate
        }

def practical_matching_analysis():
    """Main function for practical matching analysis"""
    print("=== Practical Matching Analysis ===\n")
    
    # Load balanced data
    df = pd.read_csv('uplift_model_data_balanced.csv')
    print(f"Loaded balanced dataset: {len(df):,} samples")
    
    # Initialize analyzer
    analyzer = PracticalMatchingAnalysis()
    
    # Prepare features
    exclude_cols = ['treatment_ai_content', 'response', 'user_id', 'post_id', 'propensity_score']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    print(f"Using {len(feature_cols)} features for matching")
    
    # Evaluate original balance
    print("\n=== Original Balance Evaluation ===")
    original_balance = analyzer.evaluate_balance(df, feature_cols)
    
    # Try different matching methods
    matching_results = {}
    
    # 1. Improved Propensity Score Matching
    try:
        ps_matched_df = analyzer.improved_propensity_matching(df, feature_cols)
        matching_results['propensity_score'] = ps_matched_df
        print(f"Propensity score matching completed: {len(ps_matched_df):,} samples")
    except Exception as e:
        print(f"Propensity score matching failed: {e}")
    
    # 2. Improved Nearest Neighbor Matching
    try:
        nn_matched_df = analyzer.nearest_neighbor_matching_improved(df, feature_cols)
        matching_results['nearest_neighbor'] = nn_matched_df
        print(f"Nearest neighbor matching completed: {len(nn_matched_df):,} samples")
    except Exception as e:
        print(f"Nearest neighbor matching failed: {e}")
    
    # Evaluate and compare results
    best_method = None
    best_balance_score = float('inf')
    best_results = {}
    
    for method_name, matched_df in matching_results.items():
        print(f"\n=== {method_name.upper()} Matching Results ===")
        
        # Evaluate balance
        balance_metrics = analyzer.evaluate_balance(matched_df, feature_cols)
        
        # Calculate overall balance score (average SMD)
        avg_smd = np.mean([metrics['smd'] for metrics in balance_metrics.values()])
        print(f"Average SMD: {avg_smd:.4f}")
        
        if avg_smd < best_balance_score:
            best_balance_score = avg_smd
            best_method = method_name
        
        # Create visualizations
        analyzer.create_balance_visualization(matched_df, feature_cols, balance_metrics, method_name)
        analyzer.create_improved_tsne(matched_df, feature_cols, method_name)
        
        # Perform uplift analysis
        uplift_results = analyzer.create_uplift_analysis(matched_df, method_name)
        best_results[method_name] = {
            'balance_metrics': balance_metrics,
            'avg_smd': avg_smd,
            'uplift_results': uplift_results
        }
        
        # Save matched dataset
        output_file = f'uplift_model_data_{method_name}_matched.csv'
        matched_df.to_csv(output_file, index=False)
        print(f"Matched dataset saved to: {output_file}")
    
    # Final recommendation
    if best_method:
        print(f"\n=== FINAL RECOMMENDATION ===")
        print(f"Best matching method: {best_method}")
        print(f"Best balance score (avg SMD): {best_balance_score:.4f}")
        
        if best_balance_score < 0.1:
            print("✅ Excellent balance achieved!")
        elif best_balance_score < 0.25:
            print("✅ Good balance achieved!")
        else:
            print("⚠️  Balance could be improved further")
        
        # Show uplift results for best method
        if best_method in best_results and best_results[best_method]['uplift_results']:
            uplift_info = best_results[best_method]['uplift_results']
            print(f"\nUplift Analysis for {best_method}:")
            print(f"  Uplift: {uplift_info['uplift']:.4f}")
            print(f"  95% CI: [{uplift_info['uplift_ci_95'][0]:.4f}, {uplift_info['uplift_ci_95'][1]:.4f}]")
            print(f"  Treatment Rate: {uplift_info['treatment_rate']:.4f}")
            print(f"  Control Rate: {uplift_info['control_rate']:.4f}")
    
    return matching_results, best_results

if __name__ == "__main__":
    practical_matching_analysis() 