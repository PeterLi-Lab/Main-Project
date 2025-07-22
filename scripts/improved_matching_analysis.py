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

class ImprovedMatchingAnalysis:
    """Improved matching analysis to address treatment/control distribution differences"""
    
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
    
    def calculate_kl_divergence(self, treatment_data, control_data, feature, bins=50):
        """Calculate KL divergence between treatment and control distributions"""
        try:
            # Create histograms
            treatment_hist, _ = np.histogram(treatment_data[feature], bins=bins, density=True)
            control_hist, _ = np.histogram(control_data[feature], bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            treatment_hist += epsilon
            control_hist += epsilon
            
            # Normalize
            treatment_hist /= treatment_hist.sum()
            control_hist /= control_hist.sum()
            
            # Calculate KL divergence
            kl_div = np.sum(treatment_hist * np.log(treatment_hist / control_hist))
            return kl_div
        except:
            return np.nan
    
    def calculate_wasserstein_distance(self, treatment_data, control_data, feature):
        """Calculate Wasserstein distance between treatment and control distributions"""
        try:
            return wasserstein_distance(treatment_data[feature], control_data[feature])
        except:
            return np.nan
    
    def propensity_score_matching(self, df, feature_cols, caliper=0.1):
        """Perform propensity score matching"""
        print("Performing propensity score matching...")
        
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
        
        # Perform matching
        matched_treatment = []
        matched_control = []
        
        for _, treatment_row in treatment_df.iterrows():
            treatment_ps = treatment_row['propensity_score']
            
            # Find control samples within caliper
            control_candidates = control_df[
                abs(control_df['propensity_score'] - treatment_ps) <= caliper
            ]
            
            if len(control_candidates) > 0:
                # Select the closest match
                distances = abs(control_candidates['propensity_score'] - treatment_ps)
                best_match_idx = distances.idxmin()
                best_match = control_candidates.loc[best_match_idx]
                
                matched_treatment.append(treatment_row)
                matched_control.append(best_match)
                
                # Remove matched control to avoid duplicates
                control_df = control_df.drop(best_match_idx)
        
        if matched_treatment and matched_control:
            matched_df = pd.concat([
                pd.DataFrame(matched_treatment),
                pd.DataFrame(matched_control)
            ], ignore_index=True)
            
            print(f"Matched - Treatment: {len(matched_treatment):,}, Control: {len(matched_control):,}")
            return matched_df
        else:
            print("No matches found with current caliper. Trying with larger caliper...")
            return self.propensity_score_matching(df, feature_cols, caliper=caliper*2)
    
    def nearest_neighbor_matching(self, df, feature_cols, n_neighbors=1):
        """Perform nearest neighbor matching"""
        print("Performing nearest neighbor matching...")
        
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
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
        nn_model.fit(control_features)
        
        # Find nearest neighbors for each treatment sample
        distances, indices = nn_model.kneighbors(treatment_features)
        
        # Select matched pairs
        matched_treatment = []
        matched_control = []
        used_control_indices = set()
        
        for i, (treatment_idx, neighbor_indices) in enumerate(zip(treatment_df.index, indices)):
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
    
    def mahalanobis_matching(self, df, feature_cols):
        """Perform Mahalanobis distance matching"""
        print("Performing Mahalanobis distance matching...")
        
        # Prepare features
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into treatment and control
        treatment_df = df[df['treatment_ai_content'] == 1].copy()
        control_df = df[df['treatment_ai_content'] == 0].copy()
        
        treatment_features = X_scaled[df['treatment_ai_content'] == 1]
        control_features = X_scaled[df['treatment_ai_content'] == 0]
        
        print(f"Original - Treatment: {len(treatment_df):,}, Control: {len(control_df):,}")
        
        # Calculate covariance matrix for Mahalanobis distance
        cov_matrix = np.cov(control_features.T)
        
        # Find best matches
        matched_treatment = []
        matched_control = []
        used_control_indices = set()
        
        for i, treatment_feature in enumerate(treatment_features):
            best_distance = float('inf')
            best_control_idx = None
            
            for j, control_feature in enumerate(control_features):
                if j not in used_control_indices:
                    try:
                        distance = mahalanobis(treatment_feature, control_feature, cov_matrix)
                        if distance < best_distance:
                            best_distance = distance
                            best_control_idx = j
                    except:
                        continue
            
            if best_control_idx is not None:
                matched_treatment.append(treatment_df.iloc[i])
                matched_control.append(control_df.iloc[best_control_idx])
                used_control_indices.add(best_control_idx)
        
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
            kl_div = self.calculate_kl_divergence(treatment_data, control_data, feature)
            wasserstein_dist = self.calculate_wasserstein_distance(treatment_data, control_data, feature)
            
            balance_metrics[feature] = {
                'smd': smd,
                'kl_divergence': kl_div,
                'wasserstein_distance': wasserstein_dist
            }
            
            print(f"{feature}:")
            print(f"  SMD: {smd:.4f}")
            print(f"  KL Divergence: {kl_div:.4f}")
            print(f"  Wasserstein Distance: {wasserstein_dist:.4f}")
            print()
        
        return balance_metrics
    
    def create_balance_visualization(self, df, feature_cols, balance_metrics):
        """Create visualization of balance metrics"""
        print("Creating balance visualization...")
        
        # Prepare data for plotting
        features = list(balance_metrics.keys())
        smd_values = [balance_metrics[f]['smd'] for f in features]
        kl_values = [balance_metrics[f]['kl_divergence'] for f in features]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # SMD plot
        axes[0, 0].barh(features, smd_values, color='skyblue')
        axes[0, 0].axvline(x=0.1, color='red', linestyle='--', label='SMD < 0.1 (Good)')
        axes[0, 0].axvline(x=0.25, color='orange', linestyle='--', label='SMD < 0.25 (Acceptable)')
        axes[0, 0].set_xlabel('Standardized Mean Difference')
        axes[0, 0].set_title('Balance Assessment: SMD')
        axes[0, 0].legend()
        
        # KL Divergence plot
        axes[0, 1].barh(features, kl_values, color='lightgreen')
        axes[0, 1].set_xlabel('KL Divergence')
        axes[0, 1].set_title('Balance Assessment: KL Divergence')
        
        # Feature distribution comparison
        treatment_data = df[df['treatment_ai_content'] == 1]
        control_data = df[df['treatment_ai_content'] == 0]
        
        for i, feature in enumerate(features[:4]):
            row = i // 2
            col = i % 2 + 1
            
            if col < 2:
                axes[row, col].hist(treatment_data[feature], alpha=0.7, label='Treatment', bins=30)
                axes[row, col].hist(control_data[feature], alpha=0.7, label='Control', bins=30)
                axes[row, col].set_xlabel(feature)
                axes[row, col].set_ylabel('Frequency')
                axes[row, col].set_title(f'Distribution: {feature}')
                axes[row, col].legend()
        
        plt.tight_layout()
        plt.savefig('balance_evaluation.png', dpi=300, bbox_inches='tight')
        print("Balance evaluation visualization saved as 'balance_evaluation.png'")
    
    def create_improved_tsne(self, df, feature_cols):
        """Create improved t-SNE visualization"""
        print("Creating improved t-SNE visualization...")
        
        # Prepare features
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
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
        plt.title('Improved t-SNE: Treatment vs Control Distribution (After Matching)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('improved_tsne_matching.png', dpi=300, bbox_inches='tight')
        print("Improved t-SNE visualization saved as 'improved_tsne_matching.png'")
        
        return X_tsne

def improved_matching_analysis():
    """Main function for improved matching analysis"""
    print("=== Improved Matching Analysis ===\n")
    
    # Load balanced data
    df = pd.read_csv('uplift_model_data_balanced.csv')
    print(f"Loaded balanced dataset: {len(df):,} samples")
    
    # Initialize analyzer
    analyzer = ImprovedMatchingAnalysis()
    
    # Prepare features
    exclude_cols = ['treatment_ai_content', 'response', 'user_id', 'post_id', 'propensity_score']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    print(f"Using {len(feature_cols)} features for matching")
    
    # Evaluate original balance
    print("\n=== Original Balance Evaluation ===")
    original_balance = analyzer.evaluate_balance(df, feature_cols)
    
    # Try different matching methods
    matching_results = {}
    
    # 1. Propensity Score Matching
    try:
        ps_matched_df = analyzer.propensity_score_matching(df, feature_cols)
        matching_results['propensity_score'] = ps_matched_df
        print(f"Propensity score matching completed: {len(ps_matched_df):,} samples")
    except Exception as e:
        print(f"Propensity score matching failed: {e}")
    
    # 2. Nearest Neighbor Matching
    try:
        nn_matched_df = analyzer.nearest_neighbor_matching(df, feature_cols)
        matching_results['nearest_neighbor'] = nn_matched_df
        print(f"Nearest neighbor matching completed: {len(nn_matched_df):,} samples")
    except Exception as e:
        print(f"Nearest neighbor matching failed: {e}")
    
    # 3. Mahalanobis Matching
    try:
        mahal_matched_df = analyzer.mahalanobis_matching(df, feature_cols)
        matching_results['mahalanobis'] = mahal_matched_df
        print(f"Mahalanobis matching completed: {len(mahal_matched_df):,} samples")
    except Exception as e:
        print(f"Mahalanobis matching failed: {e}")
    
    # Evaluate and compare results
    best_method = None
    best_balance_score = float('inf')
    
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
        analyzer.create_balance_visualization(matched_df, feature_cols, balance_metrics)
        analyzer.create_improved_tsne(matched_df, feature_cols)
        
        # Save matched dataset
        output_file = f'uplift_model_data_{method_name}_matched.csv'
        matched_df.to_csv(output_file, index=False)
        print(f"Matched dataset saved to: {output_file}")
    
    # Final recommendation
    if best_method:
        print(f"\n=== RECOMMENDATION ===")
        print(f"Best matching method: {best_method}")
        print(f"Best balance score (avg SMD): {best_balance_score:.4f}")
        
        if best_balance_score < 0.1:
            print("✅ Excellent balance achieved!")
        elif best_balance_score < 0.25:
            print("✅ Good balance achieved!")
        else:
            print("⚠️  Balance could be improved further")
    
    return matching_results

if __name__ == "__main__":
    improved_matching_analysis() 