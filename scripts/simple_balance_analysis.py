import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

class SimpleBalanceAnalysis:
    """Simple balance analysis to address treatment/control distribution differences"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
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
    
    def create_balanced_sample(self, df, feature_cols, target_size=2000):
        """Create a balanced sample with better treatment/control distribution"""
        print("Creating balanced sample with improved distribution...")
        
        # Prepare features
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into treatment and control
        treatment_df = df[df['treatment_ai_content'] == 1].copy()
        control_df = df[df['treatment_ai_content'] == 0].copy()
        
        print(f"Original - Treatment: {len(treatment_df):,}, Control: {len(control_df):,}")
        
        # Sample equal numbers from each group
        n_samples = min(target_size // 2, len(treatment_df), len(control_df))
        
        treatment_sample = treatment_df.sample(n=n_samples, random_state=42)
        control_sample = control_df.sample(n=n_samples, random_state=42)
        
        # Combine samples
        balanced_df = pd.concat([treatment_sample, control_sample], ignore_index=True)
        
        print(f"Balanced - Treatment: {len(treatment_sample):,}, Control: {len(control_sample):,}")
        
        return balanced_df
    
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
        """Create balance visualization"""
        print(f"Creating balance visualization for {method_name}...")
        
        # Prepare data for plotting
        features = list(balance_metrics.keys())
        smd_values = [balance_metrics[f]['smd'] for f in features]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # SMD plot
        axes[0, 0].barh(features, smd_values, color='skyblue')
        axes[0, 0].axvline(x=0.1, color='red', linestyle='--', label='SMD < 0.1 (Excellent)')
        axes[0, 0].axvline(x=0.25, color='orange', linestyle='--', label='SMD < 0.25 (Good)')
        axes[0, 0].set_xlabel('Standardized Mean Difference')
        axes[0, 0].set_title(f'Balance Assessment: {method_name}')
        axes[0, 0].legend()
        
        # Feature distribution comparison (top 3 features)
        treatment_data = df[df['treatment_ai_content'] == 1]
        control_data = df[df['treatment_ai_content'] == 0]
        
        for i, feature in enumerate(features[:3]):
            row = i + 1
            col = 0
            
            axes[row, col].hist(treatment_data[feature], alpha=0.7, label='Treatment', bins=30, color='red')
            axes[row, col].hist(control_data[feature], alpha=0.7, label='Control', bins=30, color='blue')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].set_title(f'{feature} Distribution')
            axes[row, col].legend()
        
        plt.tight_layout()
        plt.savefig(f'balance_evaluation_{method_name}.png', dpi=300, bbox_inches='tight')
        print(f"Balance evaluation visualization saved as 'balance_evaluation_{method_name}.png'")
    
    def create_tsne_visualization(self, df, feature_cols, method_name):
        """Create t-SNE visualization"""
        print(f"Creating t-SNE visualization for {method_name}...")
        
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
        plt.title(f't-SNE: {method_name} Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f'tsne_{method_name}.png', dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved as 'tsne_{method_name}.png'")
        
        return X_tsne
    
    def create_uplift_analysis(self, df, method_name):
        """Create uplift analysis"""
        print(f"Creating uplift analysis for {method_name}...")
        
        if 'response' not in df.columns:
            print("No response column found, skipping uplift analysis")
            return None
        
        treatment_response_rate = df[df['treatment_ai_content'] == 1]['response'].mean()
        control_response_rate = df[df['treatment_ai_content'] == 0]['response'].mean()
        uplift = treatment_response_rate - control_response_rate
        
        # Bootstrap confidence intervals
        n_bootstrap = 500
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

def simple_balance_analysis():
    """Main function for simple balance analysis"""
    print("=== Simple Balance Analysis ===\n")
    
    # Load balanced data
    df = pd.read_csv('uplift_model_data_balanced.csv')
    print(f"Loaded balanced dataset: {len(df):,} samples")
    
    # Initialize analyzer
    analyzer = SimpleBalanceAnalysis()
    
    # Prepare features
    exclude_cols = ['treatment_ai_content', 'response', 'user_id', 'post_id', 'propensity_score']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    print(f"Using {len(feature_cols)} features for analysis")
    
    # Evaluate original balance
    print("\n=== Original Balance Evaluation ===")
    original_balance = analyzer.evaluate_balance(df, feature_cols)
    
    # Create balanced sample
    balanced_df = analyzer.create_balanced_sample(df, feature_cols, target_size=2000)
    
    print(f"\n=== BALANCED SAMPLE RESULTS ===")
    
    # Evaluate balance after balancing
    balance_metrics = analyzer.evaluate_balance(balanced_df, feature_cols)
    
    # Calculate overall balance score (average SMD)
    avg_smd = np.mean([metrics['smd'] for metrics in balance_metrics.values()])
    print(f"Average SMD: {avg_smd:.4f}")
    
    # Create visualizations
    analyzer.create_balance_visualization(balanced_df, feature_cols, balance_metrics, 'balanced_sample')
    analyzer.create_tsne_visualization(balanced_df, feature_cols, 'balanced_sample')
    
    # Perform uplift analysis
    uplift_results = analyzer.create_uplift_analysis(balanced_df, 'balanced_sample')
    
    # Save balanced dataset
    output_file = 'uplift_model_data_simple_balanced.csv'
    balanced_df.to_csv(output_file, index=False)
    print(f"Balanced dataset saved to: {output_file}")
    
    # Final recommendation
    print(f"\n=== FINAL RECOMMENDATION ===")
    print(f"Average SMD: {avg_smd:.4f}")
    
    if avg_smd < 0.1:
        print("✅ Excellent balance achieved!")
    elif avg_smd < 0.25:
        print("✅ Good balance achieved!")
    else:
        print("⚠️  Balance could be improved further")
    
    if uplift_results:
        print(f"\nUplift Analysis:")
        print(f"  Uplift: {uplift_results['uplift']:.4f}")
        print(f"  95% CI: [{uplift_results['uplift_ci_95'][0]:.4f}, {uplift_results['uplift_ci_95'][1]:.4f}]")
        print(f"  Treatment Rate: {uplift_results['treatment_rate']:.4f}")
        print(f"  Control Rate: {uplift_results['control_rate']:.4f}")
    
    return balanced_df, balance_metrics, uplift_results

if __name__ == "__main__":
    simple_balance_analysis() 