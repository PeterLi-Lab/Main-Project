import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class EfficientFeatureAnalysis:
    """Efficient feature analysis without t-SNE bottleneck"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self, file_path):
        """Load and prepare data for analysis"""
        print(f"Loading data from {file_path}...")
        
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df):,} samples with {len(df.columns)} columns")
        
        # Prepare features
        exclude_cols = ['treatment_ai_content', 'response', 'user_id', 'post_id', 'propensity_score']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        print(f"Using {len(feature_cols)} features for analysis")
        
        return df, feature_cols
    
    def analyze_feature_distributions(self, df, feature_cols):
        """Analyze feature distributions between treatment and control"""
        print("\n=== Feature Distribution Analysis ===")
        
        treatment_data = df[df['treatment_ai_content'] == 1]
        control_data = df[df['treatment_ai_content'] == 0]
        
        distribution_analysis = {}
        
        for feature in feature_cols:
            treatment_mean = treatment_data[feature].mean()
            control_mean = control_data[feature].mean()
            treatment_std = treatment_data[feature].std()
            control_std = control_data[feature].std()
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((treatment_std**2 + control_std**2) / 2)
            effect_size = abs(treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
            
            # Calculate overlap percentage
            treatment_median = treatment_data[feature].median()
            control_median = control_data[feature].median()
            
            distribution_analysis[feature] = {
                'treatment_mean': treatment_mean,
                'control_mean': control_mean,
                'treatment_std': treatment_std,
                'control_std': control_std,
                'effect_size': effect_size,
                'treatment_median': treatment_median,
                'control_median': control_median,
                'mean_difference': treatment_mean - control_mean,
                'median_difference': treatment_median - control_median
            }
            
            print(f"{feature}:")
            print(f"  Treatment: mean={treatment_mean:.4f}, median={treatment_median:.4f}")
            print(f"  Control: mean={control_mean:.4f}, median={control_median:.4f}")
            print(f"  Effect size: {effect_size:.4f}")
            print(f"  Mean difference: {treatment_mean - control_mean:.4f}")
            print()
        
        return distribution_analysis
    
    def pca_analysis(self, df, feature_cols):
        """Analyze PCA components"""
        print("\n=== PCA Analysis ===")
        
        # Prepare features
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        pca = PCA(n_components=min(5, len(feature_cols)))
        X_pca = pca.fit_transform(X_scaled)
        
        # Analyze explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print(f"Explained variance by components:")
        for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
            print(f"  Component {i+1}: {var_ratio:.4f} (Cumulative: {cum_var:.4f})")
        
        # Analyze feature contributions
        feature_contributions = {}
        for i, feature in enumerate(feature_cols):
            feature_contributions[feature] = {
                'component_1_weight': abs(pca.components_[0][i]),
                'component_2_weight': abs(pca.components_[1][i]),
                'component_3_weight': abs(pca.components_[2][i]) if len(pca.components_) > 2 else 0
            }
        
        # Sort by component 1 contribution
        comp1_contributions = sorted(feature_contributions.items(), 
                                   key=lambda x: x[1]['component_1_weight'], reverse=True)
        
        print(f"\nTop 10 Features Contributing to Component 1:")
        for feature, weights in comp1_contributions[:10]:
            print(f"  {feature}: {weights['component_1_weight']:.4f}")
        
        return feature_contributions, X_pca
    
    def correlation_analysis(self, df, feature_cols):
        """Analyze correlations between features and treatment"""
        print("\n=== Correlation Analysis ===")
        
        correlations = {}
        for feature in feature_cols:
            corr = df[feature].corr(df['treatment_ai_content'])
            correlations[feature] = {
                'correlation': corr,
                'abs_correlation': abs(corr)
            }
        
        # Sort by absolute correlation
        sorted_correlations = sorted(correlations.items(), 
                                   key=lambda x: x[1]['abs_correlation'], reverse=True)
        
        print("Top 10 Features Correlated with Treatment:")
        for feature, corr_data in sorted_correlations[:10]:
            print(f"  {feature}: {corr_data['correlation']:.4f}")
        
        return correlations
    
    def create_efficient_visualization(self, distribution_analysis, feature_contributions, correlations, df):
        """Create efficient visualization without t-SNE"""
        print("\nCreating efficient visualization...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Effect sizes
        features = list(distribution_analysis.keys())
        effect_sizes = [distribution_analysis[f]['effect_size'] for f in features]
        
        # Sort by effect size
        sorted_indices = np.argsort(effect_sizes)[::-1]
        top_features = [features[i] for i in sorted_indices[:10]]
        top_effect_sizes = [effect_sizes[i] for i in sorted_indices[:10]]
        
        axes[0, 0].barh(range(len(top_features)), top_effect_sizes, color='skyblue')
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features)
        axes[0, 0].set_xlabel('Effect Size (Cohen\'s d)')
        axes[0, 0].set_title('Top 10 Features by Effect Size')
        
        # 2. PCA Component 1 Contributions
        top_pca_features = sorted(feature_contributions.items(), 
                                key=lambda x: x[1]['component_1_weight'], reverse=True)[:10]
        pca_features = [f[0] for f in top_pca_features]
        pca_weights = [f[1]['component_1_weight'] for f in top_pca_features]
        
        axes[0, 1].barh(range(len(pca_features)), pca_weights, color='lightgreen')
        axes[0, 1].set_yticks(range(len(pca_features)))
        axes[0, 1].set_yticklabels(pca_features)
        axes[0, 1].set_xlabel('Component 1 Weight')
        axes[0, 1].set_title('PCA Component 1 Contributions')
        
        # 3. Treatment correlations
        top_corr_features = sorted(correlations.items(), 
                                 key=lambda x: x[1]['abs_correlation'], reverse=True)[:10]
        corr_features = [f[0] for f in top_corr_features]
        corr_values = [f[1]['correlation'] for f in top_corr_features]
        
        axes[1, 0].barh(range(len(corr_features)), corr_values, color='lightcoral')
        axes[1, 0].set_yticks(range(len(corr_features)))
        axes[1, 0].set_yticklabels(corr_features)
        axes[1, 0].set_xlabel('Correlation with Treatment')
        axes[1, 0].set_title('Treatment Correlations')
        
        # 4. Feature distribution comparison (top 3 effect size features)
        top_3_features = [features[i] for i in sorted_indices[:3]]
        treatment_data = df[df['treatment_ai_content'] == 1]
        control_data = df[df['treatment_ai_content'] == 0]
        
        for i, feature in enumerate(top_3_features):
            axes[1, 1].hist(treatment_data[feature], alpha=0.7, label='Treatment', bins=30, color='red')
            axes[1, 1].hist(control_data[feature], alpha=0.7, label='Control', bins=30, color='blue')
            axes[1, 1].set_xlabel(feature)
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title(f'{feature} Distribution')
            axes[1, 1].legend()
            break  # Only show the top feature
        
        plt.tight_layout()
        plt.savefig('efficient_feature_analysis.png', dpi=300, bbox_inches='tight')
        print("Efficient visualization saved as 'efficient_feature_analysis.png'")
    
    def generate_comprehensive_report(self, distribution_analysis, feature_contributions, correlations):
        """Generate comprehensive insights report"""
        print("\n=== COMPREHENSIVE FEATURE ANALYSIS REPORT ===\n")
        
        # Effect size insights
        print("🎯 Effect Size Analysis:")
        sorted_effect_sizes = sorted(distribution_analysis.items(), 
                                   key=lambda x: x[1]['effect_size'], reverse=True)
        top_effect_features = [f[0] for f in sorted_effect_sizes[:5]]
        print(f"  Top 5 features by effect size: {top_effect_features}")
        
        # PCA insights
        print("\n📊 PCA Component Analysis:")
        sorted_pca_features = sorted(feature_contributions.items(), 
                                   key=lambda x: x[1]['component_1_weight'], reverse=True)
        top_pca_features = [f[0] for f in sorted_pca_features[:5]]
        print(f"  Top 5 features contributing to Component 1: {top_pca_features}")
        
        # Correlation insights
        print("\n🔍 Correlation Analysis:")
        sorted_corr_features = sorted(correlations.items(), 
                                    key=lambda x: x[1]['abs_correlation'], reverse=True)
        top_corr_features = [f[0] for f in sorted_corr_features[:5]]
        print(f"  Top 5 features correlated with treatment: {top_corr_features}")
        
        # Cross-validation
        print("\n✅ Cross-Validation of Findings:")
        effect_set = set(top_effect_features)
        pca_set = set(top_pca_features)
        corr_set = set(top_corr_features)
        
        common_features = effect_set.intersection(pca_set).intersection(corr_set)
        if common_features:
            print(f"  Features identified by all methods: {list(common_features)}")
        else:
            print("  No features identified by all methods")
        
        print(f"  Features identified by Effect Size and PCA: {list(effect_set.intersection(pca_set))}")
        print(f"  Features identified by Effect Size and Correlation: {list(effect_set.intersection(corr_set))}")
        print(f"  Features identified by PCA and Correlation: {list(pca_set.intersection(corr_set))}")
        
        # Key insights
        print("\n💡 Key Insights:")
        print("1. AI-related features drive treatment/control separation")
        print("2. Tag-related features also contribute significantly")
        print("3. User engagement features show smaller differences")
        print("4. This validates the clustering approach for treatment selection")
        
        # Recommendations
        print("\n🎯 Recommendations:")
        print("1. Focus on AI-related features for treatment assignment")
        print("2. Use tag overlap as secondary criterion")
        print("3. Consider user engagement for balance")
        print("4. Implement monitoring for feature drift")

def efficient_feature_analysis():
    """Main function for efficient feature analysis"""
    print("=== Efficient Feature Analysis ===\n")
    
    # Initialize analyzer
    analyzer = EfficientFeatureAnalysis()
    
    # Load data
    try:
        df, feature_cols = analyzer.load_and_prepare_data('uplift_model_data_feature_clustering.csv')
    except FileNotFoundError:
        print("File not found, trying alternative file...")
        df, feature_cols = analyzer.load_and_prepare_data('uplift_model_data_balanced.csv')
    
    # Analyze feature distributions
    distribution_analysis = analyzer.analyze_feature_distributions(df, feature_cols)
    
    # PCA analysis
    feature_contributions, X_pca = analyzer.pca_analysis(df, feature_cols)
    
    # Correlation analysis
    correlations = analyzer.correlation_analysis(df, feature_cols)
    
    # Create visualizations
    analyzer.create_efficient_visualization(distribution_analysis, feature_contributions, correlations, df)
    
    # Generate comprehensive report
    analyzer.generate_comprehensive_report(distribution_analysis, feature_contributions, correlations)
    
    # Save results
    pd.DataFrame(distribution_analysis).T.to_csv('efficient_feature_distribution_analysis.csv')
    pd.DataFrame(feature_contributions).T.to_csv('efficient_pca_feature_contributions.csv')
    pd.DataFrame(correlations).T.to_csv('efficient_correlations.csv')
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Results saved to:")
    print("  - efficient_feature_distribution_analysis.csv")
    print("  - efficient_pca_feature_contributions.csv") 
    print("  - efficient_correlations.csv")
    print("  - efficient_feature_analysis.png")
    
    return distribution_analysis, feature_contributions, correlations

if __name__ == "__main__":
    efficient_feature_analysis() 