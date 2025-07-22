import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceAnalysis:
    """Feature importance analysis to understand treatment/control separation"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.xgb_model = None
        
    def load_and_prepare_data(self, file_path):
        """Load and prepare data for analysis"""
        print(f"Loading data from {file_path}...")
        
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df):,} samples with {len(df.columns)} columns")
        
        # Prepare features
        exclude_cols = ['treatment_ai_content', 'response', 'user_id', 'post_id', 'propensity_score']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        print(f"Using {len(feature_cols)} features for analysis")
        
        # Prepare feature matrix
        X = df[feature_cols].fillna(0)
        y = df['treatment_ai_content']
        
        return df, X, y, feature_cols
    
    def xgboost_feature_importance(self, X, y, feature_cols):
        """Analyze feature importance using XGBoost"""
        print("\n=== XGBoost Feature Importance Analysis ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train XGBoost model
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.xgb_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.xgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        importance_scores = self.xgb_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return feature_importance
    
    def pca_analysis(self, X, feature_cols):
        """Analyze PCA components to understand feature contributions"""
        print("\n=== PCA Analysis ===")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit PCA
        self.pca = PCA(n_components=min(10, len(feature_cols)))
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Analyze explained variance
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        print(f"Explained variance by components:")
        for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
            print(f"  Component {i+1}: {var_ratio:.4f} (Cumulative: {cum_var:.4f})")
        
        # Analyze feature contributions to first component
        feature_contributions = pd.DataFrame({
            'feature': feature_cols,
            'component_1_weight': abs(self.pca.components_[0]),
            'component_2_weight': abs(self.pca.components_[1])
        }).sort_values('component_1_weight', ascending=False)
        
        print(f"\nTop 10 Features Contributing to Component 1:")
        for i, row in feature_contributions.head(10).iterrows():
            print(f"  {row['feature']}: {row['component_1_weight']:.4f}")
        
        print(f"\nTop 10 Features Contributing to Component 2:")
        for i, row in feature_contributions.sort_values('component_2_weight', ascending=False).head(10).iterrows():
            print(f"  {row['feature']}: {row['component_2_weight']:.4f}")
        
        return feature_contributions, X_pca
    
    def tsne_component_analysis(self, X, feature_cols, df):
        """Analyze which features correlate with t-SNE components"""
        print("\n=== t-SNE Component Analysis ===")
        
        # Prepare features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Add t-SNE components to dataframe
        df_analysis = df.copy()
        df_analysis['tsne_component_1'] = X_tsne[:, 0]
        df_analysis['tsne_component_2'] = X_tsne[:, 1]
        
        # Calculate correlations with original features
        correlations = {}
        for feature in feature_cols:
            corr_comp1 = df_analysis[feature].corr(df_analysis['tsne_component_1'])
            corr_comp2 = df_analysis[feature].corr(df_analysis['tsne_component_2'])
            correlations[feature] = {
                'correlation_comp1': corr_comp1,
                'correlation_comp2': corr_comp2,
                'abs_corr_comp1': abs(corr_comp1),
                'abs_corr_comp2': abs(corr_comp2)
            }
        
        # Sort by correlation with component 1
        comp1_correlations = pd.DataFrame(correlations).T.sort_values('abs_corr_comp1', ascending=False)
        
        print("Top 10 Features Correlated with t-SNE Component 1:")
        for feature in comp1_correlations.head(10).index:
            corr = comp1_correlations.loc[feature, 'correlation_comp1']
            print(f"  {feature}: {corr:.4f}")
        
        print("\nTop 10 Features Correlated with t-SNE Component 2:")
        comp2_correlations = pd.DataFrame(correlations).T.sort_values('abs_corr_comp2', ascending=False)
        for feature in comp2_correlations.head(10).index:
            corr = comp2_correlations.loc[feature, 'correlation_comp2']
            print(f"  {feature}: {corr:.4f}")
        
        return correlations, X_tsne
    
    def create_comprehensive_visualization(self, feature_importance, feature_contributions, correlations, X_tsne, df):
        """Create comprehensive visualization of feature importance analysis"""
        print("\nCreating comprehensive visualization...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. XGBoost Feature Importance
        top_features = feature_importance.head(10)
        axes[0, 0].barh(range(len(top_features)), top_features['importance'], color='skyblue')
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features['feature'])
        axes[0, 0].set_xlabel('Importance Score')
        axes[0, 0].set_title('XGBoost Feature Importance')
        
        # 2. PCA Component 1 Contributions
        top_pca_features = feature_contributions.head(10)
        axes[0, 1].barh(range(len(top_pca_features)), top_pca_features['component_1_weight'], color='lightgreen')
        axes[0, 1].set_yticks(range(len(top_pca_features)))
        axes[0, 1].set_yticklabels(top_pca_features['feature'])
        axes[0, 1].set_xlabel('Component 1 Weight')
        axes[0, 1].set_title('PCA Component 1 Contributions')
        
        # 3. t-SNE Component 1 Correlations
        top_corr_features = pd.DataFrame(correlations).T.sort_values('abs_corr_comp1', ascending=False).head(10)
        axes[0, 2].barh(range(len(top_corr_features)), top_corr_features['correlation_comp1'], color='lightcoral')
        axes[0, 2].set_yticks(range(len(top_corr_features)))
        axes[0, 2].set_yticklabels(top_corr_features.index)
        axes[0, 2].set_xlabel('Correlation with t-SNE Component 1')
        axes[0, 2].set_title('t-SNE Component 1 Correlations')
        
        # 4. t-SNE visualization
        treatment_mask = df['treatment_ai_content'] == 1
        axes[1, 0].scatter(X_tsne[~treatment_mask, 0], X_tsne[~treatment_mask, 1], 
                           c='blue', alpha=0.6, s=20, label='Control')
        axes[1, 0].scatter(X_tsne[treatment_mask, 0], X_tsne[treatment_mask, 1], 
                           c='red', alpha=0.6, s=20, label='Treatment')
        axes[1, 0].set_xlabel('t-SNE Component 1')
        axes[1, 0].set_ylabel('t-SNE Component 2')
        axes[1, 0].set_title('t-SNE: Treatment vs Control')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Feature distribution comparison (top 3 important features)
        top_3_features = feature_importance.head(3)['feature'].tolist()
        treatment_data = df[df['treatment_ai_content'] == 1]
        control_data = df[df['treatment_ai_content'] == 0]
        
        for i, feature in enumerate(top_3_features):
            row = 1
            col = i + 1
            
            axes[row, col].hist(treatment_data[feature], alpha=0.7, label='Treatment', bins=30, color='red')
            axes[row, col].hist(control_data[feature], alpha=0.7, label='Control', bins=30, color='blue')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].set_title(f'{feature} Distribution')
            axes[row, col].legend()
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        print("Comprehensive feature importance visualization saved as 'feature_importance_analysis.png'")
    
    def create_correlation_heatmap(self, correlations, feature_cols):
        """Create correlation heatmap for t-SNE components"""
        print("Creating correlation heatmap...")
        
        # Prepare correlation data
        corr_data = pd.DataFrame(correlations).T[['correlation_comp1', 'correlation_comp2']]
        corr_data.columns = ['t-SNE Component 1', 't-SNE Component 2']
        
        # Select top features
        top_features = corr_data.abs().max(axis=1).sort_values(ascending=False).head(15).index
        corr_subset = corr_data.loc[top_features]
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_subset.T, annot=True, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Feature Correlations with t-SNE Components')
        plt.tight_layout()
        plt.savefig('tsne_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Correlation heatmap saved as 'tsne_correlation_heatmap.png'")
    
    def generate_insights_report(self, feature_importance, feature_contributions, correlations):
        """Generate insights report"""
        print("\n=== FEATURE IMPORTANCE INSIGHTS ===\n")
        
        # XGBoost insights
        print("🎯 XGBoost Feature Importance Insights:")
        top_xgb_features = feature_importance.head(5)['feature'].tolist()
        print(f"  Top 5 features driving treatment/control separation: {top_xgb_features}")
        
        # PCA insights
        print("\n📊 PCA Component Analysis Insights:")
        top_pca_features = feature_contributions.head(5)['feature'].tolist()
        print(f"  Top 5 features contributing to Component 1: {top_pca_features}")
        
        # t-SNE insights
        print("\n🔍 t-SNE Component Analysis Insights:")
        top_corr_features = pd.DataFrame(correlations).T.sort_values('abs_corr_comp1', ascending=False).head(5).index.tolist()
        print(f"  Top 5 features correlated with t-SNE Component 1: {top_corr_features}")
        
        # Cross-validation
        print("\n✅ Cross-Validation of Findings:")
        xgb_set = set(top_xgb_features)
        pca_set = set(top_pca_features)
        tsne_set = set(top_corr_features)
        
        common_features = xgb_set.intersection(pca_set).intersection(tsne_set)
        if common_features:
            print(f"  Features identified by all methods: {list(common_features)}")
        else:
            print("  No features identified by all methods")
        
        print(f"  Features identified by XGBoost and PCA: {list(xgb_set.intersection(pca_set))}")
        print(f"  Features identified by XGBoost and t-SNE: {list(xgb_set.intersection(tsne_set))}")
        print(f"  Features identified by PCA and t-SNE: {list(pca_set.intersection(tsne_set))}")

def feature_importance_analysis():
    """Main function for feature importance analysis"""
    print("=== Feature Importance Analysis ===\n")
    
    # Initialize analyzer
    analyzer = FeatureImportanceAnalysis()
    
    # Load data
    try:
        df, X, y, feature_cols = analyzer.load_and_prepare_data('uplift_model_data_feature_clustering.csv')
    except FileNotFoundError:
        print("File not found, trying alternative file...")
        df, X, y, feature_cols = analyzer.load_and_prepare_data('uplift_model_data_balanced.csv')
    
    # XGBoost feature importance
    feature_importance = analyzer.xgboost_feature_importance(X, y, feature_cols)
    
    # PCA analysis
    feature_contributions, X_pca = analyzer.pca_analysis(X, feature_cols)
    
    # t-SNE component analysis
    correlations, X_tsne = analyzer.tsne_component_analysis(X, feature_cols, df)
    
    # Create visualizations
    analyzer.create_comprehensive_visualization(feature_importance, feature_contributions, correlations, X_tsne, df)
    analyzer.create_correlation_heatmap(correlations, feature_cols)
    
    # Generate insights report
    analyzer.generate_insights_report(feature_importance, feature_contributions, correlations)
    
    # Save results
    feature_importance.to_csv('xgboost_feature_importance.csv', index=False)
    feature_contributions.to_csv('pca_feature_contributions.csv', index=False)
    pd.DataFrame(correlations).T.to_csv('tsne_correlations.csv')
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Results saved to:")
    print("  - xgboost_feature_importance.csv")
    print("  - pca_feature_contributions.csv") 
    print("  - tsne_correlations.csv")
    print("  - feature_importance_analysis.png")
    print("  - tsne_correlation_heatmap.png")
    
    return feature_importance, feature_contributions, correlations

if __name__ == "__main__":
    feature_importance_analysis() 