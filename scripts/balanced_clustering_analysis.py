import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

class BalancedClusteringAnalysis:
    """Balanced clustering analysis with proper treatment/control ratios"""
    
    def __init__(self, n_clusters=10, target_ratio=1.0):
        self.n_clusters = n_clusters
        self.target_ratio = target_ratio  # 1.0 for 1:1, 2.0 for 2:1
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Prepare features for clustering"""
        print("Preparing features for clustering...")
        
        # Select numerical features for clustering
        exclude_cols = ['treatment_ai_content', 'response', 'user_id', 'post_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        print(f"Selected {len(feature_cols)} features for clustering")
        
        # Prepare feature matrix
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, feature_cols
    
    def perform_clustering(self, X_scaled):
        """Perform clustering on scaled features"""
        print(f"Performing K-means clustering with {self.n_clusters} clusters...")
        
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        
        clusters = kmeans.fit_predict(X_scaled)
        print(f"Clustering completed. Found {len(np.unique(clusters))} clusters")
        
        return clusters, kmeans
    
    def identify_ai_clusters(self, df, clusters, feature_cols):
        """Identify clusters that contain AI-related content"""
        print("Identifying AI-related clusters...")
        
        # Define AI-related features
        ai_related_features = [
            'user_ai_interest_score', 'user_ai_interest_weighted', 'user_ai_interactions',
            'ai_interest_x_treatment', 'user_previous_ai_click_rate'
        ]
        
        # Find available AI-related features
        available_ai_features = [col for col in ai_related_features if col in feature_cols]
        print(f"Available AI-related features: {available_ai_features}")
        
        # Calculate AI scores for each cluster
        cluster_ai_scores = {}
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_data = df[cluster_mask]
            
            ai_feature_scores = []
            for feature in available_ai_features:
                if feature in cluster_data.columns:
                    feature_mean = cluster_data[feature].mean()
                    ai_feature_scores.append(feature_mean)
            
            cluster_ai_score = np.mean(ai_feature_scores) if ai_feature_scores else 0
            cluster_ai_scores[cluster_id] = cluster_ai_score
        
        # Sort clusters by AI score
        sorted_clusters = sorted(cluster_ai_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("Cluster AI scores:")
        for cluster_id, score in sorted_clusters[:5]:
            cluster_size = (clusters == cluster_id).sum()
            print(f"  Cluster {cluster_id}: {score:.4f} (size: {cluster_size:,})")
        
        # Select top clusters as AI clusters
        ai_cluster_threshold = np.mean([score for _, score in sorted_clusters]) + np.std([score for _, score in sorted_clusters])
        ai_clusters = [cluster_id for cluster_id, score in sorted_clusters if score >= ai_cluster_threshold]
        
        if len(ai_clusters) == 0:
            ai_clusters = [cluster_id for cluster_id, _ in sorted_clusters[:3]]
        
        print(f"Selected {len(ai_clusters)} AI-related clusters (threshold: {ai_cluster_threshold:.4f})")
        
        return ai_clusters, cluster_ai_scores
    
    def balance_treatment_control(self, df, clusters, ai_clusters):
        """Balance treatment and control groups with proper ratios"""
        print("Balancing treatment and control groups...")
        
        # Select posts from AI clusters
        ai_cluster_mask = np.isin(clusters, ai_clusters)
        ai_cluster_posts = df[ai_cluster_mask].copy()
        
        print(f"Posts in AI clusters: {len(ai_cluster_posts):,}")
        
        # Create treatment labels if not exists
        if 'treatment_ai_content' not in ai_cluster_posts.columns:
            ai_feature_cols = [col for col in ai_cluster_posts.columns if 'ai' in col.lower()]
            if ai_feature_cols:
                ai_feature = ai_feature_cols[0]
                threshold = ai_cluster_posts[ai_feature].median()
                ai_cluster_posts['treatment_ai_content'] = (ai_cluster_posts[ai_feature] > threshold).astype(int)
                print(f"Created treatment labels based on {ai_feature} (threshold: {threshold:.4f})")
            else:
                ai_cluster_posts['treatment_ai_content'] = np.random.choice(
                    [0, 1], size=len(ai_cluster_posts), p=[0.7, 0.3]
                )
                print("Created random treatment labels")
        
        # Split into treatment and control
        treatment_posts = ai_cluster_posts[ai_cluster_posts['treatment_ai_content'] == 1]
        control_posts = ai_cluster_posts[ai_cluster_posts['treatment_ai_content'] == 0]
        
        print(f"Original - Treatment: {len(treatment_posts):,}, Control: {len(control_posts):,}")
        
        # Balance the groups
        if len(treatment_posts) > len(control_posts) * self.target_ratio:
            # Sample treatment posts to match target ratio
            target_treatment_size = int(len(control_posts) * self.target_ratio)
            treatment_posts = treatment_posts.sample(n=target_treatment_size, random_state=42)
            print(f"Sampled treatment posts to: {len(treatment_posts):,}")
        elif len(control_posts) > len(treatment_posts) / self.target_ratio:
            # Sample control posts to match target ratio
            target_control_size = int(len(treatment_posts) / self.target_ratio)
            control_posts = control_posts.sample(n=target_control_size, random_state=42)
            print(f"Sampled control posts to: {len(control_posts):,}")
        
        print(f"Balanced - Treatment: {len(treatment_posts):,}, Control: {len(control_posts):,}")
        print(f"Ratio: {len(treatment_posts)/len(control_posts):.2f}:1")
        
        # Combine treatment and control
        final_df = pd.concat([treatment_posts, control_posts], ignore_index=True)
        
        return final_df, treatment_posts, control_posts
    
    def create_tsne_visualization(self, df, feature_cols):
        """Create t-SNE visualization for treatment vs control"""
        print("Creating t-SNE visualization...")
        
        # Prepare features for t-SNE
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Scatter plot
        treatment_mask = df['treatment_ai_content'] == 1
        plt.scatter(X_tsne[~treatment_mask, 0], X_tsne[~treatment_mask, 1], 
                   c='blue', alpha=0.6, s=20, label='Control')
        plt.scatter(X_tsne[treatment_mask, 0], X_tsne[treatment_mask, 1], 
                   c='red', alpha=0.6, s=20, label='Treatment')
        
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Visualization: Treatment vs Control Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('tsne_treatment_control.png', dpi=300, bbox_inches='tight')
        print("t-SNE visualization saved as 'tsne_treatment_control.png'")
        
        return X_tsne
    
    def create_umap_visualization(self, df, feature_cols):
        """Create UMAP visualization for treatment vs control"""
        print("Creating UMAP visualization...")
        
        # Prepare features for UMAP
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Apply UMAP
        reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap = reducer.fit_transform(X_scaled)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Scatter plot
        treatment_mask = df['treatment_ai_content'] == 1
        plt.scatter(X_umap[~treatment_mask, 0], X_umap[~treatment_mask, 1], 
                   c='blue', alpha=0.6, s=20, label='Control')
        plt.scatter(X_umap[treatment_mask, 0], X_umap[treatment_mask, 1], 
                   c='red', alpha=0.6, s=20, label='Treatment')
        
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.title('UMAP Visualization: Treatment vs Control Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('umap_treatment_control.png', dpi=300, bbox_inches='tight')
        print("UMAP visualization saved as 'umap_treatment_control.png'")
        
        return X_umap
    
    def bootstrap_uplift_analysis(self, df, n_bootstrap=1000):
        """Perform bootstrap analysis for uplift estimation"""
        print(f"Performing bootstrap analysis with {n_bootstrap} iterations...")
        
        uplift_samples = []
        treatment_response_rates = []
        control_response_rates = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_sample = resample(df, random_state=i)
            
            # Calculate response rates
            treatment_response_rate = bootstrap_sample[bootstrap_sample['treatment_ai_content'] == 1]['response'].mean()
            control_response_rate = bootstrap_sample[bootstrap_sample['treatment_ai_content'] == 0]['response'].mean()
            uplift = treatment_response_rate - control_response_rate
            
            uplift_samples.append(uplift)
            treatment_response_rates.append(treatment_response_rate)
            control_response_rates.append(control_response_rate)
        
        # Calculate confidence intervals
        uplift_mean = np.mean(uplift_samples)
        uplift_std = np.std(uplift_samples)
        uplift_ci_95 = np.percentile(uplift_samples, [2.5, 97.5])
        uplift_ci_90 = np.percentile(uplift_samples, [5, 95])
        
        print(f"\n=== Bootstrap Uplift Analysis ===")
        print(f"Mean uplift: {uplift_mean:.4f}")
        print(f"Standard deviation: {uplift_std:.4f}")
        print(f"95% Confidence interval: [{uplift_ci_95[0]:.4f}, {uplift_ci_95[1]:.4f}]")
        print(f"90% Confidence interval: [{uplift_ci_90[0]:.4f}, {uplift_ci_90[1]:.4f}]")
        
        # Create bootstrap distribution plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(uplift_samples, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(uplift_mean, color='red', linestyle='--', label=f'Mean: {uplift_mean:.4f}')
        plt.axvline(uplift_ci_95[0], color='orange', linestyle=':', label=f'95% CI: [{uplift_ci_95[0]:.4f}, {uplift_ci_95[1]:.4f}]')
        plt.axvline(uplift_ci_95[1], color='orange', linestyle=':')
        plt.xlabel('Uplift')
        plt.ylabel('Frequency')
        plt.title('Bootstrap Uplift Distribution')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.hist(treatment_response_rates, bins=50, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('Treatment Response Rate')
        plt.ylabel('Frequency')
        plt.title('Treatment Response Rate Distribution')
        
        plt.subplot(2, 2, 3)
        plt.hist(control_response_rates, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Control Response Rate')
        plt.ylabel('Frequency')
        plt.title('Control Response Rate Distribution')
        
        plt.subplot(2, 2, 4)
        plt.scatter(control_response_rates, treatment_response_rates, alpha=0.5)
        plt.xlabel('Control Response Rate')
        plt.ylabel('Treatment Response Rate')
        plt.title('Treatment vs Control Response Rates')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Diagonal line
        
        plt.tight_layout()
        plt.savefig('bootstrap_uplift_analysis.png', dpi=300, bbox_inches='tight')
        print("Bootstrap analysis visualization saved as 'bootstrap_uplift_analysis.png'")
        
        return {
            'uplift_mean': uplift_mean,
            'uplift_std': uplift_std,
            'uplift_ci_95': uplift_ci_95,
            'uplift_ci_90': uplift_ci_90,
            'uplift_samples': uplift_samples
        }
    
    def analyze_feature_distribution(self, df, feature_cols):
        """Analyze feature distribution between treatment and control"""
        print("\n=== Feature Distribution Analysis ===")
        
        treatment_data = df[df['treatment_ai_content'] == 1]
        control_data = df[df['treatment_ai_content'] == 0]
        
        # Select top features for analysis
        top_features = feature_cols[:10]  # Analyze top 10 features
        
        for feature in top_features:
            treatment_mean = treatment_data[feature].mean()
            control_mean = control_data[feature].mean()
            treatment_std = treatment_data[feature].std()
            control_std = control_data[feature].std()
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((treatment_std**2 + control_std**2) / 2)
            effect_size = abs(treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
            
            print(f"{feature}:")
            print(f"  Treatment: mean={treatment_mean:.4f}, std={treatment_std:.4f}")
            print(f"  Control: mean={control_mean:.4f}, std={control_std:.4f}")
            print(f"  Effect size: {effect_size:.4f}")
            print()

def balanced_clustering_analysis():
    """Main function for balanced clustering analysis"""
    print("=== Balanced Clustering Analysis ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Initialize analyzer
    analyzer = BalancedClusteringAnalysis(n_clusters=10, target_ratio=1.0)  # 1:1 ratio
    
    # Prepare features
    X_scaled, feature_cols = analyzer.prepare_features(df)
    
    # Perform clustering
    clusters, clusterer = analyzer.perform_clustering(X_scaled)
    
    # Identify AI clusters
    ai_clusters, cluster_scores = analyzer.identify_ai_clusters(df, clusters, feature_cols)
    
    # Balance treatment and control
    final_df, treatment_posts, control_posts = analyzer.balance_treatment_control(
        df, clusters, ai_clusters
    )
    
    # Create visualizations
    X_tsne = analyzer.create_tsne_visualization(final_df, feature_cols)
    X_umap = analyzer.create_umap_visualization(final_df, feature_cols)
    
    # Analyze feature distribution
    analyzer.analyze_feature_distribution(final_df, feature_cols)
    
    # Perform bootstrap analysis
    bootstrap_results = analyzer.bootstrap_uplift_analysis(final_df, n_bootstrap=500)
    
    # Final analysis
    print("\n=== Final Dataset Analysis ===")
    print(f"Total samples: {len(final_df):,}")
    print(f"Treatment samples: {len(treatment_posts):,}")
    print(f"Control samples: {len(control_posts):,}")
    
    # Check response distribution
    if 'response' in final_df.columns:
        treatment_response_rate = final_df[final_df['treatment_ai_content'] == 1]['response'].mean()
        control_response_rate = final_df[final_df['treatment_ai_content'] == 0]['response'].mean()
        uplift = treatment_response_rate - control_response_rate
        
        print(f"\nUplift Analysis:")
        print(f"  Treatment response rate: {treatment_response_rate:.2%}")
        print(f"  Control response rate: {control_response_rate:.2%}")
        print(f"  Uplift: {uplift:.2%}")
        
        if uplift > 0:
            print("✅ Positive uplift detected")
        elif uplift < 0:
            print("⚠️  Negative uplift detected")
        else:
            print("➖ No uplift detected")
    
    # Save results
    output_file = 'uplift_model_data_balanced.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\nBalanced dataset saved to: {output_file}")
    
    return final_df, bootstrap_results

if __name__ == "__main__":
    balanced_clustering_analysis() 