import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureBasedClustering:
    """Feature-based clustering for treatment and control selection"""
    
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Prepare features for clustering"""
        print("Preparing features for clustering...")
        
        # Select numerical features for clustering
        # Exclude target variables and ID columns
        exclude_cols = ['treatment_ai_content', 'response', 'user_id', 'post_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
        
        print(f"Selected {len(feature_cols)} features for clustering")
        print("Feature columns:", feature_cols[:10], "..." if len(feature_cols) > 10 else "")
        
        # Prepare feature matrix
        X = df[feature_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, feature_cols
    
    def prepare_content_features(self, df):
        """Automatically detect content-related columns and create TF-IDF features for clustering"""
        try:
            print("Detecting content-related columns for clustering...")
            content_keywords = ['content', 'title', 'tag', 'text', 'body', 'description']
            text_columns = [col for col in df.columns if any(kw in col.lower() for kw in content_keywords)]
            print(f"Detected text columns: {text_columns}")
            if not text_columns:
                raise ValueError("No content-related text columns found in the dataset.")
            # Combine all text columns into a single string per row
            combined_text = df[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
            # TF-IDF vectorization
            print("Creating TF-IDF features for clustering...")
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2), min_df=2, max_df=0.95)
            tfidf_matrix = vectorizer.fit_transform(combined_text)
            print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
            return tfidf_matrix, vectorizer, text_columns
        except Exception as e:
            print(f"[ERROR] Exception in prepare_content_features: {e}")
            raise
    
    def perform_clustering(self, X_scaled):
        """Perform clustering on scaled features using MiniBatchKMeans"""
        print(f"Performing MiniBatchKMeans clustering with {self.n_clusters} clusters...")
        
        minibatch_kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=1000,
            max_iter=100
        )
        
        clusters = minibatch_kmeans.fit_predict(X_scaled)
        
        print(f"Clustering completed. Found {len(np.unique(clusters))} clusters")
        
        return clusters, minibatch_kmeans
    
    def identify_ai_clusters(self, df, clusters, feature_cols):
        """Identify clusters that contain AI-related content based on feature patterns"""
        print("Identifying AI-related clusters...")
        
        # Calculate AI-related feature scores for each cluster
        cluster_ai_scores = {}
        
        # Define AI-related features
        ai_related_features = [
            'user_ai_interest_score', 'user_ai_interest_weighted', 'user_ai_interactions',
            'ai_interest_x_treatment', 'user_previous_ai_click_rate'
        ]
        
        # Find available AI-related features
        available_ai_features = [col for col in ai_related_features if col in feature_cols]
        print(f"Available AI-related features: {available_ai_features}")
        
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_data = df[cluster_mask]
            
            # Calculate AI feature density for this cluster
            ai_feature_scores = []
            for feature in available_ai_features:
                if feature in cluster_data.columns:
                    # Calculate mean value of AI feature in this cluster
                    feature_mean = cluster_data[feature].mean()
                    ai_feature_scores.append(feature_mean)
            
            # Calculate overall AI score for this cluster
            if ai_feature_scores:
                cluster_ai_score = np.mean(ai_feature_scores)
                cluster_ai_scores[cluster_id] = cluster_ai_score
            else:
                cluster_ai_scores[cluster_id] = 0
        
        # Sort clusters by AI score
        sorted_clusters = sorted(cluster_ai_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("Cluster AI scores:")
        for cluster_id, score in sorted_clusters[:5]:
            cluster_size = (clusters == cluster_id).sum()
            print(f"  Cluster {cluster_id}: {score:.4f} (size: {cluster_size:,})")
        
        # Select top clusters as AI clusters
        # Use top 30% of clusters or clusters with score > mean + std
        ai_cluster_threshold = np.mean([score for _, score in sorted_clusters]) + np.std([score for _, score in sorted_clusters])
        ai_clusters = [cluster_id for cluster_id, score in sorted_clusters if score >= ai_cluster_threshold]
        
        # If no clusters meet threshold, take top 3
        if len(ai_clusters) == 0:
            ai_clusters = [cluster_id for cluster_id, _ in sorted_clusters[:3]]
        
        print(f"Selected {len(ai_clusters)} AI-related clusters (threshold: {ai_cluster_threshold:.4f})")
        
        return ai_clusters, cluster_ai_scores
    
    def select_treatment_and_control(self, df, clusters, ai_clusters):
        """Select treatment and control groups from AI clusters"""
        print("Selecting treatment and control groups...")
        
        # Create mask for AI cluster posts
        ai_cluster_mask = np.isin(clusters, ai_clusters)
        ai_cluster_posts = df[ai_cluster_mask].copy()
        
        print(f"Posts in AI clusters: {len(ai_cluster_posts):,}")
        
        # Check if treatment_ai_content column exists
        if 'treatment_ai_content' in ai_cluster_posts.columns:
            print("Using existing treatment_ai_content column")
        else:
            print("Creating treatment_ai_content column based on AI features")
            # Create treatment labels based on AI feature values
            ai_feature_cols = [col for col in ai_cluster_posts.columns if 'ai' in col.lower()]
            if ai_feature_cols:
                # Use the first AI feature to create treatment labels
                ai_feature = ai_feature_cols[0]
                threshold = ai_cluster_posts[ai_feature].median()
                ai_cluster_posts['treatment_ai_content'] = (ai_cluster_posts[ai_feature] > threshold).astype(int)
                print(f"Created treatment labels based on {ai_feature} (threshold: {threshold:.4f})")
            else:
                # Create random treatment labels
                ai_cluster_posts['treatment_ai_content'] = np.random.choice(
                    [0, 1], size=len(ai_cluster_posts), p=[0.7, 0.3]
                )
                print("Created random treatment labels")
        
        # Split into treatment and control
        treatment_posts = ai_cluster_posts[ai_cluster_posts['treatment_ai_content'] == 1]
        control_posts = ai_cluster_posts[ai_cluster_posts['treatment_ai_content'] == 0]
        
        print(f"Treatment posts in AI clusters: {len(treatment_posts):,}")
        print(f"Control posts in AI clusters: {len(control_posts):,}")
        
        # Balance the groups if needed
        if len(control_posts) > len(treatment_posts) * 2:
            # Sample control posts to be roughly 2x treatment size
            control_sample_size = min(len(control_posts), len(treatment_posts) * 2)
            control_posts = control_posts.sample(n=control_sample_size, random_state=42)
            print(f"Sampled control posts to: {len(control_posts):,}")
        
        # Combine treatment and control
        final_df = pd.concat([treatment_posts, control_posts], ignore_index=True)
        
        return final_df, treatment_posts, control_posts
    
    def analyze_cluster_composition(self, df, clusters, ai_clusters, feature_cols):
        """Analyze the composition of AI clusters"""
        print("\n=== AI Cluster Analysis ===")
        
        for cluster_id in ai_clusters:
            cluster_mask = clusters == cluster_id
            cluster_posts = df[cluster_mask]
            
            print(f"\nCluster {cluster_id} (size: {len(cluster_posts):,}):")
            
            # Show feature statistics
            ai_features = [col for col in feature_cols if 'ai' in col.lower()]
            if ai_features:
                print("AI feature statistics:")
                for feature in ai_features[:5]:  # Show top 5 AI features
                    mean_val = cluster_posts[feature].mean()
                    std_val = cluster_posts[feature].std()
                    print(f"  {feature}: mean={mean_val:.4f}, std={std_val:.4f}")
            
            # Show treatment distribution
            if 'treatment_ai_content' in cluster_posts.columns:
                treatment_dist = cluster_posts['treatment_ai_content'].value_counts(normalize=True)
                print("Treatment distribution:")
                for value, ratio in treatment_dist.items():
                    print(f"  {value}: {ratio:.1%}")
    
    def visualize_clusters(self, df, clusters, ai_clusters, feature_cols):
        """Visualize cluster distribution"""
        print("Creating cluster visualization...")
        
        # Use PCA for visualization
        pca = PCA(n_components=2)
        X_scaled, _ = self.prepare_features(df)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create visualization data
        cluster_data = pd.DataFrame({
            'PCA1': X_pca[:, 0],
            'PCA2': X_pca[:, 1],
            'Cluster': clusters,
            'IsAICluster': np.isin(clusters, ai_clusters)
        })
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Scatter plot
        colors = ['red' if is_ai else 'blue' for is_ai in cluster_data['IsAICluster']]
        plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], c=colors, alpha=0.6, s=10)
        
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Cluster Analysis: PCA Visualization')
        plt.legend(['AI Clusters', 'Other Clusters'])
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig('feature_cluster_analysis.png', dpi=300, bbox_inches='tight')
        print("Cluster visualization saved as 'feature_cluster_analysis.png'")
        
        return cluster_data
    
    def analyze_feature_importance(self, df, clusters, ai_clusters, feature_cols):
        """Analyze feature importance for clustering"""
        print("\n=== Feature Importance Analysis ===")
        
        # Calculate feature importance based on cluster separation
        feature_importance = {}
        
        for feature in feature_cols:
            # Calculate variance between AI clusters and other clusters
            ai_cluster_mask = np.isin(clusters, ai_clusters)
            ai_cluster_values = df[ai_cluster_mask][feature]
            other_cluster_values = df[~ai_cluster_mask][feature]
            
            if len(ai_cluster_values) > 0 and len(other_cluster_values) > 0:
                # Calculate effect size (Cohen's d)
                mean_diff = ai_cluster_values.mean() - other_cluster_values.mean()
                pooled_std = np.sqrt((ai_cluster_values.var() + other_cluster_values.var()) / 2)
                effect_size = abs(mean_diff / pooled_std) if pooled_std > 0 else 0
                feature_importance[feature] = effect_size
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("Top 10 most important features for AI cluster identification:")
        for feature, importance in sorted_features[:10]:
            print(f"  {feature}: {importance:.4f}")
        
        return feature_importance

    def analyze_cluster_distribution(self, df, clusters, feature_cols):
        """Analyze and print cluster size and feature means for each cluster"""
        print("\n=== Cluster Distribution Analysis ===")
        unique_clusters = np.unique(clusters)
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_data = df[cluster_mask]
            print(f"Cluster {cluster_id}: size = {len(cluster_data):,}")
            # Print mean of first 5 features for illustration
            for feature in feature_cols[:5]:
                mean_val = cluster_data[feature].mean()
                print(f"  {feature}: mean = {mean_val:.4f}")
        print("\nCluster distribution analysis complete.")

def feature_based_clustering():
    """Main function for content-based clustering (text only)"""
    print("=== Content-Based Clustering (Text Features Only) ===\n")
    try:
        # Load data
        df = pd.read_csv('uplift_model_data.csv')
        print(f"Total data volume: {len(df):,}")
        selector = FeatureBasedClustering(n_clusters=10)
        # Prepare content features
        tfidf_matrix, vectorizer, text_columns = selector.prepare_content_features(df)
        # Perform clustering
        clusters, clusterer = selector.perform_clustering(tfidf_matrix)
        # Analyze cluster distribution
        print("\n=== Cluster Distribution Analysis ===")
        unique_clusters = np.unique(clusters)
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_data = df[cluster_mask]
            print(f"Cluster {cluster_id}: size = {len(cluster_data):,}")
            # Show top keywords in this cluster
            if tfidf_matrix.shape[1] > 0:
                cluster_tfidf = tfidf_matrix[cluster_mask].mean(axis=0)
                top_indices = np.asarray(cluster_tfidf).ravel().argsort()[-10:][::-1]
                top_keywords = [vectorizer.get_feature_names_out()[i] for i in top_indices]
                print(f"  Top keywords: {top_keywords}")
        print("\nClustering process complete. No treatment/control selection performed.")
        return clusters
    except Exception as e:
        print(f"[ERROR] Exception in feature_based_clustering: {e}")
        return None

if __name__ == "__main__":
    feature_based_clustering() 