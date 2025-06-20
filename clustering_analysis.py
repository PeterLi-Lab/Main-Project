import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings

class ClusteringAnalyzer:
    def __init__(self, df_combined, embeddings):
        """Initialize clustering analyzer with preprocessed data"""
        self.df_combined = df_combined.copy()
        self.embeddings = embeddings
        self.cluster_labels = None
        self.kmeans_model = None
        self.dbscan_model = None
        self.pca_50d = None
        self.pca_2d = None
        self.reduced_embeddings_2d = None
        
    def perform_pca_reduction(self, n_components_50d=50, n_components_2d=2):
        """Perform PCA dimensionality reduction"""
        print("=== Performing PCA Dimensionality Reduction ===")
        
        # Suppress sklearn deprecation warnings
        warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
        
        # First, reduce to 50 dimensions for faster clustering
        self.pca_50d = PCA(n_components=n_components_50d, random_state=42)
        embeddings_50d = self.pca_50d.fit_transform(self.embeddings)
        
        print(f"Reduced from {self.embeddings.shape[1]} dimensions to {n_components_50d} dimensions")
        print(f"Explained variance ratio: {self.pca_50d.explained_variance_ratio_.sum():.3f}")
        
        # Then reduce to 2D for visualization
        self.pca_2d = PCA(n_components=n_components_2d, random_state=42)
        self.reduced_embeddings_2d = self.pca_2d.fit_transform(embeddings_50d)
        
        print(f"Further reduced to {n_components_2d} dimensions for visualization")
        
        return embeddings_50d, self.reduced_embeddings_2d
    
    def perform_kmeans_clustering(self, embeddings_50d, n_clusters=None):
        """Perform K-means clustering"""
        print("\n=== Performing K-means Clustering ===")
        
        # Determine number of clusters if not provided
        if n_clusters is None:
            n_clusters = min(15, len(embeddings_50d) // 100)  # Adaptive number of clusters
        
        print(f"Using {n_clusters} clusters for {len(embeddings_50d)} samples")
        
        # Perform K-means clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans_model.fit_predict(embeddings_50d)
        
        # Add cluster labels to dataframe
        self.df_combined['Cluster_KMeans'] = self.cluster_labels
        
        print(f"K-means Clustering Results:")
        print(f"Number of clusters: {n_clusters}")
        print(f"Clustering completed successfully!")
        
        return self.cluster_labels
    
    def perform_dbscan_clustering(self, embeddings_50d, eps=None, min_samples=None):
        """Perform DBSCAN clustering"""
        print("\n=== Performing DBSCAN Clustering ===")
        
        # Determine parameters if not provided
        if eps is None:
            # Estimate eps using k-nearest neighbors
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=5).fit(embeddings_50d)
            distances, indices = nbrs.kneighbors(embeddings_50d)
            eps = np.percentile(distances[:, -1], 90)  # 90th percentile
        
        if min_samples is None:
            # Improved min_samples calculation using multiple heuristics
            n_samples = len(embeddings_50d)
            n_dimensions = embeddings_50d.shape[1]
            
            # Heuristic 1: Based on dimensionality (2 * dimensions)
            min_samples_dim = 2 * n_dimensions
            
            # Heuristic 2: Based on dataset size (log of sample size)
            min_samples_size = max(3, int(np.log(n_samples) * 2))
            
            # Heuristic 3: Based on common DBSCAN practice (5-10% of data for small datasets)
            min_samples_percent = max(3, int(n_samples * 0.05))  # 5% of data
            
            # Heuristic 4: Minimum reasonable value for clustering
            min_samples_min = 5
            
            # Take the median of all heuristics, but ensure it's reasonable
            candidates = [min_samples_dim, min_samples_size, min_samples_percent, min_samples_min]
            min_samples = int(np.median(candidates))
            
            # Ensure it's not too large (max 10% of data)
            max_reasonable = min(100, int(n_samples * 0.1))
            min_samples = min(min_samples, max_reasonable)
            
            print(f"Min_samples calculation:")
            print(f"  - Based on dimensions ({n_dimensions}): {min_samples_dim}")
            print(f"  - Based on dataset size ({n_samples}): {min_samples_size}")
            print(f"  - Based on percentage (5%): {min_samples_percent}")
            print(f"  - Final chosen value: {min_samples}")
        
        print(f"DBSCAN parameters - eps: {eps:.4f}, min_samples: {min_samples}")
        
        # Perform DBSCAN clustering
        self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = self.dbscan_model.fit_predict(embeddings_50d)
        
        # Add cluster labels to dataframe
        self.df_combined['Cluster_DBSCAN'] = dbscan_labels
        
        # Count clusters (excluding noise points labeled as -1)
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)
        
        print(f"DBSCAN Clustering Results:")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        print(f"Clustering completed successfully!")
        
        return dbscan_labels
    
    def visualize_clusters_comparison(self):
        """Create comparison visualization for both clustering methods"""
        print("\n=== Creating Cluster Comparison Visualization ===")
        
        # Add PCA coordinates to dataframe
        self.df_combined['pca_x'] = self.reduced_embeddings_2d[:, 0]
        self.df_combined['pca_y'] = self.reduced_embeddings_2d[:, 1]
        
        # Create comparison visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # K-means clustering visualization
        scatter1 = ax1.scatter(self.df_combined['pca_x'], self.df_combined['pca_y'], 
                              c=self.df_combined['Cluster_KMeans'], cmap='tab20', s=30, alpha=0.7)
        n_clusters_kmeans = len(np.unique(self.df_combined['Cluster_KMeans']))
        ax1.set_title(f'K-means Clustering\n{n_clusters_kmeans} clusters')
        ax1.set_xlabel('PCA Component 1')
        ax1.set_ylabel('PCA Component 2')
        ax1.grid(True, alpha=0.3)
        
        # DBSCAN clustering visualization
        scatter2 = ax2.scatter(self.df_combined['pca_x'], self.df_combined['pca_y'], 
                              c=self.df_combined['Cluster_DBSCAN'], cmap='tab20', s=30, alpha=0.7)
        n_clusters_dbscan = len(set(self.df_combined['Cluster_DBSCAN'])) - (1 if -1 in self.df_combined['Cluster_DBSCAN'] else 0)
        ax2.set_title(f'DBSCAN Clustering\n{n_clusters_dbscan} clusters')
        ax2.set_xlabel('PCA Component 1')
        ax2.set_ylabel('PCA Component 2')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_cluster_characteristics(self):
        """Analyze characteristics of each cluster for both methods"""
        print("\n=== Analyzing Cluster Characteristics ===")
        
        # K-means cluster statistics
        print("\n--- K-means Clustering Statistics ---")
        kmeans_stats = self.df_combined.groupby('Cluster_KMeans').agg({
            'post_length': ['mean', 'count'],
            'vote_ratio': 'mean',
            'total_votes': 'mean',
            'title_length': 'mean',
            'title_has_question_mark': 'sum',
            'title_has_code': 'sum',
            'title_has_error': 'sum',
            'post_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(2)
        
        print("K-means Cluster Statistics:")
        print(kmeans_stats)
        
        # DBSCAN cluster statistics
        print("\n--- DBSCAN Clustering Statistics ---")
        dbscan_stats = self.df_combined.groupby('Cluster_DBSCAN').agg({
            'post_length': ['mean', 'count'],
            'vote_ratio': 'mean',
            'total_votes': 'mean',
            'title_length': 'mean',
            'title_has_question_mark': 'sum',
            'title_has_code': 'sum',
            'title_has_error': 'sum',
            'post_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(2)
        
        print("DBSCAN Cluster Statistics:")
        print(dbscan_stats)
        
        # Cluster size distribution
        kmeans_sizes = self.df_combined['Cluster_KMeans'].value_counts().sort_index()
        dbscan_sizes = self.df_combined['Cluster_DBSCAN'].value_counts().sort_index()
        
        print(f"\nK-means Cluster Size Distribution:")
        for cluster_id, size in kmeans_sizes.items():
            print(f"Cluster {cluster_id}: {size} posts ({size/len(self.df_combined)*100:.1f}%)")
        
        print(f"\nDBSCAN Cluster Size Distribution:")
        for cluster_id, size in dbscan_sizes.items():
            if cluster_id == -1:
                print(f"Noise points: {size} posts ({size/len(self.df_combined)*100:.1f}%)")
            else:
                print(f"Cluster {cluster_id}: {size} posts ({size/len(self.df_combined)*100:.1f}%)")
        
        return kmeans_stats, dbscan_stats, kmeans_sizes, dbscan_sizes
    
    def compare_clustering_algorithms(self, embeddings_50d):
        """Compare K-means and DBSCAN clustering algorithms"""
        print("\n=== Comparing Clustering Algorithms ===")
        
        # Calculate metrics for both algorithms
        comparison_data = []
        
        # K-means metrics
        try:
            kmeans_silhouette = silhouette_score(embeddings_50d, self.df_combined['Cluster_KMeans'])
            kmeans_inertia = self.kmeans_model.inertia_
            kmeans_n_clusters = len(np.unique(self.df_combined['Cluster_KMeans']))
            kmeans_noise = 0  # K-means doesn't have noise points
        except:
            kmeans_silhouette = None
            kmeans_inertia = None
            kmeans_n_clusters = len(np.unique(self.df_combined['Cluster_KMeans']))
            kmeans_noise = 0
        
        # DBSCAN metrics
        try:
            dbscan_labels = self.df_combined['Cluster_DBSCAN']
            # Only calculate silhouette for non-noise points
            non_noise_mask = dbscan_labels != -1
            if sum(non_noise_mask) > 1 and len(np.unique(dbscan_labels[non_noise_mask])) > 1:
                dbscan_silhouette = silhouette_score(embeddings_50d[non_noise_mask], dbscan_labels[non_noise_mask])
            else:
                dbscan_silhouette = None
            dbscan_n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            dbscan_noise = list(dbscan_labels).count(-1)
        except:
            dbscan_silhouette = None
            dbscan_n_clusters = len(set(self.df_combined['Cluster_DBSCAN'])) - (1 if -1 in self.df_combined['Cluster_DBSCAN'] else 0)
            dbscan_noise = list(self.df_combined['Cluster_DBSCAN']).count(-1)
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            'Metric': ['Number of Clusters', 'Silhouette Score', 'Inertia', 'Noise Points', 'Algorithm Type'],
            'K-means': [
                kmeans_n_clusters,
                f"{kmeans_silhouette:.3f}" if kmeans_silhouette is not None else "N/A",
                f"{kmeans_inertia:.2f}" if kmeans_inertia is not None else "N/A",
                kmeans_noise,
                'Centroid-based'
            ],
            'DBSCAN': [
                dbscan_n_clusters,
                f"{dbscan_silhouette:.3f}" if dbscan_silhouette is not None else "N/A",
                "N/A",  # DBSCAN doesn't have inertia
                dbscan_noise,
                'Density-based'
            ]
        })
        
        print("Algorithm Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Visualize comparison
        self.visualize_clustering_comparison_charts()
        
        return comparison_df
    
    def visualize_clustering_comparison_charts(self):
        """Create detailed comparison charts"""
        print("\n=== Creating Detailed Comparison Charts ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Cluster size distribution comparison
        kmeans_sizes = self.df_combined['Cluster_KMeans'].value_counts().sort_index()
        dbscan_sizes = self.df_combined['Cluster_DBSCAN'].value_counts().sort_index()
        
        axes[0, 0].bar(range(len(kmeans_sizes)), kmeans_sizes.values, color='skyblue', alpha=0.7, label='K-means')
        axes[0, 0].set_title('K-means Cluster Size Distribution')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of Posts')
        axes[0, 0].set_xticks(range(len(kmeans_sizes)))
        axes[0, 0].set_xticklabels([f'C{i}' for i in kmeans_sizes.index])
        
        # Filter out noise points for DBSCAN visualization
        dbscan_sizes_no_noise = dbscan_sizes[dbscan_sizes.index != -1]
        axes[0, 1].bar(range(len(dbscan_sizes_no_noise)), dbscan_sizes_no_noise.values, color='lightgreen', alpha=0.7, label='DBSCAN')
        axes[0, 1].set_title('DBSCAN Cluster Size Distribution (Excluding Noise)')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Number of Posts')
        axes[0, 1].set_xticks(range(len(dbscan_sizes_no_noise)))
        axes[0, 1].set_xticklabels([f'C{i}' for i in dbscan_sizes_no_noise.index])
        
        # 2. Average vote ratio by cluster
        vote_ratio_kmeans = self.df_combined.groupby('Cluster_KMeans')['vote_ratio'].mean()
        vote_ratio_dbscan = self.df_combined.groupby('Cluster_DBSCAN')['vote_ratio'].mean()
        vote_ratio_dbscan_no_noise = vote_ratio_dbscan[vote_ratio_dbscan.index != -1]
        
        axes[0, 2].bar(range(len(vote_ratio_kmeans)), vote_ratio_kmeans.values, color='lightcoral', alpha=0.7)
        axes[0, 2].set_title('Average Vote Ratio by Cluster (K-means)')
        axes[0, 2].set_xlabel('Cluster ID')
        axes[0, 2].set_ylabel('Average Vote Ratio')
        axes[0, 2].set_xticks(range(len(vote_ratio_kmeans)))
        axes[0, 2].set_xticklabels([f'C{i}' for i in vote_ratio_kmeans.index])
        
        # 3. Post type distribution comparison
        post_type_kmeans = pd.crosstab(self.df_combined['Cluster_KMeans'], self.df_combined['post_type'])
        post_type_dbscan = pd.crosstab(self.df_combined['Cluster_DBSCAN'], self.df_combined['post_type'])
        post_type_dbscan_no_noise = post_type_dbscan[post_type_dbscan.index != -1]
        
        post_type_kmeans.plot(kind='bar', ax=axes[1, 0], alpha=0.7)
        axes[1, 0].set_title('Post Type Distribution by Cluster (K-means)')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Number of Posts')
        axes[1, 0].legend(title='Post Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].tick_params(axis='x', rotation=0)
        
        post_type_dbscan_no_noise.plot(kind='bar', ax=axes[1, 1], alpha=0.7)
        axes[1, 1].set_title('Post Type Distribution by Cluster (DBSCAN)')
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('Number of Posts')
        axes[1, 1].legend(title='Post Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].tick_params(axis='x', rotation=0)
        
        # 4. Noise points analysis for DBSCAN
        if -1 in self.df_combined['Cluster_DBSCAN'].values:
            noise_data = self.df_combined[self.df_combined['Cluster_DBSCAN'] == -1]
            non_noise_data = self.df_combined[self.df_combined['Cluster_DBSCAN'] != -1]
            
            axes[1, 2].hist(noise_data['post_length'], bins=20, alpha=0.7, label='Noise Points', color='red')
            axes[1, 2].hist(non_noise_data['post_length'], bins=20, alpha=0.5, label='Clustered Points', color='blue')
            axes[1, 2].set_title('Post Length: Noise vs Clustered Points (DBSCAN)')
            axes[1, 2].set_xlabel('Post Length (words)')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()
    
    def show_sample_titles(self, n_samples=3):
        """Show sample titles from each cluster for both methods"""
        print(f"\n=== Sample Titles from Each Cluster (Top {n_samples} per cluster) ===")
        
        # K-means samples
        print("\n--- K-means Clustering Samples ---")
        n_clusters_kmeans = len(np.unique(self.df_combined['Cluster_KMeans']))
        for cluster_id in range(n_clusters_kmeans):
            cluster_titles = self.df_combined[self.df_combined['Cluster_KMeans'] == cluster_id]['Title'].head(n_samples).tolist()
            cluster_size = len(self.df_combined[self.df_combined['Cluster_KMeans'] == cluster_id])
            print(f"\nK-means Cluster {cluster_id} (Size: {cluster_size}):")
            for i, title in enumerate(cluster_titles, 1):
                print(f"  {i}. {title[:80]}...")
        
        # DBSCAN samples
        print("\n--- DBSCAN Clustering Samples ---")
        dbscan_clusters = sorted([c for c in self.df_combined['Cluster_DBSCAN'].unique() if c != -1])
        for cluster_id in dbscan_clusters:
            cluster_titles = self.df_combined[self.df_combined['Cluster_DBSCAN'] == cluster_id]['Title'].head(n_samples).tolist()
            cluster_size = len(self.df_combined[self.df_combined['Cluster_DBSCAN'] == cluster_id])
            print(f"\nDBSCAN Cluster {cluster_id} (Size: {cluster_size}):")
            for i, title in enumerate(cluster_titles, 1):
                print(f"  {i}. {title[:80]}...")
        
        # Noise points samples
        if -1 in self.df_combined['Cluster_DBSCAN'].values:
            noise_titles = self.df_combined[self.df_combined['Cluster_DBSCAN'] == -1]['Title'].head(n_samples).tolist()
            noise_size = len(self.df_combined[self.df_combined['Cluster_DBSCAN'] == -1])
            print(f"\nDBSCAN Noise Points (Size: {noise_size}):")
            for i, title in enumerate(noise_titles, 1):
                print(f"  {i}. {title[:80]}...")
    
    def perform_complete_clustering(self, n_clusters=None, eps=None, min_samples=None):
        """Perform complete clustering analysis pipeline with both algorithms"""
        print("=== Starting Complete Clustering Analysis Pipeline ===")
        
        # Perform PCA reduction
        embeddings_50d, reduced_embeddings_2d = self.perform_pca_reduction()
        
        # Perform K-means clustering
        kmeans_labels = self.perform_kmeans_clustering(embeddings_50d, n_clusters)
        
        # Perform DBSCAN clustering
        dbscan_labels = self.perform_dbscan_clustering(embeddings_50d, eps, min_samples)
        
        # Create visualizations
        self.visualize_clusters_comparison()
        
        # Analyze cluster characteristics
        kmeans_stats, dbscan_stats, kmeans_sizes, dbscan_sizes = self.analyze_cluster_characteristics()
        
        # Compare algorithms
        comparison_df = self.compare_clustering_algorithms(embeddings_50d)
        
        # Show sample titles
        self.show_sample_titles()
        
        print("\n=== Clustering Analysis Complete ===")
        
        return {
            'kmeans_labels': kmeans_labels,
            'dbscan_labels': dbscan_labels,
            'kmeans_model': self.kmeans_model,
            'dbscan_model': self.dbscan_model,
            'kmeans_stats': kmeans_stats,
            'dbscan_stats': dbscan_stats,
            'kmeans_sizes': kmeans_sizes,
            'dbscan_sizes': dbscan_sizes,
            'comparison_df': comparison_df,
            'df_with_clusters': self.df_combined
        }

def analyze_clustering_quality(embeddings, cluster_labels, kmeans_model):
    """Analyze the quality of clustering results"""
    print("\n=== Clustering Quality Analysis ===")
    
    # Calculate silhouette score
    try:
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        if silhouette_avg > 0.7:
            print("Excellent clustering quality")
        elif silhouette_avg > 0.5:
            print("Good clustering quality")
        elif silhouette_avg > 0.25:
            print("Fair clustering quality")
        else:
            print("Poor clustering quality")
    except:
        print("Could not calculate silhouette score")
    
    # Calculate inertia (within-cluster sum of squares) - only for K-means
    if kmeans_model is not None:
        inertia = kmeans_model.inertia_
        print(f"Inertia: {inertia:.2f}")
    
    # Calculate number of clusters
    n_clusters = len(np.unique(cluster_labels))
    print(f"Number of clusters: {n_clusters}")
    
    # Calculate cluster size statistics
    cluster_sizes = np.bincount(cluster_labels)
    print(f"Cluster size - Min: {cluster_sizes.min()}, Max: {cluster_sizes.max()}, Mean: {cluster_sizes.mean():.1f}")
    
    return {
        'silhouette_score': silhouette_avg if 'silhouette_avg' in locals() else None,
        'inertia': inertia if 'inertia' in locals() else None,
        'n_clusters': n_clusters,
        'cluster_size_stats': {
            'min': cluster_sizes.min(),
            'max': cluster_sizes.max(),
            'mean': cluster_sizes.mean(),
            'std': cluster_sizes.std()
        }
    }

if __name__ == "__main__":
    # Example usage
    print("This module should be used with preprocessed data from data_preprocessing.py")
    print("Example usage:")
    print("from data_preprocessing import DataPreprocessor")
    print("from clustering_analysis import ClusteringAnalyzer")
    print("")
    print("# Preprocess data")
    print("preprocessor = DataPreprocessor()")
    print("df_combined, embeddings, model = preprocessor.preprocess_all()")
    print("")
    print("# Perform clustering")
    print("cluster_analyzer = ClusteringAnalyzer(df_combined, embeddings)")
    print("results = cluster_analyzer.perform_complete_clustering()") 