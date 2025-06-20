import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import hdbscan
import umap
import warnings
import time

class ClusteringAnalyzer:
    def __init__(self, df_combined, embeddings):
        """Initialize clustering analyzer with preprocessed data"""
        self.df_combined = df_combined.copy()
        self.embeddings = embeddings
        self.cluster_labels = None
        self.kmeans_model = None
        self.hdbscan_model = None
        self.gmm_model = None
        self.umap_reducer = None
        self.pca_50d = None
        self.pca_2d = None
        self.reduced_embeddings_2d = None
        
    def perform_pca_reduction(self, n_components_50d=50, n_components_2d=2, variance_threshold=0.9):
        """Perform PCA dimensionality reduction"""
        print("=== Performing PCA Dimensionality Reduction ===")
        
        # Suppress sklearn deprecation warnings
        warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
        
        print(f"Finding optimal number of components to retain {variance_threshold*100}% variance...")
        print(f"Original dimensions: {self.embeddings.shape[1]}")
        
        # Use PCA with variance_threshold directly
        self.pca_50d = PCA(n_components=variance_threshold, random_state=42)
        embeddings_50d = self.pca_50d.fit_transform(self.embeddings)
        
        optimal_components = embeddings_50d.shape[1]
        print(f"Optimal components for {variance_threshold*100}% variance: {optimal_components}")
        print(f"Actual variance retained: {self.pca_50d.explained_variance_ratio_.sum():.3f}")
        
        # Use optimal components for clustering (but cap at n_components_50d if specified)
        clustering_components = min(optimal_components, n_components_50d)
        if clustering_components < optimal_components:
            print(f"Using {clustering_components} components (capped at {n_components_50d})")
            # Re-fit PCA with capped components
            self.pca_50d = PCA(n_components=clustering_components, random_state=42)
            embeddings_50d = self.pca_50d.fit_transform(self.embeddings)
        else:
            print(f"Using {clustering_components} components for clustering")
        
        print(f"Reduced from {self.embeddings.shape[1]} dimensions to {clustering_components} dimensions")
        print(f"Explained variance ratio: {self.pca_50d.explained_variance_ratio_.sum():.3f}")
        
        # Then reduce to 2D for visualization
        self.pca_2d = PCA(n_components=n_components_2d, random_state=42)
        self.reduced_embeddings_2d = self.pca_2d.fit_transform(embeddings_50d)
        
        print(f"Further reduced to {n_components_2d} dimensions for visualization")
        
        # Plot variance explained curve
        self.plot_variance_explained_curve(optimal_components, variance_threshold)
        
        return embeddings_50d, self.reduced_embeddings_2d
    
    def perform_umap_reduction(self, n_components=50, n_neighbors=15, min_dist=0.1, random_state=42):
        """Perform UMAP dimensionality reduction"""
        print("=== Performing UMAP Dimensionality Reduction ===")
        
        print(f"Original dimensions: {self.embeddings.shape[1]}")
        print(f"Target dimensions: {n_components}")
        print(f"UMAP parameters - n_neighbors: {n_neighbors}, min_dist: {min_dist}")
        
        # Perform UMAP reduction
        self.umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric='cosine'  # Use cosine distance for better performance with text embeddings
        )
        
        embeddings_umap = self.umap_reducer.fit_transform(self.embeddings)
        
        print(f"UMAP reduction completed!")
        print(f"Reduced from {self.embeddings.shape[1]} dimensions to {embeddings_umap.shape[1]} dimensions")
        
        # Also create 2D version for visualization
        umap_2d = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric='cosine'
        )
        embeddings_umap_2d = umap_2d.fit_transform(self.embeddings)
        
        print(f"Created 2D UMAP embedding for visualization")
        
        return embeddings_umap, embeddings_umap_2d
    
    def plot_variance_explained_curve(self, optimal_components, variance_threshold):
        """Plot the variance explained curve"""
        print("\n=== Plotting Variance Explained Curve ===")
        
        # Fit PCA with all components to get explained variance for plotting
        pca_full = PCA(random_state=42)
        pca_full.fit(self.embeddings)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        plt.figure(figsize=(12, 5))
        
        # Plot cumulative explained variance
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-', linewidth=2)
        plt.axhline(y=variance_threshold, color='red', linestyle='--', 
                   label=f'{variance_threshold*100}% Variance Threshold')
        plt.axvline(x=optimal_components, color='green', linestyle='--', 
                   label=f'Optimal Components: {optimal_components}')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance vs Number of Components')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot individual explained variance
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
                pca_full.explained_variance_ratio_, 'r-', linewidth=2)
        plt.axvline(x=optimal_components, color='green', linestyle='--', 
                   label=f'Optimal Components: {optimal_components}')
        plt.xlabel('Component Number')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Individual Component Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed information
        print(f"Top 10 components explained variance:")
        for i in range(min(10, len(pca_full.explained_variance_ratio_))):
            print(f"  Component {i+1}: {pca_full.explained_variance_ratio_[i]:.4f}")
        
        print(f"\nVariance retention at different component counts:")
        for n in [10, 20, 30, 50, 100]:
            if n <= len(cumulative_variance):
                variance_retained = cumulative_variance[n-1]
                print(f"  {n} components: {variance_retained:.3f} ({variance_retained*100:.1f}%)")
    
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
    
    def perform_hdbscan_clustering(self, embeddings_50d, min_cluster_size=3, min_samples=None):
        """Perform HDBSCAN clustering"""
        print("\n=== Performing HDBSCAN Clustering ===")
        
        # Determine parameters if not provided
        if min_samples is None:
            # Improved min_samples calculation for HDBSCAN
            n_samples = len(embeddings_50d)
            n_dimensions = embeddings_50d.shape[1]
            
            # HDBSCAN typically works better with smaller min_samples than DBSCAN
            # Heuristic 1: Based on dimensionality (1.5 * dimensions)
            min_samples_dim = int(1.5 * n_dimensions)
            
            # Heuristic 2: Based on dataset size (log of sample size)
            min_samples_size = max(3, int(np.log(n_samples) * 1.5))
            
            # Heuristic 3: Based on common HDBSCAN practice (1-3% of data)
            min_samples_percent = max(3, int(n_samples * 0.02))  # 2% of data
            
            # Heuristic 4: Minimum reasonable value for clustering
            min_samples_min = 3
            
            # Take the median of all heuristics, but ensure it's reasonable
            candidates = [min_samples_dim, min_samples_size, min_samples_percent, min_samples_min]
            min_samples = int(np.median(candidates))
            
            # Ensure it's not too large (max 5% of data for HDBSCAN)
            max_reasonable = min(50, int(n_samples * 0.05))
            min_samples = min(min_samples, max_reasonable)
            
            print(f"Min_samples calculation:")
            print(f"  - Based on dimensions ({n_dimensions}): {min_samples_dim}")
            print(f"  - Based on dataset size ({n_samples}): {min_samples_size}")
            print(f"  - Based on percentage (2%): {min_samples_percent}")
            print(f"  - Final chosen value: {min_samples}")
        
        # Adjust min_cluster_size based on dataset size
        if min_cluster_size == 3:
            n_samples = len(embeddings_50d)
            min_cluster_size = max(3, int(n_samples * 0.001))  # 0.1% of data
            min_cluster_size = min(min_cluster_size, 100)  # Cap at 100
        
        print(f"HDBSCAN parameters - min_cluster_size: {min_cluster_size}, min_samples: {min_samples}")
        
        # Perform HDBSCAN clustering
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, 
            min_samples=min_samples,
            cluster_selection_epsilon=0.0,  # More conservative clustering
            cluster_selection_method='eom'  # Excess of Mass method
        )
        hdbscan_labels = self.hdbscan_model.fit_predict(embeddings_50d)
        
        # Add cluster labels to dataframe
        self.df_combined['Cluster_HDBSCAN'] = hdbscan_labels
        
        # Count clusters (excluding noise points labeled as -1)
        n_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
        n_noise = list(hdbscan_labels).count(-1)
        
        print(f"HDBSCAN Clustering Results:")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        print(f"Clustering completed successfully!")
        
        return hdbscan_labels
    
    def perform_gmm_clustering(self, embeddings_reduced, n_components=None, covariance_type='full', random_state=42):
        """Perform Gaussian Mixture Model clustering"""
        print("\n=== Performing GMM Clustering ===")
        
        # Determine number of components if not provided
        if n_components is None:
            n_components = min(15, len(embeddings_reduced) // 100)  # Adaptive number of components
        
        print(f"Using {n_components} components for {len(embeddings_reduced)} samples")
        print(f"GMM parameters - covariance_type: {covariance_type}")
        
        # Perform GMM clustering
        self.gmm_model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=10
        )
        gmm_labels = self.gmm_model.fit_predict(embeddings_reduced)
        
        # Add cluster labels to dataframe
        self.df_combined['Cluster_GMM'] = gmm_labels
        
        # Calculate clustering quality metrics
        silhouette_avg = silhouette_score(embeddings_reduced, gmm_labels)
        bic_score = self.gmm_model.bic(embeddings_reduced)
        aic_score = self.gmm_model.aic(embeddings_reduced)
        
        print(f"GMM Clustering Results:")
        print(f"Number of components: {n_components}")
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"BIC Score: {bic_score:.2f}")
        print(f"AIC Score: {aic_score:.2f}")
        print(f"Clustering completed successfully!")
        
        return gmm_labels, {
            'silhouette_score': silhouette_avg,
            'bic_score': bic_score,
            'aic_score': aic_score,
            'n_components': n_components
        }
    
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
        
        # HDBSCAN clustering visualization
        scatter2 = ax2.scatter(self.df_combined['pca_x'], self.df_combined['pca_y'], 
                              c=self.df_combined['Cluster_HDBSCAN'], cmap='tab20', s=30, alpha=0.7)
        n_clusters_hdbscan = len(set(self.df_combined['Cluster_HDBSCAN'])) - (1 if -1 in self.df_combined['Cluster_HDBSCAN'] else 0)
        ax2.set_title(f'HDBSCAN Clustering\n{n_clusters_hdbscan} clusters')
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
        
        # HDBSCAN cluster statistics
        print("\n--- HDBSCAN Clustering Statistics ---")
        hdbscan_stats = self.df_combined.groupby('Cluster_HDBSCAN').agg({
            'post_length': ['mean', 'count'],
            'vote_ratio': 'mean',
            'total_votes': 'mean',
            'title_length': 'mean',
            'title_has_question_mark': 'sum',
            'title_has_code': 'sum',
            'title_has_error': 'sum',
            'post_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(2)
        
        print("HDBSCAN Cluster Statistics:")
        print(hdbscan_stats)
        
        # Cluster size distribution
        kmeans_sizes = self.df_combined['Cluster_KMeans'].value_counts().sort_index()
        hdbscan_sizes = self.df_combined['Cluster_HDBSCAN'].value_counts().sort_index()
        
        print(f"\nK-means Cluster Size Distribution:")
        for cluster_id, size in kmeans_sizes.items():
            print(f"Cluster {cluster_id}: {size} posts ({size/len(self.df_combined)*100:.1f}%)")
        
        print(f"\nHDBSCAN Cluster Size Distribution:")
        for cluster_id, size in hdbscan_sizes.items():
            if cluster_id == -1:
                print(f"Noise points: {size} posts ({size/len(self.df_combined)*100:.1f}%)")
            else:
                print(f"Cluster {cluster_id}: {size} posts ({size/len(self.df_combined)*100:.1f}%)")
        
        return kmeans_stats, hdbscan_stats, kmeans_sizes, hdbscan_sizes
    
    def compare_clustering_algorithms(self, embeddings_50d):
        """Compare K-means and HDBSCAN clustering algorithms"""
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
        
        # HDBSCAN metrics
        try:
            hdbscan_labels = self.df_combined['Cluster_HDBSCAN']
            # Only calculate silhouette for non-noise points
            non_noise_mask = hdbscan_labels != -1
            if sum(non_noise_mask) > 1 and len(np.unique(hdbscan_labels[non_noise_mask])) > 1:
                hdbscan_silhouette = silhouette_score(embeddings_50d[non_noise_mask], hdbscan_labels[non_noise_mask])
            else:
                hdbscan_silhouette = None
            hdbscan_n_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
            hdbscan_noise = list(hdbscan_labels).count(-1)
        except:
            hdbscan_silhouette = None
            hdbscan_n_clusters = len(set(self.df_combined['Cluster_HDBSCAN'])) - (1 if -1 in self.df_combined['Cluster_HDBSCAN'] else 0)
            hdbscan_noise = list(self.df_combined['Cluster_HDBSCAN']).count(-1)
        
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
            'HDBSCAN': [
                hdbscan_n_clusters,
                f"{hdbscan_silhouette:.3f}" if hdbscan_silhouette is not None else "N/A",
                "N/A",  # HDBSCAN doesn't have inertia
                hdbscan_noise,
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
        hdbscan_sizes = self.df_combined['Cluster_HDBSCAN'].value_counts().sort_index()
        
        axes[0, 0].bar(range(len(kmeans_sizes)), kmeans_sizes.values, color='skyblue', alpha=0.7, label='K-means')
        axes[0, 0].set_title('K-means Cluster Size Distribution')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of Posts')
        axes[0, 0].set_xticks(range(len(kmeans_sizes)))
        axes[0, 0].set_xticklabels([f'C{i}' for i in kmeans_sizes.index])
        
        # Filter out noise points for HDBSCAN visualization
        hdbscan_sizes_no_noise = hdbscan_sizes[hdbscan_sizes.index != -1]
        axes[0, 1].bar(range(len(hdbscan_sizes_no_noise)), hdbscan_sizes_no_noise.values, color='lightgreen', alpha=0.7, label='HDBSCAN')
        axes[0, 1].set_title('HDBSCAN Cluster Size Distribution (Excluding Noise)')
        axes[0, 1].set_xlabel('Cluster ID')
        axes[0, 1].set_ylabel('Number of Posts')
        axes[0, 1].set_xticks(range(len(hdbscan_sizes_no_noise)))
        axes[0, 1].set_xticklabels([f'C{i}' for i in hdbscan_sizes_no_noise.index])
        
        # 2. Average vote ratio by cluster
        vote_ratio_kmeans = self.df_combined.groupby('Cluster_KMeans')['vote_ratio'].mean()
        vote_ratio_hdbscan = self.df_combined.groupby('Cluster_HDBSCAN')['vote_ratio'].mean()
        vote_ratio_hdbscan_no_noise = vote_ratio_hdbscan[vote_ratio_hdbscan.index != -1]
        
        axes[0, 2].bar(range(len(vote_ratio_kmeans)), vote_ratio_kmeans.values, color='lightcoral', alpha=0.7)
        axes[0, 2].set_title('Average Vote Ratio by Cluster (K-means)')
        axes[0, 2].set_xlabel('Cluster ID')
        axes[0, 2].set_ylabel('Average Vote Ratio')
        axes[0, 2].set_xticks(range(len(vote_ratio_kmeans)))
        axes[0, 2].set_xticklabels([f'C{i}' for i in vote_ratio_kmeans.index])
        
        # 3. Post type distribution comparison
        post_type_kmeans = pd.crosstab(self.df_combined['Cluster_KMeans'], self.df_combined['post_type'])
        post_type_hdbscan = pd.crosstab(self.df_combined['Cluster_HDBSCAN'], self.df_combined['post_type'])
        post_type_hdbscan_no_noise = post_type_hdbscan[post_type_hdbscan.index != -1]
        
        post_type_kmeans.plot(kind='bar', ax=axes[1, 0], alpha=0.7)
        axes[1, 0].set_title('Post Type Distribution by Cluster (K-means)')
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Number of Posts')
        axes[1, 0].legend(title='Post Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].tick_params(axis='x', rotation=0)
        
        post_type_hdbscan_no_noise.plot(kind='bar', ax=axes[1, 1], alpha=0.7)
        axes[1, 1].set_title('Post Type Distribution by Cluster (HDBSCAN)')
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('Number of Posts')
        axes[1, 1].legend(title='Post Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].tick_params(axis='x', rotation=0)
        
        # 4. Noise points analysis for HDBSCAN
        if -1 in self.df_combined['Cluster_HDBSCAN'].values:
            noise_data = self.df_combined[self.df_combined['Cluster_HDBSCAN'] == -1]
            non_noise_data = self.df_combined[self.df_combined['Cluster_HDBSCAN'] != -1]
            
            axes[1, 2].hist(noise_data['post_length'], bins=20, alpha=0.7, label='Noise Points', color='red')
            axes[1, 2].hist(non_noise_data['post_length'], bins=20, alpha=0.5, label='Clustered Points', color='blue')
            axes[1, 2].set_title('Post Length: Noise vs Clustered Points (HDBSCAN)')
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
        
        # HDBSCAN samples
        print("\n--- HDBSCAN Clustering Samples ---")
        hdbscan_clusters = sorted([c for c in self.df_combined['Cluster_HDBSCAN'].unique() if c != -1])
        for cluster_id in hdbscan_clusters:
            cluster_titles = self.df_combined[self.df_combined['Cluster_HDBSCAN'] == cluster_id]['Title'].head(n_samples).tolist()
            cluster_size = len(self.df_combined[self.df_combined['Cluster_HDBSCAN'] == cluster_id])
            print(f"\nHDBSCAN Cluster {cluster_id} (Size: {cluster_size}):")
            for i, title in enumerate(cluster_titles, 1):
                print(f"  {i}. {title[:80]}...")
        
        # Noise points samples
        if -1 in self.df_combined['Cluster_HDBSCAN'].values:
            noise_titles = self.df_combined[self.df_combined['Cluster_HDBSCAN'] == -1]['Title'].head(n_samples).tolist()
            noise_size = len(self.df_combined[self.df_combined['Cluster_HDBSCAN'] == -1])
            print(f"\nHDBSCAN Noise Points (Size: {noise_size}):")
            for i, title in enumerate(noise_titles, 1):
                print(f"  {i}. {title[:80]}...")
    
    def perform_complete_clustering(self, n_clusters=None, min_cluster_size=3, min_samples=None, variance_threshold=0.9, use_combined_features=False, tfidf_features=None, max_dimensions=50):
        """Perform complete clustering analysis pipeline with both algorithms"""
        print("=== Starting Complete Clustering Analysis Pipeline ===")
        
        if use_combined_features and tfidf_features is not None:
            # Perform combined TF-IDF + Transformer clustering
            print("\n" + "="*50)
            print("STEP 2: COMBINED TF-IDF + TRANSFORMER CLUSTERING")
            print("="*50)
            
            combined_labels, combined_features_pca, distance_matrix = self.perform_combined_clustering(
                tfidf_features, self.embeddings, variance_threshold, max_dimensions, min_cluster_size, min_samples
            )
            
            # Create visualizations
            self.visualize_combined_clustering()
            
            # Analyze cluster characteristics
            combined_stats, combined_sizes = self.analyze_combined_cluster_characteristics()
            
            # Show sample titles
            self.show_combined_sample_titles()
            
            print("\n=== Combined Clustering Analysis Complete ===")
            
            return {
                'combined_labels': combined_labels,
                'combined_features_pca': combined_features_pca,
                'distance_matrix': distance_matrix,
                'hdbscan_combined_model': self.hdbscan_combined_model,
                'combined_stats': combined_stats,
                'combined_sizes': combined_sizes,
                'df_with_clusters': self.df_combined
            }
        
        else:
            # Perform traditional clustering (K-means + HDBSCAN)
            print("\n" + "="*50)
            print("STEP 2: TRADITIONAL CLUSTERING ANALYSIS (K-means + HDBSCAN)")
            print("="*50)
            
            # Perform PCA reduction
            embeddings_50d, reduced_embeddings_2d = self.perform_pca_reduction(variance_threshold=variance_threshold)
            
            # Perform K-means clustering
            kmeans_labels = self.perform_kmeans_clustering(embeddings_50d, n_clusters)
            
            # Perform HDBSCAN clustering
            hdbscan_labels = self.perform_hdbscan_clustering(embeddings_50d, min_cluster_size, min_samples)
            
            # Create visualizations
            self.visualize_clusters_comparison()
            
            # Analyze cluster characteristics
            kmeans_stats, hdbscan_stats, kmeans_sizes, hdbscan_sizes = self.analyze_cluster_characteristics()
            
            # Compare algorithms
            comparison_df = self.compare_clustering_algorithms(embeddings_50d)
            
            # Show sample titles
            self.show_sample_titles()
            
            print("\n=== Traditional Clustering Analysis Complete ===")
            
            return {
                'kmeans_labels': kmeans_labels,
                'hdbscan_labels': hdbscan_labels,
                'kmeans_model': self.kmeans_model,
                'hdbscan_model': self.hdbscan_model,
                'kmeans_stats': kmeans_stats,
                'hdbscan_stats': hdbscan_stats,
                'kmeans_sizes': kmeans_sizes,
                'hdbscan_sizes': hdbscan_sizes,
                'comparison_df': comparison_df,
                'df_with_clusters': self.df_combined
            }
    
    def show_combined_sample_titles(self, n_samples=3):
        """Show sample titles from combined clustering results"""
        print(f"\n=== Sample Titles from Combined Clustering (Top {n_samples} per cluster) ===")
        
        # Combined clustering samples
        print("\n--- Combined TF-IDF + Transformer Clustering Samples ---")
        combined_clusters = sorted([c for c in self.df_combined['Cluster_Combined'].unique() if c != -1])
        for cluster_id in combined_clusters:
            cluster_titles = self.df_combined[self.df_combined['Cluster_Combined'] == cluster_id]['Title'].head(n_samples).tolist()
            cluster_size = len(self.df_combined[self.df_combined['Cluster_Combined'] == cluster_id])
            print(f"\nCombined Cluster {cluster_id} (Size: {cluster_size}):")
            for i, title in enumerate(cluster_titles, 1):
                print(f"  {i}. {title[:80]}...")
        
        # Noise points samples
        if -1 in self.df_combined['Cluster_Combined'].values:
            noise_titles = self.df_combined[self.df_combined['Cluster_Combined'] == -1]['Title'].head(n_samples).tolist()
            noise_size = len(self.df_combined[self.df_combined['Cluster_Combined'] == -1])
            print(f"\nCombined Noise Points (Size: {noise_size}):")
            for i, title in enumerate(noise_titles, 1):
                print(f"  {i}. {title[:80]}...")
    
    def perform_combined_clustering(self, tfidf_features, embeddings, variance_threshold=0.9, max_dimensions=50, min_cluster_size=None, min_samples=None):
        """Perform clustering using combined TF-IDF + Transformer features"""
        print("\n=== Performing Combined TF-IDF + Transformer Clustering ===")
        
        # Concatenate TF-IDF and Transformer features
        print("Concatenating TF-IDF and Transformer features...")
        print(f"TF-IDF features shape: {tfidf_features.shape}")
        print(f"Transformer embeddings shape: {embeddings.shape}")
        
        # Ensure same number of samples
        assert tfidf_features.shape[0] == embeddings.shape[0], "Sample counts must match"
        
        # Concatenate features
        combined_features = np.concatenate([tfidf_features, embeddings], axis=1)
        print(f"Combined features shape: {combined_features.shape}")
        
        # Perform PCA on combined features
        print(f"\nPerforming PCA on combined features to retain {variance_threshold*100}% variance...")
        self.pca_combined = PCA(n_components=variance_threshold, random_state=42)
        combined_features_pca = self.pca_combined.fit_transform(combined_features)
        
        # Check if we need to limit dimensions
        if combined_features_pca.shape[1] > max_dimensions:
            print(f"PCA resulted in {combined_features_pca.shape[1]} dimensions, limiting to {max_dimensions}")
            # Re-fit PCA with limited components
            self.pca_combined = PCA(n_components=max_dimensions, random_state=42)
            combined_features_pca = self.pca_combined.fit_transform(combined_features)
            actual_variance = self.pca_combined.explained_variance_ratio_.sum()
            print(f"Limited PCA retained {actual_variance:.3f} ({actual_variance*100:.1f}%) variance")
        else:
            print(f"PCA reduced from {combined_features.shape[1]} to {combined_features_pca.shape[1]} dimensions")
            print(f"Explained variance ratio: {self.pca_combined.explained_variance_ratio_.sum():.3f}")
        
        # Perform HDBSCAN clustering with euclidean distances
        print("Performing HDBSCAN clustering with euclidean distances...")
        
        # Determine HDBSCAN parameters - use provided parameters or calculate defaults
        n_samples = len(combined_features_pca)
        
        # Use provided min_cluster_size or calculate default
        if min_cluster_size is None:
            min_cluster_size = max(3, int(n_samples * 0.0005))  # 0.05% of data
            min_cluster_size = min(min_cluster_size, 50)  # Cap at 50
            print(f"Using calculated min_cluster_size: {min_cluster_size}")
        else:
            print(f"Using provided min_cluster_size: {min_cluster_size}")
        
        # Use provided min_samples or calculate default
        if min_samples is None:
            min_samples = max(3, int(np.log(n_samples) * 1.0))  # log(n) * 1.0
            min_samples = min(min_samples, 30)  # Cap at 30
            print(f"Using calculated min_samples: {min_samples}")
        else:
            print(f"Using provided min_samples: {min_samples}")
        
        print(f"HDBSCAN parameters - min_cluster_size: {min_cluster_size}, min_samples: {min_samples}")
        print(f"Dataset size: {n_samples} samples")
        
        # Perform HDBSCAN clustering with time tracking
        print("Starting HDBSCAN clustering...")
        start_time = time.time()
        
        self.hdbscan_combined_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',  # Use euclidean distance (BallTree supports this)
            cluster_selection_epsilon=0.0,
            cluster_selection_method='eom'
        )
        
        # Pass the PCA-reduced features directly to HDBSCAN
        combined_labels = self.hdbscan_combined_model.fit_predict(combined_features_pca)
        
        end_time = time.time()
        print(f"HDBSCAN clustering completed in {end_time - start_time:.2f} seconds")
        
        # Add cluster labels to dataframe
        self.df_combined['Cluster_Combined'] = combined_labels
        
        # Count clusters (excluding noise points labeled as -1)
        n_clusters = len(set(combined_labels)) - (1 if -1 in combined_labels else 0)
        n_noise = list(combined_labels).count(-1)
        
        print(f"Combined Clustering Results:")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        print(f"Clustering completed successfully!")
        
        # Store combined features for later use
        self.combined_features_pca = combined_features_pca
        
        return combined_labels, combined_features_pca, None  # No distance matrix needed
    
    def visualize_combined_clustering(self):
        """Visualize combined clustering results"""
        print("\n=== Visualizing Combined Clustering Results ===")
        
        if not hasattr(self, 'combined_features_pca'):
            print("Combined features not available. Run perform_combined_clustering() first.")
            return
        
        # Reduce to 2D for visualization
        pca_2d = PCA(n_components=2, random_state=42)
        combined_2d = pca_2d.fit_transform(self.combined_features_pca)
        
        # Add PCA coordinates to dataframe
        self.df_combined['combined_pca_x'] = combined_2d[:, 0]
        self.df_combined['combined_pca_y'] = combined_2d[:, 1]
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Combined clustering visualization
        scatter = plt.scatter(self.df_combined['combined_pca_x'], 
                            self.df_combined['combined_pca_y'], 
                            c=self.df_combined['Cluster_Combined'], 
                            cmap='tab20', s=30, alpha=0.7)
        
        n_clusters = len(set(self.df_combined['Cluster_Combined'])) - (1 if -1 in self.df_combined['Cluster_Combined'] else 0)
        plt.title(f'Combined TF-IDF + Transformer Clustering\n{n_clusters} clusters')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True, alpha=0.3)
        
        # Add legend for noise points
        if -1 in self.df_combined['Cluster_Combined'].values:
            noise_points = plt.scatter([], [], c='black', s=30, alpha=0.7, label='Noise')
            plt.legend([noise_points], ['Noise Points'])
        
        plt.tight_layout()
        plt.show()
        
        # Calculate and plot distance distribution using HDBSCAN's internal distances
        if hasattr(self.hdbscan_combined_model, 'distances_'):
            plt.figure(figsize=(10, 6))
            
            # Get distances from HDBSCAN model
            distances = self.hdbscan_combined_model.distances_
            distances = distances[distances > 0]  # Remove zero distances
            
            plt.hist(distances, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Distribution of HDBSCAN Internal Distances')
            plt.xlabel('Distance')
            plt.ylabel('Frequency')
            plt.axvline(np.median(distances), color='red', linestyle='--', 
                       label=f'Median: {np.median(distances):.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            print(f"Distance statistics:")
            print(f"  Mean: {np.mean(distances):.3f}")
            print(f"  Median: {np.median(distances):.3f}")
            print(f"  Std: {np.std(distances):.3f}")
            print(f"  Min: {np.min(distances):.3f}")
            print(f"  Max: {np.max(distances):.3f}")
        else:
            print("Distance statistics not available (HDBSCAN internal distances not accessible)")
    
    def analyze_combined_cluster_characteristics(self):
        """Analyze characteristics of combined clustering results"""
        print("\n=== Analyzing Combined Clustering Characteristics ===")
        
        # Combined cluster statistics
        print("\n--- Combined Clustering Statistics ---")
        combined_stats = self.df_combined.groupby('Cluster_Combined').agg({
            'post_length': ['mean', 'count'],
            'vote_ratio': 'mean',
            'total_votes': 'mean',
            'title_length': 'mean',
            'title_has_question_mark': 'sum',
            'title_has_code': 'sum',
            'title_has_error': 'sum',
            'post_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(2)
        
        print("Combined Cluster Statistics:")
        print(combined_stats)
        
        # Cluster size distribution
        combined_sizes = self.df_combined['Cluster_Combined'].value_counts().sort_index()
        
        print(f"\nCombined Cluster Size Distribution:")
        for cluster_id, size in combined_sizes.items():
            percentage = (size / len(self.df_combined)) * 100
            if cluster_id == -1:
                print(f"  Noise points: {size} posts ({percentage:.1f}%)")
            else:
                print(f"  Cluster {cluster_id}: {size} posts ({percentage:.1f}%)")
        
        return combined_stats, combined_sizes
    
    def perform_umap_gmm_clustering(self, n_components=50, n_neighbors=15, min_dist=0.1, 
                                   n_gmm_components=None, covariance_type='full', random_state=42):
        """Perform UMAP reduction followed by GMM clustering"""
        print("\n=== Performing UMAP + GMM Clustering ===")
        
        # Perform UMAP reduction
        embeddings_umap, embeddings_umap_2d = self.perform_umap_reduction(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )
        
        # Perform GMM clustering on UMAP-reduced features
        gmm_labels, gmm_metrics = self.perform_gmm_clustering(
            embeddings_umap,
            n_components=n_gmm_components,
            covariance_type=covariance_type,
            random_state=random_state
        )
        
        # Store UMAP coordinates for visualization
        self.df_combined['umap_x'] = embeddings_umap_2d[:, 0]
        self.df_combined['umap_y'] = embeddings_umap_2d[:, 1]
        
        print(f"\nUMAP + GMM Clustering Summary:")
        print(f"UMAP reduced from {self.embeddings.shape[1]} to {embeddings_umap.shape[1]} dimensions")
        print(f"GMM found {gmm_metrics['n_components']} clusters")
        print(f"Silhouette Score: {gmm_metrics['silhouette_score']:.3f}")
        
        return gmm_labels, embeddings_umap, embeddings_umap_2d, gmm_metrics
    
    def visualize_umap_gmm_clustering(self):
        """Visualize UMAP + GMM clustering results"""
        print("\n=== Visualizing UMAP + GMM Clustering Results ===")
        
        if 'umap_x' not in self.df_combined.columns or 'Cluster_GMM' not in self.df_combined.columns:
            print("UMAP + GMM clustering not performed yet. Run perform_umap_gmm_clustering() first.")
            return
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # UMAP + GMM clustering visualization
        scatter = plt.scatter(self.df_combined['umap_x'], 
                            self.df_combined['umap_y'], 
                            c=self.df_combined['Cluster_GMM'], 
                            cmap='tab20', s=30, alpha=0.7)
        
        n_clusters = len(set(self.df_combined['Cluster_GMM']))
        plt.title(f'UMAP + GMM Clustering\n{n_clusters} clusters')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Cluster ID')
        
        plt.tight_layout()
        plt.show()
        
        # Plot cluster size distribution
        plt.figure(figsize=(10, 6))
        cluster_sizes = self.df_combined['Cluster_GMM'].value_counts().sort_index()
        
        plt.bar(range(len(cluster_sizes)), cluster_sizes.values, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('GMM Cluster Size Distribution')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Posts')
        plt.xticks(range(len(cluster_sizes)), cluster_sizes.index)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(cluster_sizes.values):
            plt.text(i, v + max(cluster_sizes.values) * 0.01, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        print(f"GMM Cluster Size Distribution:")
        for cluster_id, size in cluster_sizes.items():
            percentage = (size / len(self.df_combined)) * 100
            print(f"  Cluster {cluster_id}: {size} posts ({percentage:.1f}%)")
    
    def analyze_umap_gmm_cluster_characteristics(self):
        """Analyze characteristics of UMAP + GMM clustering results"""
        print("\n=== Analyzing UMAP + GMM Clustering Characteristics ===")
        
        if 'Cluster_GMM' not in self.df_combined.columns:
            print("GMM clustering not performed yet. Run perform_umap_gmm_clustering() first.")
            return None, None
        
        # GMM cluster statistics
        print("\n--- GMM Clustering Statistics ---")
        gmm_stats = self.df_combined.groupby('Cluster_GMM').agg({
            'post_length': ['mean', 'count'],
            'vote_ratio': 'mean',
            'total_votes': 'mean',
            'title_length': 'mean',
            'title_has_question_mark': 'sum',
            'title_has_code': 'sum',
            'title_has_error': 'sum',
            'post_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(2)
        
        print("GMM Cluster Statistics:")
        print(gmm_stats)
        
        # Cluster size distribution
        gmm_sizes = self.df_combined['Cluster_GMM'].value_counts().sort_index()
        
        print(f"\nGMM Cluster Size Distribution:")
        for cluster_id, size in gmm_sizes.items():
            percentage = (size / len(self.df_combined)) * 100
            print(f"  Cluster {cluster_id}: {size} posts ({percentage:.1f}%)")
        
        # Analyze cluster characteristics
        print(f"\n--- GMM Cluster Characteristics Analysis ---")
        
        # Most distinctive clusters by post length
        avg_lengths = self.df_combined.groupby('Cluster_GMM')['post_length'].mean().sort_values(ascending=False)
        print(f"Clusters by average post length (descending):")
        for cluster_id, avg_length in avg_lengths.head(5).items():
            print(f"  Cluster {cluster_id}: {avg_length:.1f} words")
        
        # Most distinctive clusters by vote ratio
        avg_votes = self.df_combined.groupby('Cluster_GMM')['vote_ratio'].mean().sort_values(ascending=False)
        print(f"\nClusters by average vote ratio (descending):")
        for cluster_id, avg_vote in avg_votes.head(5).items():
            print(f"  Cluster {cluster_id}: {avg_vote:.3f}")
        
        # Clusters with most questions
        question_counts = self.df_combined.groupby('Cluster_GMM')['title_has_question_mark'].sum().sort_values(ascending=False)
        print(f"\nClusters by question mark count (descending):")
        for cluster_id, count in question_counts.head(5).items():
            percentage = (count / gmm_sizes[cluster_id]) * 100
            print(f"  Cluster {cluster_id}: {count} questions ({percentage:.1f}%)")
        
        return gmm_stats, gmm_sizes

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