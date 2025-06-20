import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
from pathlib import Path

# Import the new modular components
from data_preprocessing import DataPreprocessor
from clustering_analysis import ClusteringAnalyzer, analyze_clustering_quality

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stack Overflow Data Analysis Pipeline')
    parser.add_argument('--mode', choices=['preprocess', 'cluster', 'combined', 'umap_gmm', 'all'], 
                       default='all', help='Pipeline mode to run')
    parser.add_argument('--data-dir', default='data', help='Directory containing XML data files')
    parser.add_argument('--cache-file', default='processed_data.pkl', help='Cache file for processed data')
    parser.add_argument('--n-clusters', type=int, default=8, help='Number of clusters for K-means')
    parser.add_argument('--min-cluster-size', type=int, default=3, help='Minimum cluster size for HDBSCAN')
    parser.add_argument('--min-samples', type=int, help='Minimum samples for HDBSCAN (auto-determined if not specified)')
    parser.add_argument('--variance-threshold', type=float, default=0.9, 
                       help='Variance threshold for PCA (0.0-1.0)')
    parser.add_argument('--max-dimensions', type=int, default=50, 
                       help='Maximum dimensions after PCA for combined clustering')
    parser.add_argument('--umap-components', type=int, default=50, 
                       help='Number of UMAP components for dimensionality reduction')
    parser.add_argument('--umap-neighbors', type=int, default=15, 
                       help='Number of neighbors for UMAP')
    parser.add_argument('--umap-min-dist', type=float, default=0.1, 
                       help='Minimum distance for UMAP')
    parser.add_argument('--gmm-components', type=int, help='Number of GMM components (auto-determined if not specified)')
    parser.add_argument('--gmm-covariance', choices=['full', 'tied', 'diag', 'spherical'], 
                       default='full', help='GMM covariance type')
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Force reprocessing even if cache exists')
    
    return parser.parse_args()

def save_processed_data(data, cache_file):
    """Save processed data to a single pickle file"""
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(exist_ok=True)
    
    # Save all data to a single pickle file
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Processed data saved to {cache_path}")

def load_processed_data(cache_file):
    """Load processed data from a single pickle file"""
    cache_path = Path(cache_file)
    
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file {cache_file} does not exist")
    
    # Load all data from a single pickle file
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Processed data loaded from {cache_path}")
    return data['df_combined'], data['embeddings'], data['tfidf_features'], data['model']

def run_preprocessing(args):
    """Run data preprocessing only"""
    print("=== Running Data Preprocessing Only ===")
    
    preprocessor = DataPreprocessor(base_path=args.data_dir)
    df_combined, embeddings, tfidf_features, model = preprocessor.preprocess_all()
    
    # Save processed data
    data = {
        'df_combined': df_combined,
        'embeddings': embeddings,
        'tfidf_features': tfidf_features,
        'model': model
    }
    save_processed_data(data, args.cache_file)
    
    return data

def run_clustering(args):
    """Run clustering analysis only"""
    print("=== Running Clustering Analysis Only ===")
    
    # Load processed data
    df_combined, embeddings, tfidf_features, model = load_processed_data(args.cache_file)
    
    # Run clustering
    cluster_analyzer = ClusteringAnalyzer(df_combined, embeddings)
    clustering_results = cluster_analyzer.perform_complete_clustering(
        n_clusters=args.n_clusters,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        variance_threshold=args.variance_threshold,
        use_combined_features=False
    )
    
    # Analyze clustering quality
    print("\n=== Clustering Quality Analysis ===")
    
    # K-means quality analysis
    print("\n--- K-means Quality Analysis ---")
    kmeans_quality = analyze_clustering_quality(
        embeddings, 
        clustering_results['kmeans_labels'], 
        clustering_results['kmeans_model']
    )
    
    # DBSCAN quality analysis (excluding noise points)
    print("\n--- HDBSCAN Quality Analysis ---")
    hdbscan_labels = clustering_results['hdbscan_labels']
    non_noise_mask = hdbscan_labels != -1
    if sum(non_noise_mask) > 1:
        hdbscan_quality = analyze_clustering_quality(
            embeddings[non_noise_mask], 
            hdbscan_labels[non_noise_mask], 
            None
        )
    else:
        hdbscan_quality = {'silhouette_score': None, 'n_clusters': 0}
    
    # Print results
    print_clustering_summary(df_combined, embeddings, tfidf_features, 
                           clustering_results, kmeans_quality, hdbscan_quality)
    
    return {
        'df_combined': df_combined,
        'embeddings': embeddings,
        'tfidf_features': tfidf_features,
        'clustering_results': clustering_results,
        'kmeans_quality': kmeans_quality,
        'hdbscan_quality': hdbscan_quality
    }

def run_visualization(args):
    """Run visualization only"""
    print("=== Running Visualization Only ===")
    
    # Load processed data
    df_combined, embeddings, tfidf_features, model = load_processed_data(args.cache_file)
    
    # Create preprocessor instance for visualization
    preprocessor = DataPreprocessor(base_path=args.data_dir)
    preprocessor.df_combined = df_combined
    preprocessor.embeddings = embeddings
    preprocessor.tfidf_features = tfidf_features
    preprocessor.model = model
    
    # Run visualizations
    preprocessor.visualize_derived_variables()
    preprocessor.visualize_tfidf_features()
    preprocessor.print_summary_statistics()
    
    return {
        'df_combined': df_combined,
        'embeddings': embeddings,
        'tfidf_features': tfidf_features
    }

def print_clustering_summary(df_combined, embeddings, tfidf_features, 
                           clustering_results, kmeans_quality, hdbscan_quality):
    """Print clustering summary"""
    print(f"\n=== Analysis Complete ===")
    print(f"Total posts analyzed: {len(df_combined)}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"TF-IDF features shape: {tfidf_features.shape}")
    
    # K-means summary
    print(f"\n--- K-means Clustering Summary ---")
    print(f"Number of clusters: {clustering_results['kmeans_sizes'].shape[0]}")
    print(f"Clustering quality - Silhouette Score: {kmeans_quality.get('silhouette_score', 'N/A')}")
    print(f"Clustering quality - Inertia: {kmeans_quality.get('inertia', 'N/A')}")
    
    # HDBSCAN summary
    print(f"\n--- HDBSCAN Clustering Summary ---")
    hdbscan_clusters = len([c for c in clustering_results['hdbscan_sizes'].index if c != -1])
    hdbscan_noise = clustering_results['hdbscan_sizes'].get(-1, 0)
    print(f"Number of clusters: {hdbscan_clusters}")
    print(f"Number of noise points: {hdbscan_noise}")
    print(f"Clustering quality - Silhouette Score: {hdbscan_quality.get('silhouette_score', 'N/A')}")
    
    # Key statistics
    print(f"\n=== Key Statistics ===")
    print(f"Average post length: {df_combined['post_length'].mean():.1f} words")
    print(f"Average vote ratio: {df_combined['vote_ratio'].mean():.3f}")
    print(f"Average total votes: {df_combined['total_votes'].mean():.1f}")
    print(f"Most common post type: {df_combined['post_type'].mode().iloc[0] if len(df_combined['post_type'].mode()) > 0 else 'Unknown'}")
    
    # TF-IDF statistics
    print(f"\n=== TF-IDF Statistics ===")
    print(f"TF-IDF mean: {df_combined['tfidf_mean'].mean():.4f}")
    print(f"TF-IDF std: {df_combined['tfidf_std'].mean():.4f}")
    print(f"TF-IDF max: {df_combined['tfidf_max'].mean():.4f}")
    print(f"TF-IDF min: {df_combined['tfidf_min'].mean():.4f}")
    
    # Cluster distribution
    print(f"\n=== Cluster Distribution Comparison ===")
    
    print(f"K-means Cluster Distribution:")
    kmeans_sizes = clustering_results['kmeans_sizes']
    for cluster_id, size in kmeans_sizes.items():
        percentage = (size / len(df_combined)) * 100
        print(f"  Cluster {cluster_id}: {size} posts ({percentage:.1f}%)")
    
    print(f"\nHDBSCAN Cluster Distribution:")
    hdbscan_sizes = clustering_results['hdbscan_sizes']
    for cluster_id, size in hdbscan_sizes.items():
        percentage = (size / len(df_combined)) * 100
        if cluster_id == -1:
            print(f"  Noise points: {size} posts ({percentage:.1f}%)")
        else:
            print(f"  Cluster {cluster_id}: {size} posts ({percentage:.1f}%)")
    
    # Algorithm comparison
    print(f"\n=== Algorithm Comparison Summary ===")
    comparison_df = clustering_results['comparison_df']
    print(comparison_df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description='Stack Overflow Data Analysis Pipeline')
    parser.add_argument('--mode', choices=['preprocess', 'cluster', 'combined', 'umap_gmm', 'all'], 
                       default='all', help='Pipeline mode to run')
    parser.add_argument('--data-dir', default='data', help='Directory containing XML data files')
    parser.add_argument('--cache-file', default='processed_data.pkl', help='Cache file for processed data')
    parser.add_argument('--n-clusters', type=int, default=8, help='Number of clusters for K-means')
    parser.add_argument('--min-cluster-size', type=int, default=3, help='Minimum cluster size for HDBSCAN')
    parser.add_argument('--min-samples', type=int, help='Minimum samples for HDBSCAN (auto-determined if not specified)')
    parser.add_argument('--variance-threshold', type=float, default=0.9, 
                       help='Variance threshold for PCA (0.0-1.0)')
    parser.add_argument('--max-dimensions', type=int, default=50, 
                       help='Maximum dimensions after PCA for combined clustering')
    parser.add_argument('--umap-components', type=int, default=50, 
                       help='Number of UMAP components for dimensionality reduction')
    parser.add_argument('--umap-neighbors', type=int, default=15, 
                       help='Number of neighbors for UMAP')
    parser.add_argument('--umap-min-dist', type=float, default=0.1, 
                       help='Minimum distance for UMAP')
    parser.add_argument('--gmm-components', type=int, help='Number of GMM components (auto-determined if not specified)')
    parser.add_argument('--gmm-covariance', choices=['full', 'tied', 'diag', 'spherical'], 
                       default='full', help='GMM covariance type')
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Force reprocessing even if cache exists')
    
    args = parser.parse_args()
    
    print("=== Stack Overflow Data Analysis Pipeline ===")
    print(f"Mode: {args.mode}")
    print(f"Data directory: {args.data_dir}")
    print(f"Cache file: {args.cache_file}")
    print(f"Variance threshold: {args.variance_threshold}")
    print(f"Max dimensions: {args.max_dimensions}")
    
    # Initialize data preprocessor
    data_preprocessor = DataPreprocessor(args.data_dir)
    
    # Initialize clustering analyzer
    clustering_analyzer = None
    
    if args.mode in ['preprocess', 'all']:
        print("\n" + "="*50)
        print("STEP 1: DATA PREPROCESSING")
        print("="*50)
        
        # Load and preprocess data
        df, embeddings, tfidf_features, model = data_preprocessor.preprocess_all()
        
        # Save processed data
        data = {
            'df_combined': df,
            'embeddings': embeddings,
            'tfidf_features': tfidf_features,
            'model': model
        }
        save_processed_data(data, args.cache_file)
        
        # Initialize clustering analyzer with processed data
        clustering_analyzer = ClusteringAnalyzer(df, embeddings)
        
        print(f"\nPreprocessing completed!")
        print(f"Dataset shape: {df.shape}")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"TF-IDF features shape: {tfidf_features.shape}")
    
    if args.mode in ['cluster', 'all']:
        if clustering_analyzer is None:
            # Load cached data for clustering only
            df, embeddings, tfidf_features, model = load_processed_data(args.cache_file)
            clustering_analyzer = ClusteringAnalyzer(df, embeddings)
        
        print("\n" + "="*50)
        print("STEP 2: TRADITIONAL CLUSTERING ANALYSIS")
        print("="*50)
        
        # Perform traditional clustering
        clustering_results = clustering_analyzer.perform_complete_clustering(
            n_clusters=args.n_clusters,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            variance_threshold=args.variance_threshold,
            use_combined_features=False
        )
        
        print("\nTraditional clustering analysis completed!")
    
    if args.mode in ['combined', 'all']:
        if clustering_analyzer is None:
            # Load cached data for combined clustering only
            df, embeddings, tfidf_features, model = load_processed_data(args.cache_file)
            clustering_analyzer = ClusteringAnalyzer(df, embeddings)
        
        print("\n" + "="*50)
        print("STEP 3: COMBINED TF-IDF + TRANSFORMER CLUSTERING")
        print("="*50)
        
        # Perform combined clustering
        combined_results = clustering_analyzer.perform_complete_clustering(
            n_clusters=args.n_clusters,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            variance_threshold=args.variance_threshold,
            use_combined_features=True,
            tfidf_features=tfidf_features,
            max_dimensions=args.max_dimensions
        )
        
        print("\nCombined clustering analysis completed!")
    
    if args.mode in ['umap_gmm', 'all']:
        if clustering_analyzer is None:
            # Load cached data for UMAP+GMM clustering only
            df, embeddings, tfidf_features, model = load_processed_data(args.cache_file)
            clustering_analyzer = ClusteringAnalyzer(df, embeddings)
        
        print("\n" + "="*50)
        print("STEP 4: UMAP + GMM CLUSTERING")
        print("="*50)
        
        # Perform UMAP + GMM clustering
        umap_gmm_results = clustering_analyzer.perform_umap_gmm_clustering(
            n_components=args.umap_components,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            n_gmm_components=args.gmm_components,
            covariance_type=args.gmm_covariance
        )
        
        # Create visualizations
        clustering_analyzer.visualize_umap_gmm_clustering()
        
        # Analyze cluster characteristics
        gmm_stats, gmm_sizes = clustering_analyzer.analyze_umap_gmm_cluster_characteristics()
        
        print("\nUMAP + GMM clustering analysis completed!")
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()