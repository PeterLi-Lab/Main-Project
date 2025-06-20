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
    parser.add_argument('--mode', '-m', 
                       choices=['full', 'preprocess', 'cluster', 'visualize'],
                       default='full',
                       help='Analysis mode: full (complete pipeline), preprocess (data preprocessing only), cluster (clustering only), visualize (visualization only)')
    parser.add_argument('--data-dir', '-d',
                       default='data',
                       help='Directory containing data files')
    parser.add_argument('--output-dir', '-o',
                       default='output',
                       help='Directory to save/load processed data')
    parser.add_argument('--force-reprocess', '-f',
                       action='store_true',
                       help='Force reprocessing even if cached data exists')
    parser.add_argument('--max-features', '-mf',
                       type=int,
                       default=1000,
                       help='Maximum number of TF-IDF features')
    parser.add_argument('--n-components', '-nc',
                       type=int,
                       default=100,
                       help='Number of TF-IDF components after SVD')
    parser.add_argument('--n-clusters', '-k',
                       type=int,
                       default=None,
                       help='Number of clusters for K-means (auto-determined if not specified)')
    parser.add_argument('--variance-threshold', '-vt',
                       type=float,
                       default=0.9,
                       help='Variance threshold for PCA (default: 0.9 = 90%%)')
    
    return parser.parse_args()

def save_processed_data(data, output_dir):
    """Save processed data to files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save dataframe
    data['df_combined'].to_pickle(output_path / 'df_combined.pkl')
    
    # Save embeddings
    with open(output_path / 'embeddings.pkl', 'wb') as f:
        pickle.dump(data['embeddings'], f)
    
    # Save TF-IDF features
    with open(output_path / 'tfidf_features.pkl', 'wb') as f:
        pickle.dump(data['tfidf_features'], f)
    
    # Save model
    with open(output_path / 'sentence_model.pkl', 'wb') as f:
        pickle.dump(data['model'], f)
    
    print(f"Processed data saved to {output_path}")

def load_processed_data(output_dir):
    """Load processed data from files"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory {output_dir} does not exist")
    
    # Load dataframe
    df_combined = pd.read_pickle(output_path / 'df_combined.pkl')
    
    # Load embeddings
    with open(output_path / 'embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    
    # Load TF-IDF features
    with open(output_path / 'tfidf_features.pkl', 'rb') as f:
        tfidf_features = pickle.load(f)
    
    # Load model
    with open(output_path / 'sentence_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print(f"Processed data loaded from {output_path}")
    return df_combined, embeddings, tfidf_features, model

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
    save_processed_data(data, args.output_dir)
    
    return data

def run_clustering(args):
    """Run clustering analysis only"""
    print("=== Running Clustering Analysis Only ===")
    
    # Load processed data
    df_combined, embeddings, tfidf_features, model = load_processed_data(args.output_dir)
    
    # Run clustering
    cluster_analyzer = ClusteringAnalyzer(df_combined, embeddings)
    clustering_results = cluster_analyzer.perform_complete_clustering(
        n_clusters=args.n_clusters,
        variance_threshold=args.variance_threshold
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
    
    # DBSCAN quality analysis
    print("\n--- DBSCAN Quality Analysis ---")
    dbscan_labels = clustering_results['dbscan_labels']
    non_noise_mask = dbscan_labels != -1
    if sum(non_noise_mask) > 1:
        dbscan_quality = analyze_clustering_quality(
            embeddings[non_noise_mask], 
            dbscan_labels[non_noise_mask], 
            None
        )
    else:
        dbscan_quality = {'silhouette_score': None, 'n_clusters': 0}
    
    # Print results
    print_clustering_summary(df_combined, embeddings, tfidf_features, 
                           clustering_results, kmeans_quality, dbscan_quality)
    
    return {
        'df_combined': df_combined,
        'embeddings': embeddings,
        'tfidf_features': tfidf_features,
        'clustering_results': clustering_results,
        'kmeans_quality': kmeans_quality,
        'dbscan_quality': dbscan_quality
    }

def run_visualization(args):
    """Run visualization only"""
    print("=== Running Visualization Only ===")
    
    # Load processed data
    df_combined, embeddings, tfidf_features, model = load_processed_data(args.output_dir)
    
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
                           clustering_results, kmeans_quality, dbscan_quality):
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
    
    # DBSCAN summary
    print(f"\n--- DBSCAN Clustering Summary ---")
    dbscan_clusters = len([c for c in clustering_results['dbscan_sizes'].index if c != -1])
    dbscan_noise = clustering_results['dbscan_sizes'].get(-1, 0)
    print(f"Number of clusters: {dbscan_clusters}")
    print(f"Number of noise points: {dbscan_noise}")
    print(f"Clustering quality - Silhouette Score: {dbscan_quality.get('silhouette_score', 'N/A')}")
    
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
    
    print(f"\nDBSCAN Cluster Distribution:")
    dbscan_sizes = clustering_results['dbscan_sizes']
    for cluster_id, size in dbscan_sizes.items():
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
    """Main function to run the complete analysis pipeline"""
    # Parse command line arguments
    args = parse_arguments()
    
    print("=== Stack Overflow Data Analysis Pipeline ===")
    print(f"Mode: {args.mode}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Check if processed data exists and force-reprocess flag
    output_path = Path(args.output_dir)
    data_exists = output_path.exists() and (output_path / 'df_combined.pkl').exists()
    
    if args.mode == 'full':
        # Full pipeline
        if data_exists and not args.force_reprocess:
            print(f"\nFound existing processed data in {args.output_dir}")
            print("Loading cached data...")
            df_combined, embeddings, tfidf_features, model = load_processed_data(args.output_dir)
            
            # Run clustering with loaded data
            cluster_analyzer = ClusteringAnalyzer(df_combined, embeddings)
            clustering_results = cluster_analyzer.perform_complete_clustering(
                n_clusters=args.n_clusters,
                variance_threshold=args.variance_threshold
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
            
            # DBSCAN quality analysis
            print("\n--- DBSCAN Quality Analysis ---")
            dbscan_labels = clustering_results['dbscan_labels']
            non_noise_mask = dbscan_labels != -1
            if sum(non_noise_mask) > 1:
                dbscan_quality = analyze_clustering_quality(
                    embeddings[non_noise_mask], 
                    dbscan_labels[non_noise_mask], 
                    None
                )
            else:
                dbscan_quality = {'silhouette_score': None, 'n_clusters': 0}
            
            # Print results
            print_clustering_summary(df_combined, embeddings, tfidf_features, 
                                   clustering_results, kmeans_quality, dbscan_quality)
            
            results = {
                'df_combined': df_combined,
                'embeddings': embeddings,
                'tfidf_features': tfidf_features,
                'clustering_results': clustering_results,
                'kmeans_quality': kmeans_quality,
                'dbscan_quality': dbscan_quality
            }
        else:
            # Run complete preprocessing and clustering
            print("\n" + "="*50)
            print("STEP 1: DATA PREPROCESSING")
            print("="*50)
            
            preprocessor = DataPreprocessor(base_path=args.data_dir)
            df_combined, embeddings, tfidf_features, model = preprocessor.preprocess_all()
            
            # Save processed data
            data = {
                'df_combined': df_combined,
                'embeddings': embeddings,
                'tfidf_features': tfidf_features,
                'model': model
            }
            save_processed_data(data, args.output_dir)
            
            # Run clustering
            print("\n" + "="*50)
            print("STEP 2: CLUSTERING ANALYSIS (K-means + DBSCAN)")
            print("="*50)
            
            cluster_analyzer = ClusteringAnalyzer(df_combined, embeddings)
            clustering_results = cluster_analyzer.perform_complete_clustering(
                n_clusters=args.n_clusters,
                variance_threshold=args.variance_threshold
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
            
            # DBSCAN quality analysis
            print("\n--- DBSCAN Quality Analysis ---")
            dbscan_labels = clustering_results['dbscan_labels']
            non_noise_mask = dbscan_labels != -1
            if sum(non_noise_mask) > 1:
                dbscan_quality = analyze_clustering_quality(
                    embeddings[non_noise_mask], 
                    dbscan_labels[non_noise_mask], 
                    None
                )
            else:
                dbscan_quality = {'silhouette_score': None, 'n_clusters': 0}
            
            # Print results
            print_clustering_summary(df_combined, embeddings, tfidf_features, 
                                   clustering_results, kmeans_quality, dbscan_quality)
            
            results = {
                'df_combined': df_combined,
                'embeddings': embeddings,
                'tfidf_features': tfidf_features,
                'clustering_results': clustering_results,
                'kmeans_quality': kmeans_quality,
                'dbscan_quality': dbscan_quality
            }
    
    elif args.mode == 'preprocess':
        results = run_preprocessing(args)
    
    elif args.mode == 'cluster':
        results = run_clustering(args)
    
    elif args.mode == 'visualize':
        results = run_visualization(args)
    
    return results

if __name__ == "__main__":
    results = main()