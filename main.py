#!/usr/bin/env python3
"""
Main script for Stack Overflow Data Analysis
Includes data preprocessing, clustering analysis, prediction models, and industrial-grade CTR system
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
from pathlib import Path
import seaborn as sns
from data_preprocessing import DataPreprocessor
from clustering_analysis import ClusteringAnalyzer, analyze_clustering_quality
from prediction_models import CTRPredictor, RetentionPredictor, run_prediction_analysis, IndustrialCTRPredictor

import warnings
warnings.filterwarnings('ignore')

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
    
    # Handle both old and new data formats
    if isinstance(data, dict) and 'df_combined' in data:
        # New format with normalization data
        return (data['df_combined'], data['embeddings'], 
                data['tfidf_features'], data['model'])
    else:
        # Old format (backward compatibility)
        return data

def run_preprocessing(args):
    """Run data preprocessing only"""
    print("=== Running Data Preprocessing Only ===")
    
    preprocessor = DataPreprocessor(base_path=args.data_dir)
    
    # Run complete preprocessing with feature normalization
    normalization_config = {
        'numerical_method': 'standard',  # Use StandardScaler for numerical features
        'categorical_method': 'label',   # Use LabelEncoder for categorical features
        'create_interactions': True,     # Create interaction features
        'create_polynomials': True,      # Create polynomial features
        'polynomial_degree': 2           # Degree of polynomial features
    }
    
    df_combined, embeddings, tfidf_features, model = preprocessor.preprocess_all(
        include_normalization=True,
        normalization_config=normalization_config
    )
    
    # Save processed data
    data = {
        'df_combined': df_combined,
        'embeddings': embeddings,
        'tfidf_features': tfidf_features,
        'model': model,
        'scalers': getattr(preprocessor, 'scalers', {}),
        'label_encoders': getattr(preprocessor, 'label_encoders', {})
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

def run_industrial_ctr_analysis(df_combined):
    """Run industrial-grade CTR analysis"""
    print("\n=== Industrial-Grade CTR System ===")
    print("Reference architecture from Alibaba, ByteDance, Google, Meta and other internet giants")
    
    # Create industrial features
    preprocessor = DataPreprocessor()
    df_industrial = preprocessor.create_industrial_features(df_combined.copy())
    
    # Perform negative sampling
    df_balanced = preprocessor.create_negative_sampling(df_industrial)
    
    # Train industrial models
    industrial_predictor = IndustrialCTRPredictor()
    models = industrial_predictor.train_industrial_models(df_balanced)
    
    # Evaluate models
    industrial_predictor.evaluate_industrial_models()
    
    # Online prediction example
    print("\n=== Online Prediction Example ===")
    sample_features = {
        'Score': 10,
        'ViewCount': 100,
        'AnswerCount': 2,
        'CommentCount': 5,
        'title_length': 15,
        'post_length': 200,
        'num_tags': 3,
        'post_age_days': 30,
        'user_post_count': 50,
        'user_reputation': 1000,
        'total_votes': 20,
        'upvotes': 18,
        'vote_ratio': 0.9,
        'total_influence_score': 500,
        'high_quality_influence': 200,
        'total_badges': 10,
        'badge_quality_score': 0.8,
        'badge_rate_per_day': 0.1,
        'hour_of_day': 0.5,
        'day_of_week': 0.3,
        'month_of_year': 0.6,
        'is_weekend': 0,
        'is_peak_hours': 1,
        'quality_view_ratio': 0.1,
        'vote_quality_ratio': 0.9,
        'engagement_efficiency': 20,
        'content_complexity': 13.3,
        'score_per_day': 0.33
    }
    
    # Add hash encoded features
    sample_features.update({
        'OwnerUserId_hash': hash('user_123') % 10000,
        'first_tag_hash': hash('python') % 10000,
        'influence_level_hash': hash('medium') % 10000,
        'badge_level_hash': hash('silver') % 10000,
        'multi_domain_influence_hash': hash('single') % 10000,
        'OwnerUserId_first_tag_cross': hash('user_123_python') % 1000,
        'influence_level_badge_level_cross': hash('medium_silver') % 1000,
        'user_post_count_user_reputation_cross': hash('50_1000') % 1000,
        'Score_ViewCount_cross': hash('10_100') % 1000
    })
    
    # Add sequence features (20 features: 5*4)
    for i in range(5):
        for j in range(4):
            sample_features[f'seq_{i}_{j}'] = 0
    
    prediction_result = industrial_predictor.online_predict(sample_features)
    print(f"Prediction result: {prediction_result}")
    
    return industrial_predictor

def main():
    """Main function to run the complete analysis pipeline"""
    print("=== Stack Overflow Data Analysis System ===")
    print("1. Data Preprocessing")
    print("2. Clustering Analysis")
    print("3. Basic Prediction Models")
    print("4. Industrial-Grade CTR System")
    print("5. Run Complete Pipeline")
    
    choice = input("\nPlease select an option (1-5): ").strip()
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    print("\n=== Loading and Preprocessing Data ===")
    df_posts, df_users, df_tags, df_votes, df_badges = preprocessor.load_data()
    
    if df_posts is None:
        print("Failed to load data. Please check your data files.")
        return
    
    # Run preprocessing
    df_combined = preprocessor.preprocess_all(include_normalization=True)
    
    if df_combined is None:
        print("Failed to preprocess data.")
        return
    
    print(f"Preprocessing completed. Combined dataset shape: {df_combined.shape}")
    
    # Execute based on user choice
    if choice == "1":
        print("\n=== Data Preprocessing Only ===")
        preprocessor.print_summary_statistics()
        
    elif choice == "2":
        print("\n=== Clustering Analysis ===")
        analyzer = ClusteringAnalyzer()
        analyzer.run_clustering_analysis(df_combined)
        
    elif choice == "3":
        print("\n=== Basic Prediction Models ===")
        run_prediction_analysis(df_combined)
        
    elif choice == "4":
        print("\n=== Industrial-Grade CTR System ===")
        industrial_predictor = run_industrial_ctr_analysis(df_combined)
        
    elif choice == "5":
        print("\n=== Running Complete Pipeline ===")
        
        # 1. Data preprocessing summary
        print("\n--- Step 1: Data Preprocessing Summary ---")
        preprocessor.print_summary_statistics()
        
        # 2. Clustering analysis
        print("\n--- Step 2: Clustering Analysis ---")
        analyzer = ClusteringAnalyzer()
        analyzer.run_clustering_analysis(df_combined)
        
        # 3. Basic prediction models
        print("\n--- Step 3: Basic Prediction Models ---")
        run_prediction_analysis(df_combined)
        
        # 4. Industrial-grade CTR system
        print("\n--- Step 4: Industrial-Grade CTR System ---")
        industrial_predictor = run_industrial_ctr_analysis(df_combined)
        
        print("\n=== Complete Pipeline Finished ===")
        print("All analyses have been completed successfully!")
        print("Check the 'output' folder for generated visualizations and results.")
        
    else:
        print("Invalid choice. Please select a number between 1 and 5.")

# Example usage of feature normalization
def example_feature_normalization():
    """Example of how to use feature normalization"""
    print("=== Feature Normalization Example ===")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Option 1: Run complete preprocessing with normalization
    print("\nOption 1: Complete preprocessing with normalization")
    df_combined, embeddings, tfidf_features, model = preprocessor.preprocess_all(
        include_normalization=True,
        normalization_config={
            'numerical_method': 'standard',
            'categorical_method': 'label',
            'create_interactions': True,
            'create_polynomials': True,
            'polynomial_degree': 2
        }
    )
    
    # Option 2: Run preprocessing without normalization first, then add it
    print("\nOption 2: Add normalization after preprocessing")
    preprocessor2 = DataPreprocessor()
    df_combined2, embeddings2, tfidf_features2, model2 = preprocessor2.preprocess_all(
        include_normalization=False
    )
    
    # Then add normalization
    df_normalized = preprocessor2.normalize_features()
    
    # Option 3: Custom normalization configuration
    print("\nOption 3: Custom normalization configuration")
    custom_config = {
        'numerical_method': 'robust',  # Use RobustScaler for numerical features
        'categorical_method': 'onehot',  # Use OneHotEncoder for categorical features
        'create_interactions': True,
        'create_polynomials': False,  # Don't create polynomial features
        'polynomial_degree': 2
    }
    
    df_custom = preprocessor2.normalize_features(custom_config)
    
    print("\nFeature normalization examples completed!")
    return df_combined, df_normalized, df_custom

if __name__ == "__main__":
    main()