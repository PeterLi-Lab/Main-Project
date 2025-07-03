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
from prediction_models import CTRPredictor, RetentionPredictor, run_prediction_analysis, IndustrialCTRPredictor, RetentionDurationPredictor, UpliftModeling

import warnings
warnings.filterwarnings('ignore')

# Import the new modular components
from data_preprocessing import DataPreprocessor
from clustering_analysis import ClusteringAnalyzer, analyze_clustering_quality

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stack Overflow Data Analysis Pipeline')
    parser.add_argument('--mode', choices=['preprocess', 'cluster', 'combined', 'umap_gmm', 'all', 'ctr', 'retention', 'duration', 'uplift'], 
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
    
    # Prediction model arguments
    parser.add_argument('--model-type', choices=['xgboost', 'lightgbm', 'random_forest', 'logistic_regression', 'linear_regression'], 
                       default='xgboost', help='Model type for prediction tasks')
    parser.add_argument('--target-col', default='ctr_proxy_normalized', 
                       help='Target column for prediction tasks')
    
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

def run_ctr_prediction(args):
    """Run CTR prediction only"""
    print("=== Running CTR Prediction Only ===")
    
    # Load processed data
    df_combined, embeddings, tfidf_features, model = load_processed_data(args.cache_file)
    
    # Train CTR model
    ctr_predictor = CTRPredictor()
    ctr_results = ctr_predictor.train_ctr_model(
        df_combined, 
        target_col=args.target_col, 
        model_type=args.model_type
    )
    
    # Visualize results
    if ctr_results:
        ctr_predictor.visualize_ctr_results(ctr_results)
    
    return {
        'df_combined': df_combined,
        'ctr_results': ctr_results,
        'ctr_predictor': ctr_predictor
    }

def run_retention_prediction(args):
    """Run retention prediction only"""
    print("=== Running Retention Prediction Only ===")
    
    # Load processed data
    df_combined, embeddings, tfidf_features, model = load_processed_data(args.cache_file)
    
    # Create preprocessor instance and generate retention samples
    preprocessor = DataPreprocessor(base_path=args.data_dir)
    preprocessor.df_combined = df_combined
    preprocessor.df_posts = preprocessor.load_data()['posts']
    preprocessor.df_votes = preprocessor.load_data()['votes']
    
    # Generate 7-day retention samples
    retention_samples = preprocessor.create_7day_retention_samples()
    
    # Train retention model with preprocessor
    retention_predictor = RetentionPredictor(preprocessor=preprocessor)
    retention_results = retention_predictor.train_retention_model(
        df_combined, 
        target_col='is_retained', 
        model_type=args.model_type,
        retention_window_days=7
    )
    
    # Visualize results
    if retention_results:
        retention_predictor.visualize_retention_results(retention_results)
    
    return {
        'df_combined': df_combined,
        'retention_results': retention_results,
        'retention_predictor': retention_predictor,
        'retention_samples': retention_samples
    }

def run_duration_prediction(args):
    """Run retention duration prediction only"""
    print("=== Running Retention Duration Prediction Only ===")
    
    # Load processed data
    df_combined, embeddings, tfidf_features, model = load_processed_data(args.cache_file)
    
    # Train duration model
    duration_predictor = RetentionDurationPredictor()
    duration_results = duration_predictor.train_duration_model(
        df_combined, 
        target_col='days_to_next_action', 
        model_type=args.model_type
    )
    
    # Visualize results
    if duration_results:
        duration_predictor.visualize_duration_results(duration_results)
    
    return {
        'df_combined': df_combined,
        'duration_results': duration_results,
        'duration_predictor': duration_predictor
    }

def run_uplift_modeling(args):
    """Run uplift modeling only"""
    print("=== Running Uplift Modeling Only ===")
    
    # Load processed data
    df_combined, embeddings, tfidf_features, model = load_processed_data(args.cache_file)
    
    # Train uplift models
    uplift_model = UpliftModeling()
    uplift_results = uplift_model.train_uplift_models(
        df_combined, 
        model_type=args.model_type
    )
    
    # Visualize results
    if uplift_results:
        uplift_model.visualize_uplift_results(uplift_results)
    
    return {
        'df_combined': df_combined,
        'uplift_results': uplift_results,
        'uplift_model': uplift_model
    }

def main():
    """Main function"""
    args = parse_arguments()
    
    print("=== Stack Overflow Data Analysis Pipeline ===")
    print(f"Mode: {args.mode}")
    print(f"Data directory: {args.data_dir}")
    print(f"Cache file: {args.cache_file}")
    
    try:
        if args.mode == 'preprocess':
            # Run preprocessing only
            data = run_preprocessing(args)
            
        elif args.mode == 'cluster':
            # Run clustering only
            data = run_clustering(args)
            
        elif args.mode == 'ctr':
            # Run CTR prediction only
            data = run_ctr_prediction(args)
            
        elif args.mode == 'retention':
            # Run retention prediction only
            data = run_retention_prediction(args)
            
        elif args.mode == 'duration':
            # Run retention duration prediction only
            data = run_duration_prediction(args)
            
        elif args.mode == 'uplift':
            # Run uplift modeling only
            data = run_uplift_modeling(args)
            
        elif args.mode == 'combined':
            # Run combined clustering
            data = run_combined_clustering(args)
            
        elif args.mode == 'umap_gmm':
            # Run UMAP + GMM clustering
            data = run_umap_gmm_clustering(args)
            
        elif args.mode == 'all':
            # Run complete pipeline
            print("\n=== Running Complete Pipeline ===")
            
            # 1. Preprocessing
            print("\n1. Data Preprocessing")
            data = run_preprocessing(args)
            
            # 2. Clustering
            print("\n2. Clustering Analysis")
            cluster_data = run_clustering(args)
            
            # 3. CTR Prediction
            print("\n3. CTR Prediction")
            ctr_data = run_ctr_prediction(args)
            
            # 4. Retention Prediction
            print("\n4. Retention Prediction")
            retention_data = run_retention_prediction(args)
            
            # 5. Duration Prediction
            print("\n5. Retention Duration Prediction")
            duration_data = run_duration_prediction(args)
            
            # 6. Uplift Modeling
            print("\n6. Uplift Modeling")
            uplift_data = run_uplift_modeling(args)
            
            # Combine all results
            data = {
                'preprocessing': data,
                'clustering': cluster_data,
                'ctr': ctr_data,
                'retention': retention_data,
                'duration': duration_data,
                'uplift': uplift_data
            }
        
        print(f"\n=== Pipeline completed successfully ===")
        
        # Save results summary
        save_results_summary(data, args.mode)
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def save_results_summary(data, mode):
    """Save a summary of the results"""
    import json
    from datetime import datetime
    
    summary = {
        'mode': mode,
        'timestamp': datetime.now().isoformat(),
        'results': {}
    }
    
    if mode == 'all':
        # Extract key metrics from each component
        if 'ctr' in data and data['ctr'].get('ctr_results'):
            summary['results']['ctr'] = {
                'accuracy': data['ctr']['ctr_results'].get('accuracy', 'N/A'),
                'model_type': 'CTR Prediction'
            }
        
        if 'retention' in data and data['retention'].get('retention_results'):
            summary['results']['retention'] = {
                'accuracy': data['retention']['retention_results'].get('accuracy', 'N/A'),
                'model_type': 'Retention Prediction'
            }
        
        if 'duration' in data and data['duration'].get('duration_results'):
            summary['results']['duration'] = {
                'mse': data['duration']['duration_results'].get('mse', 'N/A'),
                'r2': data['duration']['duration_results'].get('r2', 'N/A'),
                'model_type': 'Duration Prediction'
            }
        
        if 'uplift' in data and data['uplift'].get('uplift_results'):
            summary['results']['uplift'] = {
                'control_mse': data['uplift']['uplift_results'].get('control_mse', 'N/A'),
                'treatment_mse': data['uplift']['uplift_results'].get('treatment_mse', 'N/A'),
                'model_type': 'Uplift Modeling'
            }
    
    # Save summary
    os.makedirs('output', exist_ok=True)
    with open('output/results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results summary saved to: output/results_summary.json")

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