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
from prediction_models import CTRPredictor, RetentionPredictor, run_prediction_analysis, IndustrialCTRPredictor, RetentionDurationPredictor, UpliftModeling, MultiTaskPredictor
from visualization import VisualizationModule

import warnings
warnings.filterwarnings('ignore')

# Import the new modular components
from data_preprocessing import DataPreprocessor
from clustering_analysis import ClusteringAnalyzer, analyze_clustering_quality

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stack Overflow Data Analysis Pipeline')
    parser.add_argument('--mode', choices=['preprocess', 'cluster', 'combined', 'umap_gmm', 'all', 'ctr', 'retention', 'duration', 'uplift', 'multitask'], 
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
    
    # Multi-task learning arguments
    parser.add_argument('--multitask-architecture', choices=['mmoe', 'ple', 'simplified'], 
                       default='mmoe', help='Multi-task learning architecture')
    parser.add_argument('--visualize', action='store_true', help='Generate comprehensive visualizations')
    parser.add_argument('--save-plots', action='store_true', help='Save all plots to output directory')
    
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
    
    # Load raw data for retention sample generation
    df_posts, df_users, df_tags, df_votes, df_badges = preprocessor.load_data()
    preprocessor.df_posts = df_posts
    preprocessor.df_votes = df_votes
    
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
    
    # Create preprocessor instance and load raw data
    preprocessor = DataPreprocessor(base_path=args.data_dir)
    df_posts, df_users, df_tags, df_votes, df_badges = preprocessor.load_data()
    preprocessor.df_posts = df_posts
    # Generate duration samples
    duration_samples = preprocessor.create_retention_duration_samples()
    # Train regression model
    duration_predictor = RetentionDurationPredictor()
    duration_results = duration_predictor.train_duration_model(
        duration_samples, df_users=df_users, model_type=args.model_type
    )
    # Visualize results
    if duration_results:
        y_test = duration_results['y_test']
        y_pred = duration_results['y_pred']
        plt.figure(figsize=(8,6))
        plt.scatter(y_test, y_pred, alpha=0.2)
        plt.xlabel('True days_to_next_action')
        plt.ylabel('Predicted')
        plt.title('Retention Duration Regression')
        plt.show()
    return {
        'df_combined': df_combined,
        'duration_results': duration_results,
        'duration_predictor': duration_predictor,
        'duration_samples': duration_samples
    }

def run_uplift_modeling(args):
    """Run uplift modeling only"""
    print("=== Running Uplift Modeling Only ===")
    
    # Load processed data
    df_combined, embeddings, tfidf_features, model = load_processed_data(args.cache_file)
    
    # Create preprocessor instance and load raw data
    preprocessor = DataPreprocessor(base_path=args.data_dir)
    df_posts, df_users, df_tags, df_votes, df_badges = preprocessor.load_data()
    preprocessor.df_posts = df_posts
    preprocessor.df_votes = df_votes
    
    # Generate uplift samples
    all_users = set(df_users['Id'].astype(str))
    positive_pairs = set(zip(df_posts['user_id'], df_posts['post_id']))
    negative_samples = []
    np.random.seed(42)
    for post_id, group in df_posts.groupby('post_id'):
        liked_users = set(group['user_id'])
        possible_users = list(all_users - liked_users)
        n_neg = len(group)  # Number of negative samples = number of positive samples
        if n_neg == 0 or len(possible_users) == 0:
            continue
        sampled_users = np.random.choice(possible_users, size=min(n_neg, len(possible_users)), replace=False)
        for i, user_id in enumerate(sampled_users):
            # 50% probability to assign to treatment/control
            if i < len(sampled_users) // 2:
                # Recommendation group, time_diff random [0, 24]
                time_diff = np.random.uniform(0, 24)
                treatment = 1
            else:
                # Control group, time_diff random [24, max]
                time_diff = np.random.uniform(24, df_posts['time_diff_hours'].max())
                treatment = 0
            negative_samples.append({
                'user_id': user_id,
                'post_id': post_id,
                'treatment': treatment,
                'is_click': 0,
                'time_diff_hours': time_diff
            })
    if negative_samples:
        df_neg = pd.DataFrame(negative_samples)
        df_uplift = pd.concat([df_posts, df_neg], ignore_index=True)
    
    # Train uplift models
    uplift_model = UpliftModeling()
    uplift_results = uplift_model.train_uplift_models(
        df_uplift, model_type=args.model_type
    )
    
    # Visualize results
    if uplift_results:
        uplift_model.visualize_uplift_results(uplift_results)
    
    return {
        'df_combined': df_combined,
        'uplift_results': uplift_results,
        'uplift_model': uplift_model,
        'uplift_samples': df_uplift
    }

def run_multitask_learning(args):
    """Run multi-task learning pipeline"""
    print("\n=== Running Multi-Task Learning Pipeline ===")
    
    try:
        # Load preprocessed data
        cache_file = 'data/processed_data_cache.pkl'
        if os.path.exists(cache_file):
            df_combined, embeddings, tfidf_features, model = load_processed_data(cache_file)
            print("Loaded cached preprocessed data")
        else:
            print("No cached data found. Please run preprocessing first.")
            return None
        
        # Initialize multi-task predictor
        from prediction_models import MultiTaskPredictor
        
        # Prepare features for multi-task learning
        # Combine numerical and categorical features
        feature_columns = [col for col in df_combined.columns 
                         if col not in ['PostId', 'UserId', 'Title', 'Body', 'Tags', 'CreationDate']]
        
        X = df_combined[feature_columns].fillna(0)
        
        # Generate synthetic labels for multi-task learning
        np.random.seed(42)
        n_samples = len(X)
        
        # Task 1: Click prediction (binary)
        click_labels = np.random.binomial(1, 0.3, n_samples)
        
        # Task 2: Conversion prediction (binary)
        conversion_labels = np.random.binomial(1, 0.1, n_samples)
        
        # Task 3: Engagement score (regression)
        engagement_scores = np.random.normal(0.5, 0.2, n_samples)
        engagement_scores = np.clip(engagement_scores, 0, 1)
        
        # Initialize multi-task predictor
        multitask_predictor = MultiTaskPredictor(
            input_dim=len(feature_columns),
            task_configs={
                'click': {'type': 'binary', 'output_dim': 1},
                'conversion': {'type': 'binary', 'output_dim': 1},
                'engagement': {'type': 'regression', 'output_dim': 1}
            },
            architecture='mmoe',  # or 'ple'
            num_experts=4,
            expert_dim=64
        )
        
        # Train the model
        print("Training multi-task learning model...")
        history = multitask_predictor.train(
            X, 
            {
                'click': click_labels,
                'conversion': conversion_labels,
                'engagement': engagement_scores
            },
            validation_split=0.2,
            epochs=50,
            batch_size=32
        )
        
        # Evaluate the model
        print("Evaluating multi-task learning model...")
        predictions = multitask_predictor.predict(X)
        
        # Calculate metrics for each task
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
        
        multitask_results = {}
        
        # Click prediction metrics
        click_accuracy = accuracy_score(click_labels, predictions['click'] > 0.5)
        multitask_results['click'] = {
            'accuracy': click_accuracy,
            'task_type': 'binary_classification'
        }
        
        # Conversion prediction metrics
        conversion_accuracy = accuracy_score(conversion_labels, predictions['conversion'] > 0.5)
        multitask_results['conversion'] = {
            'accuracy': conversion_accuracy,
            'task_type': 'binary_classification'
        }
        
        # Engagement prediction metrics
        engagement_mse = mean_squared_error(engagement_scores, predictions['engagement'])
        engagement_r2 = r2_score(engagement_scores, predictions['engagement'])
        multitask_results['engagement'] = {
            'mse': engagement_mse,
            'r2': engagement_r2,
            'task_type': 'regression'
        }
        
        print("\nMulti-Task Learning Results:")
        print(f"Click Prediction Accuracy: {click_accuracy:.4f}")
        print(f"Conversion Prediction Accuracy: {conversion_accuracy:.4f}")
        print(f"Engagement Prediction MSE: {engagement_mse:.4f}")
        print(f"Engagement Prediction R²: {engagement_r2:.4f}")
        
        # Generate visualizations if requested
        if args.visualize:
            print("\nGenerating multi-task learning visualizations...")
            viz = VisualizationModule()
            
            # Create results dictionary for visualization
            viz_results = {
                'MultiTask_Click': {'predictions': predictions['click'], 'labels': click_labels},
                'MultiTask_Conversion': {'predictions': predictions['conversion'], 'labels': conversion_labels},
                'MultiTask_Engagement': {'predictions': predictions['engagement'], 'labels': engagement_scores}
            }
            
            viz.plot_roc_curves(viz_results)
            viz.plot_model_comparison(viz_results, metric='accuracy')
            
            if args.save_plots:
                viz.save_all_plots(viz_results, prefix="multitask_analysis")
        
        return {
            'multitask_results': multitask_results,
            'model': multitask_predictor,
            'history': history
        }
        
    except Exception as e:
        print(f"Error in multi-task learning: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_combined_clustering(args):
    """Run combined clustering analysis"""
    print("\n=== Running Combined Clustering Analysis ===")
    
    try:
        # Load preprocessed data
        cache_file = 'data/processed_data_cache.pkl'
        if os.path.exists(cache_file):
            df_combined, embeddings, tfidf_features, model = load_processed_data(cache_file)
            print("Loaded cached preprocessed data")
        else:
            print("No cached data found. Please run preprocessing first.")
            return None
        
        # Initialize clustering analyzer
        from clustering_analysis import ClusteringAnalyzer
        analyzer = ClusteringAnalyzer(df_combined, embeddings)
        
        # Perform combined clustering
        print("Performing combined clustering analysis...")
        cluster_labels, cluster_centers, cluster_quality = analyzer.perform_combined_clustering(
            tfidf_features, 
            embeddings,
            variance_threshold=0.9,
            max_dimensions=50
        )
        
        # Analyze cluster characteristics
        print("Analyzing cluster characteristics...")
        cluster_analysis = analyzer.analyze_combined_cluster_characteristics()
        
        # Generate visualizations if requested
        if args.visualize:
            print("Generating clustering visualizations...")
            analyzer.visualize_combined_clustering()
            
            if args.save_plots:
                # Save clustering plots
                plt.savefig('output/combined_clustering_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers,
            'cluster_quality': cluster_quality,
            'cluster_analysis': cluster_analysis
        }
        
    except Exception as e:
        print(f"Error in combined clustering: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_umap_gmm_clustering(args):
    """Run UMAP + GMM clustering analysis"""
    print("\n=== Running UMAP + GMM Clustering Analysis ===")
    
    try:
        # Load preprocessed data
        cache_file = 'data/processed_data_cache.pkl'
        if os.path.exists(cache_file):
            df_combined, embeddings, tfidf_features, model = load_processed_data(cache_file)
            print("Loaded cached preprocessed data")
        else:
            print("No cached data found. Please run preprocessing first.")
            return None
        
        # Initialize clustering analyzer
        from clustering_analysis import ClusteringAnalyzer
        analyzer = ClusteringAnalyzer(df_combined, embeddings)
        
        # Perform UMAP + GMM clustering
        print("Performing UMAP + GMM clustering analysis...")
        cluster_labels, cluster_centers, cluster_quality = analyzer.perform_umap_gmm_clustering(
            n_components=50,
            n_neighbors=15,
            min_dist=0.1,
            n_gmm_components=10
        )
        
        # Analyze cluster characteristics
        print("Analyzing cluster characteristics...")
        cluster_analysis = analyzer.analyze_umap_gmm_cluster_characteristics()
        
        # Generate visualizations if requested
        if args.visualize:
            print("Generating UMAP + GMM clustering visualizations...")
            analyzer.visualize_umap_gmm_clustering()
            
            if args.save_plots:
                # Save clustering plots
                plt.savefig('output/umap_gmm_clustering_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers,
            'cluster_quality': cluster_quality,
            'cluster_analysis': cluster_analysis
        }
        
    except Exception as e:
        print(f"Error in UMAP + GMM clustering: {e}")
        import traceback
        traceback.print_exc()
        return None

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
            
        elif args.mode == 'multitask':
            # Run multi-task learning
            data = run_multitask_learning(args)
            
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
            
            # 7. Multi-Task Learning
            print("\n7. Multi-Task Learning")
            multitask_data = run_multitask_learning(args)
            
            # Combine all results
            data = {
                'preprocessing': data,
                'clustering': cluster_data,
                'ctr': ctr_data,
                'retention': retention_data,
                'duration': duration_data,
                'uplift': uplift_data,
                'multitask': multitask_data
            }
            
            # Generate comprehensive visualizations if requested
            if args.visualize:
                print("\n8. Generating Comprehensive Visualizations")
                viz = VisualizationModule()
                
                # Create combined results for visualization
                combined_results = {}
                if 'ctr' in data and data['ctr'].get('ctr_results'):
                    combined_results['CTR'] = data['ctr']['ctr_results']
                if 'retention' in data and data['retention'].get('retention_results'):
                    combined_results['Retention'] = data['retention']['retention_results']
                if 'multitask' in data and data['multitask'].get('multitask_results'):
                    combined_results['MultiTask'] = data['multitask']['multitask_results']
                
                if combined_results:
                    viz.plot_model_comparison(combined_results, metric='auc')
                    viz.plot_roc_curves(combined_results)
                    
                    if args.save_plots:
                        viz.save_all_plots(combined_results, prefix="comprehensive_analysis")

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