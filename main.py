import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import the new modular components
from data_preprocessing import DataPreprocessor
from clustering_analysis import ClusteringAnalyzer, analyze_clustering_quality

def main():
    """Main function to run the complete analysis pipeline"""
    print("=== Stack Overflow Data Analysis Pipeline ===")
    
    # Step 1: Data Preprocessing
    print("\n" + "="*50)
    print("STEP 1: DATA PREPROCESSING")
    print("="*50)
    
    preprocessor = DataPreprocessor()
    df_combined, embeddings, model = preprocessor.preprocess_all()
    
    # Step 2: Clustering Analysis (K-means + DBSCAN)
    print("\n" + "="*50)
    print("STEP 2: CLUSTERING ANALYSIS (K-means + DBSCAN)")
    print("="*50)
    
    cluster_analyzer = ClusteringAnalyzer(df_combined, embeddings)
    clustering_results = cluster_analyzer.perform_complete_clustering()
    
    # Analyze clustering quality for both algorithms
    print("\n=== Clustering Quality Analysis ===")
    
    # K-means quality analysis
    print("\n--- K-means Quality Analysis ---")
    kmeans_quality = analyze_clustering_quality(
        embeddings, 
        clustering_results['kmeans_labels'], 
        clustering_results['kmeans_model']
    )
    
    # DBSCAN quality analysis (excluding noise points)
    print("\n--- DBSCAN Quality Analysis ---")
    dbscan_labels = clustering_results['dbscan_labels']
    non_noise_mask = dbscan_labels != -1
    if sum(non_noise_mask) > 1:
        dbscan_quality = analyze_clustering_quality(
            embeddings[non_noise_mask], 
            dbscan_labels[non_noise_mask], 
            None  # DBSCAN doesn't have inertia
        )
    else:
        dbscan_quality = {'silhouette_score': None, 'n_clusters': 0}
    
    # Step 3: Final Summary
    print("\n" + "="*50)
    print("STEP 3: FINAL SUMMARY")
    print("="*50)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Total posts analyzed: {len(df_combined)}")
    
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
    
    # Display some key statistics
    print(f"\n=== Key Statistics ===")
    print(f"Average post length: {df_combined['post_length'].mean():.1f} words")
    print(f"Average vote ratio: {df_combined['vote_ratio'].mean():.3f}")
    print(f"Average total votes: {df_combined['total_votes'].mean():.1f}")
    print(f"Most common post type: {df_combined['post_type'].mode().iloc[0] if len(df_combined['post_type'].mode()) > 0 else 'Unknown'}")
    
    # Show cluster distribution comparison
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
    
    # Algorithm comparison summary
    print(f"\n=== Algorithm Comparison Summary ===")
    comparison_df = clustering_results['comparison_df']
    print(comparison_df.to_string(index=False))
    
    return {
        'df_combined': df_combined,
        'embeddings': embeddings,
        'clustering_results': clustering_results,
        'kmeans_quality': kmeans_quality,
        'dbscan_quality': dbscan_quality
    }

if __name__ == "__main__":
    results = main()