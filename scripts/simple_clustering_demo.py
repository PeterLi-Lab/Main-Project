import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def simple_clustering_demo():
    """Simple clustering demo for treatment selection"""
    print("=== Simple Clustering Demo ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Take a sample for demo
    sample_size = min(10000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    print(f"Using sample of {sample_size:,} records for demo")
    
    # Select features for clustering
    exclude_cols = ['treatment_ai_content', 'response', 'user_id', 'post_id']
    feature_cols = [col for col in df_sample.columns 
                   if col not in exclude_cols and df_sample[col].dtype in ['int64', 'float64']]
    
    print(f"Using {len(feature_cols)} features for clustering")
    
    # Prepare features
    X = df_sample[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    print("Performing clustering...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    print(f"Created {len(np.unique(clusters))} clusters")
    
    # Identify AI-related clusters
    ai_features = ['user_ai_interest_score', 'user_ai_interest_weighted', 'user_ai_interactions']
    available_ai_features = [col for col in ai_features if col in feature_cols]
    
    print(f"Available AI features: {available_ai_features}")
    
    # Calculate AI scores for each cluster
    cluster_ai_scores = {}
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        cluster_data = df_sample[cluster_mask]
        
        ai_scores = []
        for feature in available_ai_features:
            feature_mean = cluster_data[feature].mean()
            ai_scores.append(feature_mean)
        
        cluster_ai_score = np.mean(ai_scores) if ai_scores else 0
        cluster_ai_scores[cluster_id] = cluster_ai_score
    
    # Sort clusters by AI score
    sorted_clusters = sorted(cluster_ai_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\nCluster AI scores:")
    for cluster_id, score in sorted_clusters:
        cluster_size = (clusters == cluster_id).sum()
        print(f"  Cluster {cluster_id}: {score:.4f} (size: {cluster_size:,})")
    
    # Select top 2 clusters as AI clusters
    ai_clusters = [cluster_id for cluster_id, _ in sorted_clusters[:2]]
    print(f"\nSelected AI clusters: {ai_clusters}")
    
    # Select posts from AI clusters
    ai_cluster_mask = np.isin(clusters, ai_clusters)
    ai_cluster_posts = df_sample[ai_cluster_mask].copy()
    
    print(f"Posts in AI clusters: {len(ai_cluster_posts):,}")
    
    # Create treatment labels if not exists
    if 'treatment_ai_content' not in ai_cluster_posts.columns:
        # Use AI interest score to create treatment labels
        if 'user_ai_interest_score' in ai_cluster_posts.columns:
            threshold = ai_cluster_posts['user_ai_interest_score'].median()
            ai_cluster_posts['treatment_ai_content'] = (ai_cluster_posts['user_ai_interest_score'] > threshold).astype(int)
            print(f"Created treatment labels based on user_ai_interest_score (threshold: {threshold:.4f})")
        else:
            ai_cluster_posts['treatment_ai_content'] = np.random.choice([0, 1], size=len(ai_cluster_posts), p=[0.7, 0.3])
            print("Created random treatment labels")
    
    # Split into treatment and control
    treatment_posts = ai_cluster_posts[ai_cluster_posts['treatment_ai_content'] == 1]
    control_posts = ai_cluster_posts[ai_cluster_posts['treatment_ai_content'] == 0]
    
    print(f"\nTreatment posts: {len(treatment_posts):,}")
    print(f"Control posts: {len(control_posts):,}")
    
    # Analyze AI features in treatment vs control
    print("\n=== AI Feature Analysis ===")
    for feature in available_ai_features:
        treatment_mean = treatment_posts[feature].mean()
        control_mean = control_posts[feature].mean()
        print(f"{feature}:")
        print(f"  Treatment mean: {treatment_mean:.4f}")
        print(f"  Control mean: {control_mean:.4f}")
        print(f"  Difference: {treatment_mean - control_mean:.4f}")
    
    # Check response distribution
    if 'response' in ai_cluster_posts.columns:
        treatment_response_rate = treatment_posts['response'].mean()
        control_response_rate = control_posts['response'].mean()
        uplift = treatment_response_rate - control_response_rate
        
        print(f"\n=== Uplift Analysis ===")
        print(f"Treatment response rate: {treatment_response_rate:.2%}")
        print(f"Control response rate: {control_response_rate:.2%}")
        print(f"Uplift: {uplift:.2%}")
        
        if uplift > 0:
            print("✅ Positive uplift detected")
        elif uplift < 0:
            print("⚠️  Negative uplift detected")
        else:
            print("➖ No uplift detected")
    
    # Create visualization
    print("\nCreating visualization...")
    
    # Use first two features for visualization
    if len(feature_cols) >= 2:
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        colors = ['red' if i in ai_clusters else 'blue' for i in clusters]
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=colors, alpha=0.6, s=20)
        
        plt.xlabel(f'Feature 1: {feature_cols[0]}')
        plt.ylabel(f'Feature 2: {feature_cols[1]}')
        plt.title('Cluster Analysis Demo')
        plt.legend(['AI Clusters', 'Other Clusters'])
        plt.grid(True, alpha=0.3)
        
        plt.savefig('simple_cluster_demo.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'simple_cluster_demo.png'")
    
    # Save results
    final_df = pd.concat([treatment_posts, control_posts], ignore_index=True)
    output_file = 'uplift_model_data_simple_clustering.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    print(f"\n=== Summary ===")
    print(f"Total samples analyzed: {len(df_sample):,}")
    print(f"AI cluster samples: {len(ai_cluster_posts):,}")
    print(f"Treatment samples: {len(treatment_posts):,}")
    print(f"Control samples: {len(control_posts):,}")
    
    return final_df

if __name__ == "__main__":
    simple_clustering_demo() 