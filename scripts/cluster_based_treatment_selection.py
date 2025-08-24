import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ClusterBasedTreatmentSelection:
    """Cluster-based treatment and control selection for uplift modeling"""
    
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.ai_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning',
            'neural network', 'algorithm', 'automation', 'chatbot', 'gpt', 'llm',
            'data science', 'predictive', 'automated', 'intelligent', 'smart',
            'tensorflow', 'pytorch', 'scikit-learn', 'openai', 'claude',
            'nlp', 'computer vision', 'reinforcement learning', 'transformer'
        ]
        
    def extract_text_features(self, df, text_columns):
        """Extract and combine text features"""
        print("Extracting text features for clustering...")
        
        # Combine all text columns
        combined_text = df[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
        
        # Clean text
        combined_text = combined_text.str.lower()
        combined_text = combined_text.str.replace(r'[^\w\s]', ' ', regex=True)
        combined_text = combined_text.str.replace(r'\s+', ' ', regex=True)
        
        return combined_text
    
    def create_text_embeddings(self, text_data):
        """Create TF-IDF embeddings for clustering"""
        print("Creating TF-IDF embeddings...")
        
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        tfidf_matrix = vectorizer.fit_transform(text_data)
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        return tfidf_matrix, vectorizer
    
    def perform_clustering(self, tfidf_matrix, method='kmeans'):
        """Perform clustering on text embeddings"""
        print(f"Performing {method} clustering...")
        
        if method == 'kmeans':
            clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
        elif method == 'dbscan':
            clusterer = DBSCAN(
                eps=0.3,
                min_samples=5
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        clusters = clusterer.fit_predict(tfidf_matrix)
        
        print(f"Clustering completed. Found {len(np.unique(clusters))} clusters")
        
        return clusters, clusterer
    
    def identify_ai_clusters(self, df, clusters, text_data):
        """Identify clusters that contain AI-related content"""
        print("Identifying AI-related clusters...")
        
        # Calculate AI keyword density for each cluster
        cluster_ai_scores = {}
        
        for cluster_id in np.unique(clusters):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
                
            cluster_mask = clusters == cluster_id
            cluster_texts = text_data[cluster_mask]
            
            # Calculate AI keyword density
            ai_keyword_count = 0
            total_words = 0
            
            for text in cluster_texts:
                text_lower = text.lower()
                for keyword in self.ai_keywords:
                    if keyword in text_lower:
                        ai_keyword_count += 1
                total_words += len(text.split())
            
            # Calculate AI density score
            if total_words > 0:
                ai_density = ai_keyword_count / len(cluster_texts)  # Average keywords per post
                cluster_ai_scores[cluster_id] = ai_density
            else:
                cluster_ai_scores[cluster_id] = 0
        
        # Sort clusters by AI density
        sorted_clusters = sorted(cluster_ai_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("Cluster AI density scores:")
        for cluster_id, score in sorted_clusters[:5]:
            cluster_size = (clusters == cluster_id).sum()
            print(f"  Cluster {cluster_id}: {score:.3f} (size: {cluster_size})")
        
        # Select top clusters as potential AI clusters
        ai_cluster_threshold = np.mean([score for _, score in sorted_clusters]) + np.std([score for _, score in sorted_clusters])
        ai_clusters = [cluster_id for cluster_id, score in sorted_clusters if score >= ai_cluster_threshold]
        
        print(f"Selected {len(ai_clusters)} AI-related clusters (threshold: {ai_cluster_threshold:.3f})")
        
        return ai_clusters, cluster_ai_scores
    
    def select_treatment_and_control(self, df, clusters, ai_clusters, tag_columns):
        """Select treatment and control groups from AI clusters"""
        print("Selecting treatment and control groups...")
        
        # Create mask for AI cluster posts
        ai_cluster_mask = np.isin(clusters, ai_clusters)
        ai_cluster_posts = df[ai_cluster_mask].copy()
        
        print(f"Posts in AI clusters: {len(ai_cluster_posts):,}")
        
        # Create treatment labels based on tag
        if tag_columns:
            tag_col = tag_columns[0]
            ai_cluster_posts['treatment_ai_content'] = ai_cluster_posts[tag_col].str.contains(
                'ai content', case=False, na=False
            ).astype(int)
        else:
            # If no tag column, create dummy treatment labels
            ai_cluster_posts['treatment_ai_content'] = np.random.choice(
                [0, 1], size=len(ai_cluster_posts), p=[0.7, 0.3]
            )
        
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
    
    def analyze_cluster_composition(self, df, clusters, ai_clusters):
        """Analyze the composition of AI clusters"""
        print("\n=== AI Cluster Analysis ===")
        
        for cluster_id in ai_clusters:
            cluster_mask = clusters == cluster_id
            cluster_posts = df[cluster_mask]
            
            print(f"\nCluster {cluster_id} (size: {len(cluster_posts):,}):")
            
            # Show some example texts
            text_columns = [col for col in df.columns if any(x in col.lower() for x in ['content', 'text', 'title', 'tag'])]
            if text_columns:
                combined_text = self.extract_text_features(cluster_posts, text_columns)
                print("Sample texts:")
                for i, text in enumerate(combined_text.head(3)):
                    print(f"  {i+1}. {text[:100]}...")
            
            # Check AI keyword frequency
            if text_columns:
                combined_text = self.extract_text_features(cluster_posts, text_columns)
                keyword_counts = {}
                for keyword in self.ai_keywords:
                    count = sum(1 for text in combined_text if keyword in text.lower())
                    if count > 0:
                        keyword_counts[keyword] = count
                
                if keyword_counts:
                    print("Top AI keywords:")
                    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
                    for keyword, count in sorted_keywords[:5]:
                        print(f"    {keyword}: {count}")
    
    def visualize_clusters(self, df, clusters, ai_clusters, text_data):
        """Visualize cluster distribution and AI density"""
        print("Creating cluster visualization...")
        
        # Calculate AI density for each cluster
        cluster_ai_density = {}
        for cluster_id in np.unique(clusters):
            if cluster_id == -1:
                continue
            cluster_mask = clusters == cluster_id
            cluster_texts = text_data[cluster_mask]
            
            ai_count = sum(1 for text in cluster_texts 
                          for keyword in self.ai_keywords 
                          if keyword in text.lower())
            density = ai_count / len(cluster_texts) if len(cluster_texts) > 0 else 0
            cluster_ai_density[cluster_id] = density
        
        # Create visualization data
        cluster_sizes = [np.sum(clusters == i) for i in np.unique(clusters) if i != -1]
        cluster_densities = [cluster_ai_density.get(i, 0) for i in np.unique(clusters) if i != -1]
        cluster_ids = [i for i in np.unique(clusters) if i != -1]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Scatter plot: cluster size vs AI density
        colors = ['red' if i in ai_clusters else 'blue' for i in cluster_ids]
        plt.scatter(cluster_sizes, cluster_densities, c=colors, alpha=0.7, s=100)
        
        # Add labels
        for i, (size, density, cluster_id) in enumerate(zip(cluster_sizes, cluster_densities, cluster_ids)):
            plt.annotate(f'C{cluster_id}', (size, density), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        plt.xlabel('Cluster Size')
        plt.ylabel('AI Keyword Density')
        plt.title('Cluster Analysis: Size vs AI Density')
        plt.legend(['AI Clusters', 'Other Clusters'])
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
        print("Cluster visualization saved as 'cluster_analysis.png'")
        
        return cluster_ai_density

def cluster_based_treatment_selection():
    """Main function for cluster-based treatment selection"""
    print("=== Cluster-Based Treatment Selection ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Find text columns
    text_columns = [col for col in df.columns if any(x in col.lower() for x in ['content', 'text', 'title', 'tag'])]
    print(f"Text columns found: {text_columns}")
    
    if not text_columns:
        print("No text columns found")
        return None
    
    # Find tag columns
    tag_columns = [col for col in df.columns if 'tag' in col.lower()]
    print(f"Tag columns found: {tag_columns}")
    
    # Initialize cluster selector
    selector = ClusterBasedTreatmentSelection(n_clusters=10)
    
    # Extract text features
    text_data = selector.extract_text_features(df, text_columns)
    
    # Create embeddings
    tfidf_matrix, vectorizer = selector.create_text_embeddings(text_data)
    
    # Perform clustering
    clusters, clusterer = selector.perform_clustering(tfidf_matrix, method='kmeans')
    
    # Identify AI clusters
    ai_clusters, cluster_scores = selector.identify_ai_clusters(df, clusters, text_data)
    
    # Analyze cluster composition
    selector.analyze_cluster_composition(df, clusters, ai_clusters)
    
    # Visualize clusters
    cluster_density = selector.visualize_clusters(df, clusters, ai_clusters, text_data)
    
    # Select treatment and control groups
    final_df, treatment_posts, control_posts = selector.select_treatment_and_control(
        df, clusters, ai_clusters, tag_columns
    )
    
    # Final analysis
    print("\n=== Final Dataset Analysis ===")
    print(f"Total samples: {len(final_df):,}")
    print(f"Treatment samples: {len(treatment_posts):,}")
    print(f"Control samples: {len(control_posts):,}")
    
    # Check response distribution if available
    if 'response' in final_df.columns:
        treatment_response_rate = final_df[final_df['treatment_ai_content'] == 1]['response'].mean()
        control_response_rate = final_df[final_df['treatment_ai_content'] == 0]['response'].mean()
        uplift = treatment_response_rate - control_response_rate
        
        print(f"\nUplift Analysis:")
        print(f"  Treatment response rate: {treatment_response_rate:.2%}")
        print(f"  Control response rate: {control_response_rate:.2%}")
        print(f"  Uplift: {uplift:.2%}")
    
    # Save results
    output_file = 'uplift_model_data_cluster_based.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\nCluster-based dataset saved to: {output_file}")
    
    return final_df

if __name__ == "__main__":
    cluster_based_treatment_selection() 