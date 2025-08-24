import pandas as pd
import numpy as np
import re

def analyze_ai_clusters():
    """Analyze which clusters actually contain AI content"""
    print("ANALYZING CORRECT AI CLUSTERS")
    print("="*50)
    
    # Load cluster data
    df = pd.read_csv('post_clusters.csv')
    print(f"Total posts: {len(df):,}")
    print(f"Number of clusters: {df['cluster_id'].nunique()}")
    
    # Define AI keywords
    ai_keywords = [
        'ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
        'gpt', 'llm', 'data science', 'predictive', 'automated', 'intelligent', 'smart',
        'tensorflow', 'pytorch', 'scikit-learn', 'openai', 'nlp', 'computer vision', 
        'reinforcement learning', 'transformer', 'keras', 'svm', 'random forest', 'xgboost'
    ]
    
    # Analyze each cluster for AI content
    print(f"\nAI CONTENT ANALYSIS BY CLUSTER")
    print("-" * 50)
    
    cluster_ai_analysis = []
    
    for cluster_id in sorted(df['cluster_id'].unique()):
        cluster_data = df[df['cluster_id'] == cluster_id]
        cluster_size = len(cluster_data)
        
        # Count AI keyword occurrences
        ai_counts = {}
        total_ai_posts = 0
        
        for keyword in ai_keywords:
            # Count posts containing this keyword
            keyword_count = cluster_data['merged_content'].str.contains(keyword, case=False, na=False).sum()
            ai_counts[keyword] = keyword_count
            
            if keyword_count > 0:
                total_ai_posts += keyword_count
        
        # Calculate AI density
        ai_density = total_ai_posts / cluster_size if cluster_size > 0 else 0
        
        # Find top AI keywords for this cluster
        top_ai_keywords = sorted(ai_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_ai_keywords = [f"{kw}:{count}" for kw, count in top_ai_keywords if count > 0]
        
        cluster_ai_analysis.append({
            'cluster_id': cluster_id,
            'size': cluster_size,
            'ai_density': ai_density,
            'total_ai_posts': total_ai_posts,
            'top_ai_keywords': top_ai_keywords
        })
        
        print(f"Cluster {cluster_id}:")
        print(f"  Size: {cluster_size:,}")
        print(f"  AI density: {ai_density:.3f}")
        print(f"  Total AI posts: {total_ai_posts:,}")
        print(f"  Top AI keywords: {top_ai_keywords}")
        print()
    
    # Sort clusters by AI density
    cluster_ai_analysis.sort(key=lambda x: x['ai_density'], reverse=True)
    
    print(f"CLUSTERS RANKED BY AI DENSITY")
    print("-" * 50)
    for i, cluster_info in enumerate(cluster_ai_analysis):
        print(f"{i+1}. Cluster {cluster_info['cluster_id']}: AI density = {cluster_info['ai_density']:.3f}")
    
    # Recommend the best AI cluster
    best_cluster = cluster_ai_analysis[0]
    print(f"\nRECOMMENDATION:")
    print(f"Best AI cluster: Cluster {best_cluster['cluster_id']}")
    print(f"  AI density: {best_cluster['ai_density']:.3f}")
    print(f"  Size: {best_cluster['size']:,}")
    print(f"  Top AI keywords: {best_cluster['top_ai_keywords']}")
    
    # Compare with current Cluster 7
    cluster7_info = next((c for c in cluster_ai_analysis if c['cluster_id'] == 7), None)
    if cluster7_info:
        print(f"\nCURRENT CLUSTER 7 ANALYSIS:")
        print(f"  AI density: {cluster7_info['ai_density']:.3f}")
        print(f"  Rank: #{[c['cluster_id'] for c in cluster_ai_analysis].index(7) + 1}")
        print(f"  Problem: Very low AI density!")
    
    return cluster_ai_analysis

def create_correct_ai_cluster():
    """Create treatment/control split for the best AI cluster"""
    print(f"\nCREATING CORRECT AI CLUSTER ANALYSIS")
    print("-" * 50)
    
    # Load data
    df = pd.read_csv('post_clusters.csv')
    
    # Find the best AI cluster (Cluster 1 based on previous analysis)
    best_cluster_id = 1  # Machine learning cluster
    cluster_data = df[df['cluster_id'] == best_cluster_id].copy()
    
    print(f"Using Cluster {best_cluster_id} (Machine Learning cluster)")
    print(f"Cluster size: {len(cluster_data):,}")
    
    # Define AI keywords for treatment/control split
    ai_keywords = [
        'ai', 'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
        'gpt', 'llm', 'data science', 'predictive', 'automated', 'intelligent', 'smart',
        'tensorflow', 'pytorch', 'scikit-learn', 'openai', 'nlp', 'computer vision', 
        'reinforcement learning', 'transformer'
    ]
    ai_pattern = re.compile('|'.join(ai_keywords), re.IGNORECASE)
    
    # Create treatment/control split
    def assign_group(text):
        if pd.isnull(text):
            return 'control'
        return 'treatment' if ai_pattern.search(text) else 'control'
    
    cluster_data['group'] = cluster_data['merged_content'].apply(assign_group)
    
    # Analyze results
    treatment_count = (cluster_data['group'] == 'treatment').sum()
    control_count = (cluster_data['group'] == 'control').sum()
    
    print(f"Treatment: {treatment_count:,} ({treatment_count/len(cluster_data):.1%})")
    print(f"Control: {control_count:,} ({control_count/len(cluster_data):.1%})")
    
    # Save results
    output_file = f'cluster{best_cluster_id}_treatment_control.csv'
    cluster_data[['Id', 'Title', 'Body', 'Tags', 'merged_content', 'group']].to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")
    
    return cluster_data

if __name__ == "__main__":
    cluster_analysis = analyze_ai_clusters()
    correct_cluster = create_correct_ai_cluster()







