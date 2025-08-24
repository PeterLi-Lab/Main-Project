import pandas as pd
import numpy as np

# Load optimized clusters
df = pd.read_csv('optimized_post_clusters.csv')
print(f"Total posts: {len(df):,}")

# Show cluster distribution
print("\nCluster distribution:")
cluster_counts = df['cluster_id'].value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    percentage = count / len(df) * 100
    print(f"Cluster {cluster_id}: {count:,} posts ({percentage:.1f}%)")

# AI keywords for content identification
ai_keywords = [
    'artificial intelligence', 'ai', 'machine learning', 'ml', 'deep learning',
    'neural network', 'neural networks', 'neural', 'tensorflow', 'pytorch',
    'keras', 'scikit-learn', 'sklearn', 'classification', 'regression',
    'clustering', 'supervised', 'unsupervised', 'reinforcement learning',
    'natural language processing', 'nlp', 'computer vision', 'cv'
]

print("\n=== AI CONTENT DENSITY ANALYSIS ===")

cluster_ai_density = []

for cluster_id in sorted(df['cluster_id'].unique()):
    cluster_data = df[df['cluster_id'] == cluster_id]
    cluster_size = len(cluster_data)
    
    # Count AI keyword occurrences
    ai_count = 0
    for _, row in cluster_data.iterrows():
        content = f"{row['Title']} {row['Body']} {row['Tags']}".lower()
        for keyword in ai_keywords:
            if keyword in content:
                ai_count += 1
                break  # Count each post only once
    
    ai_density = ai_count / cluster_size if cluster_size > 0 else 0
    
    cluster_ai_density.append({
        'cluster_id': cluster_id,
        'size': cluster_size,
        'ai_count': ai_count,
        'ai_density': ai_density,
        'percentage': cluster_size / len(df) * 100
    })
    
    print(f"Cluster {cluster_id}:")
    print(f"  Size: {cluster_size:,} ({cluster_size/len(df)*100:.1f}%)")
    print(f"  AI posts: {ai_count:,}")
    print(f"  AI density: {ai_density:.3f}")
    print()

# Sort by AI density
cluster_ai_density.sort(key=lambda x: x['ai_density'], reverse=True)

print("=== CLUSTERS RANKED BY AI DENSITY ===")
for i, cluster in enumerate(cluster_ai_density):
    print(f"{i+1}. Cluster {cluster['cluster_id']}: AI density = {cluster['ai_density']:.3f}")
    print(f"   Size: {cluster['size']:,} posts, AI posts: {cluster['ai_count']:,}")

# Recommend best cluster
best_cluster = cluster_ai_density[0]
print(f"\n=== RECOMMENDATION ===")
print(f"Best cluster for AI content analysis: Cluster {best_cluster['cluster_id']}")
print(f"AI density: {best_cluster['ai_density']:.3f}")
print(f"Total posts: {best_cluster['size']:,}")
print(f"AI posts: {best_cluster['ai_count']:,}")

# Create treatment/control for best cluster
best_cluster_id = best_cluster['cluster_id']
cluster_data = df[df['cluster_id'] == best_cluster_id].copy()

# Create treatment labels
def is_ai_content(content):
    content_lower = content.lower()
    return any(keyword in content_lower for keyword in ai_keywords)

cluster_data['is_ai_content'] = cluster_data.apply(
    lambda row: is_ai_content(f"{row['Title']} {row['Body']} {row['Tags']}"), 
    axis=1
)

# Create treatment/control groups
cluster_data['treatment'] = cluster_data['is_ai_content'].map({True: 'treatment', False: 'control'})

# Analysis
treatment_count = (cluster_data['treatment'] == 'treatment').sum()
control_count = (cluster_data['treatment'] == 'control').sum()

print(f"\n=== TREATMENT/CONTROL SPLIT FOR CLUSTER {best_cluster_id} ===")
print(f"Treatment posts: {treatment_count:,} ({treatment_count/len(cluster_data)*100:.1f}%)")
print(f"Control posts: {control_count:,} ({control_count/len(cluster_data)*100:.1f}%)")

# Export
output_file = f'cluster{best_cluster_id}_treatment_control.csv'
cluster_data.to_csv(output_file, index=False)
print(f"Treatment/control data exported to {output_file}")







