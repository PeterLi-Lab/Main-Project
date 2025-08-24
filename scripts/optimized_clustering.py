import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import html
import re

DATA_DIR = 'data'
POSTS_FILE = os.path.join(DATA_DIR, 'Posts.xml')
TAGS_FILE = os.path.join(DATA_DIR, 'Tags.xml')
COMMENTS_FILE = os.path.join(DATA_DIR, 'Comments.xml')

TOP_KEYWORDS = 10
TOP_EXAMPLES = 3

def parse_posts(posts_file):
    print(f"Parsing {posts_file} ...")
    posts = []
    for event, elem in ET.iterparse(posts_file, events=("end",)):
        if elem.tag == "row":
            post = {
                'Id': elem.attrib.get('Id'),
                'Title': elem.attrib.get('Title', ''),
                'Body': elem.attrib.get('Body', ''),
                'Tags': elem.attrib.get('Tags', ''),
                'PostTypeId': elem.attrib.get('PostTypeId', ''),
            }
            posts.append(post)
        elem.clear()
    print(f"Parsed {len(posts):,} posts.")
    return pd.DataFrame(posts)

def parse_tags(tags_file):
    print(f"Parsing {tags_file} ...")
    tags = {}
    for event, elem in ET.iterparse(tags_file, events=("end",)):
        if elem.tag == "row":
            tag_name = elem.attrib.get('TagName')
            tag_id = elem.attrib.get('Id')
            if tag_name and tag_id:
                tags[tag_name] = tag_id
        elem.clear()
    print(f"Parsed {len(tags):,} tags.")
    return tags

def parse_comments(comments_file):
    print(f"Parsing {comments_file} ...")
    post_comments = {}
    for event, elem in ET.iterparse(comments_file, events=("end",)):
        if elem.tag == "row":
            post_id = elem.attrib.get('PostId')
            text = elem.attrib.get('Text', '')
            if post_id:
                if post_id not in post_comments:
                    post_comments[post_id] = []
                post_comments[post_id].append(text)
        elem.clear()
    print(f"Parsed comments for {len(post_comments):,} posts.")
    return post_comments

def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def merge_post_content(row, post_comments):
    parts = [row['Title'], row['Body'], row['Tags']]
    if row['Id'] in post_comments:
        parts.append(' '.join(post_comments[row['Id']]))
    merged = ' '.join([clean_text(p) for p in parts if p])
    return merged

def optimize_k(tfidf_matrix, k_range=range(2, 16)):
    """Find optimal k using Elbow Method and Silhouette Score"""
    print(f"Optimizing k from {k_range.start} to {k_range.stop-1}...")
    
    # Calculate within-cluster sum of squares for Elbow Method
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        print(f"Testing k={k}...")
        
        # Use MiniBatchKMeans for efficiency
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000, max_iter=100)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Calculate inertia (within-cluster sum of squares)
        inertias.append(kmeans.inertia_)
        
        # Calculate silhouette score (skip k=1)
        if k > 1:
            silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(0)
    
    # Find elbow point
    # Calculate the rate of change of inertia
    inertia_changes = np.diff(inertias)
    inertia_change_rates = np.diff(inertia_changes)
    elbow_k = k_range[np.argmax(inertia_change_rates) + 1]
    
    # Find best silhouette score
    best_silhouette_k = k_range[np.argmax(silhouette_scores)]
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Elbow plot
    ax1.plot(k_range, inertias, 'bo-')
    ax1.axvline(x=elbow_k, color='red', linestyle='--', label=f'Elbow at k={elbow_k}')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.legend()
    ax1.grid(True)
    
    # Silhouette plot
    ax2.plot(k_range, silhouette_scores, 'go-')
    ax2.axvline(x=best_silhouette_k, color='red', linestyle='--', label=f'Best Silhouette at k={best_silhouette_k}')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('k_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Elbow Method suggests k={elbow_k}")
    print(f"Silhouette Score suggests k={best_silhouette_k}")
    
    # Choose the optimal k (prefer silhouette score if reasonable)
    if abs(best_silhouette_k - elbow_k) <= 2:
        optimal_k = best_silhouette_k
        method = "Silhouette Score"
    else:
        optimal_k = elbow_k
        method = "Elbow Method"
    
    print(f"Selected optimal k={optimal_k} using {method}")
    
    return optimal_k, inertias, silhouette_scores

def main():
    # Parse data
    posts_df = parse_posts(POSTS_FILE)
    tags_dict = parse_tags(TAGS_FILE)
    post_comments = parse_comments(COMMENTS_FILE)

    # Merge content
    print("Merging post content ...")
    posts_df['merged_content'] = posts_df.apply(lambda row: merge_post_content(row, post_comments), axis=1)

    # Only keep posts with content
    posts_df = posts_df[posts_df['merged_content'].str.strip() != '']
    print(f"Posts with non-empty content: {len(posts_df):,}")

    # TF-IDF embedding
    print("Creating TF-IDF features ...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2), min_df=2, max_df=0.95)
    tfidf_matrix = vectorizer.fit_transform(posts_df['merged_content'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Optimize k
    optimal_k, inertias, silhouette_scores = optimize_k(tfidf_matrix)

    # Clustering with optimal k
    print(f"Clustering posts into {optimal_k} clusters ...")
    clusterer = MiniBatchKMeans(n_clusters=optimal_k, random_state=42, batch_size=1000, max_iter=100)
    clusters = clusterer.fit_predict(tfidf_matrix)
    posts_df['cluster_id'] = clusters
    print(f"Clustering completed. Found {len(np.unique(clusters))} clusters.")

    # Analyze cluster distribution
    print("\n=== OPTIMIZED CLUSTER DISTRIBUTION ANALYSIS ===")
    cluster_sizes = []
    
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        cluster_data = posts_df[cluster_mask]
        cluster_size = len(cluster_data)
        cluster_sizes.append(cluster_size)
        
        print(f"Cluster {cluster_id}: size = {cluster_size:,} ({cluster_size/len(posts_df)*100:.1f}%)")
        
        # Top keywords
        cluster_tfidf = tfidf_matrix[cluster_mask].mean(axis=0)
        top_indices = np.asarray(cluster_tfidf).ravel().argsort()[-TOP_KEYWORDS:][::-1]
        top_keywords = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        print(f"  Top keywords: {top_keywords}")
        
        # Representative posts
        print("  Representative posts:")
        for i, (_, row) in enumerate(cluster_data.head(TOP_EXAMPLES).iterrows()):
            print(f"    - Title: {row['Title'][:60]}")
            print(f"      Body: {row['Body'][:80]} ...")
        print()

    # Cluster balance analysis
    cluster_sizes = np.array(cluster_sizes)
    print(f"Cluster Balance Analysis:")
    print(f"  Min cluster size: {cluster_sizes.min():,}")
    print(f"  Max cluster size: {cluster_sizes.max():,}")
    print(f"  Median cluster size: {np.median(cluster_sizes):,.0f}")
    print(f"  Standard deviation: {cluster_sizes.std():,.0f}")
    print(f"  Balance ratio (max/min): {cluster_sizes.max()/cluster_sizes.min():.2f}")

    # Export cluster labels
    out_csv = 'optimized_post_clusters.csv'
    posts_df[['Id', 'Title', 'Body', 'Tags', 'cluster_id', 'merged_content']].to_csv(out_csv, index=False)
    print(f"\nOptimized cluster labels exported to {out_csv}")

if __name__ == "__main__":
    main()







