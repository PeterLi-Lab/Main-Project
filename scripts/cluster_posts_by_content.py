import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import html
import re

DATA_DIR = 'data'
POSTS_FILE = os.path.join(DATA_DIR, 'Posts.xml')
TAGS_FILE = os.path.join(DATA_DIR, 'Tags.xml')
COMMENTS_FILE = os.path.join(DATA_DIR, 'Comments.xml')

N_CLUSTERS = 10
TOP_KEYWORDS = 10
TOP_EXAMPLES = 3

# 1. 解析 Posts.xml

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

# 2. 解析 Tags.xml

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

# 3. 解析 Comments.xml

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

# 4. 文本清洗

def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip().lower()

# 5. 合并post内容

def merge_post_content(row, post_comments):
    parts = [row['Title'], row['Body'], row['Tags']]
    if row['Id'] in post_comments:
        parts.append(' '.join(post_comments[row['Id']]))
    merged = ' '.join([clean_text(p) for p in parts if p])
    return merged

# 6. 主流程

def main():
    # 解析数据
    posts_df = parse_posts(POSTS_FILE)
    tags_dict = parse_tags(TAGS_FILE)
    post_comments = parse_comments(COMMENTS_FILE)

    # 合并内容
    print("Merging post content ...")
    posts_df['merged_content'] = posts_df.apply(lambda row: merge_post_content(row, post_comments), axis=1)

    # 只保留有内容的post
    posts_df = posts_df[posts_df['merged_content'].str.strip() != '']
    print(f"Posts with non-empty content: {len(posts_df):,}")

    # TF-IDF embedding
    print("Creating TF-IDF features ...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2), min_df=2, max_df=0.95)
    tfidf_matrix = vectorizer.fit_transform(posts_df['merged_content'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # MiniBatchKMeans聚类
    print(f"Clustering posts into {N_CLUSTERS} clusters ...")
    clusterer = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, batch_size=1000, max_iter=100)
    clusters = clusterer.fit_predict(tfidf_matrix)
    posts_df['cluster_id'] = clusters
    print(f"Clustering completed. Found {len(np.unique(clusters))} clusters.")

    # 输出每个簇的样本数、top关键词、代表性post
    print("\n=== Cluster Distribution Analysis ===")
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        cluster_data = posts_df[cluster_mask]
        print(f"Cluster {cluster_id}: size = {len(cluster_data):,}")
        # Top关键词
        cluster_tfidf = tfidf_matrix[cluster_mask].mean(axis=0)
        top_indices = np.asarray(cluster_tfidf).ravel().argsort()[-TOP_KEYWORDS:][::-1]
        top_keywords = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        print(f"  Top keywords: {top_keywords}")
        # 代表性post
        print("  Representative posts:")
        for i, (_, row) in enumerate(cluster_data.head(TOP_EXAMPLES).iterrows()):
            print(f"    - Title: {row['Title'][:60]}")
            print(f"      Body: {row['Body'][:80]} ...")
        print()

    # 导出聚类标签到csv
    out_csv = 'post_clusters.csv'
    posts_df[['Id', 'Title', 'Body', 'Tags', 'cluster_id', 'merged_content']].to_csv(out_csv, index=False)
    print(f"\nCluster labels exported to {out_csv}")

if __name__ == "__main__":
    main() 