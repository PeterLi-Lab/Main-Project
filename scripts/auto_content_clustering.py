import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import html
import re

# 中文说明：
# 该脚本会自动扫描data/目录下所有XML文件，检测所有内容相关字段（如title、body、content、tag、description等），
# 提取并预处理文本，合并为一段内容，进行TF-IDF编码和MiniBatchKMeans聚类（10个簇），输出每个簇的样本量和Top关键词。
# 增强异常捕获和逻辑检查，保证健壮性和持续运行。

def extract_text_fields_from_xml(xml_file, content_keywords):
    """Extract content-related text fields from an XML file."""
    print(f"Parsing: {xml_file}")
    records = []
    try:
        for event, elem in ET.iterparse(xml_file, events=("end",)):
            if elem.tag == "row":
                record = {}
                for k, v in elem.attrib.items():
                    for kw in content_keywords:
                        if kw in k.lower():
                            record[k] = v
                if record:
                    records.append(record)
                elem.clear()
    except Exception as e:
        print(f"[ERROR] Failed to parse {xml_file}: {e}")
    return records

def preprocess_record(record):
    """Preprocess a single record: HTML decode, tag clean, merge text."""
    text_parts = []
    for k, v in record.items():
        key = k.lower()
        try:
            if 'body' in key:
                # HTML decode
                text_parts.append(html.unescape(v))
            elif 'tag' in key:
                # Remove < and >, split tags
                tags = v.replace('<', ' ').replace('>', ' ').replace('|', ' ')
                text_parts.append(tags)
            else:
                text_parts.append(str(v))
        except Exception as e:
            print(f"[WARN] Error processing field {k}: {e}")
    return ' '.join(text_parts)

def is_meaningless_keyword(word):
    """Check if a keyword is likely meaningless (all digits, all punctuation, or very short)."""
    if len(word) < 2:
        return True
    if re.fullmatch(r'\d+', word):
        return True
    if re.fullmatch(r'\W+', word):
        return True
    return False

def main():
    print("=== Auto Content Clustering from Raw XML Data ===\n")
    data_dir = 'data'
    content_keywords = ['content', 'title', 'body', 'tag', 'description', 'text']
    all_records = []
    # Scan all XML files
    xml_files = glob.glob(os.path.join(data_dir, '*.xml'))
    print(f"Found XML files: {xml_files}")
    for xml_file in xml_files:
        try:
            records = extract_text_fields_from_xml(xml_file, content_keywords)
            all_records.extend(records)
        except Exception as e:
            print(f"[ERROR] Exception while extracting from {xml_file}: {e}")
    print(f"Total records with content fields: {len(all_records):,}")
    if not all_records:
        print("No content records found. Exiting.")
        return
    # Preprocess and merge text
    merged_texts = []
    empty_count = 0
    for rec in all_records:
        try:
            text = preprocess_record(rec)
            if not text.strip():
                empty_count += 1
            merged_texts.append(text)
        except Exception as e:
            print(f"[WARN] Error preprocessing record: {e}")
            merged_texts.append("")
    print(f"Empty content records: {empty_count}")
    if empty_count > 0.5 * len(merged_texts):
        print("[WARN] More than half of the records have empty content! Check your data and field selection.")
    # TF-IDF vectorization
    try:
        print("Creating TF-IDF features for clustering...")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2), min_df=2, max_df=0.95)
        tfidf_matrix = vectorizer.fit_transform(merged_texts)
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    except Exception as e:
        print(f"[ERROR] TF-IDF vectorization failed: {e}")
        return
    if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
        print("[ERROR] TF-IDF matrix is empty. Exiting.")
        return
    # MiniBatchKMeans clustering
    n_clusters = 10
    try:
        print(f"Performing MiniBatchKMeans clustering with {n_clusters} clusters...")
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000, max_iter=100)
        clusters = clusterer.fit_predict(tfidf_matrix)
        print(f"Clustering completed. Found {len(np.unique(clusters))} clusters\n")
    except Exception as e:
        print(f"[ERROR] Clustering failed: {e}")
        return
    # Analyze cluster distribution
    print("=== Cluster Distribution Analysis ===")
    cluster_sizes = [np.sum(clusters == cid) for cid in np.unique(clusters)]
    if max(cluster_sizes) > 10 * np.median(cluster_sizes):
        print("[WARN] Cluster size imbalance detected: some clusters are much larger than others.")
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        cluster_size = np.sum(cluster_mask)
        print(f"Cluster {cluster_id}: size = {cluster_size:,}")
        # Show top keywords in this cluster
        if tfidf_matrix.shape[1] > 0:
            cluster_tfidf = tfidf_matrix[cluster_mask].mean(axis=0)
            top_indices = np.asarray(cluster_tfidf).ravel().argsort()[-10:][::-1]
            top_keywords = [vectorizer.get_feature_names_out()[i] for i in top_indices]
            n_meaningless = sum(is_meaningless_keyword(w) for w in top_keywords)
            if n_meaningless > 7:
                print("  [WARN] Most top keywords are likely meaningless. Check your content extraction and preprocessing.")
            print(f"  Top keywords: {top_keywords}")
    print("\nClustering process complete. No treatment/control selection performed.")

if __name__ == "__main__":
    while True:
        try:
            main()
            break
        except Exception as e:
            print(f"[FATAL] Unhandled exception: {e}. Retrying...") 