#!/usr/bin/env python3
"""
Debug script to check post_id to tags mapping issue
"""

import pandas as pd
import xml.etree.ElementTree as ET

def debug_post_mapping():
    """Debug the post_id to tags mapping issue"""
    print("=== Debugging Post ID to Tags Mapping ===")
    
    # Load user_post_click_samples.csv
    df_samples = pd.read_csv('user_post_click_samples.csv')
    print(f"Loaded {len(df_samples)} samples")
    
    # Load Posts.xml and create post_tags mapping
    tree = ET.parse('data/Posts.xml')
    root = tree.getroot()
    
    post_tags = {}
    for row in root:
        post_id = row.get('Id')
        tags_str = row.get('Tags', '')
        # Parse tags like in uplift_treatment_labeling.py
        if not tags_str or tags_str == '':
            parsed_tags = []
        else:
            cleaned_tags = tags_str.strip('|')
            if not cleaned_tags:
                parsed_tags = []
            else:
                parsed_tags = [tag for tag in cleaned_tags.split('|') if tag]
        post_tags[post_id] = parsed_tags
    
    print(f"Created post_tags mapping for {len(post_tags)} posts")
    
    # Check the mapping
    print("\nChecking post_id to tags mapping:")
    sample_post_ids = df_samples['post_id'].head(10).tolist()
    
    for post_id in sample_post_ids:
        post_id_str = str(post_id)
        if post_id_str in post_tags:
            tags = post_tags[post_id_str]
            print(f"  Post {post_id} (str): {tags}")
        else:
            print(f"  Post {post_id} (str): NOT FOUND")
    
    # Test the map function
    print("\nTesting map function:")
    df_samples['post_tags'] = df_samples['post_id'].map(post_tags)
    df_samples['post_tags'] = df_samples['post_tags'].fillna('[]')
    
    sample_tags = df_samples['post_tags'].head(5).tolist()
    for i, tags in enumerate(sample_tags):
        print(f"  Sample {i+1}: {tags}")
    
    # Check if there are any non-empty tags
    non_empty_tags = df_samples[df_samples['post_tags'].apply(lambda x: len(x) > 0)]
    print(f"\nSamples with non-empty tags: {len(non_empty_tags)}")
    
    if len(non_empty_tags) > 0:
        print("Sample non-empty tags:")
        for _, row in non_empty_tags.head(3).iterrows():
            print(f"  Post {row['post_id']}: {row['post_tags']}")

if __name__ == "__main__":
    debug_post_mapping() 