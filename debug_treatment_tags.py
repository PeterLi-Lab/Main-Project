#!/usr/bin/env python3
"""
Debug script to check treatment tag matching
"""

import pandas as pd
import xml.etree.ElementTree as ET

def parse_tags(tags_str):
    """Parse tags from pipe-separated format: '|tag1|tag2|' -> ['tag1', 'tag2']"""
    if not tags_str or tags_str == '':
        return []
    return [tag for tag in tags_str.strip('|').split('|') if tag]

def debug_tags():
    # Load post tags
    print("Loading post tags...")
    tree = ET.parse('data/Posts.xml')
    root = tree.getroot()
    
    # Test tag parsing
    print("Testing tag parsing...")
    test_tags = [
        '|machine-learning|',
        '|javascript|html|css|',
        '|android|ios|',
        '|sql|mysql|',
        ''
    ]
    
    for tags_str in test_tags:
        parsed = parse_tags(tags_str)
        print(f"'{tags_str}' -> {parsed}")
    
    # Check actual posts with tags
    posts_with_tags = 0
    total_posts = 0
    sample_tagged_posts = []
    
    for row in root:
        total_posts += 1
        raw_tags = row.get('Tags', '')
        if raw_tags and raw_tags.strip():
            posts_with_tags += 1
            if len(sample_tagged_posts) < 10:
                post_id = row.get('Id')
                tag_list = parse_tags(raw_tags)
                sample_tagged_posts.append((post_id, tag_list))
    
    print(f"\nPosts with tags: {posts_with_tags}/{total_posts}")
    print("Sample tagged posts:")
    for post_id, tags in sample_tagged_posts:
        print(f"Post {post_id}: {tags}")
    
    # Check for AI-related tags
    ai_tags = ['ai', 'artificial-intelligence', 'machine-learning', 'deep-learning', 
               'neural-network', 'tensorflow', 'pytorch', 'scikit-learn', 'nlp', 
               'computer-vision', 'data-science']
    
    ai_posts = []
    for row in root:
        raw_tags = row.get('Tags', '')
        if raw_tags:
            tag_list = parse_tags(raw_tags)
            if any(tag.lower() in ai_tags for tag in tag_list):
                post_id = row.get('Id')
                ai_posts.append((post_id, tag_list))
    
    print(f"\nPosts with AI tags: {len(ai_posts)}")
    if ai_posts:
        print("Sample AI posts:")
        for post_id, tags in ai_posts[:5]:
            print(f"Post {post_id}: {tags}")
    
    # Load user-post samples and check matching
    print("\n=== Checking User-Post Samples ===")
    df_samples = pd.read_csv('user_post_click_samples.csv')
    print(f"User-post samples shape: {df_samples.shape}")
    
    # Check sample post IDs
    sample_post_ids = df_samples['post_id'].head(10).tolist()
    print("Sample post IDs from dataset:")
    for post_id in sample_post_ids:
        print(f"Post {post_id}")
    
    # Check if these post IDs exist in XML
    xml_post_ids = set()
    for row in root:
        xml_post_ids.add(row.get('Id'))
    
    print(f"\nXML has {len(xml_post_ids)} unique post IDs")
    print(f"Sample dataset has {df_samples['post_id'].nunique()} unique post IDs")
    
    # Check overlap
    sample_post_ids_set = set(df_samples['post_id'].astype(str))
    overlap = sample_post_ids_set & xml_post_ids
    print(f"Overlap: {len(overlap)} post IDs")
    
    # Test treatment assignment on a few samples
    print("\n=== Testing Treatment Assignment ===")
    post_tags = {}
    for row in root:
        post_id = row.get('Id')
        raw_tags = row.get('Tags', '')
        post_tags[post_id] = parse_tags(raw_tags)
    
    # Test on first few samples
    for i, row in df_samples.head(5).iterrows():
        post_id = str(row['post_id'])
        tags = post_tags.get(post_id, [])
        has_ai = any(tag.lower() in ['machine-learning', 'ai'] for tag in tags)
        print(f"Post {post_id}: tags={tags}, has_ai={has_ai}")

if __name__ == "__main__":
    debug_tags() 