#!/usr/bin/env python3
"""
Debug script to check uplift treatment labeling issues
"""

import pandas as pd
import xml.etree.ElementTree as ET

def check_post_id_mapping():
    """Check if post_ids from user_post_click_samples.csv exist in Posts.xml"""
    print("=== Checking Post ID Mapping ===")
    
    # Load user_post_click_samples.csv
    df_samples = pd.read_csv('user_post_click_samples.csv')
    sample_post_ids = df_samples['post_id'].head(10).tolist()
    print(f"Sample post_ids from user_post_click_samples.csv: {sample_post_ids}")
    
    # Load Posts.xml
    tree = ET.parse('data/Posts.xml')
    root = tree.getroot()
    
    # Create mapping of post_id to tags
    post_tags_map = {}
    for row in root:
        post_id = row.get('Id')
        tags = row.get('Tags', '')
        post_tags_map[post_id] = tags
    
    print(f"Total posts in Posts.xml: {len(post_tags_map)}")
    
    # Check if sample post_ids exist in Posts.xml
    print("\n=== Checking Sample Post IDs ===")
    for post_id in sample_post_ids:
        post_id_str = str(post_id)
        if post_id_str in post_tags_map:
            tags = post_tags_map[post_id_str]
            print(f"✓ Post {post_id}: Found in Posts.xml, Tags: {tags}")
        else:
            print(f"✗ Post {post_id}: NOT found in Posts.xml")
    
    # Check for AI-related tags in Posts.xml
    print("\n=== Checking for AI-related Tags ===")
    ai_tags = ['machine-learning', 'deep-learning', 'neural-network', 'tensorflow', 'pytorch', 'scikit-learn', 'nlp', 'ai', 'artificial-intelligence']
    
    ai_posts = []
    for post_id, tags in post_tags_map.items():
        if tags:  # Only check posts with tags
            for ai_tag in ai_tags:
                if ai_tag in tags.lower():
                    ai_posts.append((post_id, tags))
                    break
    
    print(f"Found {len(ai_posts)} posts with AI-related tags")
    if ai_posts:
        print("Sample AI posts:")
        for post_id, tags in ai_posts[:5]:
            print(f"  Post {post_id}: {tags}")
    
    # Check overlap between user_post_click_samples and AI posts
    print("\n=== Checking Overlap ===")
    sample_post_ids_set = set(str(pid) for pid in df_samples['post_id'].unique())
    ai_post_ids_set = set(pid for pid, _ in ai_posts)
    
    overlap = sample_post_ids_set.intersection(ai_post_ids_set)
    print(f"Posts in user_post_click_samples: {len(sample_post_ids_set)}")
    print(f"Posts with AI tags: {len(ai_post_ids_set)}")
    print(f"Overlap: {len(overlap)}")
    
    if overlap:
        print("Sample overlapping posts:")
        for post_id in list(overlap)[:5]:
            tags = post_tags_map[post_id]
            print(f"  Post {post_id}: {tags}")
    else:
        print("❌ NO OVERLAP FOUND - This is the problem!")
        print("The user_post_click_samples.csv contains posts that don't have AI tags")

def debug_post_tags_mapping():
    """Debug the post_tags mapping in uplift_treatment_labeling.py"""
    print("\n=== Debugging Post Tags Mapping ===")
    
    # Load user_post_click_samples.csv
    df_samples = pd.read_csv('user_post_click_samples.csv')
    
    # Load Posts.xml and create post_tags mapping like in uplift_treatment_labeling.py
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
    
    # Check mapping for sample post_ids
    sample_post_ids = df_samples['post_id'].head(10).tolist()
    print("Checking post_tags mapping for sample post_ids:")
    for post_id in sample_post_ids:
        post_id_str = str(post_id)
        if post_id_str in post_tags:
            tags = post_tags[post_id_str]
            print(f"  Post {post_id}: {tags}")
        else:
            print(f"  Post {post_id}: NOT FOUND in post_tags")
    
    # Check if post_id types match
    print(f"\nPost ID type check:")
    print(f"  Sample post_id type: {type(sample_post_ids[0])}")
    print(f"  post_tags keys type: {type(list(post_tags.keys())[0])}")
    
    # Check for AI tags in the mapped posts
    ai_tags = ['machine-learning', 'deep-learning', 'neural-network', 'tensorflow', 'pytorch', 'scikit-learn', 'nlp', 'ai', 'artificial-intelligence']
    
    ai_matched = 0
    for post_id in sample_post_ids:
        post_id_str = str(post_id)
        if post_id_str in post_tags:
            tags = post_tags[post_id_str]
            for ai_tag in ai_tags:
                if ai_tag in [tag.lower() for tag in tags]:
                    ai_matched += 1
                    print(f"  ✓ Post {post_id} has AI tag '{ai_tag}': {tags}")
                    break
    
    print(f"\nAI tag matches in sample: {ai_matched}/{len(sample_post_ids)}")

if __name__ == "__main__":
    check_post_id_mapping()
    debug_post_tags_mapping() 