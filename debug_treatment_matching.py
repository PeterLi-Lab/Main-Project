#!/usr/bin/env python3
"""
Debug script to check treatment matching logic step by step
"""

import pandas as pd
import xml.etree.ElementTree as ET

def debug_treatment_matching():
    """Debug the treatment matching logic in detail"""
    print("=== Debugging Treatment Matching Logic ===")
    
    # Step 1: Load user_post_click_samples.csv
    print("\n1. Loading user_post_click_samples.csv...")
    df_samples = pd.read_csv('user_post_click_samples.csv')
    print(f"   Loaded {len(df_samples)} samples")
    print(f"   Sample post_ids: {df_samples['post_id'].head(5).tolist()}")
    
    # Step 2: Load Posts.xml and create post_tags mapping
    print("\n2. Loading Posts.xml and creating post_tags mapping...")
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
    
    print(f"   Created post_tags mapping for {len(post_tags)} posts")
    
    # Step 3: Check mapping for sample post_ids
    print("\n3. Checking post_tags mapping for sample post_ids...")
    sample_post_ids = df_samples['post_id'].head(10).tolist()
    for post_id in sample_post_ids:
        post_id_str = str(post_id)
        if post_id_str in post_tags:
            tags = post_tags[post_id_str]
            print(f"   Post {post_id}: {tags}")
        else:
            print(f"   Post {post_id}: NOT FOUND in post_tags")
    
    # Step 4: Define AI tags and check matching
    print("\n4. Checking AI tag matching...")
    ai_tags = ['machine-learning', 'deep-learning', 'neural-network', 'tensorflow', 'pytorch', 'scikit-learn', 'nlp', 'ai', 'artificial-intelligence']
    print(f"   AI tags to match: {ai_tags}")
    
    # Step 5: Test treatment matching logic
    print("\n5. Testing treatment matching logic...")
    def check_treatment_match(tags):
        if not tags or tags == []:
            return False
        # Convert tags to lowercase for matching
        tags_lower = [tag.lower() for tag in tags]
        config_tags_lower = [tag.lower() for tag in ai_tags]
        return any(tag in config_tags_lower for tag in tags_lower)
    
    # Test on sample post_ids
    ai_matched_count = 0
    for post_id in sample_post_ids:
        post_id_str = str(post_id)
        if post_id_str in post_tags:
            tags = post_tags[post_id_str]
            is_ai = check_treatment_match(tags)
            if is_ai:
                ai_matched_count += 1
                print(f"   ✓ Post {post_id} has AI tags: {tags}")
            else:
                print(f"   ✗ Post {post_id} no AI tags: {tags}")
        else:
            print(f"   ✗ Post {post_id} not found in post_tags")
    
    print(f"\n   AI matched posts in sample: {ai_matched_count}/{len(sample_post_ids)}")
    
    # Step 6: Check overall AI tag distribution
    print("\n6. Checking overall AI tag distribution...")
    total_ai_posts = 0
    for post_id, tags in post_tags.items():
        if check_treatment_match(tags):
            total_ai_posts += 1
    
    print(f"   Total posts with AI tags: {total_ai_posts}")
    
    # Step 7: Check overlap between user_post_click_samples and AI posts
    print("\n7. Checking overlap between samples and AI posts...")
    sample_post_ids_set = set(str(pid) for pid in df_samples['post_id'].unique())
    ai_post_ids_set = set(pid for pid, tags in post_tags.items() if check_treatment_match(tags))
    
    overlap = sample_post_ids_set.intersection(ai_post_ids_set)
    print(f"   Posts in user_post_click_samples: {len(sample_post_ids_set)}")
    print(f"   Posts with AI tags: {len(ai_post_ids_set)}")
    print(f"   Overlap: {len(overlap)}")
    
    if overlap:
        print("   Sample overlapping posts:")
        for post_id in list(overlap)[:5]:
            tags = post_tags[post_id]
            print(f"     Post {post_id}: {tags}")
    else:
        print("   ❌ NO OVERLAP FOUND!")

if __name__ == "__main__":
    debug_treatment_matching() 