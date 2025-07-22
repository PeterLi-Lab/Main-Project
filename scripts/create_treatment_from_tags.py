import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_treatment_from_tags():
    """Create treatment labels based on tag containing 'ai content' with smart control group selection"""
    print("=== Create Treatment Labels from Tags (Smart Control Group) ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Check data structure
    print(f"Data columns: {list(df.columns)}")
    print(f"Data shape: {df.shape}")
    
    # Check for tag columns
    tag_columns = [col for col in df.columns if 'tag' in col.lower()]
    print(f"\nTag columns found: {tag_columns}")
    
    if not tag_columns:
        print("❌ No tag columns found")
        return None
    
    # 1. Create initial treatment labels based on tag containing 'ai content'
    print("\n=== 1. Creating Initial Treatment Labels ===")
    
    tag_col = tag_columns[0]  # Use the first tag column
    print(f"Using tag column: {tag_col}")
    
    # Check if tag contains 'ai content' (case insensitive)
    df['treatment_ai_content'] = df[tag_col].str.contains('ai content', case=False, na=False).astype(int)
    
    # Display initial results
    treatment_count = (df['treatment_ai_content'] == 1).sum()
    control_count = (df['treatment_ai_content'] == 0).sum()
    total_count = len(df)
    
    print(f"Initial treatment samples (tag contains 'ai content'): {treatment_count:,} ({treatment_count/total_count:.1%})")
    print(f"Initial control samples (tag does not contain 'ai content'): {control_count:,} ({control_count/total_count:.1%})")
    
    # 2. Smart control group selection - find posts similar to AI content
    print("\n=== 2. Smart Control Group Selection ===")
    
    # Find AI-related keywords that might indicate similar content
    ai_related_keywords = [
        'ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning',
        'neural network', 'algorithm', 'automation', 'chatbot', 'gpt', 'llm',
        'data science', 'predictive', 'automated', 'intelligent', 'smart'
    ]
    
    # Create a similarity score for each post
    df['ai_similarity_score'] = 0
    
    for keyword in ai_related_keywords:
        # Check if tag contains AI-related keywords (but not 'ai content')
        keyword_matches = df[tag_col].str.contains(keyword, case=False, na=False)
        # Only count if it doesn't already have 'ai content' tag
        non_ai_content_matches = keyword_matches & (df['treatment_ai_content'] == 0)
        df.loc[non_ai_content_matches, 'ai_similarity_score'] += 1
    
    # Also check content columns for AI-related terms
    content_columns = [col for col in df.columns if 'content' in col.lower() or 'text' in col.lower() or 'title' in col.lower()]
    
    for content_col in content_columns:
        if content_col in df.columns:
            for keyword in ai_related_keywords:
                content_matches = df[content_col].str.contains(keyword, case=False, na=False)
                non_ai_content_matches = content_matches & (df['treatment_ai_content'] == 0)
                df.loc[non_ai_content_matches, 'ai_similarity_score'] += 0.5  # Lower weight for content matches
    
    # 3. Select control group from similar posts
    print("\n=== 3. Selecting Control Group ===")
    
    # Find posts with high AI similarity but no 'ai content' tag
    high_similarity_posts = df[(df['ai_similarity_score'] >= 2) & (df['treatment_ai_content'] == 0)]
    medium_similarity_posts = df[(df['ai_similarity_score'] >= 1) & (df['treatment_ai_content'] == 0)]
    
    print(f"Posts with high AI similarity (score >= 2): {len(high_similarity_posts):,}")
    print(f"Posts with medium AI similarity (score >= 1): {len(medium_similarity_posts):,}")
    
    # Create refined control group
    # Option 1: Use high similarity posts as control
    if len(high_similarity_posts) >= treatment_count * 0.5:  # At least 50% of treatment size
        control_group = high_similarity_posts
        print(f"Using high similarity posts as control group: {len(control_group):,}")
    # Option 2: Use medium similarity posts as control
    elif len(medium_similarity_posts) >= treatment_count * 0.5:
        control_group = medium_similarity_posts
        print(f"Using medium similarity posts as control group: {len(control_group):,}")
    else:
        # Option 3: Use all non-treatment posts as control
        control_group = df[df['treatment_ai_content'] == 0]
        print(f"Using all non-treatment posts as control group: {len(control_group):,}")
    
    # Create refined dataset with balanced treatment and control
    treatment_group = df[df['treatment_ai_content'] == 1]
    
    # Balance the groups if needed
    if len(control_group) > len(treatment_group) * 2:
        # Sample control group to be roughly 2x treatment size
        control_sample_size = min(len(control_group), len(treatment_group) * 2)
        control_group = control_group.sample(n=control_sample_size, random_state=42)
        print(f"Sampled control group to: {len(control_group):,}")
    
    # Combine treatment and control groups
    refined_df = pd.concat([treatment_group, control_group], ignore_index=True)
    
    print(f"\nFinal dataset:")
    print(f"  Treatment samples: {len(treatment_group):,}")
    print(f"  Control samples: {len(control_group):,}")
    print(f"  Total samples: {len(refined_df):,}")
    
    # 4. Analyze the refined groups
    print("\n=== 4. Refined Group Analysis ===")
    
    # Check treatment balance
    treatment_ratio = len(treatment_group) / len(refined_df)
    control_ratio = len(control_group) / len(refined_df)
    
    print(f"Treatment ratio: {treatment_ratio:.1%}")
    print(f"Control ratio: {control_ratio:.1%}")
    
    if 0.3 <= treatment_ratio <= 0.7:
        print("✅ Treatment groups are well balanced")
    else:
        print("⚠️  Treatment groups are imbalanced")
    
    # 5. Check response distribution by treatment
    if 'response' in refined_df.columns:
        print("\n=== 5. Response Distribution by Treatment ===")
        
        treatment_response_rate = refined_df[refined_df['treatment_ai_content'] == 1]['response'].mean()
        control_response_rate = refined_df[refined_df['treatment_ai_content'] == 0]['response'].mean()
        uplift = treatment_response_rate - control_response_rate
        
        print(f"Treatment response rate: {treatment_response_rate:.2%}")
        print(f"Control response rate: {control_response_rate:.2%}")
        print(f"Uplift: {uplift:.2%}")
        
        if uplift > 0:
            print("✅ Positive uplift detected")
        elif uplift < 0:
            print("⚠️  Negative uplift detected")
        else:
            print("➖ No uplift detected")
    
    # 6. Save refined data
    print("\n=== 6. Saving Refined Data ===")
    
    output_file = 'uplift_model_data_refined.csv'
    refined_df.to_csv(output_file, index=False)
    print(f"Refined data saved to: {output_file}")
    
    return refined_df

def analyze_tag_patterns(df, tag_col):
    """Analyze patterns in tags"""
    print(f"\n=== Tag Pattern Analysis for {tag_col} ===")
    
    # Check for common patterns
    all_tags = df[tag_col].dropna().astype(str)
    
    # Count tag frequencies
    tag_counts = all_tags.value_counts()
    print(f"Total unique tags: {len(tag_counts)}")
    
    # Show most common tags
    print("\nMost common tags:")
    for tag, count in tag_counts.head(10).items():
        print(f"  {tag}: {count:,}")
    
    # Check for AI-related patterns
    ai_patterns = ['ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning']
    
    print("\nAI-related tag patterns:")
    for pattern in ai_patterns:
        pattern_count = all_tags.str.contains(pattern, case=False, na=False).sum()
        print(f"  '{pattern}': {pattern_count:,} occurrences")
    
    return tag_counts

if __name__ == "__main__":
    # Create treatment labels from tags with smart control group selection
    df = create_treatment_from_tags()
    
    if df is not None:
        # Analyze tag patterns
        tag_columns = [col for col in df.columns if 'tag' in col.lower()]
        if tag_columns:
            analyze_tag_patterns(df, tag_columns[0]) 