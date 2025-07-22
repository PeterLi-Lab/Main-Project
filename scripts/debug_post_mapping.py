import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def debug_post_mapping():
    """Debug post mapping and relationships"""
    print("=== Post Mapping Debug ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Check data structure
    print(f"Data columns: {list(df.columns)}")
    print(f"Data shape: {df.shape}")
    
    # Check for post_id column
    if 'post_id' not in df.columns:
        print("❌ No post_id column found")
        return None
    
    print("✅ post_id column found")
    
    # 1. Analyze post distribution
    print("\n=== 1. Post Distribution Analysis ===")
    
    unique_posts = df['post_id'].nunique()
    total_interactions = len(df)
    
    print(f"Unique posts: {unique_posts:,}")
    print(f"Total interactions: {total_interactions:,}")
    print(f"Average interactions per post: {total_interactions/unique_posts:.1f}")
    
    # Post interaction distribution
    post_interactions = df['post_id'].value_counts()
    print(f"\nPost interaction statistics:")
    print(f"  Min interactions: {post_interactions.min()}")
    print(f"  Max interactions: {post_interactions.max()}")
    print(f"  Mean interactions: {post_interactions.mean():.1f}")
    print(f"  Median interactions: {post_interactions.median():.1f}")
    
    # Check for posts with very few interactions
    low_interaction_posts = (post_interactions < 10).sum()
    print(f"Posts with < 10 interactions: {low_interaction_posts} ({low_interaction_posts/unique_posts:.1%})")
    
    # Check for posts with very many interactions
    high_interaction_posts = (post_interactions > 100).sum()
    print(f"Posts with > 100 interactions: {high_interaction_posts} ({high_interaction_posts/unique_posts:.1%})")
    
    # 2. Check post-treatment relationship
    print("\n=== 2. Post-Treatment Relationship ===")
    
    if 'treatment_ai_content' in df.columns:
        # Check if posts have consistent treatment assignment
        post_treatment = df.groupby('post_id')['treatment_ai_content'].agg(['mean', 'std'])
        
        # Posts with consistent treatment
        consistent_treatment_posts = (post_treatment['std'] == 0).sum()
        total_posts = len(post_treatment)
        
        print(f"Posts with consistent treatment: {consistent_treatment_posts}/{total_posts} ({consistent_treatment_posts/total_posts:.1%})")
        
        if consistent_treatment_posts / total_posts > 0.8:
            print("⚠️  Most posts have consistent treatment assignment - potential post-based leakage")
        else:
            print("✅ Treatment assignment varies by post (good)")
        
        # Check treatment distribution by post
        print(f"\nTreatment distribution by post:")
        treatment_by_post = post_treatment['mean'].value_counts().sort_index()
        for treatment_rate, count in treatment_by_post.items():
            print(f"  {treatment_rate:.1%} treatment rate: {count} posts")
    
    # 3. Check post-response relationship
    print("\n=== 3. Post-Response Relationship ===")
    
    if 'response' in df.columns:
        # Check response rates by post
        post_response = df.groupby('post_id')['response'].agg(['mean', 'count'])
        
        print(f"Post response rate statistics:")
        print(f"  Min response rate: {post_response['mean'].min():.2%}")
        print(f"  Max response rate: {post_response['mean'].max():.2%}")
        print(f"  Mean response rate: {post_response['mean'].mean():.2%}")
        print(f"  Median response rate: {post_response['mean'].median():.2%}")
        
        # Check for posts with extreme response rates
        zero_response_posts = (post_response['mean'] == 0).sum()
        full_response_posts = (post_response['mean'] == 1).sum()
        
        print(f"\nPosts with 0% response rate: {zero_response_posts} ({zero_response_posts/unique_posts:.1%})")
        print(f"Posts with 100% response rate: {full_response_posts} ({full_response_posts/unique_posts:.1%})")
    
    # 4. Check post-user relationship
    print("\n=== 4. Post-User Relationship ===")
    
    if 'user_id' in df.columns:
        # Check unique users per post
        users_per_post = df.groupby('post_id')['user_id'].nunique()
        
        print(f"User interaction statistics per post:")
        print(f"  Min unique users: {users_per_post.min()}")
        print(f"  Max unique users: {users_per_post.max()}")
        print(f"  Mean unique users: {users_per_post.mean():.1f}")
        print(f"  Median unique users: {users_per_post.median():.1f}")
        
        # Check for posts with very few users
        low_user_posts = (users_per_post < 5).sum()
        print(f"Posts with < 5 unique users: {low_user_posts} ({low_user_posts/unique_posts:.1%})")
        
        # Check for posts with many users
        high_user_posts = (users_per_post > 50).sum()
        print(f"Posts with > 50 unique users: {high_user_posts} ({high_user_posts/unique_posts:.1%})")
    
    # 5. Check for data quality issues
    print("\n=== 5. Data Quality Issues ===")
    
    issues = []
    
    # Check for missing post_ids
    missing_post_ids = df['post_id'].isnull().sum()
    if missing_post_ids > 0:
        issues.append(f"Found {missing_post_ids} missing post_ids")
    
    # Check for duplicate post-user interactions
    if 'user_id' in df.columns:
        duplicate_interactions = df.groupby(['post_id', 'user_id']).size()
        duplicate_count = (duplicate_interactions > 1).sum()
        if duplicate_count > 0:
            issues.append(f"Found {duplicate_count} post-user pairs with multiple interactions")
    
    # Check for posts with no interactions
    posts_with_interactions = df['post_id'].nunique()
    if posts_with_interactions == 0:
        issues.append("No posts found in data")
    
    if issues:
        print("⚠️  Found the following issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ No obvious data quality issues found")
    
    # 6. Summary and recommendations
    print("\n=== 6. Summary and Recommendations ===")
    
    recommendations = []
    
    if low_interaction_posts / unique_posts > 0.5:
        recommendations.append("Consider filtering out posts with very few interactions")
    
    if high_interaction_posts / unique_posts > 0.1:
        recommendations.append("Consider sampling from posts with very many interactions")
    
    if 'treatment_ai_content' in df.columns and consistent_treatment_posts / total_posts > 0.8:
        recommendations.append("Review treatment assignment strategy - most posts have consistent treatment")
    
    if zero_response_posts / unique_posts > 0.3:
        recommendations.append("Many posts have 0% response rate - consider data quality")
    
    if full_response_posts / unique_posts > 0.3:
        recommendations.append("Many posts have 100% response rate - consider data quality")
    
    if recommendations:
        print("Recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
    else:
        print("✅ No specific recommendations needed")
    
    return {
        'unique_posts': unique_posts,
        'total_interactions': total_interactions,
        'avg_interactions_per_post': total_interactions/unique_posts,
        'low_interaction_posts': low_interaction_posts,
        'high_interaction_posts': high_interaction_posts,
        'issues': issues,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    results = debug_post_mapping() 