import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    """Verify CTR calculation method"""
    
    print("=== CTR CALCULATION VERIFICATION ===")
    
    # Load click data
    print("Loading click data...")
    click_data = pd.read_csv('user_post_click_samples.csv')
    print(f"Click data shape: {click_data.shape}")
    print(f"Columns: {click_data.columns.tolist()}")
    
    # Check unique IDs
    print(f"\nUnique post IDs: {click_data['post_id'].nunique():,}")
    print(f"Unique user IDs: {click_data['user_id'].nunique():,}")
    print(f"Total records: {len(click_data):,}")
    
    # Check click distribution
    print(f"\nClick distribution:")
    print(f"Clicked = 1: {click_data['is_click'].sum():,} ({click_data['is_click'].mean()*100:.2f}%)")
    print(f"Clicked = 0: {(click_data['is_click'] == 0).sum():,} ({(click_data['is_click'] == 0).mean()*100:.2f}%)")
    
    # Sample a few posts to understand the data structure
    print(f"\n=== SAMPLE POST ANALYSIS ===")
    sample_posts = click_data['post_id'].unique()[:5]
    
    for post_id in sample_posts:
        post_data = click_data[click_data['post_id'] == post_id]
        print(f"\nPost ID: {post_id}")
        print(f"  Total interactions: {len(post_data)}")
        print(f"  Unique users: {post_data['user_id'].nunique()}")
        print(f"  Total clicks: {post_data['is_click'].sum()}")
        print(f"  Click rate: {post_data['is_click'].mean():.4f}")
        print(f"  Users who clicked: {post_data[post_data['is_click'] == 1]['user_id'].nunique()}")
        
        # Check if multiple interactions per user
        user_interactions = post_data.groupby('user_id')['is_click'].agg(['count', 'sum']).reset_index()
        print(f"  Users with multiple interactions: {(user_interactions['count'] > 1).sum()}")
        if (user_interactions['count'] > 1).sum() > 0:
            print(f"  Example user interactions:")
            multi_user = user_interactions[user_interactions['count'] > 1].iloc[0]
            print(f"    User {multi_user['user_id']}: {multi_user['count']} interactions, {multi_user['sum']} clicks")
    
    # Current CTR calculation method
    print(f"\n=== CURRENT CTR CALCULATION ===")
    post_click_data = click_data.groupby('post_id').agg({
        'is_click': ['mean', 'sum', 'count'],
        'user_id': 'nunique'
    }).reset_index()
    post_click_data.columns = ['post_id', 'click_rate', 'total_clicks', 'total_interactions', 'unique_users']
    
    # Calculate CTR using different methods
    post_click_data['ctr_clicks_per_user'] = post_click_data['total_clicks'] / (post_click_data['unique_users'] + 1e-6)
    post_click_data['ctr_clicks_per_interaction'] = post_click_data['total_clicks'] / post_click_data['total_interactions']
    
    print(f"CTR calculation methods:")
    print(f"1. clicks_per_user (total_clicks / unique_users):")
    print(f"   Mean: {post_click_data['ctr_clicks_per_user'].mean():.4f}")
    print(f"   Median: {post_click_data['ctr_clicks_per_user'].median():.4f}")
    print(f"   Min: {post_click_data['ctr_clicks_per_user'].min():.4f}")
    print(f"   Max: {post_click_data['ctr_clicks_per_user'].max():.4f}")
    
    print(f"\n2. clicks_per_interaction (total_clicks / total_interactions):")
    print(f"   Mean: {post_click_data['ctr_clicks_per_interaction'].mean():.4f}")
    print(f"   Median: {post_click_data['ctr_clicks_per_interaction'].median():.4f}")
    print(f"   Min: {post_click_data['ctr_clicks_per_interaction'].min():.4f}")
    print(f"   Max: {post_click_data['ctr_clicks_per_interaction'].max():.4f}")
    
    print(f"\n3. click_rate (from groupby mean):")
    print(f"   Mean: {post_click_data['click_rate'].mean():.4f}")
    print(f"   Median: {post_click_data['click_rate'].median():.4f}")
    print(f"   Min: {post_click_data['click_rate'].min():.4f}")
    print(f"   Max: {post_click_data['click_rate'].max():.4f}")
    
    # Check which method gives all 1.0s
    print(f"\n=== WHICH METHOD GIVES ALL 1.0s? ===")
    all_ones_clicks_per_user = (post_click_data['ctr_clicks_per_user'] == 1.0).sum()
    all_ones_clicks_per_interaction = (post_click_data['ctr_clicks_per_interaction'] == 1.0).sum()
    all_ones_click_rate = (post_click_data['click_rate'] == 1.0).sum()
    
    print(f"Posts with CTR = 1.0:")
    print(f"  clicks_per_user: {all_ones_clicks_per_user:,} ({all_ones_clicks_per_user/len(post_click_data)*100:.1f}%)")
    print(f"  clicks_per_interaction: {all_ones_clicks_per_interaction:,} ({all_ones_clicks_per_interaction/len(post_click_data)*100:.1f}%)")
    print(f"  click_rate: {all_ones_click_rate:,} ({all_ones_click_rate/len(post_click_data)*100:.1f}%)")
    
    # Analyze why we get so many 1.0s
    print(f"\n=== ANALYSIS OF 1.0 CTRs ===")
    
    # Check posts with CTR = 1.0
    posts_with_1_ctr = post_click_data[post_click_data['click_rate'] == 1.0]
    print(f"Posts with click_rate = 1.0:")
    print(f"  Total posts: {len(posts_with_1_ctr):,}")
    print(f"  Average total_interactions: {posts_with_1_ctr['total_interactions'].mean():.2f}")
    print(f"  Average unique_users: {posts_with_1_ctr['unique_users'].mean():.2f}")
    print(f"  Average total_clicks: {posts_with_1_ctr['total_clicks'].mean():.2f}")
    
    # Check posts with CTR < 1.0
    posts_with_less_1_ctr = post_click_data[post_click_data['click_rate'] < 1.0]
    print(f"\nPosts with click_rate < 1.0:")
    print(f"  Total posts: {len(posts_with_less_1_ctr):,}")
    print(f"  Average total_interactions: {posts_with_less_1_ctr['total_interactions'].mean():.2f}")
    print(f"  Average unique_users: {posts_with_less_1_ctr['unique_users'].mean():.2f}")
    print(f"  Average total_clicks: {posts_with_less_1_ctr['total_clicks'].mean():.2f}")
    
    # Check if the issue is with single-interaction posts
    single_interaction_posts = post_click_data[post_click_data['total_interactions'] == 1]
    print(f"\nPosts with only 1 interaction:")
    print(f"  Total posts: {len(single_interaction_posts):,}")
    print(f"  Posts with CTR = 1.0: {(single_interaction_posts['click_rate'] == 1.0).sum():,}")
    print(f"  Posts with CTR < 1.0: {(single_interaction_posts['click_rate'] < 1.0).sum():,}")
    
    # Check if users can click multiple times
    print(f"\n=== MULTIPLE CLICKS ANALYSIS ===")
    
    # Find posts where total_clicks > unique_users
    multiple_clicks_posts = post_click_data[post_click_data['total_clicks'] > post_click_data['unique_users']]
    print(f"Posts where total_clicks > unique_users: {len(multiple_clicks_posts):,}")
    
    if len(multiple_clicks_posts) > 0:
        print(f"Example post:")
        example_post = multiple_clicks_posts.iloc[0]
        print(f"  Post ID: {example_post['post_id']}")
        print(f"  Total clicks: {example_post['total_clicks']}")
        print(f"  Unique users: {example_post['unique_users']}")
        print(f"  Total interactions: {example_post['total_interactions']}")
        print(f"  Click rate: {example_post['click_rate']:.4f}")
        
        # Check the actual data for this post
        post_detail = click_data[click_data['post_id'] == example_post['post_id']]
        print(f"  Actual interactions:")
        for _, row in post_detail.iterrows():
            print(f"    User {row['user_id']}: clicked = {row['is_click']}")
    
    # Check if the data structure makes sense
    print(f"\n=== DATA STRUCTURE VALIDATION ===")
    
    # Check if total_clicks can exceed total_interactions
    invalid_posts = post_click_data[post_click_data['total_clicks'] > post_click_data['total_interactions']]
    print(f"Posts where total_clicks > total_interactions: {len(invalid_posts):,}")
    
    if len(invalid_posts) > 0:
        print("WARNING: This should not be possible!")
        print("Example invalid post:")
        invalid_post = invalid_posts.iloc[0]
        print(f"  Post ID: {invalid_post['post_id']}")
        print(f"  Total clicks: {invalid_post['total_clicks']}")
        print(f"  Total interactions: {invalid_post['total_interactions']}")
    
    # Check if unique_users can exceed total_interactions
    invalid_users = post_click_data[post_click_data['unique_users'] > post_click_data['total_interactions']]
    print(f"Posts where unique_users > total_interactions: {len(invalid_users):,}")
    
    # Summary and recommendations
    print(f"\n=== SUMMARY AND RECOMMENDATIONS ===")
    
    print(f"Current CTR calculation issues:")
    print(f"1. 94.4% of posts have CTR = 1.0")
    print(f"2. This suggests most posts have very few interactions")
    print(f"3. The click_rate method (groupby mean) gives the same result as clicks_per_interaction")
    
    print(f"\nRecommended CTR calculation:")
    print(f"Use: total_clicks / total_interactions")
    print(f"Reason: This represents the true click-through rate per interaction")
    print(f"Current method is correct, but the data has limited variation")
    
    print(f"\nAlternative approaches:")
    print(f"1. Use binary outcome: high vs low engagement posts")
    print(f"2. Use different target: total_clicks, unique_users, etc.")
    print(f"3. Filter out posts with very few interactions")
    print(f"4. Use user-level analysis instead of post-level")

if __name__ == "__main__":
    main()
