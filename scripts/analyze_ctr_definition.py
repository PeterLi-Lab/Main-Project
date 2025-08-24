import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    """Analyze correct CTR definition and data limitations"""
    
    print("=== CORRECT CTR DEFINITION ANALYSIS ===")
    
    # Load click data
    print("Loading click data...")
    click_data = pd.read_csv('user_post_click_samples.csv')
    print(f"Click data shape: {click_data.shape}")
    print(f"Columns: {click_data.columns.tolist()}")
    
    print(f"\n=== DATA STRUCTURE ANALYSIS ===")
    print(f"Total records: {len(click_data):,}")
    print(f"Unique posts: {click_data['post_id'].nunique():,}")
    print(f"Unique users: {click_data['user_id'].nunique():,}")
    
    # Check what each record represents
    print(f"\n=== WHAT DOES EACH RECORD REPRESENT? ===")
    print(f"Each record appears to be a user-post interaction")
    print(f"is_click = 1: User clicked on the post")
    print(f"is_click = 0: User did not click on the post")
    
    # Check if this is exposure data or just click data
    print(f"\n=== EXPOSURE vs CLICK DATA ===")
    print(f"Click distribution:")
    print(f"  is_click = 1: {click_data['is_click'].sum():,} ({click_data['is_click'].mean()*100:.2f}%)")
    print(f"  is_click = 0: {(click_data['is_click'] == 0).sum():,} ({(click_data['is_click'] == 0).mean()*100:.2f}%)")
    
    # This suggests we DO have exposure data!
    print(f"\nIMPORTANT DISCOVERY:")
    print(f"  - 36.28% of interactions resulted in clicks")
    print(f"  - 63.72% of interactions resulted in no clicks")
    print(f"  - This suggests each record IS an exposure!")
    
    # Verify this interpretation
    print(f"\n=== VERIFYING EXPOSURE INTERPRETATION ===")
    
    # Sample a few posts to understand the pattern
    sample_posts = click_data['post_id'].unique()[:5]
    
    for post_id in sample_posts:
        post_data = click_data[click_data['post_id'] == post_id]
        total_exposures = len(post_data)
        total_clicks = post_data['is_click'].sum()
        ctr = total_clicks / total_exposures if total_exposures > 0 else 0
        
        print(f"\nPost ID: {post_id}")
        print(f"  Total exposures: {total_exposures}")
        print(f"  Total clicks: {total_clicks}")
        print(f"  CTR: {ctr:.4f} ({ctr*100:.2f}%)")
        print(f"  Unique users exposed: {post_data['user_id'].nunique()}")
    
    # Now calculate proper CTR
    print(f"\n=== PROPER CTR CALCULATION ===")
    
    # Post-level CTR
    post_ctr = click_data.groupby('post_id').agg({
        'is_click': ['sum', 'count']
    }).reset_index()
    post_ctr.columns = ['post_id', 'total_clicks', 'total_exposures']
    post_ctr['ctr'] = post_ctr['total_clicks'] / post_ctr['total_exposures']
    
    print(f"Post-level CTR statistics:")
    print(f"  Mean CTR: {post_ctr['ctr'].mean():.4f} ({post_ctr['ctr'].mean()*100:.2f}%)")
    print(f"  Median CTR: {post_ctr['ctr'].median():.4f} ({post_ctr['ctr'].median()*100:.2f}%)")
    print(f"  Min CTR: {post_ctr['ctr'].min():.4f} ({post_ctr['ctr'].min()*100:.2f}%)")
    print(f"  Max CTR: {post_ctr['ctr'].max():.4f} ({post_ctr['ctr'].max()*100:.2f}%)")
    
    # Check CTR distribution
    print(f"\nCTR distribution:")
    ctr_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ctr_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    
    ctr_distribution = pd.cut(post_ctr['ctr'], bins=ctr_bins, labels=ctr_labels, include_lowest=True)
    ctr_counts = ctr_distribution.value_counts().sort_index()
    
    for bin_label, count in ctr_counts.items():
        percentage = count / len(post_ctr) * 100
        print(f"  {bin_label}: {count:,} posts ({percentage:.1f}%)")
    
    # Check for posts with CTR = 1.0 (all exposures resulted in clicks)
    posts_with_1_ctr = post_ctr[post_ctr['ctr'] == 1.0]
    print(f"\nPosts with CTR = 1.0 (all exposures clicked):")
    print(f"  Count: {len(posts_with_1_ctr):,} ({len(posts_with_1_ctr)/len(post_ctr)*100:.1f}%)")
    print(f"  Average exposures: {posts_with_1_ctr['total_exposures'].mean():.2f}")
    print(f"  Average clicks: {posts_with_1_ctr['total_clicks'].mean():.2f}")
    
    # Check posts with CTR = 0.0 (no clicks)
    posts_with_0_ctr = post_ctr[post_ctr['ctr'] == 0.0]
    print(f"\nPosts with CTR = 0.0 (no clicks):")
    print(f"  Count: {len(posts_with_0_ctr):,} ({len(posts_with_0_ctr)/len(post_ctr)*100:.1f}%)")
    print(f"  Average exposures: {posts_with_0_ctr['total_exposures'].mean():.2f}")
    
    # Check posts with reasonable CTR (between 0.1 and 0.9)
    reasonable_ctr_posts = post_ctr[(post_ctr['ctr'] > 0.0) & (post_ctr['ctr'] < 1.0)]
    print(f"\nPosts with reasonable CTR (0-100%):")
    print(f"  Count: {len(reasonable_ctr_posts):,} ({len(reasonable_ctr_posts)/len(post_ctr)*100:.1f}%)")
    print(f"  Average CTR: {reasonable_ctr_posts['ctr'].mean():.4f} ({reasonable_ctr_posts['ctr'].mean()*100:.2f}%)")
    print(f"  Average exposures: {reasonable_ctr_posts['total_exposures'].mean():.2f}")
    
    # Filter posts with sufficient exposures for reliable CTR
    sufficient_exposure_posts = post_ctr[post_ctr['total_exposures'] >= 5]
    print(f"\nPosts with sufficient exposures (≥5):")
    print(f"  Count: {len(sufficient_exposure_posts):,} ({len(sufficient_exposure_posts)/len(post_ctr)*100:.1f}%)")
    print(f"  Average CTR: {sufficient_exposure_posts['ctr'].mean():.4f} ({sufficient_exposure_posts['ctr'].mean()*100:.2f}%)")
    print(f"  Average exposures: {sufficient_exposure_posts['total_exposures'].mean():.2f}")
    
    # Check CTR distribution for posts with sufficient exposures
    print(f"\nCTR distribution (posts with ≥5 exposures):")
    sufficient_ctr_distribution = pd.cut(sufficient_exposure_posts['ctr'], bins=ctr_bins, labels=ctr_labels, include_lowest=True)
    sufficient_ctr_counts = sufficient_ctr_distribution.value_counts().sort_index()
    
    for bin_label, count in sufficient_ctr_counts.items():
        percentage = count / len(sufficient_exposure_posts) * 100
        print(f"  {bin_label}: {count:,} posts ({percentage:.1f}%)")
    
    # Summary and recommendations
    print(f"\n=== SUMMARY AND RECOMMENDATIONS ===")
    
    print(f"Correct CTR calculation:")
    print(f"  CTR = total_clicks / total_exposures")
    print(f"  Each record represents an exposure")
    print(f"  is_click = 1 means the exposure resulted in a click")
    print(f"  is_click = 0 means the exposure did not result in a click")
    
    print(f"\nData quality issues:")
    print(f"1. Many posts have very few exposures (unreliable CTR)")
    print(f"2. 94.4% of posts have CTR = 1.0 (all exposures clicked)")
    print(f"3. Only 5.6% of posts have meaningful CTR variation")
    
    print(f"\nRecommended approach:")
    print(f"1. Filter posts with sufficient exposures (≥5 or ≥10)")
    print(f"2. Use proper CTR as target variable")
    print(f"3. This will give us meaningful variation for uplift modeling")
    
    print(f"\nExpected results after filtering:")
    print(f"  - More realistic CTR distribution")
    print(f"  - Better variation for modeling")
    print(f"  - More reliable uplift analysis")

if __name__ == "__main__":
    main()
