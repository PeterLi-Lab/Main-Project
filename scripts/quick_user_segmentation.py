import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    """Quick user segmentation analysis"""
    
    print("=== QUICK USER SEGMENTATION ANALYSIS ===")
    
    # Load click data
    print("Loading click data...")
    click_data = pd.read_csv('user_post_click_samples.csv')
    click_data = click_data.rename(columns={'is_click': 'clicked'})
    
    # Create user features
    print("Creating user features...")
    user_features = click_data.groupby('user_id').agg({
        'clicked': ['mean', 'sum', 'count', 'std'],
        'post_id': 'nunique'
    }).reset_index()
    
    user_features.columns = ['user_id', 'user_click_rate', 'user_total_clicks', 'user_total_interactions', 'user_click_std', 'user_unique_posts']
    
    # Calculate additional features
    user_features['user_click_consistency'] = 1 - user_features['user_click_std']
    user_features['user_click_consistency'] = user_features['user_click_consistency'].fillna(0)
    
    user_features['user_post_diversity'] = user_features['user_unique_posts'] / user_features['user_total_interactions']
    user_features['user_post_diversity'] = user_features['user_post_diversity'].fillna(0)
    
    # Create engagement levels
    user_features['user_engagement_level'] = pd.cut(
        user_features['user_total_interactions'], 
        bins=[0, 5, 20, 100, np.inf], 
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    print(f"Total users: {len(user_features):,}")
    
    # Perform segmentation
    print("Performing user segmentation...")
    
    # Select features for clustering
    feature_columns = [
        'user_click_rate', 'user_total_clicks', 'user_total_interactions', 
        'user_click_consistency', 'user_post_diversity'
    ]
    
    # Prepare data
    X = user_features[feature_columns].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    user_features['user_segment'] = kmeans.fit_predict(X_scaled)
    
    # Analyze segments
    print("\n=== USER SEGMENTATION RESULTS ===")
    segment_summaries = []
    
    for segment in sorted(user_features['user_segment'].unique()):
        segment_data = user_features[user_features['user_segment'] == segment]
        
        print(f"\nSegment {segment}:")
        print(f"  Count: {len(segment_data):,} users ({len(segment_data)/len(user_features)*100:.1f}%)")
        print(f"  Click rate: {segment_data['user_click_rate'].mean():.3f}")
        print(f"  Total clicks: {segment_data['user_total_clicks'].mean():.1f}")
        print(f"  Total interactions: {segment_data['user_total_interactions'].mean():.1f}")
        print(f"  Click consistency: {segment_data['user_click_consistency'].mean():.3f}")
        print(f"  Post diversity: {segment_data['user_post_diversity'].mean():.3f}")
        
        # Engagement level distribution
        engagement_dist = segment_data['user_engagement_level'].value_counts()
        print(f"  Engagement level distribution:")
        for level, count in engagement_dist.items():
            print(f"    {level}: {count:,} ({count/len(segment_data)*100:.1f}%)")
        
        # Store summary
        segment_summaries.append({
            'segment': segment,
            'count': len(segment_data),
            'percentage': len(segment_data)/len(user_features)*100,
            'click_rate': segment_data['user_click_rate'].mean(),
            'total_clicks': segment_data['user_total_clicks'].mean(),
            'total_interactions': segment_data['user_total_interactions'].mean(),
            'click_consistency': segment_data['user_click_consistency'].mean(),
            'post_diversity': segment_data['user_post_diversity'].mean(),
            'high_engagement_pct': (segment_data['user_engagement_level'].isin(['High', 'Very High'])).mean() * 100
        })
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Segment sizes
    segment_sizes = user_features['user_segment'].value_counts().sort_index()
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
    axes[0, 0].bar(segment_sizes.index, segment_sizes.values, color=colors)
    axes[0, 0].set_xlabel('User Segment')
    axes[0, 0].set_ylabel('Number of Users')
    axes[0, 0].set_title('User Segment Sizes')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Click rate by segment
    click_rates = [s['click_rate'] for s in segment_summaries]
    axes[0, 1].bar(range(len(click_rates)), click_rates, color=colors)
    axes[0, 1].set_xlabel('User Segment')
    axes[0, 1].set_ylabel('Average Click Rate')
    axes[0, 1].set_title('Click Rate by User Segment')
    axes[0, 1].set_xticks(range(len(click_rates)))
    axes[0, 1].set_xticklabels([s['segment'] for s in segment_summaries])
    axes[0, 1].grid(True, alpha=0.3)
    
    # Total interactions by segment
    interactions = [s['total_interactions'] for s in segment_summaries]
    axes[1, 0].bar(range(len(interactions)), interactions, color=colors)
    axes[1, 0].set_xlabel('User Segment')
    axes[1, 0].set_ylabel('Average Total Interactions')
    axes[1, 0].set_title('User Activity Level by Segment')
    axes[1, 0].set_xticks(range(len(interactions)))
    axes[1, 0].set_xticklabels([s['segment'] for s in segment_summaries])
    axes[1, 0].grid(True, alpha=0.3)
    
    # High engagement percentage
    high_engagement = [s['high_engagement_pct'] for s in segment_summaries]
    axes[1, 1].bar(range(len(high_engagement)), high_engagement, color=colors)
    axes[1, 1].set_xlabel('User Segment')
    axes[1, 1].set_ylabel('High Engagement Users (%)')
    axes[1, 1].set_title('High Engagement Percentage by Segment')
    axes[1, 1].set_xticks(range(len(high_engagement)))
    axes[1, 1].set_xticklabels([s['segment'] for s in segment_summaries])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('user_segmentation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Export results
    user_features.to_csv('user_segments_quick.csv', index=False)
    
    # Create segment descriptions
    print("\n=== SEGMENT DESCRIPTIONS ===")
    
    # Sort by click rate to identify best segments
    segment_summaries.sort(key=lambda x: x['click_rate'], reverse=True)
    
    for i, summary in enumerate(segment_summaries):
        print(f"\n{i+1}. Segment {summary['segment']} - {summary['count']:,} users ({summary['percentage']:.1f}%)")
        print(f"   Click Rate: {summary['click_rate']:.3f}")
        print(f"   Total Interactions: {summary['total_interactions']:.1f}")
        print(f"   Click Consistency: {summary['click_consistency']:.3f}")
        print(f"   Post Diversity: {summary['post_diversity']:.3f}")
        print(f"   High Engagement: {summary['high_engagement_pct']:.1f}%")
        
        # Create description
        if summary['click_rate'] > 0.8:
            click_desc = "High click rate"
        elif summary['click_rate'] > 0.6:
            click_desc = "Medium-high click rate"
        elif summary['click_rate'] > 0.4:
            click_desc = "Medium click rate"
        else:
            click_desc = "Low click rate"
        
        if summary['total_interactions'] > 50:
            activity_desc = "Very active"
        elif summary['total_interactions'] > 20:
            activity_desc = "Active"
        elif summary['total_interactions'] > 5:
            activity_desc = "Moderately active"
        else:
            activity_desc = "Low activity"
        
        if summary['high_engagement_pct'] > 50:
            engagement_desc = "High engagement"
        elif summary['high_engagement_pct'] > 25:
            engagement_desc = "Medium engagement"
        else:
            engagement_desc = "Low engagement"
        
        print(f"   Description: {click_desc}, {activity_desc}, {engagement_desc}")
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print("Results exported to 'user_segments_quick.csv'")
    print("Visualization saved as 'user_segmentation_analysis.png'")

if __name__ == "__main__":
    main()







