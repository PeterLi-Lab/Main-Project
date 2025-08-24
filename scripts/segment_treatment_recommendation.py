import pandas as pd
import numpy as np

def main():
    """Recommend which user segments are best for AI tag treatment"""
    
    print("=== USER SEGMENT TREATMENT RECOMMENDATION ===")
    print("Based on user segmentation and uplift analysis")
    
    # Load user segments
    user_segments = pd.read_csv('user_segments_quick.csv')
    
    # Load uplift results
    uplift_results = pd.read_csv('simple_uplift_results.csv')
    
    print(f"User segments loaded: {len(user_segments):,} users")
    print(f"Uplift results loaded: {len(uplift_results):,} posts")
    
    # Analyze each segment
    print("\n=== SEGMENT ANALYSIS FOR AI TAG TREATMENT ===")
    
    segment_analysis = []
    
    for segment in sorted(user_segments['user_segment'].unique()):
        segment_data = user_segments[user_segments['user_segment'] == segment]
        
        print(f"\n--- Segment {segment} ---")
        print(f"Users: {len(segment_data):,} ({len(segment_data)/len(user_segments)*100:.1f}%)")
        
        # Key metrics
        click_rate = segment_data['user_click_rate'].mean()
        total_interactions = segment_data['user_total_interactions'].mean()
        click_consistency = segment_data['user_click_consistency'].mean()
        high_engagement_pct = (segment_data['user_engagement_level'].isin(['High', 'Very High'])).mean() * 100
        
        print(f"Click Rate: {click_rate:.3f}")
        print(f"Total Interactions: {total_interactions:.1f}")
        print(f"Click Consistency: {click_consistency:.3f}")
        print(f"High Engagement: {high_engagement_pct:.1f}%")
        
        # Treatment recommendation logic
        treatment_score = 0
        reasons = []
        
        # Factor 1: Click rate (higher is better for treatment)
        if click_rate > 0.8:
            treatment_score += 3
            reasons.append("High click rate - likely to respond to treatments")
        elif click_rate > 0.6:
            treatment_score += 2
            reasons.append("Medium-high click rate - good treatment potential")
        elif click_rate > 0.4:
            treatment_score += 1
            reasons.append("Medium click rate - moderate treatment potential")
        else:
            treatment_score += 0
            reasons.append("Low click rate - poor treatment potential")
        
        # Factor 2: Activity level (moderate is best - not too low, not too high)
        if 5 <= total_interactions <= 50:
            treatment_score += 3
            reasons.append("Optimal activity level - engaged but not overwhelmed")
        elif total_interactions > 50:
            treatment_score += 1
            reasons.append("Very high activity - may be less sensitive to treatments")
        else:
            treatment_score += 0
            reasons.append("Low activity - may not notice treatments")
        
        # Factor 3: Click consistency (higher is better)
        if click_consistency > 0.8:
            treatment_score += 2
            reasons.append("High click consistency - predictable behavior")
        elif click_consistency > 0.5:
            treatment_score += 1
            reasons.append("Medium click consistency - somewhat predictable")
        else:
            treatment_score += 0
            reasons.append("Low click consistency - unpredictable behavior")
        
        # Factor 4: Engagement level (moderate is best)
        if 10 <= high_engagement_pct <= 50:
            treatment_score += 2
            reasons.append("Moderate engagement - good balance")
        elif high_engagement_pct > 50:
            treatment_score += 1
            reasons.append("High engagement - may be less sensitive to treatments")
        else:
            treatment_score += 0
            reasons.append("Low engagement - may not respond to treatments")
        
        # Segment size factor (larger segments are better for business impact)
        segment_size_pct = len(segment_data) / len(user_segments) * 100
        if segment_size_pct > 20:
            treatment_score += 2
            reasons.append("Large segment size - high business impact potential")
        elif segment_size_pct > 5:
            treatment_score += 1
            reasons.append("Medium segment size - moderate business impact")
        else:
            treatment_score += 0
            reasons.append("Small segment size - limited business impact")
        
        print(f"Treatment Score: {treatment_score}/12")
        print("Reasons:")
        for reason in reasons:
            print(f"  - {reason}")
        
        # Determine recommendation
        if treatment_score >= 8:
            recommendation = "STRONGLY RECOMMEND"
            recommendation_color = "GREEN"
        elif treatment_score >= 6:
            recommendation = "RECOMMEND"
            recommendation_color = "YELLOW"
        elif treatment_score >= 4:
            recommendation = "CONSIDER"
            recommendation_color = "ORANGE"
        else:
            recommendation = "NOT RECOMMENDED"
            recommendation_color = "RED"
        
        print(f"Recommendation: {recommendation}")
        
        segment_analysis.append({
            'segment': segment,
            'users': len(segment_data),
            'percentage': segment_size_pct,
            'click_rate': click_rate,
            'total_interactions': total_interactions,
            'click_consistency': click_consistency,
            'high_engagement_pct': high_engagement_pct,
            'treatment_score': treatment_score,
            'recommendation': recommendation,
            'reasons': reasons
        })
    
    # Sort by treatment score
    segment_analysis.sort(key=lambda x: x['treatment_score'], reverse=True)
    
    print(f"\n=== FINAL TREATMENT RECOMMENDATIONS ===")
    print("Ranked by treatment suitability:")
    
    for i, analysis in enumerate(segment_analysis):
        print(f"\n{i+1}. Segment {analysis['segment']} - {analysis['recommendation']}")
        print(f"   Users: {analysis['users']:,} ({analysis['percentage']:.1f}%)")
        print(f"   Treatment Score: {analysis['treatment_score']}/12")
        print(f"   Click Rate: {analysis['click_rate']:.3f}")
        print(f"   Activity Level: {analysis['total_interactions']:.1f} interactions")
        print(f"   Click Consistency: {analysis['click_consistency']:.3f}")
        print(f"   High Engagement: {analysis['high_engagement_pct']:.1f}%")
    
    # Business impact analysis
    print(f"\n=== BUSINESS IMPACT ANALYSIS ===")
    
    total_users = len(user_segments)
    recommended_users = 0
    
    for analysis in segment_analysis:
        if "RECOMMEND" in analysis['recommendation']:
            recommended_users += analysis['users']
            print(f"Segment {analysis['segment']}: {analysis['users']:,} users ({analysis['percentage']:.1f}%)")
    
    print(f"\nTotal recommended users: {recommended_users:,} ({recommended_users/total_users*100:.1f}%)")
    
    # Implementation strategy
    print(f"\n=== IMPLEMENTATION STRATEGY ===")
    
    print("Phase 1 - High Priority Segments:")
    high_priority = [s for s in segment_analysis if s['treatment_score'] >= 8]
    for segment in high_priority:
        print(f"  - Segment {segment['segment']}: {segment['users']:,} users")
    
    print("\nPhase 2 - Medium Priority Segments:")
    medium_priority = [s for s in segment_analysis if 6 <= s['treatment_score'] < 8]
    for segment in medium_priority:
        print(f"  - Segment {segment['segment']}: {segment['users']:,} users")
    
    print("\nPhase 3 - Test Segments:")
    test_priority = [s for s in segment_analysis if 4 <= s['treatment_score'] < 6]
    for segment in test_priority:
        print(f"  - Segment {segment['segment']}: {segment['users']:,} users")
    
    print("\nAvoid Segments:")
    avoid_segments = [s for s in segment_analysis if s['treatment_score'] < 4]
    for segment in avoid_segments:
        print(f"  - Segment {segment['segment']}: {segment['users']:,} users")
    
    # Export recommendations
    recommendations_df = pd.DataFrame(segment_analysis)
    recommendations_df.to_csv('treatment_recommendations.csv', index=False)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print("Recommendations exported to 'treatment_recommendations.csv'")

if __name__ == "__main__":
    main()







