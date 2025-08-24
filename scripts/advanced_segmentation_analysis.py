import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class AdvancedSegmentationAnalysis:
    """
    Advanced segmentation analysis for uplift modeling with detailed insights
    """
    
    def __init__(self):
        self.segments = {}
        self.feature_importance = {}
        
    def load_data(self, file_path='cluster7_user_post_uplift_prediction.csv'):
        """Load uplift prediction data"""
        print("=== Loading Data for Advanced Segmentation ===")
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df):,} records")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def create_multidimensional_segments(self, df, n_segments=6):
        """Create segments based on multiple dimensions"""
        print(f"\n=== Creating Multidimensional Segments (n={n_segments}) ===")
        
        # Prepare features for segmentation
        feature_cols = []
        
        # Uplift score
        feature_cols.append('uplift_pred')
        
        # Click behavior
        feature_cols.append('is_click')
        
        # Content length features
        if 'Title' in df.columns:
            df['title_length'] = df['Title'].astype(str).str.len()
            feature_cols.append('title_length')
        
        if 'Body' in df.columns:
            df['body_length'] = df['Body'].astype(str).str.len()
            feature_cols.append('body_length')
        
        if 'Tags' in df.columns:
            df['tags_length'] = df['Tags'].astype(str).str.len()
            feature_cols.append('tags_length')
        
        # Create interaction features
        df['title_body_ratio'] = df['title_length'] / (df['body_length'] + 1)
        feature_cols.append('title_body_ratio')
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df[feature_cols])
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
        df['multidimensional_segment'] = kmeans.fit_predict(features_scaled)
        
        # Analyze segments
        segment_analysis = {}
        for segment in range(n_segments):
            segment_data = df[df['multidimensional_segment'] == segment]
            
            segment_analysis[segment] = {
                'count': len(segment_data),
                'percentage': len(segment_data) / len(df) * 100,
                'mean_uplift': segment_data['uplift_pred'].mean(),
                'median_uplift': segment_data['uplift_pred'].median(),
                'std_uplift': segment_data['uplift_pred'].std(),
                'click_rate': segment_data['is_click'].mean(),
                'mean_title_length': segment_data['title_length'].mean(),
                'mean_body_length': segment_data['body_length'].mean(),
                'mean_tags_length': segment_data['tags_length'].mean(),
                'mean_title_body_ratio': segment_data['title_body_ratio'].mean(),
                'positive_uplift_pct': (segment_data['uplift_pred'] > 0).mean() * 100,
                'high_uplift_pct': (segment_data['uplift_pred'] > segment_data['uplift_pred'].quantile(0.9)).mean() * 100
            }
        
        # Sort segments by mean uplift
        sorted_segments = sorted(segment_analysis.items(), key=lambda x: x[1]['mean_uplift'])
        
        print("Multidimensional Segments Analysis:")
        for i, (segment, metrics) in enumerate(sorted_segments):
            segment_name = f"Segment {i+1}"
            print(f"\n{segment_name}:")
            print(f"  Count: {metrics['count']:,} ({metrics['percentage']:.1f}%)")
            print(f"  Mean Uplift: {metrics['mean_uplift']:.4f}")
            print(f"  Click Rate: {metrics['click_rate']:.2%}")
            print(f"  Positive Uplift %: {metrics['positive_uplift_pct']:.1f}%")
            print(f"  High Uplift %: {metrics['high_uplift_pct']:.1f}%")
            print(f"  Avg Title Length: {metrics['mean_title_length']:.1f}")
            print(f"  Avg Body Length: {metrics['mean_body_length']:.1f}")
            print(f"  Title/Body Ratio: {metrics['mean_title_body_ratio']:.3f}")
        
        return segment_analysis, sorted_segments, df
    
    def analyze_feature_importance(self, df):
        """Analyze feature importance for uplift prediction"""
        print("\n=== Feature Importance Analysis ===")
        
        # Prepare features
        feature_cols = []
        if 'title_length' in df.columns:
            feature_cols.append('title_length')
        if 'body_length' in df.columns:
            feature_cols.append('body_length')
        if 'tags_length' in df.columns:
            feature_cols.append('tags_length')
        if 'title_body_ratio' in df.columns:
            feature_cols.append('title_body_ratio')
        
        if not feature_cols:
            print("No features available for importance analysis")
            return None
        
        # Train Random Forest to get feature importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(df[feature_cols], df['uplift_pred'])
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Feature Importance for Uplift Prediction:")
        for _, row in importance_df.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(importance_df['feature'], importance_df['importance'], color='skyblue')
        plt.title('Feature Importance for Uplift Prediction')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def create_uplift_performance_matrix(self, df, segment_analysis):
        """Create uplift performance matrix"""
        print("\n=== Uplift Performance Matrix ===")
        
        # Create performance matrix
        performance_data = []
        for segment, metrics in segment_analysis.items():
            performance_data.append({
                'Segment': f"Segment {segment+1}",
                'Mean Uplift': metrics['mean_uplift'],
                'Click Rate': metrics['click_rate'],
                'Positive Uplift %': metrics['positive_uplift_pct'],
                'High Uplift %': metrics['high_uplift_pct'],
                'Count': metrics['count'],
                'Percentage': metrics['percentage']
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Sort by mean uplift
        performance_df = performance_df.sort_values('Mean Uplift', ascending=False)
        
        print("Uplift Performance Matrix:")
        print(performance_df.to_string(index=False, float_format='%.4f'))
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        
        # Prepare data for heatmap
        heatmap_data = performance_df[['Mean Uplift', 'Click Rate', 'Positive Uplift %', 'High Uplift %']].values
        heatmap_df = pd.DataFrame(heatmap_data, 
                                columns=['Mean Uplift', 'Click Rate', 'Positive Uplift %', 'High Uplift %'],
                                index=performance_df['Segment'])
        
        # Create heatmap
        sns.heatmap(heatmap_df.T, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Value'})
        plt.title('Uplift Performance Matrix Heatmap')
        plt.tight_layout()
        plt.savefig('uplift_performance_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return performance_df
    
    def analyze_content_patterns(self, df, segment_analysis):
        """Analyze content patterns by segment"""
        print("\n=== Content Pattern Analysis ===")
        
        content_patterns = {}
        for segment, metrics in segment_analysis.items():
            segment_data = df[df['multidimensional_segment'] == segment]
            
            # Analyze content characteristics
            content_patterns[segment] = {
                'segment_name': f"Segment {segment+1}",
                'mean_uplift': metrics['mean_uplift'],
                'click_rate': metrics['click_rate'],
                'avg_title_length': metrics['mean_title_length'],
                'avg_body_length': metrics['mean_body_length'],
                'avg_tags_length': metrics['mean_tags_length'],
                'title_body_ratio': metrics['mean_title_body_ratio'],
                'positive_uplift_pct': metrics['positive_uplift_pct']
            }
        
        # Create content pattern visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Title length vs Uplift
        segments = list(content_patterns.keys())
        title_lengths = [content_patterns[s]['avg_title_length'] for s in segments]
        uplifts = [content_patterns[s]['mean_uplift'] for s in segments]
        colors = ['red' if u < 0 else 'green' for u in uplifts]
        
        axes[0, 0].scatter(title_lengths, uplifts, c=colors, s=100, alpha=0.7)
        axes[0, 0].set_xlabel('Average Title Length')
        axes[0, 0].set_ylabel('Mean Uplift')
        axes[0, 0].set_title('Title Length vs Uplift by Segment')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Body length vs Uplift
        body_lengths = [content_patterns[s]['avg_body_length'] for s in segments]
        axes[0, 1].scatter(body_lengths, uplifts, c=colors, s=100, alpha=0.7)
        axes[0, 1].set_xlabel('Average Body Length')
        axes[0, 1].set_ylabel('Mean Uplift')
        axes[0, 1].set_title('Body Length vs Uplift by Segment')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Title/Body ratio vs Uplift
        ratios = [content_patterns[s]['title_body_ratio'] for s in segments]
        axes[1, 0].scatter(ratios, uplifts, c=colors, s=100, alpha=0.7)
        axes[1, 0].set_xlabel('Title/Body Ratio')
        axes[1, 0].set_ylabel('Mean Uplift')
        axes[1, 0].set_title('Title/Body Ratio vs Uplift by Segment')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Click rate vs Positive uplift percentage
        click_rates = [content_patterns[s]['click_rate'] for s in segments]
        positive_pcts = [content_patterns[s]['positive_uplift_pct'] for s in segments]
        axes[1, 1].scatter(click_rates, positive_pcts, c=colors, s=100, alpha=0.7)
        axes[1, 1].set_xlabel('Click Rate')
        axes[1, 1].set_ylabel('Positive Uplift %')
        axes[1, 1].set_title('Click Rate vs Positive Uplift % by Segment')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('content_pattern_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return content_patterns
    
    def generate_segment_recommendations(self, segment_analysis, content_patterns):
        """Generate detailed recommendations for each segment"""
        print("\n=== Segment-Specific Recommendations ===")
        
        # Sort segments by mean uplift
        sorted_segments = sorted(segment_analysis.items(), key=lambda x: x[1]['mean_uplift'])
        
        recommendations = {}
        
        for i, (segment, metrics) in enumerate(sorted_segments):
            segment_name = f"Segment {i+1}"
            content_pattern = content_patterns.get(segment, {})
            
            print(f"\n{segment_name} Recommendations:")
            print(f"  Mean Uplift: {metrics['mean_uplift']:.4f}")
            print(f"  Click Rate: {metrics['click_rate']:.2%}")
            print(f"  Positive Uplift %: {metrics['positive_uplift_pct']:.1f}%")
            
            # Generate recommendations based on performance
            if metrics['mean_uplift'] > 0:
                print("  HIGH PERFORMING SEGMENT")
                print("     Recommendations:")
                print("     - Scale AI content deployment")
                print("     - Replicate successful content patterns")
                print("     - Optimize for maximum engagement")
                print("     - Use as benchmark for other segments")
            elif metrics['mean_uplift'] > -0.1:
                print("  MODERATE PERFORMING SEGMENT")
                print("     Recommendations:")
                print("     - Test different AI content formats")
                print("     - Optimize content quality")
                print("     - A/B test content variations")
                print("     - Monitor for improvement opportunities")
            else:
                print("  🚨 LOW PERFORMING SEGMENT")
                print("     Recommendations:")
                print("     - Avoid AI content deployment")
                print("     - Investigate root causes")
                print("     - Consider content quality improvements")
                print("     - Test alternative content strategies")
            
            # Content-specific recommendations
            if content_pattern:
                avg_title_length = content_pattern.get('avg_title_length', 0)
                avg_body_length = content_pattern.get('avg_body_length', 0)
                title_body_ratio = content_pattern.get('title_body_ratio', 0)
                
                print("     Content Optimization:")
                if avg_title_length < 30:
                    print("     - Consider longer, more descriptive titles")
                elif avg_title_length > 60:
                    print("     - Consider shorter, more concise titles")
                
                if avg_body_length < 500:
                    print("     - Consider more detailed content")
                elif avg_body_length > 2000:
                    print("     - Consider more concise content")
                
                if title_body_ratio > 0.1:
                    print("     - Consider more balanced title/body ratio")
            
            recommendations[segment] = {
                'segment_name': segment_name,
                'mean_uplift': metrics['mean_uplift'],
                'click_rate': metrics['click_rate'],
                'positive_uplift_pct': metrics['positive_uplift_pct'],
                'performance_category': 'HIGH' if metrics['mean_uplift'] > 0 else 'MODERATE' if metrics['mean_uplift'] > -0.1 else 'LOW'
            }
        
        return recommendations
    
    def create_action_plan(self, segment_analysis, recommendations):
        """Create actionable implementation plan"""
        print("\n=== ACTION PLAN ===")
        
        # Categorize segments
        high_performing = []
        moderate_performing = []
        low_performing = []
        
        for segment, metrics in segment_analysis.items():
            if metrics['mean_uplift'] > 0:
                high_performing.append((segment, metrics))
            elif metrics['mean_uplift'] > -0.1:
                moderate_performing.append((segment, metrics))
            else:
                low_performing.append((segment, metrics))
        
        print("IMMEDIATE ACTIONS (Next 30 days):")
        print("  1. High-Performing Segments:")
        for segment, metrics in high_performing:
            print(f"     - Scale AI content for Segment {segment+1} ({metrics['count']:,} users)")
            print(f"       Target uplift: {metrics['mean_uplift']:.4f}")
        
        print("  2. Moderate-Performing Segments:")
        for segment, metrics in moderate_performing:
            print(f"     - A/B test AI content for Segment {segment+1} ({metrics['count']:,} users)")
            print(f"       Current uplift: {metrics['mean_uplift']:.4f}")
        
        print("  3. Low-Performing Segments:")
        for segment, metrics in low_performing:
            print(f"     - Pause AI content for Segment {segment+1} ({metrics['count']:,} users)")
            print(f"       Current uplift: {metrics['mean_uplift']:.4f}")
        
        print("\n📈 MEDIUM-TERM STRATEGY (Next 90 days):")
        print("  1. Content Optimization:")
        print("     - Analyze successful content patterns")
        print("     - Develop segment-specific content guidelines")
        print("     - Implement automated content quality checks")
        
        print("  2. Model Improvements:")
        print("     - Retrain models with new data")
        print("     - Add new features based on analysis")
        print("     - Implement real-time uplift prediction")
        
        print("  3. Monitoring & Alerting:")
        print("     - Set up segment performance dashboards")
        print("     - Implement automated alerts for performance drops")
        print("     - Regular segment performance reviews")
        
        print("\nSUCCESS METRICS:")
        print("  - Overall mean uplift improvement")
        print("  - Segment-specific uplift improvements")
        print("  - Click rate improvements by segment")
        print("  - Reduction in negative uplift segments")
        
        return {
            'high_performing': high_performing,
            'moderate_performing': moderate_performing,
            'low_performing': low_performing
        }
    
    def save_advanced_report(self, segment_analysis, content_patterns, recommendations, action_plan):
        """Save advanced segmentation report"""
        print("\n=== Saving Advanced Segmentation Report ===")
        
        report_content = []
        report_content.append("ADVANCED SEGMENTATION ANALYSIS REPORT")
        report_content.append("="*60)
        report_content.append("")
        
        # Executive Summary
        report_content.append("📋 EXECUTIVE SUMMARY")
        report_content.append("-" * 30)
        
        total_segments = len(segment_analysis)
        high_performing_count = len(action_plan['high_performing'])
        moderate_performing_count = len(action_plan['moderate_performing'])
        low_performing_count = len(action_plan['low_performing'])
        
        report_content.append(f"Total segments analyzed: {total_segments}")
        report_content.append(f"High-performing segments: {high_performing_count}")
        report_content.append(f"Moderate-performing segments: {moderate_performing_count}")
        report_content.append(f"Low-performing segments: {low_performing_count}")
        report_content.append("")
        
        # Segment Details
        report_content.append("SEGMENT DETAILS")
        report_content.append("-" * 30)
        
        for segment, metrics in segment_analysis.items():
            report_content.append(f"Segment {segment+1}:")
            report_content.append(f"  Count: {metrics['count']:,} ({metrics['percentage']:.1f}%)")
            report_content.append(f"  Mean uplift: {metrics['mean_uplift']:.4f}")
            report_content.append(f"  Click rate: {metrics['click_rate']:.2%}")
            report_content.append(f"  Positive uplift %: {metrics['positive_uplift_pct']:.1f}%")
            report_content.append("")
        
        # Content Patterns
        report_content.append("📝 CONTENT PATTERNS")
        report_content.append("-" * 30)
        
        for segment, pattern in content_patterns.items():
            report_content.append(f"Segment {segment+1} Content Characteristics:")
            report_content.append(f"  Avg title length: {pattern['avg_title_length']:.1f}")
            report_content.append(f"  Avg body length: {pattern['avg_body_length']:.1f}")
            report_content.append(f"  Title/body ratio: {pattern['title_body_ratio']:.3f}")
            report_content.append("")
        
        # Recommendations
        report_content.append("RECOMMENDATIONS")
        report_content.append("-" * 30)
        
        for segment, rec in recommendations.items():
            report_content.append(f"{rec['segment_name']} ({rec['performance_category']}):")
            report_content.append(f"  Mean uplift: {rec['mean_uplift']:.4f}")
            report_content.append(f"  Click rate: {rec['click_rate']:.2%}")
            report_content.append("")
        
        # Action Plan
        report_content.append("📅 ACTION PLAN")
        report_content.append("-" * 30)
        report_content.append("Immediate Actions (30 days):")
        report_content.append("1. Scale high-performing segments")
        report_content.append("2. A/B test moderate-performing segments")
        report_content.append("3. Pause low-performing segments")
        report_content.append("")
        report_content.append("Medium-term Strategy (90 days):")
        report_content.append("1. Content optimization")
        report_content.append("2. Model improvements")
        report_content.append("3. Monitoring & alerting")
        
        # Save report
        with open('advanced_segmentation_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        print("Advanced segmentation report saved to: advanced_segmentation_report.txt")
    
    def run_complete_analysis(self):
        """Run complete advanced segmentation analysis"""
        print("ADVANCED SEGMENTATION ANALYSIS")
        print("="*60)
        
        # Load data
        df = self.load_data()
        if df is None:
            return
        
        # Create multidimensional segments
        segment_analysis, sorted_segments, df = self.create_multidimensional_segments(df)
        
        # Analyze feature importance
        feature_importance = self.analyze_feature_importance(df)
        
        # Create performance matrix
        performance_matrix = self.create_uplift_performance_matrix(df, segment_analysis)
        
        # Analyze content patterns
        content_patterns = self.analyze_content_patterns(df, segment_analysis)
        
        # Generate recommendations
        recommendations = self.generate_segment_recommendations(segment_analysis, content_patterns)
        
        # Create action plan
        action_plan = self.create_action_plan(segment_analysis, recommendations)
        
        # Save advanced report
        self.save_advanced_report(segment_analysis, content_patterns, recommendations, action_plan)
        
        print("\nAdvanced segmentation analysis completed!")
        print("Generated visualizations:")
        print("  - feature_importance_analysis.png")
        print("  - uplift_performance_matrix.png")
        print("  - content_pattern_analysis.png")
        print("📋 Generated reports:")
        print("  - advanced_segmentation_report.txt")
        
        return {
            'segment_analysis': segment_analysis,
            'content_patterns': content_patterns,
            'recommendations': recommendations,
            'action_plan': action_plan,
            'feature_importance': feature_importance
        }

def main():
    """Main function to run advanced segmentation analysis"""
    analyzer = AdvancedSegmentationAnalysis()
    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    main() 