import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class TwoStageAnalysis:
    """
    Two-stage analysis to separate tag effects from content effects
    Stage 1: Tag effect (within similar AI content)
    Stage 2: Content type effect (AI content vs non-AI content)
    """
    
    def __init__(self):
        self.ai_keywords = [
            'ai', 'artificial intelligence', 'machine learning', 'deep learning', 
            'neural network', 'gpt', 'llm', 'data science', 'predictive', 
            'automated', 'intelligent', 'smart', 'tensorflow', 'pytorch', 
            'scikit-learn', 'openai', 'nlp', 'computer vision', 'reinforcement learning',
            'transformer', 'chatbot', 'automation', 'algorithm'
        ]
        
    def load_data(self, file_path='cluster7_user_post_uplift_prediction.csv'):
        """Load data"""
        print("=== Loading Data ===")
        df = pd.read_csv(file_path)
        print(f"Total data volume: {len(df):,}")
        return df
    
    def identify_ai_content(self, df):
        """Identify AI content based on text analysis"""
        print("\n=== Identifying AI Content ===")
        
        # Method 1: Check tags for AI keywords
        df['ai_keyword_count_tags'] = 0
        for keyword in self.ai_keywords:
            keyword_matches = df['Tags'].str.contains(keyword, case=False, na=False)
            df.loc[keyword_matches, 'ai_keyword_count_tags'] += 1
        
        # Method 2: Check merged_content for AI keywords
        df['ai_keyword_count_content'] = 0
        for keyword in self.ai_keywords:
            keyword_matches = df['merged_content'].str.contains(keyword, case=False, na=False)
            df.loc[keyword_matches, 'ai_keyword_count_content'] += 1
        
        # Create AI content flag
        df['is_ai_content'] = ((df['ai_keyword_count_tags'] >= 1) | 
                              (df['ai_keyword_count_content'] >= 2)).astype(int)
        
        ai_content_count = df['is_ai_content'].sum()
        print(f"AI content posts: {ai_content_count:,} ({ai_content_count/len(df):.1%})")
        
        return df
    
    def identify_ai_tags(self, df):
        """Identify posts with AI tags"""
        print("\n=== Identifying AI Tags ===")
        
        # Check for AI tags in Tags column
        df['has_ai_tag'] = df['Tags'].str.contains('ai', case=False, na=False).astype(int)
        
        ai_tag_count = df['has_ai_tag'].sum()
        print(f"Posts with AI tags: {ai_tag_count:,} ({ai_tag_count/len(df):.1%})")
        
        return df
    
    def stage1_tag_effect_analysis(self, df):
        """Stage 1: Analyze tag effect within similar AI content"""
        print("\n" + "="*60)
        print("STAGE 1: TAG EFFECT ANALYSIS")
        print("="*60)
        print("Goal: Measure the effect of AI tags within similar AI content")
        
        # Filter for AI content only
        ai_content_df = df[df['is_ai_content'] == 1].copy()
        print(f"\nAI content posts for analysis: {len(ai_content_df):,}")
        
        if len(ai_content_df) < 100:
            print("Insufficient AI content posts for analysis")
            return None
        
        # Create treatment/control based on AI tags
        ai_content_df['treatment_ai_tag'] = ai_content_df['has_ai_tag'].astype(int)
        
        treatment_count = ai_content_df['treatment_ai_tag'].sum()
        control_count = len(ai_content_df) - treatment_count
        
        print(f"\nTreatment/Control split within AI content:")
        print(f"  Treatment (AI content + AI tag): {treatment_count:,} ({treatment_count/len(ai_content_df):.1%})")
        print(f"  Control (AI content + no AI tag): {control_count:,} ({control_count/len(ai_content_df):.1%})")
        
        if treatment_count < 50 or control_count < 50:
            print("Insufficient samples in treatment or control groups")
            return None
        
        # Analyze click rates
        treatment_click_rate = ai_content_df[ai_content_df['treatment_ai_tag'] == 1]['is_click'].mean()
        control_click_rate = ai_content_df[ai_content_df['treatment_ai_tag'] == 0]['is_click'].mean()
        
        print(f"\nClick rate analysis:")
        print(f"  Treatment (AI content + AI tag): {treatment_click_rate:.3f}")
        print(f"  Control (AI content + no AI tag): {control_click_rate:.3f}")
        print(f"  Difference: {treatment_click_rate - control_click_rate:.3f}")
        
        # Analyze uplift predictions
        treatment_uplift = ai_content_df[ai_content_df['treatment_ai_tag'] == 1]['uplift_pred'].mean()
        control_uplift = ai_content_df[ai_content_df['treatment_ai_tag'] == 0]['uplift_pred'].mean()
        tag_uplift_effect = treatment_uplift - control_uplift
        
        print(f"\nUplift analysis:")
        print(f"  Treatment uplift mean: {treatment_uplift:.4f}")
        print(f"  Control uplift mean: {control_uplift:.4f}")
        print(f"  Tag uplift effect: {tag_uplift_effect:.4f}")
        
        if tag_uplift_effect > 0:
            print(f"  POSITIVE: AI tags improve performance within AI content")
        else:
            print(f"  NEGATIVE: AI tags hurt performance within AI content")
        
        # Statistical significance test
        from scipy import stats
        treatment_clicks = ai_content_df[ai_content_df['treatment_ai_tag'] == 1]['is_click']
        control_clicks = ai_content_df[ai_content_df['treatment_ai_tag'] == 0]['is_click']
        
        t_stat, p_value = stats.ttest_ind(treatment_clicks, control_clicks)
        print(f"\nStatistical test:")
        print(f"  T-statistic: {t_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
        
        return {
            'treatment_count': treatment_count,
            'control_count': control_count,
            'treatment_click_rate': treatment_click_rate,
            'control_click_rate': control_click_rate,
            'click_rate_difference': treatment_click_rate - control_click_rate,
            'tag_uplift_effect': tag_uplift_effect,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def stage2_content_effect_analysis(self, df):
        """Stage 2: Analyze content type effect (AI vs non-AI content)"""
        print("\n" + "="*60)
        print("STAGE 2: CONTENT TYPE EFFECT ANALYSIS")
        print("="*60)
        print("Goal: Measure the effect of AI content vs non-AI content (ignoring tags)")
        
        # Create content type groups (ignoring tags)
        ai_content_df = df[df['is_ai_content'] == 1].copy()
        non_ai_content_df = df[df['is_ai_content'] == 0].copy()
        
        print(f"\nContent type split:")
        print(f"  AI content: {len(ai_content_df):,} ({len(ai_content_df)/len(df):.1%})")
        print(f"  Non-AI content: {len(non_ai_content_df):,} ({len(non_ai_content_df)/len(df):.1%})")
        
        if len(ai_content_df) < 100 or len(non_ai_content_df) < 100:
            print("Insufficient samples in content type groups")
            return None
        
        # Analyze click rates by content type
        ai_click_rate = ai_content_df['is_click'].mean()
        non_ai_click_rate = non_ai_content_df['is_click'].mean()
        
        print(f"\nClick rate analysis by content type:")
        print(f"  AI content: {ai_click_rate:.3f}")
        print(f"  Non-AI content: {non_ai_click_rate:.3f}")
        print(f"  Difference: {ai_click_rate - non_ai_click_rate:.3f}")
        
        # Analyze uplift predictions by content type
        ai_uplift = ai_content_df['uplift_pred'].mean()
        non_ai_uplift = non_ai_content_df['uplift_pred'].mean()
        content_uplift_effect = ai_uplift - non_ai_uplift
        
        print(f"\nUplift analysis by content type:")
        print(f"  AI content uplift mean: {ai_uplift:.4f}")
        print(f"  Non-AI content uplift mean: {non_ai_uplift:.4f}")
        print(f"  Content type uplift effect: {content_uplift_effect:.4f}")
        
        if content_uplift_effect > 0:
            print(f"  POSITIVE: AI content performs better than non-AI content")
        else:
            print(f"  NEGATIVE: AI content performs worse than non-AI content")
        
        # Statistical significance test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(ai_content_df['is_click'], non_ai_content_df['is_click'])
        
        print(f"\nStatistical test:")
        print(f"  T-statistic: {t_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Additional analysis: AI content with vs without tags
        ai_with_tag = ai_content_df[ai_content_df['has_ai_tag'] == 1]
        ai_without_tag = ai_content_df[ai_content_df['has_ai_tag'] == 0]
        
        if len(ai_with_tag) > 0 and len(ai_without_tag) > 0:
            print(f"\nAI content breakdown:")
            print(f"  AI content with AI tag: {len(ai_with_tag):,} (CTR: {ai_with_tag['is_click'].mean():.3f})")
            print(f"  AI content without AI tag: {len(ai_without_tag):,} (CTR: {ai_without_tag['is_click'].mean():.3f})")
        
        return {
            'ai_content_count': len(ai_content_df),
            'non_ai_content_count': len(non_ai_content_df),
            'ai_click_rate': ai_click_rate,
            'non_ai_click_rate': non_ai_click_rate,
            'click_rate_difference': ai_click_rate - non_ai_click_rate,
            'content_uplift_effect': content_uplift_effect,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def create_similar_content_clusters(self, df):
        """Create clusters of similar content for more refined analysis"""
        print("\n=== Creating Similar Content Clusters ===")
        
        # Use TF-IDF for text clustering
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['merged_content'].fillna(''))
        
        # Perform clustering
        n_clusters = min(10, len(df) // 100)  # Ensure reasonable cluster sizes
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        df['content_cluster'] = clusters
        
        print(f"Created {n_clusters} content clusters")
        
        # Analyze clusters
        for cluster_id in range(n_clusters):
            cluster_data = df[df['content_cluster'] == cluster_id]
            ai_content_in_cluster = cluster_data['is_ai_content'].sum()
            ai_tag_in_cluster = cluster_data['has_ai_tag'].sum()
            
            print(f"  Cluster {cluster_id}: {len(cluster_data):,} posts")
            print(f"    AI content: {ai_content_in_cluster:,} ({ai_content_in_cluster/len(cluster_data):.1%})")
            print(f"    AI tags: {ai_tag_in_cluster:,} ({ai_tag_in_cluster/len(cluster_data):.1%})")
        
        return df
    
    def stage1_with_clusters(self, df):
        """Stage 1 analysis within content clusters"""
        print("\n" + "="*60)
        print("STAGE 1 WITH CLUSTERS: TAG EFFECT IN SIMILAR CONTENT")
        print("="*60)
        
        results = []
        
        for cluster_id in df['content_cluster'].unique():
            cluster_data = df[df['content_cluster'] == cluster_id].copy()
            
            # Only analyze clusters with sufficient AI content
            ai_content_in_cluster = cluster_data['is_ai_content'].sum()
            if ai_content_in_cluster < 20:
                continue
            
            # Filter for AI content within cluster
            ai_cluster_data = cluster_data[cluster_data['is_ai_content'] == 1].copy()
            
            # Check if we have both treatment and control
            treatment_count = ai_cluster_data['has_ai_tag'].sum()
            control_count = len(ai_cluster_data) - treatment_count
            
            if treatment_count < 5 or control_count < 5:
                continue
            
            # Analyze tag effect within cluster
            treatment_click_rate = ai_cluster_data[ai_cluster_data['has_ai_tag'] == 1]['is_click'].mean()
            control_click_rate = ai_cluster_data[ai_cluster_data['has_ai_tag'] == 0]['is_click'].mean()
            
            print(f"\nCluster {cluster_id} (AI content only):")
            print(f"  Treatment (AI tag): {treatment_count:,} (CTR: {treatment_click_rate:.3f})")
            print(f"  Control (no AI tag): {control_count:,} (CTR: {control_click_rate:.3f})")
            print(f"  Tag effect: {treatment_click_rate - control_click_rate:.3f}")
            
            results.append({
                'cluster_id': cluster_id,
                'treatment_count': treatment_count,
                'control_count': control_count,
                'treatment_click_rate': treatment_click_rate,
                'control_click_rate': control_click_rate,
                'tag_effect': treatment_click_rate - control_click_rate
            })
        
        if results:
            avg_tag_effect = np.mean([r['tag_effect'] for r in results])
            print(f"\nAverage tag effect across clusters: {avg_tag_effect:.3f}")
        
        return results
    
    def generate_insights(self, stage1_results, stage2_results):
        """Generate insights from both stages"""
        print("\n" + "="*60)
        print("INSIGHTS AND RECOMMENDATIONS")
        print("="*60)
        
        print("\nStage 1 Results (Tag Effect):")
        if stage1_results:
            print(f"  Tag uplift effect: {stage1_results['tag_uplift_effect']:.4f}")
            print(f"  Click rate difference: {stage1_results['click_rate_difference']:.3f}")
            print(f"  Statistically significant: {stage1_results['significant']}")
            
            if stage1_results['tag_uplift_effect'] > 0:
                print("  AI tags have positive effect within AI content")
            else:
                print("  AI tags have negative effect within AI content")
        
        print("\nStage 2 Results (Content Effect):")
        if stage2_results:
            print(f"  Content type uplift effect: {stage2_results['content_uplift_effect']:.4f}")
            print(f"  Click rate difference: {stage2_results['click_rate_difference']:.3f}")
            print(f"  Statistically significant: {stage2_results['significant']}")
            
            if stage2_results['content_uplift_effect'] > 0:
                print("  AI content performs better than non-AI content")
            else:
                print("  AI content performs worse than non-AI content")
        
        print("\nStrategic Recommendations:")
        
        if stage1_results and stage2_results:
            tag_effect = stage1_results['tag_uplift_effect']
            content_effect = stage2_results['content_uplift_effect']
            
            if tag_effect > 0 and content_effect > 0:
                print("  Both AI content and AI tags are effective")
                print("     - Continue using AI content")
                print("     - Continue using AI tags")
                print("     - Consider expanding AI content strategy")
            
            elif tag_effect > 0 and content_effect <= 0:
                print("  AI tags are effective, but AI content quality needs improvement")
                print("     - Keep using AI tags")
                print("     - Improve AI content quality")
                print("     - Focus on content optimization")
            
            elif tag_effect <= 0 and content_effect > 0:
                print("  AI content is good, but AI tags are counterproductive")
                print("     - Continue using AI content")
                print("     - Remove or redesign AI tags")
                print("     - Test alternative labeling strategies")
            
            else:
                print("  Both AI content and AI tags need improvement")
                print("     - Redesign AI content strategy")
                print("     - Redesign AI tag strategy")
                print("     - Consider alternative approaches")
    
    def run_complete_analysis(self):
        """Run complete two-stage analysis"""
        print("TWO-STAGE ANALYSIS: SEPARATING TAG EFFECTS FROM CONTENT EFFECTS")
        print("="*80)
        
        # Load and prepare data
        df = self.load_data()
        df = self.identify_ai_content(df)
        df = self.identify_ai_tags(df)
        
        # Create content clusters for refined analysis
        df = self.create_similar_content_clusters(df)
        
        # Stage 1: Tag effect analysis
        stage1_results = self.stage1_tag_effect_analysis(df)
        
        # Stage 1 with clusters
        stage1_cluster_results = self.stage1_with_clusters(df)
        
        # Stage 2: Content type effect analysis
        stage2_results = self.stage2_content_effect_analysis(df)
        
        # Generate insights
        self.generate_insights(stage1_results, stage2_results)
        
        # Save results
        results_df = pd.DataFrame({
            'analysis_type': ['tag_effect', 'content_effect'],
            'uplift_effect': [
                stage1_results['tag_uplift_effect'] if stage1_results else None,
                stage2_results['content_uplift_effect'] if stage2_results else None
            ],
            'click_rate_difference': [
                stage1_results['click_rate_difference'] if stage1_results else None,
                stage2_results['click_rate_difference'] if stage2_results else None
            ],
            'p_value': [
                stage1_results['p_value'] if stage1_results else None,
                stage2_results['p_value'] if stage2_results else None
            ],
            'significant': [
                stage1_results['significant'] if stage1_results else None,
                stage2_results['significant'] if stage2_results else None
            ]
        })
        
        results_df.to_csv('two_stage_analysis_results.csv', index=False)
        print(f"\nResults saved to: two_stage_analysis_results.csv")
        
        return {
            'stage1': stage1_results,
            'stage1_clusters': stage1_cluster_results,
            'stage2': stage2_results
        }

def main():
    """Main function"""
    analyzer = TwoStageAnalysis()
    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    main()
