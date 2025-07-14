#!/usr/bin/env python3
"""
Uplift Treatment Labeling Script
Adds treatment labels to user-post click dataset based on content tags
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import os
from collections import defaultdict

class UpliftTreatmentLabeling:
    def __init__(self):
        """Initialize with treatment configurations"""
        self.df_samples = None
        self.post_tags = {}
        
        # Define treatment configurations
        self.treatments = {
            'ai_content': {
                'tags': ['machine-learning', 'deep-learning', 'neural-network', 
                        'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'nlp'],
                'description': 'AI/ML related content (precise)'
            },
            'web_development': {
                'tags': ['javascript', 'html', 'css', 'react', 'angular', 'vue', 'nodejs', 
                        'web-development', 'frontend', 'backend', 'api'],
                'description': 'Web development content'
            },
            'mobile_development': {
                'tags': ['android', 'ios', 'swift', 'kotlin', 'react-native', 'flutter', 
                        'mobile-development', 'app-development'],
                'description': 'Mobile development content'
            },
            'database': {
                'tags': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'database', 
                        'nosql', 'oracle'],
                'description': 'Database related content'
            }
        }
        self.df_users = None
        self.df_posts = None
        self.user_tag_map = None
        self.post_tags_map = None
    
    def parse_tags(self, tags_str):
        """Parse tags from pipe-separated format: '|tag1|tag2|' -> ['tag1', 'tag2']"""
        if not tags_str or tags_str == '':
            return []
        # Remove leading and trailing pipes, then split by pipe
        cleaned_tags = tags_str.strip('|')
        if not cleaned_tags:
            return []
        return [tag for tag in cleaned_tags.split('|') if tag]
    
    def load_data(self):
        """Load user-post samples and post tags"""
        print("=== Loading Data ===")
        
        # Load user-post samples
        print("Loading user-post samples...")
        self.df_samples = pd.read_csv('user_post_click_samples.csv')
        print(f"Loaded {len(self.df_samples)} samples")
        
        # Load post tags from XML
        print("Loading post tags...")
        tree = ET.parse('data/Posts.xml')
        root = tree.getroot()
        
        for row in root:
            post_id = row.get('Id')
            tags_str = row.get('Tags', '')
            parsed_tags = self.parse_tags(tags_str)
            self.post_tags[post_id] = parsed_tags
        
        print(f"Loaded tags for {len(self.post_tags)} posts")
        
        # Debug: Show some sample parsed tags
        print("Sample parsed tags:")
        sample_posts = list(self.post_tags.items())[:5]
        for post_id, tags in sample_posts:
            print(f"  Post {post_id}: {tags}")
        
        return True

    def load_feature_tables(self):
        """Load user and post features from preprocessing output"""
        print("Loading user and post features from preprocessing...")
        self.df_users = pd.read_csv('output/user_features.csv')
        self.df_posts = pd.read_csv('output/post_features.csv')
        print(f"Loaded {len(self.df_users)} user features, {len(self.df_posts)} post features")
        return True

    def build_tag_maps(self):
        """Build user tag map and post tag map from samples"""
        print("Building user and post tag maps...")
        self.user_tag_map = defaultdict(set)
        self.post_tags_map = {}
        for _, row in self.df_samples.iterrows():
            self.user_tag_map[row['user_id']].update(row['post_tags'])
            self.post_tags_map[str(row['post_id'])] = row['post_tags']
        print(f"Built user_tag_map for {len(self.user_tag_map)} users, post_tags_map for {len(self.post_tags_map)} posts")
        return True

    def compute_user_ai_interest(self, ai_tags):
        """Compute user AI/ML interest score based on past interactions"""
        # More sophisticated AI interest calculation
        self.df_samples['is_ai_post'] = self.df_samples['post_tags'].apply(
            lambda tags: any(tag in ai_tags for tag in tags)
        )
        
        # Calculate user AI interest with more weight for recent interactions
        user_ai_interest = self.df_samples.groupby('user_id').agg({
            'is_ai_post': ['mean', 'sum', 'count']
        }).reset_index()
        user_ai_interest.columns = ['user_id', 'user_ai_interest_score', 'user_ai_interactions', 'user_total_interactions']
        
        # Calculate weighted AI interest score
        user_ai_interest['user_ai_interest_weighted'] = (
            user_ai_interest['user_ai_interest_score'] * 
            (user_ai_interest['user_ai_interactions'] / user_ai_interest['user_total_interactions'])
        )
        
        self.df_samples = self.df_samples.merge(user_ai_interest, on='user_id', how='left')
        return True

    def compute_user_post_tag_overlap(self):
        def overlap(row):
            user_tags = self.user_tag_map.get(row['user_id'], set())
            post_tags = set(self.post_tags_map.get(str(row['post_id']), []))
            return len(user_tags & post_tags)
        self.df_samples['user_post_tag_overlap'] = self.df_samples.apply(overlap, axis=1)
        return True

    def compute_user_previous_ai_click_rate(self):
        user_prev_ai_click = self.df_samples.groupby('user_id')['treatment_ai_content'].mean().reset_index()
        user_prev_ai_click.rename(columns={'treatment_ai_content': 'user_previous_ai_click_rate'}, inplace=True)
        self.df_samples = self.df_samples.merge(user_prev_ai_click, on='user_id', how='left')
        return True

    def add_interaction_term(self):
        self.df_samples['ai_interest_x_treatment'] = self.df_samples['user_ai_interest_score'] * self.df_samples['treatment_ai_content']
        return True

    def merge_real_features(self):
        print("Merging real user and post features...")
        self.df_samples = self.df_samples.merge(self.df_users, left_on='user_id', right_on='Id', how='left')
        self.df_samples = self.df_samples.merge(self.df_posts, left_on='post_id', right_on='Id_x', how='left')
        return True

    def add_treatment_labels(self):
        """Add treatment labels based on post tags"""
        print("\n=== Adding Treatment Labels ===")
        
        # Add post tags to samples - convert post_id to string for mapping
        self.df_samples['post_id_str'] = self.df_samples['post_id'].astype(str)
        self.df_samples['post_tags'] = self.df_samples['post_id_str'].map(self.post_tags)
        self.df_samples['post_tags'] = self.df_samples['post_tags'].fillna('[]')
        
        # Debug: Check some sample tags
        print("Sample post tags:")
        sample_tags = self.df_samples['post_tags'].head(5).tolist()
        for i, tags in enumerate(sample_tags):
            print(f"  Sample {i+1}: {tags}")
        
        # Add treatment labels for each configuration
        for treatment_name, config in self.treatments.items():
            print(f"Adding {treatment_name} treatment...")
            
            # Check if any tag matches treatment tags (case-insensitive)
            def check_treatment_match(tags):
                if not tags or tags == '[]':
                    return False
                # Convert tags to lowercase for matching
                tags_lower = [tag.lower() for tag in tags]
                config_tags_lower = [tag.lower() for tag in config['tags']]
                return any(tag in config_tags_lower for tag in tags_lower)
            
            self.df_samples[f'treatment_{treatment_name}'] = self.df_samples['post_tags'].apply(check_treatment_match).astype(int)
            
            # Print statistics
            treatment_count = self.df_samples[f'treatment_{treatment_name}'].sum()
            treatment_rate = treatment_count / len(self.df_samples)
            print(f"  - {treatment_name}: {treatment_count} samples ({treatment_rate:.3f})")
            
            # Debug: Show some matched samples
            if treatment_count > 0:
                matched_samples = self.df_samples[self.df_samples[f'treatment_{treatment_name}'] == 1].head(3)
                print(f"  - Sample matched tags:")
                for _, row in matched_samples.iterrows():
                    print(f"    Post {row['post_id']}: {row['post_tags']}")
            else:
                print(f"  - No samples matched for {treatment_name}")
        
        # Debug: Show overall treatment distribution
        print("\nOverall treatment distribution:")
        for treatment_name in self.treatments.keys():
            treatment_col = f'treatment_{treatment_name}'
            if treatment_col in self.df_samples.columns:
                print(f"  {treatment_name}: {self.df_samples[treatment_col].value_counts().to_dict()}")
        
        return True
    
    def create_uplift_features(self):
        """Create features for uplift modeling"""
        print("\n=== Creating Uplift Features ===")
        
        # Use click behavior as response
        self.df_samples['response'] = self.df_samples['is_click']
        
        # Create interaction features
        self.df_samples['user_post_interaction'] = (
            self.df_samples['user_post_count'] * self.df_samples['post_title_length']
        )
        
        # Create engagement score
        self.df_samples['engagement_score'] = (
            self.df_samples['is_click'] * 1.0 + 
            self.df_samples['interest_score'] * 0.5
        )
        
        print("Uplift features created!")
        return True

    def add_content_quality_features(self):
        """Add content quality and engagement features"""
        # Content quality based on post features
        if 'Score' in self.df_samples.columns and 'ViewCount' in self.df_samples.columns:
            self.df_samples['content_quality_score'] = (
                self.df_samples['Score'] / (self.df_samples['ViewCount'] + 1)
            )
        
        # Engagement rate
        if 'total_votes' in self.df_samples.columns and 'post_age_days' in self.df_samples.columns:
            self.df_samples['engagement_rate'] = (
                self.df_samples['total_votes'] / (self.df_samples['post_age_days'] + 1)
            )
        
        # Content complexity
        if 'post_length' in self.df_samples.columns and 'title_length' in self.df_samples.columns:
            self.df_samples['content_complexity'] = (
                self.df_samples['post_length'] / (self.df_samples['title_length'] + 1)
            )
        
        return True

    def add_user_behavior_features(self):
        """Add user behavior and interaction features"""
        # User activity level
        if 'user_post_count' in self.df_samples.columns:
            self.df_samples['user_activity_level'] = pd.cut(
                self.df_samples['user_post_count'],
                bins=[0, 1, 5, 10, 25, float('inf')],
                labels=['Inactive', 'Low', 'Medium', 'High', 'Very High']
            )
        
        # User reputation level
        if 'user_reputation' in self.df_samples.columns:
            self.df_samples['user_reputation_level'] = pd.cut(
                self.df_samples['user_reputation'],
                bins=[0, 100, 500, 1000, 5000, float('inf')],
                labels=['New', 'Beginner', 'Intermediate', 'Advanced', 'Expert']
            )
        
        return True
    
    def analyze_treatment_effects(self):
        """Analyze treatment effects"""
        print("\n=== Treatment Effect Analysis ===")
        
        for treatment_name in self.treatments.keys():
            treatment_col = f'treatment_{treatment_name}'
            
            if treatment_col not in self.df_samples.columns:
                continue
            
            # Split into treatment and control groups
            treatment_group = self.df_samples[self.df_samples[treatment_col] == 1]
            control_group = self.df_samples[self.df_samples[treatment_col] == 0]
            
            if len(treatment_group) > 0 and len(control_group) > 0:
                treatment_rate = treatment_group['is_click'].mean()
                control_rate = control_group['is_click'].mean()
                uplift = treatment_rate - control_rate
                
                print(f"\n{treatment_name.upper()} Treatment Effect:")
                print(f"  - Treatment group: {len(treatment_group)} samples")
                print(f"  - Control group: {len(control_group)} samples")
                print(f"  - Treatment response rate: {treatment_rate:.3f}")
                print(f"  - Control response rate: {control_rate:.3f}")
                print(f"  - Uplift: {uplift:.3f}")
        
        return True
    
    def save_uplift_dataset(self, output_path='uplift_dataset.csv'):
        """Save the uplift dataset with improved feature selection"""
        print(f"\n=== Saving Uplift Dataset ===")
        
        # Define feature columns that should exist
        potential_feature_columns = [
            'user_id', 'post_id', 'is_click', 'response', 'engagement_score',
            'user_post_count', 'user_account_age_days', 'post_title_length', 
            'post_tag_count', 'interest_score', 'user_post_interaction',
            'user_ai_interest_score', 'user_ai_interest_weighted', 'user_ai_interactions',
            'content_quality_score', 'engagement_rate', 'content_complexity',
            'user_activity_level', 'user_reputation_level',
            'user_reputation', 'Score', 'ViewCount', 'AnswerCount', 'CommentCount', 
            'title_length', 'post_length', 'num_tags', 'post_age_days', 
            'total_votes', 'upvotes', 'user_post_tag_overlap', 
            'user_previous_ai_click_rate', 'ai_interest_x_treatment'
        ]
        
        # Add treatment columns
        treatment_columns = [f'treatment_{name}' for name in self.treatments.keys()]
        potential_feature_columns.extend(treatment_columns)
        
        # Only keep columns that actually exist
        feature_columns = [col for col in potential_feature_columns if col in self.df_samples.columns]
        
        print(f"Available features: {len(feature_columns)}")
        print(f"Feature columns: {feature_columns[:10]}...")  # Show first 10
        
        # Create and save uplift dataset
        uplift_dataset = self.df_samples[feature_columns].copy()
        uplift_dataset.to_csv(output_path, index=False)
        
        print(f"Uplift dataset saved to {output_path}")
        print(f"Dataset shape: {uplift_dataset.shape}")
        
        return uplift_dataset
    
    def save_standard_uplift_table(self, output_path='uplift_model_data.csv'):
        """Save standard uplift modeling table: user_id, post_id, treatment, response (AI only)"""
        print(f"\n=== Saving Standard Uplift Table (AI treatment only) ===")
        # treatment=1: AI-related content, 0: normal content
        self.df_samples['treatment'] = self.df_samples['treatment_ai_content']
        self.df_samples['response'] = self.df_samples['is_click']
        uplift_table = self.df_samples[['user_id', 'post_id', 'treatment', 'response']].copy()
        uplift_table.to_csv(output_path, index=False)
        print(f"Standard uplift table saved to {output_path}, shape: {uplift_table.shape}")
        return uplift_table

    def save_final_uplift_table(self, output_path='uplift_model_data.csv'):
        """Save final uplift modeling table with improved features"""
        print(f"\n=== Saving Final Uplift Modeling Table ===")
        
        # Define improved feature set
        potential_uplift_features = [
            'user_ai_interest_score', 'user_ai_interest_weighted', 'user_ai_interactions',
            'user_reputation', 'user_post_count', 'user_account_age_days',
            'total_badges', 'gold_badges', 'silver_badges', 'bronze_badges', 
            'unique_badge_types', 'badge_rate_per_day', 'recent_badges_30d',
            'badge_quality_score', 'Score', 'ViewCount', 'AnswerCount', 'CommentCount', 
            'title_length', 'post_length', 'num_tags', 'post_age_days', 
            'total_votes', 'upvotes', 'user_post_tag_overlap', 
            'user_previous_ai_click_rate', 'ai_interest_x_treatment',
            'content_quality_score', 'engagement_rate', 'content_complexity',
            'user_activity_level', 'user_reputation_level',
            'treatment_ai_content', 'response'
        ]
        
        # Only keep columns that exist
        uplift_features = [col for col in potential_uplift_features if col in self.df_samples.columns]
        
        # Fill missing values properly for different data types
        for col in uplift_features:
            if col in self.df_samples.columns:
                if self.df_samples[col].dtype.name == 'category':
                    # For categorical columns, fill with the most common category
                    most_common = self.df_samples[col].mode().iloc[0] if len(self.df_samples[col].mode()) > 0 else self.df_samples[col].iloc[0]
                    self.df_samples[col] = self.df_samples[col].fillna(most_common)
                else:
                    # For numeric columns, fill with 0
                    self.df_samples[col] = self.df_samples[col].fillna(0)
        
        final_uplift_df = self.df_samples[uplift_features].copy()
        final_uplift_df.to_csv(output_path, index=False)
        
        print(f"Final uplift modeling table saved to {output_path}, shape: {final_uplift_df.shape}")
        print(f"Features used: {len(uplift_features)}")
        return final_uplift_df

    def run_pipeline(self):
        """Run the complete uplift treatment labeling pipeline with improvements"""
        print("=== Uplift Treatment Labeling Pipeline (Improved) ===")
        
        # Step 1: Load data
        self.load_data()
        self.load_feature_tables()
        self.add_treatment_labels()
        self.merge_real_features()
        self.build_tag_maps()
        
        # Step 2: Improved feature engineering
        ai_tags = self.treatments['ai_content']['tags']
        self.compute_user_ai_interest(ai_tags)
        self.compute_user_post_tag_overlap()
        self.compute_user_previous_ai_click_rate()
        self.add_interaction_term()
        self.add_content_quality_features()
        self.add_user_behavior_features()
        self.create_uplift_features()
        
        # Step 3: Analysis and saving
        self.analyze_treatment_effects()
        self.save_uplift_dataset()
        self.save_standard_uplift_table()
        self.save_final_uplift_table()
        
        print("\n=== Pipeline Complete ===")
        return True

def main():
    """Main function"""
    labeler = UpliftTreatmentLabeling()
    uplift_dataset = labeler.run_pipeline()
    return uplift_dataset

if __name__ == "__main__":
    main() 