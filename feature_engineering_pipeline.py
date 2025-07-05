#!/usr/bin/env python3
"""
Feature Engineering Pipeline for StackExchange Data
Creates a modeling-ready dataset from raw XML logs
"""

import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineeringPipeline:
    def __init__(self, data_dir='data'):
        """Initialize the feature engineering pipeline"""
        self.data_dir = data_dir
        self.df_posts = None
        self.df_users = None
        self.df_tags = None
        self.df_votes = None
        self.df_comments = None
        self.df_badges = None
        self.df_combined = None
        self.feature_table = None
        
    def parse_xml(self, path):
        """Parse XML file and return DataFrame"""
        print(f"Parsing {path}...")
        tree = ET.parse(path)
        root = tree.getroot()
        df = pd.DataFrame([row.attrib for row in root])
        print(f"Loaded {len(df)} records from {path}")
        return df
    
    def load_data(self):
        """Load all XML data files"""
        print("=== Loading Data Files ===")
        
        # Load core files
        self.df_posts = self.parse_xml(os.path.join(self.data_dir, 'Posts.xml'))
        self.df_users = self.parse_xml(os.path.join(self.data_dir, 'Users.xml'))
        self.df_tags = self.parse_xml(os.path.join(self.data_dir, 'Tags.xml'))
        self.df_votes = self.parse_xml(os.path.join(self.data_dir, 'Votes.xml'))
        
        # Load additional files if available
        try:
            self.df_comments = self.parse_xml(os.path.join(self.data_dir, 'Comments.xml'))
        except:
            print("Comments.xml not found, skipping...")
            
        try:
            self.df_badges = self.parse_xml(os.path.join(self.data_dir, 'Badges.xml'))
        except:
            print("Badges.xml not found, skipping...")
        
        print("Data loading completed!")
        return True
    
    def clean_data(self):
        """Clean and prepare the data"""
        print("\n=== Data Cleaning ===")
        
        # Clean posts data
        print("Cleaning posts data...")
        self.df_posts = self.df_posts.dropna(subset=['Title', 'OwnerUserId']).copy()
        self.df_posts['Tags'] = self.df_posts['Tags'].fillna('')
        
        # Parse tags properly
        self.df_posts['TagList'] = self.df_posts['Tags'].apply(
            lambda x: re.findall(r'<(.*?)>', x) if pd.notna(x) else []
        )
        
        # Clean users data
        print("Cleaning users data...")
        self.df_users['user_reputation'] = pd.to_numeric(self.df_users['Reputation'], errors='coerce').fillna(0)
        self.df_users['Id'] = self.df_users['Id'].astype(str)
        
        # Clean tags data
        print("Cleaning tags data...")
        self.df_tags['Count'] = pd.to_numeric(self.df_tags['Count'], errors='coerce').fillna(0)
        
        # Clean votes data
        print("Cleaning votes data...")
        self.df_votes['VoteTypeId'] = pd.to_numeric(self.df_votes['VoteTypeId'], errors='coerce')
        self.df_votes['PostId'] = pd.to_numeric(self.df_votes['PostId'], errors='coerce')
        
        print("Data cleaning completed!")
        return True
    
    def merge_data(self):
        """Merge all data sources"""
        print("\n=== Merging Data ===")
        
        # Merge posts and users
        self.df_combined = self.df_posts.merge(
            self.df_users, 
            left_on='OwnerUserId', 
            right_on='Id', 
            how='left'
        )
        
        print(f"Combined dataset size: {len(self.df_combined)} records")
        return True
    
    def create_user_level_features(self):
        """Create user-level features"""
        print("\n=== Creating User-Level Features ===")
        
        # User account age
        self.df_combined['CreationDate_y'] = pd.to_datetime(self.df_combined['CreationDate_y'])
        self.df_combined['user_account_age_days'] = (
            pd.Timestamp.now() - self.df_combined['CreationDate_y']
        ).dt.days
        
        # User reputation (already cleaned)
        self.df_combined['user_reputation'] = self.df_combined['user_reputation'].fillna(0)
        
        # User post count
        user_post_counts = self.df_combined.groupby('OwnerUserId').size().reset_index()
        user_post_counts.columns = ['OwnerUserId', 'user_post_count']
        self.df_combined = self.df_combined.merge(user_post_counts, on='OwnerUserId', how='left')
        
        # User badge count (if badges data available)
        if self.df_badges is not None:
            user_badge_counts = self.df_badges.groupby('UserId').size().reset_index()
            user_badge_counts.columns = ['UserId', 'user_badge_count']
            self.df_combined = self.df_combined.merge(
                user_badge_counts, 
                left_on='OwnerUserId', 
                right_on='UserId', 
                how='left'
            )
            self.df_combined['user_badge_count'] = self.df_combined['user_badge_count'].fillna(0)
        else:
            self.df_combined['user_badge_count'] = 0
        
        print("User-level features created!")
        return True
    
    def create_post_level_features(self):
        """Create post-level features"""
        print("\n=== Creating Post-Level Features ===")
        
        # Post length
        self.df_combined['post_length'] = self.df_combined['Body'].fillna('').apply(
            lambda x: len(str(x).split())
        )
        
        # Post age
        self.df_combined['CreationDate_x'] = pd.to_datetime(self.df_combined['CreationDate_x'])
        self.df_combined['post_age_days'] = (
            pd.Timestamp.now() - self.df_combined['CreationDate_x']
        ).dt.days
        
        # Title length
        self.df_combined['title_length'] = self.df_combined['Title'].apply(
            lambda x: len(str(x).split())
        )
        
        # Number of tags
        self.df_combined['num_tags'] = self.df_combined['TagList'].apply(len)
        
        # First tag
        self.df_combined['first_tag'] = self.df_combined['TagList'].apply(
            lambda x: x[0] if len(x) > 0 else 'None'
        )
        
        # Check if first tag is popular
        top_tags = self.df_tags.sort_values('Count', ascending=False).head(20)
        self.df_combined['is_first_tag_popular'] = self.df_combined['first_tag'].isin(
            top_tags['TagName'].tolist()
        )
        
        print("Post-level features created!")
        return True
    
    def create_interaction_level_features(self):
        """Create interaction-level features"""
        print("\n=== Creating Interaction-Level Features ===")
        
        # Vote statistics
        vote_stats = self.df_votes.groupby('PostId').agg({
            'VoteTypeId': 'count'
        }).reset_index()
        vote_stats.columns = ['PostId', 'total_votes']
        
        # Upvotes (VoteTypeId == 2)
        upvotes = self.df_votes[self.df_votes['VoteTypeId'] == 2].groupby('PostId').size().reset_index()
        upvotes.columns = ['PostId', 'upvotes']
        
        # Downvotes (VoteTypeId == 3)
        downvotes = self.df_votes[self.df_votes['VoteTypeId'] == 3].groupby('PostId').size().reset_index()
        downvotes.columns = ['PostId', 'downvotes']
        
        # Merge vote statistics
        self.df_combined = self.df_combined.merge(
            vote_stats, left_on='Id_x', right_on='PostId', how='left'
        )
        self.df_combined = self.df_combined.merge(
            upvotes, left_on='Id_x', right_on='PostId', how='left'
        )
        self.df_combined = self.df_combined.merge(
            downvotes, left_on='Id_x', right_on='PostId', how='left'
        )
        
        # Fill NaN values
        self.df_combined['total_votes'] = self.df_combined['total_votes'].fillna(0)
        self.df_combined['upvotes'] = self.df_combined['upvotes'].fillna(0)
        self.df_combined['downvotes'] = self.df_combined['downvotes'].fillna(0)
        
        # Calculate vote ratios
        self.df_combined['upvote_ratio'] = self.df_combined.apply(
            lambda row: row['upvotes'] / row['total_votes'] if row['total_votes'] > 0 else 0, 
            axis=1
        )
        
        self.df_combined['downvote_ratio'] = self.df_combined.apply(
            lambda row: row['downvotes'] / row['total_votes'] if row['total_votes'] > 0 else 0, 
            axis=1
        )
        
        # Comment count (if available)
        if self.df_comments is not None:
            comment_counts = self.df_comments.groupby('PostId').size().reset_index()
            comment_counts.columns = ['PostId', 'comment_count']
            self.df_combined = self.df_combined.merge(
                comment_counts, left_on='Id_x', right_on='PostId', how='left'
            )
            self.df_combined['comment_count'] = self.df_combined['comment_count'].fillna(0)
        else:
            self.df_combined['comment_count'] = 0
        
        # Post engagement score (weighted combination)
        self.df_combined['post_engagement'] = (
            self.df_combined['upvotes'] * 1 + 
            self.df_combined['comment_count'] * 0.5 + 
            self.df_combined['total_votes'] * 0.3
        )
        
        print("Interaction-level features created!")
        return True
    
    def create_semantic_features(self):
        """Create semantic features using sentence embeddings"""
        print("\n=== Creating Semantic Features ===")
        
        try:
            # Load sentence transformer model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get titles for embedding
            titles = self.df_combined['Title'].fillna('').tolist()
            print(f"Creating embeddings for {len(titles)} titles...")
            
            # Create embeddings
            embeddings = model.encode(titles, show_progress_bar=True)
            
            # Cluster embeddings
            print("Clustering embeddings...")
            kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
            title_clusters = kmeans.fit_predict(embeddings)
            
            # Add cluster feature
            self.df_combined['title_embedding_cluster'] = title_clusters
            
            # Create cluster labels
            cluster_labels = [f"topic_cluster_{i}" for i in range(15)]
            self.df_combined['title_topic_label'] = [
                cluster_labels[cluster] for cluster in title_clusters
            ]
            
            print("Semantic features created!")
            return True
            
        except Exception as e:
            print(f"Error creating semantic features: {e}")
            # Add dummy features if embedding fails
            self.df_combined['title_embedding_cluster'] = 0
            self.df_combined['title_topic_label'] = 'topic_cluster_0'
            return False
    
    def create_modeling_targets(self):
        """Create modeling targets for different tasks"""
        print("\n=== Creating Modeling Targets ===")
        
        # CTR Prediction Target (proxy: upvote ratio > median)
        median_upvote_ratio = self.df_combined['upvote_ratio'].median()
        self.df_combined['is_click'] = (self.df_combined['upvote_ratio'] > median_upvote_ratio).astype(int)
        
        # Retention Target (proxy: user has multiple posts)
        self.df_combined['is_retained'] = (self.df_combined['user_post_count'] > 1).astype(int)
        
        # Uplift Modeling Targets
        # Treatment: AI-related content
        ai_tags = ['python', 'machine-learning', 'artificial-intelligence', 'deep-learning', 
                  'neural-network', 'tensorflow', 'pytorch', 'scikit-learn', 'nlp', 'computer-vision']
        
        self.df_combined['treatment'] = self.df_combined['TagList'].apply(
            lambda tags: any(tag.lower() in ai_tags for tag in tags)
        ).astype(int)
        
        # Response: engagement level
        self.df_combined['response'] = (
            self.df_combined['upvotes'] + self.df_combined['comment_count']
        )
        
        print("Modeling targets created!")
        return True
    
    def create_advanced_features(self):
        """Create advanced features"""
        print("\n=== Creating Advanced Features ===")
        
        # Post complexity (title length + body length)
        self.df_combined['post_complexity'] = (
            self.df_combined['title_length'] + self.df_combined['post_length']
        )
        
        # User experience level
        self.df_combined['user_experience_level'] = pd.cut(
            self.df_combined['user_reputation'],
            bins=[-np.inf, 100, 1000, 10000, np.inf],
            labels=['new', 'intermediate', 'experienced', 'expert']
        )
        
        # Post quality score
        self.df_combined['post_quality_score'] = (
            self.df_combined['upvote_ratio'] * 0.4 +
            (self.df_combined['comment_count'] / (self.df_combined['comment_count'].max() + 1)) * 0.3 +
            (self.df_combined['num_tags'] / 10) * 0.3
        )
        
        # Time-based features
        self.df_combined['post_hour'] = self.df_combined['CreationDate_x'].dt.hour
        self.df_combined['post_day_of_week'] = self.df_combined['CreationDate_x'].dt.dayofweek
        self.df_combined['post_month'] = self.df_combined['CreationDate_x'].dt.month
        
        print("Advanced features created!")
        return True
    
    def create_final_feature_table(self):
        """Create the final feature table"""
        print("\n=== Creating Final Feature Table ===")
        
        # Select relevant columns for modeling
        feature_columns = [
            # User-level features
            'user_account_age_days', 'user_reputation', 'user_post_count', 'user_badge_count',
            'user_experience_level',
            
            # Post-level features
            'post_length', 'post_age_days', 'title_length', 'num_tags', 'first_tag',
            'is_first_tag_popular', 'post_complexity', 'post_quality_score',
            'post_hour', 'post_day_of_week', 'post_month',
            
            # Interaction-level features
            'total_votes', 'upvotes', 'downvotes', 'upvote_ratio', 'downvote_ratio',
            'comment_count', 'post_engagement',
            
            # Semantic features
            'title_embedding_cluster', 'title_topic_label',
            
            # Modeling targets
            'is_click', 'is_retained', 'treatment', 'response'
        ]
        
        # Create final feature table
        self.feature_table = self.df_combined[feature_columns].copy()
        
        # Handle missing values
        numeric_columns = self.feature_table.select_dtypes(include=[np.number]).columns
        self.feature_table[numeric_columns] = self.feature_table[numeric_columns].fillna(0)
        
        categorical_columns = self.feature_table.select_dtypes(include=['object']).columns
        self.feature_table[categorical_columns] = self.feature_table[categorical_columns].fillna('Unknown')
        
        print(f"Final feature table created with {len(self.feature_table)} rows and {len(self.feature_table.columns)} columns")
        return True
    
    def save_feature_table(self, output_path='feature_table.csv'):
        """Save the feature table to CSV"""
        print(f"\n=== Saving Feature Table ===")
        
        self.feature_table.to_csv(output_path, index=False)
        print(f"Feature table saved to {output_path}")
        
        # Also save a sample for inspection
        sample_path = 'feature_table_sample.csv'
        self.feature_table.head(1000).to_csv(sample_path, index=False)
        print(f"Sample feature table saved to {sample_path}")
        
        return True
    
    def create_feature_documentation(self, output_path='features.md'):
        """Create feature documentation"""
        print(f"\n=== Creating Feature Documentation ===")
        
        documentation = """# Feature Documentation

This document describes all features in the modeling-ready dataset created from StackExchange data.

## Dataset Overview
- **Total Records**: {total_records}
- **Total Features**: {total_features}
- **Target Variables**: is_click, is_retained, treatment, response

## Feature Descriptions

""".format(
            total_records=len(self.feature_table),
            total_features=len(self.feature_table.columns)
        )
        
        # Add feature descriptions
        feature_descriptions = {
            # User-level features
            'user_account_age_days': {
                'type': 'Numerical',
                'description': 'Number of days since user account creation',
                'sample': f"Mean: {self.feature_table['user_account_age_days'].mean():.1f} days"
            },
            'user_reputation': {
                'type': 'Numerical',
                'description': 'User reputation score from StackExchange',
                'sample': f"Mean: {self.feature_table['user_reputation'].mean():.1f}"
            },
            'user_post_count': {
                'type': 'Numerical',
                'description': 'Total number of posts by the user',
                'sample': f"Mean: {self.feature_table['user_post_count'].mean():.1f}"
            },
            'user_badge_count': {
                'type': 'Numerical',
                'description': 'Total number of badges earned by the user',
                'sample': f"Mean: {self.feature_table['user_badge_count'].mean():.1f}"
            },
            'user_experience_level': {
                'type': 'Categorical',
                'description': 'User experience level based on reputation',
                'sample': self.feature_table['user_experience_level'].value_counts().to_dict()
            },
            
            # Post-level features
            'post_length': {
                'type': 'Numerical',
                'description': 'Number of words in the post body',
                'sample': f"Mean: {self.feature_table['post_length'].mean():.1f} words"
            },
            'post_age_days': {
                'type': 'Numerical',
                'description': 'Number of days since post creation',
                'sample': f"Mean: {self.feature_table['post_age_days'].mean():.1f} days"
            },
            'title_length': {
                'type': 'Numerical',
                'description': 'Number of words in the post title',
                'sample': f"Mean: {self.feature_table['title_length'].mean():.1f} words"
            },
            'num_tags': {
                'type': 'Numerical',
                'description': 'Number of tags assigned to the post',
                'sample': f"Mean: {self.feature_table['num_tags'].mean():.1f} tags"
            },
            'first_tag': {
                'type': 'Categorical',
                'description': 'First tag in the tag list',
                'sample': self.feature_table['first_tag'].value_counts().head(5).to_dict()
            },
            'is_first_tag_popular': {
                'type': 'Categorical',
                'description': 'Whether the first tag is in the top 20 most popular tags',
                'sample': self.feature_table['is_first_tag_popular'].value_counts().to_dict()
            },
            'post_complexity': {
                'type': 'Numerical',
                'description': 'Combined complexity score (title + body length)',
                'sample': f"Mean: {self.feature_table['post_complexity'].mean():.1f}"
            },
            'post_quality_score': {
                'type': 'Numerical',
                'description': 'Weighted quality score based on engagement metrics',
                'sample': f"Mean: {self.feature_table['post_quality_score'].mean():.3f}"
            },
            'post_hour': {
                'type': 'Numerical',
                'description': 'Hour of day when post was created (0-23)',
                'sample': f"Mean: {self.feature_table['post_hour'].mean():.1f}"
            },
            'post_day_of_week': {
                'type': 'Numerical',
                'description': 'Day of week when post was created (0=Monday, 6=Sunday)',
                'sample': f"Mean: {self.feature_table['post_day_of_week'].mean():.1f}"
            },
            'post_month': {
                'type': 'Numerical',
                'description': 'Month when post was created (1-12)',
                'sample': f"Mean: {self.feature_table['post_month'].mean():.1f}"
            },
            
            # Interaction-level features
            'total_votes': {
                'type': 'Numerical',
                'description': 'Total number of votes on the post',
                'sample': f"Mean: {self.feature_table['total_votes'].mean():.1f}"
            },
            'upvotes': {
                'type': 'Numerical',
                'description': 'Number of upvotes on the post',
                'sample': f"Mean: {self.feature_table['upvotes'].mean():.1f}"
            },
            'downvotes': {
                'type': 'Numerical',
                'description': 'Number of downvotes on the post',
                'sample': f"Mean: {self.feature_table['downvotes'].mean():.1f}"
            },
            'upvote_ratio': {
                'type': 'Numerical',
                'description': 'Ratio of upvotes to total votes',
                'sample': f"Mean: {self.feature_table['upvote_ratio'].mean():.3f}"
            },
            'downvote_ratio': {
                'type': 'Numerical',
                'description': 'Ratio of downvotes to total votes',
                'sample': f"Mean: {self.feature_table['downvote_ratio'].mean():.3f}"
            },
            'comment_count': {
                'type': 'Numerical',
                'description': 'Number of comments on the post',
                'sample': f"Mean: {self.feature_table['comment_count'].mean():.1f}"
            },
            'post_engagement': {
                'type': 'Numerical',
                'description': 'Weighted engagement score combining votes and comments',
                'sample': f"Mean: {self.feature_table['post_engagement'].mean():.1f}"
            },
            
            # Semantic features
            'title_embedding_cluster': {
                'type': 'Numerical',
                'description': 'Cluster ID from semantic embedding of post title',
                'sample': f"Range: 0-{self.feature_table['title_embedding_cluster'].max()}"
            },
            'title_topic_label': {
                'type': 'Categorical',
                'description': 'Topic label based on semantic clustering of titles',
                'sample': self.feature_table['title_topic_label'].value_counts().to_dict()
            },
            
            # Modeling targets
            'is_click': {
                'type': 'Binary',
                'description': 'CTR prediction target: whether post has above-median upvote ratio',
                'sample': f"Positive rate: {self.feature_table['is_click'].mean():.3f}"
            },
            'is_retained': {
                'type': 'Binary',
                'description': 'Retention prediction target: whether user has multiple posts',
                'sample': f"Positive rate: {self.feature_table['is_retained'].mean():.3f}"
            },
            'treatment': {
                'type': 'Binary',
                'description': 'Uplift modeling treatment: whether post contains AI-related tags',
                'sample': f"Treatment rate: {self.feature_table['treatment'].mean():.3f}"
            },
            'response': {
                'type': 'Numerical',
                'description': 'Uplift modeling response: engagement level (upvotes + comments)',
                'sample': f"Mean: {self.feature_table['response'].mean():.1f}"
            }
        }
        
        # Add feature descriptions to documentation
        for feature_name, feature_info in feature_descriptions.items():
            documentation += f"### Feature Name: {feature_name}\n"
            documentation += f"- Type: {feature_info['type']}\n"
            documentation += f"- Description: {feature_info['description']}\n"
            documentation += f"- Sample: {feature_info['sample']}\n\n"
        
        # Save documentation
        with open(output_path, 'w') as f:
            f.write(documentation)
        
        print(f"Feature documentation saved to {output_path}")
        return True
    
    def run_full_pipeline(self):
        """Run the complete feature engineering pipeline"""
        print("=== Starting Feature Engineering Pipeline ===")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Clean data
        self.clean_data()
        
        # Step 3: Merge data
        self.merge_data()
        
        # Step 4: Create features
        self.create_user_level_features()
        self.create_post_level_features()
        self.create_interaction_level_features()
        self.create_semantic_features()
        self.create_advanced_features()
        
        # Step 5: Create modeling targets
        self.create_modeling_targets()
        
        # Step 6: Create final feature table
        self.create_final_feature_table()
        
        # Step 7: Save results
        self.save_feature_table()
        self.create_feature_documentation()
        
        print("\n=== Feature Engineering Pipeline Complete ===")
        print(f"Final dataset: {len(self.feature_table)} rows, {len(self.feature_table.columns)} columns")
        
        return self.feature_table

def main():
    """Main function to run the feature engineering pipeline"""
    pipeline = FeatureEngineeringPipeline()
    feature_table = pipeline.run_full_pipeline()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Dataset shape: {feature_table.shape}")
    print(f"Memory usage: {feature_table.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Print target distributions
    print("\nTarget Distributions:")
    print(f"CTR Target (is_click): {feature_table['is_click'].value_counts().to_dict()}")
    print(f"Retention Target (is_retained): {feature_table['is_retained'].value_counts().to_dict()}")
    print(f"Uplift Treatment: {feature_table['treatment'].value_counts().to_dict()}")
    
    return feature_table

if __name__ == "__main__":
    main() 