#!/usr/bin/env python3
"""
User-Post Click Labeling Script
Creates user-post pairs with click labels using behavioral rules
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import random
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class UserPostClickLabeling:
    def __init__(self, data_dir='data'):
        """Initialize the click labeling pipeline"""
        self.data_dir = data_dir
        self.df_posts = None
        self.df_votes = None
        self.df_comments = None
        self.df_samples = None
        
    def parse_xml(self, path):
        """Parse XML file and return DataFrame"""
        print(f"Parsing {path}...")
        tree = ET.parse(path)
        root = tree.getroot()
        df = pd.DataFrame([row.attrib for row in root])
        print(f"Loaded {len(df)} records from {path}")
        return df
    
    def load_and_clean_data(self):
        """Load and clean all data files"""
        print("=== Loading and Cleaning Data ===")
        
        # Load data
        self.df_posts = self.parse_xml(f'{self.data_dir}/Posts.xml')
        self.df_votes = self.parse_xml(f'{self.data_dir}/Votes.xml')
        self.df_comments = self.parse_xml(f'{self.data_dir}/Comments.xml')
        
        # Clean posts data
        print("Cleaning posts data...")
        # Filter out posts with null OwnerUserId
        self.df_posts = self.df_posts.dropna(subset=['OwnerUserId']).copy()
        print(f"Posts after filtering null OwnerUserId: {len(self.df_posts)}")
        
        # Handle missing tags
        self.df_posts['Tags'] = self.df_posts['Tags'].fillna('')
        
        # Parse tags properly
        self.df_posts['TagList'] = self.df_posts['Tags'].apply(
            lambda x: re.findall(r'<(.*?)>', x) if pd.notna(x) and x != '' else []
        )
        
        # Convert CreationDate to datetime
        self.df_posts['CreationDate'] = pd.to_datetime(self.df_posts['CreationDate'])
        self.df_votes['CreationDate'] = pd.to_datetime(self.df_votes['CreationDate'])
        self.df_comments['CreationDate'] = pd.to_datetime(self.df_comments['CreationDate'])
        
        # Only filter UserId for active user identification, not for upvote counting
        self.df_posts['Id'] = self.df_posts['Id'].astype(str)
        self.df_posts['OwnerUserId'] = self.df_posts['OwnerUserId'].astype(str)
        self.df_votes['PostId'] = self.df_votes['PostId'].astype(str)
        # UserId may be null in votes, so only convert non-nulls
        self.df_votes['UserId'] = self.df_votes['UserId'].astype(str, errors='ignore')
        self.df_comments['UserId'] = self.df_comments['UserId'].astype(str)
        
        print("Data cleaning completed!")
        return True
    
    def identify_active_users(self, recent_days=30):
        """Identify active users in the recent period"""
        print(f"\n=== Identifying Active Users (Last {recent_days} Days) ===")
        
        # For historical data, use the most recent period in the dataset
        max_date = self.df_posts['CreationDate'].max()
        cutoff_date = max_date - pd.Timedelta(days=recent_days)
        
        print(f"Using cutoff date: {cutoff_date}")
        print(f"Max date in dataset: {max_date}")
        
        # Users who posted in recent period
        recent_posts = self.df_posts[self.df_posts['CreationDate'] > cutoff_date]
        post_users = set(recent_posts['OwnerUserId'])
        
        # Users who commented in recent period
        recent_comments = self.df_comments[self.df_comments['CreationDate'] > cutoff_date]
        comment_users = set(recent_comments['UserId'])
        
        # Only use votes with non-null UserId for active users
        recent_votes = self.df_votes[(self.df_votes['CreationDate'] > cutoff_date) & (self.df_votes['UserId'].notnull())]
        vote_users = set(recent_votes['UserId'])
        
        # Combine all active users
        active_users = post_users | comment_users | vote_users
        print(f"Active users identified: {len(active_users)}")
        
        # If no active users found, use all users who have any activity
        if len(active_users) == 0:
            print("No active users found in recent period, using all users with activity...")
            all_post_users = set(self.df_posts['OwnerUserId'])
            all_comment_users = set(self.df_comments['UserId'])
            all_vote_users = set(self.df_votes[self.df_votes['UserId'].notnull()]['UserId'])
            active_users = all_post_users | all_comment_users | all_vote_users
            print(f"Total users with activity: {len(active_users)}")
        
        return active_users
    
    def build_user_tag_history(self):
        """Build user tag history for interest calculation"""
        print("\n=== Building User Tag History ===")
        
        user_tag_history = defaultdict(set)
        
        for _, row in self.df_posts.iterrows():
            user_id = str(row['OwnerUserId'])
            tags = row['TagList']
            if tags:  # Only add if tags exist
                user_tag_history[user_id].update(tags)
        
        print(f"User tag history built for {len(user_tag_history)} users")
        return user_tag_history
    
    def create_positive_samples(self, active_users, upvoted_posts, N=5):
        """Create positive samples for upvoted posts (assign random active users as clickers)"""
        print(f"\n=== Creating Positive Samples (N={N}) ===")
        positive_samples = []
        assigned_users_per_post = dict()
        for post_id in upvoted_posts:
            available_users = set(active_users)
            if len(available_users) >= N:
                sampled_users = random.sample(list(available_users), N)
            else:
                sampled_users = list(available_users)
            assigned_users_per_post[post_id] = set(sampled_users)
            for user_id in sampled_users:
                positive_samples.append({
                    'user_id': user_id,
                    'post_id': post_id,
                    'is_click': 1
                })
        print(f"Created {len(positive_samples)} positive samples")
        return positive_samples, assigned_users_per_post
    
    def create_negative_samples(self, active_users, upvoted_posts, user_tag_history, assigned_users_per_post, N=5):
        """Create negative samples with interest-based prioritization"""
        print(f"\n=== Creating Negative Samples (N={N}) ===")
        negative_samples = []
        for post_id in upvoted_posts:
            post_row = self.df_posts[self.df_posts['Id'] == post_id]
            if len(post_row) == 0:
                continue
            post_tags = set(post_row.iloc[0]['TagList'])
            # Exclude users assigned as positive for this post
            candidate_users = set(active_users) - assigned_users_per_post.get(post_id, set())
            # Calculate interest scores
            user_interest = []
            for user_id in candidate_users:
                user_tags = user_tag_history.get(user_id, set())
                interest_score = len(post_tags & user_tags)
                user_interest.append((user_id, interest_score))
            user_interest.sort(key=lambda x: -x[1])
            sampled_users = user_interest[:N]
            for user_id, interest_score in sampled_users:
                negative_samples.append({
                    'user_id': user_id,
                    'post_id': post_id,
                    'is_click': 0,
                    'interest_score': interest_score
                })
        print(f"Created {len(negative_samples)} negative samples")
        return negative_samples
    
    def create_user_post_samples(self, N=5, recent_days=30):
        """Create user-post click samples"""
        print("=== Starting User-Post Click Labeling ===")
        self.load_and_clean_data()
        active_users = self.identify_active_users(recent_days)
        user_tag_history = self.build_user_tag_history()
        # Use all posts with at least one upvote (VoteTypeId==2)
        upvotes = self.df_votes[self.df_votes['VoteTypeId'] == '2']
        upvoted_posts = set(upvotes['PostId'])
        print(f"Upvoted posts: {len(upvoted_posts)}")
        if not upvoted_posts or not active_users:
            print("No upvoted posts or active users found. No samples will be created.")
            self.df_samples = pd.DataFrame()
            return self.df_samples
        positive_samples, assigned_users_per_post = self.create_positive_samples(active_users, upvoted_posts, N)
        negative_samples = self.create_negative_samples(active_users, upvoted_posts, user_tag_history, assigned_users_per_post, N)
        all_samples = positive_samples + negative_samples
        self.df_samples = pd.DataFrame(all_samples)
        print(f"\n=== Sample Creation Complete ===")
        print(f"Total samples: {len(self.df_samples)}")
        print(f"Positive samples: {len(positive_samples)}")
        print(f"Negative samples: {len(negative_samples)}")
        return self.df_samples
    
    def add_features(self):
        """Add user and post features to the samples"""
        print("\n=== Adding Features ===")
        
        # Check if we have samples
        if len(self.df_samples) == 0:
            print("No samples to add features to!")
            return False
        
        # Add user features
        user_features = self.df_posts.groupby('OwnerUserId').agg({
            'Id': 'count',
            'CreationDate': 'min'
        }).reset_index()
        user_features.columns = ['user_id', 'user_post_count', 'user_first_post_date']
        user_features['user_account_age_days'] = (
            pd.Timestamp.now() - pd.to_datetime(user_features['user_first_post_date'])
        ).dt.days
        
        # Add post features
        post_features = self.df_posts[['Id', 'Title', 'TagList', 'CreationDate']].copy()
        post_features.columns = ['post_id', 'post_title', 'post_tags', 'post_creation_date']
        post_features['post_age_days'] = (
            pd.Timestamp.now() - pd.to_datetime(post_features['post_creation_date'])
        ).dt.days
        post_features['post_title_length'] = post_features['post_title'].str.len()
        post_features['post_tag_count'] = post_features['post_tags'].apply(len)
        
        # Merge features
        self.df_samples = self.df_samples.merge(user_features, on='user_id', how='left')
        self.df_samples = self.df_samples.merge(post_features, on='post_id', how='left')
        
        # Fill missing values
        self.df_samples = self.df_samples.fillna(0)
        
        print(f"Features added. Final shape: {self.df_samples.shape}")
        return True
    
    def save_samples(self, output_path='user_post_click_samples.csv'):
        """Save the samples to CSV"""
        print(f"\n=== Saving Samples ===")
        
        self.df_samples.to_csv(output_path, index=False)
        print(f"Samples saved to {output_path}")
        
        # Also save a sample for inspection
        sample_path = 'user_post_click_samples_sample.csv'
        self.df_samples.head(1000).to_csv(sample_path, index=False)
        print(f"Sample saved to {sample_path}")
        
        return True
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n=== Summary Statistics ===")
        print(f"Dataset shape: {self.df_samples.shape}")
        print(f"Memory usage: {self.df_samples.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Target distribution
        print(f"\nTarget Distribution:")
        print(self.df_samples['is_click'].value_counts())
        
        # Feature statistics
        print(f"\nFeature Statistics:")
        numeric_cols = self.df_samples.select_dtypes(include=[np.number]).columns
        print(self.df_samples[numeric_cols].describe())
        
        return True
    
    def run_full_pipeline(self, N=5, recent_days=30):
        """Run the complete user-post click labeling pipeline"""
        print("=== User-Post Click Labeling Pipeline ===")
        
        # Create samples
        self.create_user_post_samples(N, recent_days)
        
        # Add features
        self.add_features()
        
        # Save results
        self.save_samples()
        
        # Print summary
        self.print_summary()
        
        print("\n=== Pipeline Complete ===")
        return self.df_samples

def main():
    """Main function to run the user-post click labeling pipeline"""
    labeler = UserPostClickLabeling()
    samples = labeler.run_full_pipeline(N=5, recent_days=30)
    
    return samples

if __name__ == "__main__":
    main() 