#!/usr/bin/env python3
"""
Retention Prediction Labeling Script
Creates retention labels for user behavior prediction
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings('ignore')

class RetentionPredictionLabeling:
    def __init__(self, data_dir='data'):
        """Initialize the retention labeling pipeline"""
        self.data_dir = data_dir
        self.df_posts = None
        self.df_users = None
        self.df_comments = None
        self.df_votes = None
        
    def parse_xml(self, path):
        """Parse XML file and return DataFrame"""
        print(f"Parsing {path}...")
        tree = ET.parse(path)
        root = tree.getroot()
        df = pd.DataFrame([row.attrib for row in root])
        print(f"Loaded {len(df)} records from {path}")
        return df
    
    def load_data(self):
        """Load and clean all data files"""
        print("=== Loading Data ===")
        
        # Load data
        self.df_posts = self.parse_xml(f'{self.data_dir}/Posts.xml')
        self.df_users = self.parse_xml(f'{self.data_dir}/Users.xml')
        self.df_comments = self.parse_xml(f'{self.data_dir}/Comments.xml')
        self.df_votes = self.parse_xml(f'{self.data_dir}/Votes.xml')
        
        # Clean data
        print("Cleaning data...")
        
        # Convert dates
        self.df_posts['CreationDate'] = pd.to_datetime(self.df_posts['CreationDate'])
        self.df_users['CreationDate'] = pd.to_datetime(self.df_users['CreationDate'])
        self.df_comments['CreationDate'] = pd.to_datetime(self.df_comments['CreationDate'])
        self.df_votes['CreationDate'] = pd.to_datetime(self.df_votes['CreationDate'])
        
        # Convert IDs to string
        self.df_posts['Id'] = self.df_posts['Id'].astype(str)
        self.df_posts['OwnerUserId'] = self.df_posts['OwnerUserId'].astype(str)
        self.df_users['Id'] = self.df_users['Id'].astype(str)
        self.df_comments['UserId'] = self.df_comments['UserId'].astype(str)
        self.df_votes['UserId'] = self.df_votes['UserId'].astype(str, errors='ignore')
        
        print("Data cleaning completed!")
        return True
    
    def create_retention_labels(self, retention_days=7):
        """Create retention prediction labels"""
        print(f"\n=== Creating Retention Labels (Retention Days: {retention_days}) ===")
        
        # Get all user activity timestamps
        user_activities = []
        
        # Posts activity
        posts_activity = self.df_posts[['OwnerUserId', 'CreationDate']].copy()
        posts_activity['activity_type'] = 'post'
        user_activities.append(posts_activity)
        
        # Comments activity
        comments_activity = self.df_comments[['UserId', 'CreationDate']].copy()
        comments_activity['activity_type'] = 'comment'
        comments_activity.rename(columns={'UserId': 'OwnerUserId'}, inplace=True)
        user_activities.append(comments_activity)
        
        # Votes activity (only upvotes/downvotes, not favorites)
        votes_activity = self.df_votes[self.df_votes['VoteTypeId'].isin(['2', '3'])][['UserId', 'CreationDate']].copy()
        votes_activity['activity_type'] = 'vote'
        votes_activity.rename(columns={'UserId': 'OwnerUserId'}, inplace=True)
        user_activities.append(votes_activity)
        
        # Combine all activities
        all_activities = pd.concat(user_activities, ignore_index=True)
        all_activities = all_activities.sort_values(['OwnerUserId', 'CreationDate'])
        
        # Calculate retention labels for each user
        retention_data = []
        
        for user_id in all_activities['OwnerUserId'].unique():
            user_acts = all_activities[all_activities['OwnerUserId'] == user_id].copy()
            user_acts = user_acts.sort_values('CreationDate')
            
            if len(user_acts) < 2:
                continue  # Skip users with only one activity
            
            # For each activity, check if user returns within retention_days
            for i, (idx, row) in enumerate(user_acts.iterrows()):
                current_time = row['CreationDate']
                
                # Look for next activity within retention_days
                future_acts = user_acts[user_acts['CreationDate'] > current_time]
                future_acts = future_acts[future_acts['CreationDate'] <= current_time + pd.Timedelta(days=retention_days)]
                
                # Retention label
                is_retained = 1 if len(future_acts) > 0 else 0
                
                # Duration to next action (in days)
                if len(future_acts) > 0:
                    next_time = future_acts.iloc[0]['CreationDate']
                    days_to_next_action = (next_time - current_time).days
                else:
                    days_to_next_action = retention_days  # Capped at retention_days
                
                retention_data.append({
                    'user_id': user_id,
                    'activity_date': current_time,
                    'activity_type': row['activity_type'],
                    'is_retained': is_retained,
                    'days_to_next_action': days_to_next_action
                })
        
        self.df_retention = pd.DataFrame(retention_data)
        print(f"Created retention labels for {len(self.df_retention)} user activities")
        print(f"Retention rate: {self.df_retention['is_retained'].mean():.3f}")
        print(f"Average days to next action: {self.df_retention['days_to_next_action'].mean():.2f}")
        
        return self.df_retention
    
    def add_user_features(self):
        """Add user-level features for retention prediction"""
        print("\n=== Adding User Features ===")
        
        # User account age
        self.df_users['account_age_days'] = (
            pd.Timestamp.now() - pd.to_datetime(self.df_users['CreationDate'])
        ).dt.days
        
        # User activity counts
        user_post_counts = self.df_posts.groupby('OwnerUserId').size().reset_index(name='user_post_count')
        user_comment_counts = self.df_comments.groupby('UserId').size().reset_index(name='user_comment_count')
        user_vote_counts = self.df_votes.groupby('UserId').size().reset_index(name='user_vote_count')
        
        # Merge user features
        user_features = self.df_users[['Id', 'Reputation', 'account_age_days']].copy()
        user_features = user_features.merge(user_post_counts, left_on='Id', right_on='OwnerUserId', how='left')
        user_features = user_features.merge(user_comment_counts, left_on='Id', right_on='UserId', how='left')
        user_features = user_features.merge(user_vote_counts, left_on='Id', right_on='UserId', how='left')
        
        # Fill missing values
        user_features = user_features.fillna(0)
        
        # Add to retention data
        self.df_retention = self.df_retention.merge(
            user_features[['Id', 'Reputation', 'account_age_days', 'user_post_count', 'user_comment_count', 'user_vote_count']],
            left_on='user_id', right_on='Id', how='left'
        )
        
        print(f"Added user features. Final shape: {self.df_retention.shape}")
        return True
    
    def save_retention_data(self, output_path='retention_prediction_data.csv'):
        """Save the retention prediction dataset"""
        print(f"\n=== Saving Retention Data ===")
        
        self.df_retention.to_csv(output_path, index=False)
        print(f"Retention data saved to {output_path}")
        
        # Also save a sample for inspection
        sample_path = 'retention_prediction_data_sample.csv'
        self.df_retention.head(1000).to_csv(sample_path, index=False)
        print(f"Sample saved to {sample_path}")
        
        return True
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n=== Retention Prediction Summary ===")
        print(f"Dataset shape: {self.df_retention.shape}")
        
        # Target distribution
        print(f"\nTarget Distribution:")
        print(f"is_retained: {self.df_retention['is_retained'].value_counts().to_dict()}")
        print(f"days_to_next_action stats:")
        print(self.df_retention['days_to_next_action'].describe())
        
        # Activity type distribution
        print(f"\nActivity Type Distribution:")
        print(self.df_retention['activity_type'].value_counts())
        
        return True
    
    def run_full_pipeline(self, retention_days=7):
        """Run the complete retention prediction labeling pipeline"""
        print("=== Retention Prediction Labeling Pipeline ===")
        
        # Load data
        self.load_data()
        
        # Create retention labels
        self.create_retention_labels(retention_days)
        
        # Add user features
        self.add_user_features()
        
        # Save results
        self.save_retention_data()
        
        # Print summary
        self.print_summary()
        
        print("\n=== Pipeline Complete ===")
        return self.df_retention

def main():
    """Main function"""
    labeler = RetentionPredictionLabeling()
    retention_data = labeler.run_full_pipeline(retention_days=7)
    return retention_data

if __name__ == "__main__":
    main() 