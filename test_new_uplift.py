import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import os
from datetime import datetime, timedelta

def parse_xml(path):
    """Parse XML file and return DataFrame"""
    tree = ET.parse(path)
    root = tree.getroot()
    return pd.DataFrame([row.attrib for row in root])

print("=== Testing New Uplift Sampling Logic ===")

# Load data
print("Loading data...")
df_posts = parse_xml("data/Posts.xml")
df_votes = parse_xml("data/Votes.xml")
df_comments = parse_xml("data/Comments.xml")
df_post_history = parse_xml("data/PostHistory.xml")

print(f"Posts: {len(df_posts)}")
print(f"Votes: {len(df_votes)}")
print(f"Comments: {len(df_comments)}")
print(f"PostHistory: {len(df_post_history)}")

# Process data
df_posts['CreationDate'] = pd.to_datetime(df_posts['CreationDate'])
df_votes['CreationDate'] = pd.to_datetime(df_votes['CreationDate'])
df_votes['PostId'] = df_votes['PostId'].astype(str)

# Get likes
df_likes = df_votes[df_votes['VoteTypeId'] == '2'].copy()
print(f"Likes: {len(df_likes)}")

# Define active users
post_users = set(df_posts['OwnerUserId'].unique())
comment_users = set(df_comments['UserId'].unique()) if 'UserId' in df_comments.columns else set()
post_history_users = set(df_post_history['UserId'].unique()) if 'UserId' in df_post_history.columns else set()

active_users = list(post_users | comment_users | post_history_users)
active_users = [str(uid) for uid in active_users if pd.notna(uid)]
print(f"Active users: {len(active_users)}")

# Create uplift samples
print(f"\nCreating uplift samples...")
np.random.seed(42)
uplift_samples = []

# Sample a subset for testing
sample_likes = df_likes.head(1000)

for idx, vote_row in sample_likes.iterrows():
    # Randomly assign an active user
    assigned_user = np.random.choice(active_users)
    
    post_id = vote_row['PostId']
    vote_time = vote_row['CreationDate']
    
    post_info = df_posts[df_posts['Id'] == post_id]
    if len(post_info) == 0:
        continue
        
    post_time = post_info.iloc[0]['CreationDate']
    time_diff_hours = (vote_time - post_time).total_seconds() / 3600
    
    # Skip if user voted on their own post
    if assigned_user == str(post_info.iloc[0]['OwnerUserId']):
        continue
    
    # Define treatment
    time_threshold = 2943.34
    treatment = 1 if time_diff_hours <= time_threshold else 0
    
    # Add noise
    if np.random.random() < 0.1:
        treatment = 1 - treatment
    
    uplift_samples.append({
        'user_id': assigned_user,
        'post_id': post_id,
        'treatment': treatment,
        'is_click': 1,
        'time_diff_hours': time_diff_hours
    })

df_uplift = pd.DataFrame(uplift_samples)
print(f"Created {len(df_uplift)} uplift samples")

# Check results
print(f"\nUplift Sample Analysis:")
print(f"Total samples: {len(df_uplift)}")
print(f"Treatment group: {df_uplift['treatment'].sum()}")
print(f"Control group: {(df_uplift['treatment'] == 0).sum()}")
print(f"Click rate: {df_uplift['is_click'].mean():.3f}")
print(f"Unique users: {df_uplift['user_id'].nunique()}")
print(f"Unique posts: {df_uplift['post_id'].nunique()}")

print(f"\nTime difference stats:")
print(df_uplift['time_diff_hours'].describe()) 