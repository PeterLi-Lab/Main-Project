import pandas as pd
import xml.etree.ElementTree as ET
import os
from datetime import datetime, timedelta

def parse_xml(path):
    """Parse XML file and return DataFrame"""
    tree = ET.parse(path)
    root = tree.getroot()
    return pd.DataFrame([row.attrib for row in root])

print("=== Testing Active User Definition ===")

# Load data
print("Loading data...")
df_posts = parse_xml("data/Posts.xml")
df_comments = parse_xml("data/Comments.xml")
df_post_history = parse_xml("data/PostHistory.xml")

print(f"Posts: {len(df_posts)}")
print(f"Comments: {len(df_comments)}")
print(f"PostHistory: {len(df_post_history)}")

# Check columns
print(f"\nPosts columns: {df_posts.columns.tolist()}")
print(f"Comments columns: {df_comments.columns.tolist()}")
print(f"PostHistory columns: {df_post_history.columns.tolist()}")

# Get active users
print(f"\n=== Define Active Users ===")

# Users who posted
post_users = set(df_posts['OwnerUserId'].unique())
print(f"Users who posted: {len(post_users)}")

# Users who commented
comment_users = set()
if 'UserId' in df_comments.columns:
    comment_users = set(df_comments['UserId'].unique())
    print(f"Users who commented: {len(comment_users)}")
else:
    print("No UserId in comments data")

# Users with recent post history
post_history_users = set()
if 'UserId' in df_post_history.columns:
    df_post_history['CreationDate'] = pd.to_datetime(df_post_history['CreationDate'])
    recent_date = datetime.now() - timedelta(days=180)
    recent_post_history = df_post_history[df_post_history['CreationDate'] >= recent_date]
    post_history_users = set(recent_post_history['UserId'].unique())
    print(f"Users with recent post history: {len(post_history_users)}")
else:
    print("No UserId in post history data")

# Combine all active users
active_users = list(post_users | comment_users | post_history_users)
print(f"Total active users: {len(active_users)}")

# Clean data
active_users = [str(uid) for uid in active_users if pd.notna(uid)]
print(f"Active users after cleaning: {len(active_users)}")

print(f"\nSample active users: {active_users[:10]}") 