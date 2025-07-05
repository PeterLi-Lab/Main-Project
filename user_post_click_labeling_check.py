import pandas as pd
import os
import xml.etree.ElementTree as ET

def parse_xml(path):
    """Parse XML file and return DataFrame"""
    print(f"Parsing {path}...")
    tree = ET.parse(path)
    root = tree.getroot()
    df = pd.DataFrame([row.attrib for row in root])
    print(f"Loaded {len(df)} records from {path}")
    return df

def check_data():
    # Load data from XML files
    try:
        df_posts = parse_xml('data/Posts.xml')
        df_votes = parse_xml('data/Votes.xml')
        df_comments = parse_xml('data/Comments.xml')
    except Exception as e:
        print(f"Error loading XML files: {e}")
        return

    # Check posts
    print("\n=== Posts DataFrame Check ===")
    print("Available columns:", df_posts.columns.tolist())
    required_post_cols = {'OwnerUserId', 'Id', 'Tags', 'CreationDate'}
    missing_cols = required_post_cols - set(df_posts.columns)
    print('Posts missing columns:', missing_cols)
    
    if 'OwnerUserId' in df_posts.columns:
        print('Null OwnerUserId in posts:', df_posts["OwnerUserId"].isnull().sum())
    if 'Tags' in df_posts.columns:
        print('Null Tags in posts:', df_posts["Tags"].isnull().sum())
    if 'CreationDate' in df_posts.columns:
        print('Posts CreationDate dtype:', df_posts["CreationDate"].dtype)

    # Check votes
    print("\n=== Votes DataFrame Check ===")
    print("Available columns:", df_votes.columns.tolist())
    required_vote_cols = {'UserId', 'PostId', 'VoteTypeId', 'CreationDate'}
    missing_cols = required_vote_cols - set(df_votes.columns)
    print('Votes missing columns:', missing_cols)
    
    if 'UserId' in df_votes.columns:
        print('Null UserId in votes:', df_votes["UserId"].isnull().sum())
    if 'CreationDate' in df_votes.columns:
        print('Votes CreationDate dtype:', df_votes["CreationDate"].dtype)

    # Check comments
    print("\n=== Comments DataFrame Check ===")
    print("Available columns:", df_comments.columns.tolist())
    required_comment_cols = {'UserId', 'CreationDate'}
    missing_cols = required_comment_cols - set(df_comments.columns)
    print('Comments missing columns:', missing_cols)
    
    if 'UserId' in df_comments.columns:
        print('Null UserId in comments:', df_comments["UserId"].isnull().sum())
    if 'CreationDate' in df_comments.columns:
        print('Comments CreationDate dtype:', df_comments["CreationDate"].dtype)

    # Check data types and sample data
    print("\n=== Sample Data ===")
    if len(df_posts) > 0:
        print("Sample post data:")
        print(df_posts.head(1).to_dict('records')[0])
    
    if len(df_votes) > 0:
        print("Sample vote data:")
        print(df_votes.head(1).to_dict('records')[0])

if __name__ == '__main__':
    check_data() 