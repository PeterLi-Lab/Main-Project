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

print("=== Tag-Based Uplift Modeling ===")

# Load data
print("Loading data...")
df_posts = parse_xml("data/Posts.xml")
df_votes = parse_xml("data/Votes.xml")
df_tags = parse_xml("data/Tags.xml")

print(f"Posts: {len(df_posts)}")
print(f"Votes: {len(df_votes)}")
print(f"Tags: {len(df_tags)}")

# Process data
df_posts['CreationDate'] = pd.to_datetime(df_posts['CreationDate'])
df_posts['Tags'] = df_posts['Tags'].fillna('')
df_posts['TagList'] = df_posts['Tags'].str.strip('|').str.split('|')

# Get likes
df_likes = df_votes[df_votes['VoteTypeId'] == '2'].copy()
print(f"Likes: {len(df_likes)}")

# Analyze popular tags
print(f"\n=== Tag Analysis ===")
all_tags = []
for tag_list in df_posts['TagList']:
    if tag_list and tag_list != ['']:
        all_tags.extend(tag_list)

tag_counts = pd.Series(all_tags).value_counts()
print(f"Top 20 tags:")
print(tag_counts.head(20))

# Define treatment tags (content that might be "recommended" vs "not recommended")
# Let's brainstorm some tag categories:

# 1. AI/ML related tags (likely to be recommended)
ai_ml_tags = [
    'machine-learning', 'artificial-intelligence', 'deep-learning', 'neural-network',
    'tensorflow', 'pytorch', 'scikit-learn', 'nlp', 'computer-vision',
    'reinforcement-learning', 'data-science', 'big-data', 'ai'
]

# 2. Programming language tags (neutral)
programming_tags = [
    'python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'go',
    'swift', 'kotlin', 'rust', 'scala', 'r', 'matlab'
]

# 3. Framework/Technology tags (might be recommended)
framework_tags = [
    'react', 'angular', 'vue.js', 'django', 'flask', 'spring', 'express',
    'node.js', 'asp.net', 'laravel', 'rails', 'fastapi'
]

# 4. Database tags (neutral)
database_tags = [
    'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle',
    'sql-server', 'cassandra', 'elasticsearch'
]

# 5. DevOps/Cloud tags (likely to be recommended)
devops_tags = [
    'docker', 'kubernetes', 'aws', 'azure', 'google-cloud', 'jenkins',
    'git', 'ci-cd', 'microservices', 'serverless'
]

# 6. Legacy/Old technology tags (less likely to be recommended)
legacy_tags = [
    'jquery', 'php', 'asp.net', 'vb.net', 'perl', 'cobol', 'fortran',
    'assembly', 'pascal', 'basic'
]

print(f"\n=== Treatment Tag Categories ===")
print(f"AI/ML tags: {len(ai_ml_tags)}")
print(f"Programming tags: {len(programming_tags)}")
print(f"Framework tags: {len(framework_tags)}")
print(f"Database tags: {len(database_tags)}")
print(f"DevOps tags: {len(devops_tags)}")
print(f"Legacy tags: {len(legacy_tags)}")

# Create treatment assignment function
def assign_treatment(tag_list):
    """Assign treatment based on post tags"""
    if not tag_list or tag_list == ['']:
        return 0  # No tags = control
    
    # Count tags in each category
    ai_ml_count = sum(1 for tag in tag_list if tag in ai_ml_tags)
    framework_count = sum(1 for tag in tag_list if tag in framework_tags)
    devops_count = sum(1 for tag in tag_list if tag in devops_tags)
    legacy_count = sum(1 for tag in tag_list if tag in legacy_tags)
    
    # Treatment logic:
    # - AI/ML content = likely recommended (treatment = 1)
    # - Modern frameworks = likely recommended (treatment = 1)
    # - DevOps/Cloud = likely recommended (treatment = 1)
    # - Legacy tech = less likely recommended (treatment = 0)
    # - Programming languages = neutral (treatment = 0)
    # - Databases = neutral (treatment = 0)
    
    if ai_ml_count > 0 or framework_count > 0 or devops_count > 0:
        return 1  # Treatment group (likely recommended)
    elif legacy_count > 0:
        return 0  # Control group (less likely recommended)
    else:
        return 0  # Control group (neutral)

# Create uplift dataset
print(f"\n=== Creating Tag-Based Uplift Dataset ===")

uplift_samples = []

# Process posts with votes
posts_with_votes = df_likes['PostId'].unique()
print(f"Posts with votes: {len(posts_with_votes)}")

for post_id in posts_with_votes:
    post_info = df_posts[df_posts['Id'] == post_id]
    if len(post_info) == 0:
        continue
    
    post_row = post_info.iloc[0]
    tag_list = post_row['TagList']
    
    # Assign treatment based on tags
    treatment = assign_treatment(tag_list)
    
    # Get vote count for this post
    vote_count = len(df_likes[df_likes['PostId'] == post_id])
    
    # Create sample
    uplift_samples.append({
        'post_id': post_id,
        'treatment': treatment,
        'is_click': 1,  # All posts with votes are positive samples
        'vote_count': vote_count,
        'tags': '|'.join(tag_list) if tag_list else '',
        'title': post_row['Title'][:100],  # Truncate for display
        'score': post_row.get('Score', 0),
        'view_count': post_row.get('ViewCount', 0)
    })

df_uplift = pd.DataFrame(uplift_samples)

# Analysis
print(f"\n=== Uplift Dataset Analysis ===")
print(f"Total samples: {len(df_uplift)}")
print(f"Treatment group (likely recommended): {df_uplift['treatment'].sum()}")
print(f"Control group (less likely recommended): {(df_uplift['treatment'] == 0).sum()}")
print(f"Click rate: {df_uplift['is_click'].mean():.3f}")

# Treatment vs Control analysis
treatment_group = df_uplift[df_uplift['treatment'] == 1]
control_group = df_uplift[df_uplift['treatment'] == 0]

print(f"\n=== Treatment vs Control Analysis ===")
print(f"Treatment group:")
print(f"  - Sample count: {len(treatment_group)}")
print(f"  - Avg vote count: {treatment_group['vote_count'].mean():.2f}")
print(f"  - Avg score: {treatment_group['score'].mean():.2f}")
print(f"  - Avg view count: {treatment_group['view_count'].mean():.2f}")

print(f"\nControl group:")
print(f"  - Sample count: {len(control_group)}")
print(f"  - Avg vote count: {control_group['vote_count'].mean():.2f}")
print(f"  - Avg score: {control_group['score'].mean():.2f}")
print(f"  - Avg view count: {control_group['view_count'].mean():.2f}")

# Calculate uplift
treatment_click_rate = treatment_group['is_click'].mean()
control_click_rate = control_group['is_click'].mean()
uplift = treatment_click_rate - control_click_rate

print(f"\n=== Uplift Results ===")
print(f"Treatment click rate: {treatment_click_rate:.3f}")
print(f"Control click rate: {control_click_rate:.3f}")
print(f"Uplift: {uplift:.3f}")

# Tag distribution analysis
print(f"\n=== Tag Distribution Analysis ===")
treatment_tags = []
control_tags = []

for _, row in treatment_group.iterrows():
    if row['tags']:
        treatment_tags.extend(row['tags'].split('|'))

for _, row in control_group.iterrows():
    if row['tags']:
        control_tags.extend(row['tags'].split('|'))

treatment_tag_counts = pd.Series(treatment_tags).value_counts()
control_tag_counts = pd.Series(control_tags).value_counts()

print(f"Top tags in treatment group:")
print(treatment_tag_counts.head(10))

print(f"\nTop tags in control group:")
print(control_tag_counts.head(10))

print(f"\n=== Summary ===")
print(f"Tag-based uplift modeling completed!")
print(f"Treatment: AI/ML, Modern Frameworks, DevOps content")
print(f"Control: Legacy tech, Programming languages, Databases")
print(f"Uplift effect: {uplift:.3f}") 