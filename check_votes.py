import pandas as pd
import xml.etree.ElementTree as ET
import os

def parse_xml(path):
    """Parse XML file and return DataFrame"""
    tree = ET.parse(path)
    root = tree.getroot()
    return pd.DataFrame([row.attrib for row in root])

# Check Votes.xml
print("=== Check Votes.xml Fields ===")
df_votes = parse_xml("data/Votes.xml")
print(f"Votes columns: {df_votes.columns.tolist()}")
print(f"Votes data shape: {df_votes.shape}")
print(f"Votes first 5 rows:")
print(df_votes.head())
print(f"\nVoteTypeId unique values: {df_votes['VoteTypeId'].unique()}")
print(f"VoteTypeId value counts:")
print(df_votes['VoteTypeId'].value_counts())

# Check if UserId field exists
if 'UserId' in df_votes.columns:
    print(f"\nUserId field exists, unique count: {df_votes['UserId'].nunique()}")
    print(f"UserId non-null count: {df_votes['UserId'].notna().sum()}")
    print(f"UserId first 10 values:")
    print(df_votes['UserId'].head(10))
else:
    print(f"\nUserId field does not exist!")
    print(f"Available fields: {df_votes.columns.tolist()}") 