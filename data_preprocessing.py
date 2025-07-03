import os
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, base_path=None):
        """Initialize the data preprocessor with base path"""
        if base_path is None:
            # Default to data folder in current directory
            self.base_path = os.path.join(os.getcwd(), 'data')
        else:
            self.base_path = base_path
        self.df_posts = None
        self.df_users = None
        self.df_tags = None
        self.df_votes = None
        self.df_badges = None
        self.df_combined = None
        self.embeddings = None
        self.model = None
        self.tfidf_features = None
        self.tfidf_vectorizer = None
        self.tfidf_svd = None
        
    def parse_xml(self, path):
        """Parse XML file and return DataFrame"""
        tree = ET.parse(path)
        root = tree.getroot()
        return pd.DataFrame([row.attrib for row in root])
    
    def load_data(self):
        """Load all XML data files"""
        print("=== Loading Data Files ===")
        print(f"Looking for data files in: {self.base_path}")
        
        # Check if data directory exists
        if not os.path.exists(self.base_path):
            print(f"Error: Data directory '{self.base_path}' does not exist!")
            print("Please ensure your XML data files are in the 'data' folder.")
            return None, None, None, None, None
        
        # List of required XML files
        required_files = ['Posts.xml', 'Users.xml', 'Tags.xml', 'Votes.xml', 'Badges.xml']
        
        # Check if all required files exist
        missing_files = []
        for file_name in required_files:
            file_path = os.path.join(self.base_path, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
        
        if missing_files:
            print(f"Error: Missing required data files: {missing_files}")
            print(f"Please ensure all XML files are in: {self.base_path}")
            return None, None, None, None, None
        
        try:
            self.df_posts = self.parse_xml(os.path.join(self.base_path, 'Posts.xml'))
            self.df_users = self.parse_xml(os.path.join(self.base_path, 'Users.xml'))
            self.df_tags = self.parse_xml(os.path.join(self.base_path, 'Tags.xml'))
            self.df_votes = self.parse_xml(os.path.join(self.base_path, 'Votes.xml'))
            self.df_badges = self.parse_xml(os.path.join(self.base_path, 'Badges.xml'))
            
            print(f"Successfully loaded:")
            print(f"  - {len(self.df_posts)} posts")
            print(f"  - {len(self.df_users)} users") 
            print(f"  - {len(self.df_tags)} tags")
            print(f"  - {len(self.df_votes)} votes")
            print(f"  - {len(self.df_badges)} badges")
            
            return self.df_posts, self.df_users, self.df_tags, self.df_votes, self.df_badges
                
        except Exception as e:
            print(f"Error loading data files: {e}")
            return None, None, None, None, None
    
    def basic_visualization(self):
        """Create basic visualizations of the raw data"""
        print("\n=== Basic Data Visualization ===")
        
        # Convert creation date
        self.df_posts['CreationDate'] = pd.to_datetime(self.df_posts['CreationDate'])
        
        # Post volume over time
        plt.figure(figsize=(12, 4))
        sns.histplot(self.df_posts['CreationDate'], bins=50)
        plt.title("Post Volume Over Time")
        plt.show()
        
        # Top tags
        self.df_tags['Count'] = self.df_tags['Count'].astype(int)
        top_tags = self.df_tags.sort_values('Count', ascending=False).head(20)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_tags, x='Count', y='TagName')
        plt.title("Top 20 Tags")
        plt.show()
        
        return top_tags
    
    def clean_data(self):
        """Clean and prepare the data"""
        print("\n=== Data Cleaning ===")
        
        # Clean posts data
        self.df_posts = self.df_posts.dropna(subset=['Title', 'OwnerUserId']).copy()
        self.df_posts['Tags'] = self.df_posts['Tags'].fillna('')
        self.df_posts['TagList'] = self.df_posts['Tags'].str.strip('|').str.split('|')
        
        # Clean users data
        self.df_users['user_reputation'] = self.df_users['Reputation'].astype(int)
        self.df_users['Id'] = self.df_users['Id'].astype(str)
        
        # Merge posts and users
        self.df_combined = self.df_posts.merge(self.df_users, left_on='OwnerUserId', right_on='Id', how='left')
        
        print(f"Cleaned data: {len(self.df_combined)} combined records")
        
        return self.df_combined
    
    def create_derived_variables(self):
        """Create derived variables from the data"""
        print("\n=== Creating Derived Variables ===")
        
        # Basic derived variables
        self.df_combined['title_length'] = self.df_combined['Title'].apply(lambda x: len(x.split()))
        self.df_combined['num_tags'] = self.df_combined['TagList'].apply(len)
        self.df_combined['first_tag'] = self.df_combined['TagList'].apply(lambda x: x[0] if len(x) > 0 else 'None')
        
        # Get top tags for popular tag analysis
        top_tags = self.df_tags.sort_values('Count', ascending=False).head(20)
        self.df_combined['is_first_tag_popular'] = self.df_combined['first_tag'].isin(top_tags['TagName'].tolist())
        
        # Post length
        self.df_combined['post_length'] = self.df_combined['Body'].fillna('').apply(lambda x: len(str(x).split()))
        
        # Vote statistics
        vote_stats = self.df_votes.groupby('PostId').agg({
            'VoteTypeId': 'count'
        }).reset_index()
        vote_stats.columns = ['PostId', 'total_votes']
        
        upvotes = self.df_votes[self.df_votes['VoteTypeId'] == '2'].groupby('PostId').size().reset_index()
        upvotes.columns = ['PostId', 'upvotes']
        
        # Merge vote statistics
        self.df_combined = self.df_combined.merge(vote_stats, left_on='Id_x', right_on='PostId', how='left')
        self.df_combined = self.df_combined.merge(upvotes, left_on='Id_x', right_on='PostId', how='left')
        
        # Fill NaN values
        self.df_combined['total_votes'] = self.df_combined['total_votes'].fillna(0)
        self.df_combined['upvotes'] = self.df_combined['upvotes'].fillna(0)
        
        # Calculate vote ratio
        self.df_combined['vote_ratio'] = self.df_combined.apply(
            lambda row: row['upvotes'] / row['total_votes'] if row['total_votes'] > 0 else 0, axis=1
        )
        
        # Comment count (if available)
        try:
            df_comments = self.parse_xml(os.path.join(self.base_path, 'Comments.xml'))
            comment_counts = df_comments.groupby('PostId').size().reset_index()
            comment_counts.columns = ['PostId', 'comment_count']
            self.df_combined = self.df_combined.merge(comment_counts, left_on='Id_x', right_on='PostId', how='left')
            self.df_combined['comment_count'] = self.df_combined['comment_count'].fillna(0)
            print("Comment count added successfully")
        except:
            print("Comments.xml not found, skipping comment count calculation")
        
        # Post age
        self.df_combined['post_age_days'] = (pd.Timestamp.now() - self.df_combined['CreationDate_x']).dt.days
        
        # User activity level
        user_post_counts = self.df_combined.groupby('OwnerUserId').size().reset_index()
        user_post_counts.columns = ['OwnerUserId', 'user_post_count']
        self.df_combined = self.df_combined.merge(user_post_counts, on='OwnerUserId', how='left')
        
        print("Derived variables created successfully")
        return self.df_combined
    
    def create_badge_features(self):
        """Create badge-related features for users"""
        print("\n=== Creating Badge Features ===")
        
        if self.df_badges is None or len(self.df_badges) == 0:
            print("No badges data available, skipping badge features")
            return self.df_combined
        
        # Clean badges data
        self.df_badges['UserId'] = self.df_badges['UserId'].astype(str)
        self.df_badges['Class'] = self.df_badges['Class'].astype(int)
        self.df_badges['Date'] = pd.to_datetime(self.df_badges['Date'])
        
        # Badge class mapping (1=Gold, 2=Silver, 3=Bronze)
        badge_class_names = {1: 'Gold', 2: 'Silver', 3: 'Bronze'}
        self.df_badges['BadgeClass'] = self.df_badges['Class'].map(badge_class_names)
        
        print(f"Processing {len(self.df_badges)} badges for {self.df_badges['UserId'].nunique()} users")
        
        # 1. Basic badge counts per user
        badge_counts = self.df_badges.groupby('UserId').agg({
            'Id': 'count',
            'Class': lambda x: (x == 1).sum(),  # Gold badges
            'TagBased': lambda x: x.sum() if x.dtype == bool else 0
        }).reset_index()
        badge_counts.columns = ['UserId', 'total_badges', 'gold_badges', 'tag_based_badges']
        
        # 2. Badge counts by class
        badge_by_class = self.df_badges.groupby(['UserId', 'Class']).size().unstack(fill_value=0)
        badge_by_class.columns = ['gold_badges', 'silver_badges', 'bronze_badges']
        badge_by_class = badge_by_class.reset_index()
        
        # 3. Popular badges analysis
        popular_badges = self.df_badges['Name'].value_counts().head(20)
        print(f"Top 5 most common badges: {popular_badges.head().tolist()}")
        
        # Create binary features for popular badges
        for badge_name in popular_badges.head(10).index:
            users_with_badge = self.df_badges[self.df_badges['Name'] == badge_name]['UserId'].unique()
            self.df_combined[f'has_badge_{badge_name.lower().replace(" ", "_")}'] = \
                self.df_combined['OwnerUserId'].isin(users_with_badge)
        
        # 4. Badge timing features (when badges were earned relative to post creation)
        # Get the earliest badge date for each user
        user_first_badge = self.df_badges.groupby('UserId')['Date'].min().reset_index()
        user_first_badge.columns = ['UserId', 'first_badge_date']
        
        # 5. Badge diversity (number of unique badge types)
        badge_diversity = self.df_badges.groupby('UserId')['Name'].nunique().reset_index()
        badge_diversity.columns = ['UserId', 'unique_badge_types']
        
        # 6. Badge earning rate (badges per day since first badge)
        user_badge_stats = self.df_badges.groupby('UserId').agg({
            'Date': ['min', 'max', 'count']
        }).reset_index()
        user_badge_stats.columns = ['UserId', 'first_badge_date', 'last_badge_date', 'total_badges']
        user_badge_stats['badge_span_days'] = (user_badge_stats['last_badge_date'] - user_badge_stats['first_badge_date']).dt.days
        user_badge_stats['badge_rate_per_day'] = user_badge_stats['total_badges'] / (user_badge_stats['badge_span_days'] + 1)
        
        # 7. Recent badge activity (badges in last 30 days)
        recent_date = pd.Timestamp.now() - pd.Timedelta(days=30)
        recent_badges = self.df_badges[self.df_badges['Date'] >= recent_date].groupby('UserId').size().reset_index()
        recent_badges.columns = ['UserId', 'recent_badges_30d']
        
        # 8. Badge quality score (weighted by class: Gold=3, Silver=2, Bronze=1)
        badge_quality = self.df_badges.groupby('UserId').apply(
            lambda x: (x['Class'] * 1).sum()
        ).reset_index()
        badge_quality.columns = ['UserId', 'badge_quality_score']
        
        # Merge all badge features
        badge_features = badge_counts.merge(badge_by_class, on='UserId', how='outer')
        badge_features = badge_features.merge(user_first_badge, on='UserId', how='outer')
        badge_features = badge_features.merge(badge_diversity, on='UserId', how='outer')
        badge_features = badge_features.merge(user_badge_stats[['UserId', 'badge_rate_per_day']], on='UserId', how='outer')
        badge_features = badge_features.merge(recent_badges, on='UserId', how='outer')
        badge_features = badge_features.merge(badge_quality, on='UserId', how='outer')
        
        # Fill NaN values
        badge_features = badge_features.fillna(0)
        
        # Merge with main dataframe
        self.df_combined = self.df_combined.merge(badge_features, left_on='OwnerUserId', right_on='UserId', how='left')
        
        # Fill NaN values for users without badges
        badge_columns = ['total_badges', 'gold_badges', 'silver_badges', 'bronze_badges', 
                        'tag_based_badges', 'unique_badge_types', 'badge_rate_per_day', 
                        'recent_badges_30d', 'badge_quality_score']
        for col in badge_columns:
            if col in self.df_combined.columns:
                self.df_combined[col] = self.df_combined[col].fillna(0)
        
        # 9. Create categorical badge features
        self.df_combined['badge_level'] = pd.cut(self.df_combined['total_badges'], 
                                               bins=[0, 1, 5, 10, 25, float('inf')], 
                                               labels=['No Badges', 'Beginner', 'Intermediate', 'Advanced', 'Expert'])
        
        self.df_combined['badge_quality_level'] = pd.cut(self.df_combined['badge_quality_score'], 
                                                        bins=[0, 1, 5, 15, 50, float('inf')], 
                                                        labels=['No Quality', 'Low Quality', 'Medium Quality', 'High Quality', 'Premium'])
        
        self.df_combined['has_gold_badges'] = self.df_combined['gold_badges'] > 0
        self.df_combined['has_silver_badges'] = self.df_combined['silver_badges'] > 0
        self.df_combined['has_bronze_badges'] = self.df_combined['bronze_badges'] > 0
        
        # 10. Badge activity indicators
        self.df_combined['is_badge_active'] = self.df_combined['recent_badges_30d'] > 0
        self.df_combined['is_badge_diverse'] = self.df_combined['unique_badge_types'] >= 5
        
        # 11. Badge earning rate categories
        self.df_combined['badge_earning_rate'] = pd.cut(self.df_combined['badge_rate_per_day'], 
                                                       bins=[0, 0.01, 0.05, 0.1, 0.5, float('inf')], 
                                                       labels=['Inactive', 'Slow', 'Moderate', 'Active', 'Very Active'])
        
        print("Badge features created successfully")
        print(f"Badge statistics:")
        print(f"  Users with badges: {(self.df_combined['total_badges'] > 0).sum()}")
        print(f"  Users with gold badges: {self.df_combined['has_gold_badges'].sum()}")
        print(f"  Users with recent activity: {self.df_combined['is_badge_active'].sum()}")
        
        return self.df_combined
    
    def create_user_influence_features(self):
        """Create user influence features based on badges and activity"""
        print("\n=== Creating User Influence Features ===")
        
        if self.df_badges is None or len(self.df_badges) == 0:
            print("No badges data available, skipping influence features")
            return self.df_combined
        
        # 1. Domain influence analysis
        print("Analyzing domain influence...")
        
        # Get all TagBased badges
        tag_based_badges = self.df_badges[self.df_badges['TagBased'] == 'True'].copy()
        
        if len(tag_based_badges) > 0:
            # Count badges for each user in each domain
            domain_influence = tag_based_badges.groupby(['UserId', 'Name']).agg({
                'Class': 'count',  # Number of badges in this domain
                'Id': 'count'      # Total number of badges
            }).reset_index()
            domain_influence.columns = ['UserId', 'Domain', 'domain_badges', 'total_domain_badges']
            
            # Calculate domain influence score (Gold=3 points, Silver=2 points, Bronze=1 point)
            domain_scores = tag_based_badges.groupby(['UserId', 'Name']).apply(
                lambda x: (x['Class'] * 1).sum()
            ).reset_index()
            domain_scores.columns = ['UserId', 'Domain', 'domain_influence_score']
            
            # Merge domain influence data
            domain_influence = domain_influence.merge(domain_scores, on=['UserId', 'Domain'], how='left')
            
            # Get main technical domains
            top_domains = tag_based_badges['Name'].value_counts().head(15).index.tolist()
            print(f"Top domains: {top_domains[:10]}")
            
            # Create influence features for each main domain
            for domain in top_domains:
                domain_data = domain_influence[domain_influence['Domain'] == domain]
                
                # Influence score in this domain
                self.df_combined[f'influence_{domain.lower()}'] = self.df_combined['OwnerUserId'].map(
                    domain_data.set_index('UserId')['domain_influence_score']
                ).fillna(0)
                
                # Number of badges in this domain
                self.df_combined[f'badges_{domain.lower()}'] = self.df_combined['OwnerUserId'].map(
                    domain_data.set_index('UserId')['domain_badges']
                ).fillna(0)
                
                # Whether has influence in this domain (influence score > 0)
                self.df_combined[f'has_influence_{domain.lower()}'] = \
                    self.df_combined[f'influence_{domain.lower()}'] > 0
        
        # 2. Overall influence assessment
        print("Calculating overall influence metrics...")
        
        # Total influence score (based on all badges)
        self.df_combined['total_influence_score'] = (
            self.df_combined['gold_badges'] * 3 + 
            self.df_combined['silver_badges'] * 2 + 
            self.df_combined['bronze_badges'] * 1
        )
        
        # Influence diversity (influence across different domains)
        if len(tag_based_badges) > 0:
            user_domains = tag_based_badges.groupby('UserId')['Name'].nunique().reset_index()
            user_domains.columns = ['UserId', 'influence_domains_count']
            self.df_combined = self.df_combined.merge(user_domains, left_on='OwnerUserId', right_on='UserId', how='left')
            self.df_combined['influence_domains_count'] = self.df_combined['influence_domains_count'].fillna(0)
        else:
            self.df_combined['influence_domains_count'] = 0
        
        # 3. Influence level classification
        print("Creating influence level categories...")
        
        # Overall influence level
        self.df_combined['influence_level'] = pd.cut(
            self.df_combined['total_influence_score'],
            bins=[0, 5, 15, 30, 60, float('inf')],
            labels=['Novice', 'Beginner', 'Intermediate', 'Advanced', 'Expert']
        )
        
        # Multi-domain influence level
        self.df_combined['multi_domain_influence'] = pd.cut(
            self.df_combined['influence_domains_count'],
            bins=[0, 1, 2, 3, 5, float('inf')],
            labels=['Single Domain', 'Dual Domain', 'Triple Domain', 'Multi Domain', 'Cross Domain']
        )
        
        # 4. Professional domain influence indicators
        print("Creating professional influence indicators...")
        
        # Whether is a multi-domain expert
        self.df_combined['is_multi_domain_expert'] = (
            (self.df_combined['influence_domains_count'] >= 3) & 
            (self.df_combined['total_influence_score'] >= 20)
        )
        
        # Whether is an expert in some domain (influence score >= 10)
        self.df_combined['is_domain_expert'] = self.df_combined['total_influence_score'] >= 10
        
        # Influence growth trend (based on badge acquisition time)
        if 'first_badge_date' in self.df_combined.columns:
            # Calculate days from first badge to now
            self.df_combined['influence_span_days'] = (
                pd.Timestamp.now() - self.df_combined['first_badge_date']
            ).dt.days
            
            # Influence growth rate
            self.df_combined['influence_growth_rate'] = self.df_combined['total_influence_score'] / (
                self.df_combined['influence_span_days'] + 1
            )
        else:
            self.df_combined['influence_span_days'] = 0
            self.df_combined['influence_growth_rate'] = 0
        
        # 5. Influence and content quality correlation
        print("Creating influence-content quality correlations...")
        
        # Calculate vote ratio (positive rating)
        if 'Score' in self.df_combined.columns and 'ViewCount' in self.df_combined.columns:
            # Estimate positive rating based on Score and ViewCount
            # Assume Score = upvotes - downvotes, total votes can be estimated as a proportion of ViewCount
            estimated_total_votes = self.df_combined['ViewCount'] * 0.1  # Assume 10% of viewers will vote
            self.df_combined['vote_ratio'] = (self.df_combined['Score'] + estimated_total_votes) / (
                2 * estimated_total_votes + 1e-6
            )
            self.df_combined['vote_ratio'] = self.df_combined['vote_ratio'].clip(0, 1)  # Limit to 0-1 range
        else:
            self.df_combined['vote_ratio'] = 0.5  # Default medium positive rating
        
        # High-quality influence (influence considering positive rating)
        self.df_combined['high_quality_influence'] = (
            self.df_combined['total_influence_score'] * self.df_combined['vote_ratio']
        )
        
        # Influence and post quality correlation features
        if 'ctr_proxy_normalized' in self.df_combined.columns:
            # Influence-weighted content quality
            self.df_combined['influence_weighted_ctr'] = (
                self.df_combined['ctr_proxy_normalized'] * 
                (1 + self.df_combined['total_influence_score'] * 0.01)
            )
        
        # Influence and user reputation correlation
        if 'user_reputation' in self.df_combined.columns:
            self.df_combined['influence_reputation_ratio'] = (
                self.df_combined['total_influence_score'] / 
                (self.df_combined['user_reputation'].astype(float) + 1)
            )
        
        # 6. Influence vector features (for machine learning)
        print("Creating influence vector features...")
        
        # Create influence vector (can be used for concatenation or clustering)
        influence_vector_features = [
            'total_influence_score', 'high_quality_influence', 'influence_domains_count',
            'influence_growth_rate', 'gold_badges', 'silver_badges', 'bronze_badges', 'vote_ratio'
        ]
        
        # Standardize influence vector
        scaler = StandardScaler()
        
        influence_vectors = self.df_combined[influence_vector_features].fillna(0)
        influence_vectors_scaled = scaler.fit_transform(influence_vectors)
        
        # Add standardized influence vector
        for i, feature in enumerate(influence_vector_features):
            self.df_combined[f'{feature}_scaled'] = influence_vectors_scaled[:, i]
        
        # 7. Influence statistics
        print("Influence feature statistics:")
        print(f"  Total influence score - Mean: {self.df_combined['total_influence_score'].mean():.2f}")
        print(f"  High quality influence - Mean: {self.df_combined['high_quality_influence'].mean():.2f}")
        print(f"  Vote ratio - Mean: {self.df_combined['vote_ratio'].mean():.3f}")
        print(f"  Users with domain influence: {(self.df_combined['influence_domains_count'] > 0).sum()}")
        print(f"  Multi-domain experts: {self.df_combined['is_multi_domain_expert'].sum()}")
        print(f"  Domain experts: {self.df_combined['is_domain_expert'].sum()}")
        
        # Influence level distribution
        print("\nInfluence Level Distribution:")
        influence_level_counts = self.df_combined['influence_level'].value_counts()
        for level, count in influence_level_counts.items():
            print(f"  {level}: {count} ({(count/len(self.df_combined)*100):.1f}%)")
        
        # Multi-domain influence distribution
        print("\nMulti-Domain Influence Distribution:")
        multi_domain_counts = self.df_combined['multi_domain_influence'].value_counts()
        for level, count in multi_domain_counts.items():
            print(f"  {level}: {count} ({(count/len(self.df_combined)*100):.1f}%)")
        
        print("User influence features created successfully")
        return self.df_combined
    
    def create_categorical_variables(self):
        """Create categorical and semantic variables"""
        print("\n=== Creating Categorical and Semantic Variables ===")
        
        # 1. Title-based categorical variables
        self.df_combined['title_has_question_mark'] = self.df_combined['Title'].str.contains('\?', na=False)
        self.df_combined['title_has_exclamation'] = self.df_combined['Title'].str.contains('\!', na=False)
        self.df_combined['title_has_code'] = self.df_combined['Title'].str.contains('`|```|code|function|class|def|var|const', case=False, na=False)
        self.df_combined['title_has_error'] = self.df_combined['Title'].str.contains('error|exception|bug|fail|crash|problem|issue', case=False, na=False)
        self.df_combined['title_has_how_to'] = self.df_combined['Title'].str.contains('how|what|why|when|where|which|guide|tutorial|example', case=False, na=False)
        self.df_combined['title_has_version'] = self.df_combined['Title'].str.contains('version|v\d+|\d+\.\d+', case=False, na=False)
        
        # 2. Title length categories
        self.df_combined['title_length_category'] = pd.cut(self.df_combined['title_length'], 
                                                         bins=[0, 5, 10, 20, 50, float('inf')], 
                                                         labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
        
        # 3. Tag-based categorical variables
        top_tags = self.df_tags.sort_values('Count', ascending=False).head(20)
        self.df_combined['has_popular_tags'] = self.df_combined['is_first_tag_popular']
        self.df_combined['tag_count_category'] = pd.cut(self.df_combined['num_tags'], 
                                                      bins=[0, 1, 2, 3, 5, float('inf')], 
                                                      labels=['No Tags', '1 Tag', '2 Tags', '3 Tags', '4+ Tags'])
        
        # 4. Post type classification
        def classify_post_type(title):
            title_lower = str(title).lower()
            if any(word in title_lower for word in ['how', 'what', 'why', 'when', 'where', 'which']):
                return 'Question'
            elif any(word in title_lower for word in ['error', 'exception', 'bug', 'fail', 'crash', 'problem']):
                return 'Error/Issue'
            elif any(word in title_lower for word in ['guide', 'tutorial', 'example', 'demo', 'show']):
                return 'Tutorial/Guide'
            elif any(word in title_lower for word in ['best', 'recommend', 'suggest', 'advice']):
                return 'Recommendation'
            elif any(word in title_lower for word in ['new', 'update', 'release', 'version']):
                return 'News/Update'
            else:
                return 'General'
        
        self.df_combined['post_type'] = self.df_combined['Title'].apply(classify_post_type)
        
        # 5. Time-based categorical variables
        self.df_combined['post_year'] = pd.to_datetime(self.df_combined['CreationDate_x']).dt.year
        self.df_combined['post_month'] = pd.to_datetime(self.df_combined['CreationDate_x']).dt.month
        self.df_combined['post_day_of_week'] = pd.to_datetime(self.df_combined['CreationDate_x']).dt.dayofweek
        self.df_combined['post_hour'] = pd.to_datetime(self.df_combined['CreationDate_x']).dt.hour
        
        # 6. Engagement-based categorical variables
        self.df_combined['engagement_level'] = pd.cut(self.df_combined['total_votes'], 
                                                    bins=[0, 1, 5, 10, 20, 50, float('inf')], 
                                                    labels=['No Engagement', 'Low', 'Medium', 'High', 'Very High', 'Viral'])
        
        self.df_combined['vote_ratio_category'] = pd.cut(self.df_combined['vote_ratio'], 
                                                       bins=[0, 0.3, 0.7, 1.0], 
                                                       labels=['Unpopular', 'Controversial', 'Popular'])
        
        # 7. Click-through rate (CTR) proxy features
        print("Creating click-through rate proxy features...")
        
        # Calculate engagement rate as proxy for CTR
        # Engagement rate = (votes + comments) / post_age_days
        if 'comment_count' in self.df_combined.columns:
            self.df_combined['engagement_rate'] = (self.df_combined['total_votes'] + self.df_combined['comment_count']) / (self.df_combined['post_age_days'] + 1)
        else:
            self.df_combined['engagement_rate'] = self.df_combined['total_votes'] / (self.df_combined['post_age_days'] + 1)
        
        # Click-through rate proxy based on engagement
        # Higher engagement relative to post age suggests higher CTR
        self.df_combined['ctr_proxy'] = self.df_combined['engagement_rate']
        
        # Normalize CTR proxy to 0-1 scale
        max_ctr = self.df_combined['ctr_proxy'].quantile(0.95)  # Use 95th percentile to avoid outliers
        self.df_combined['ctr_proxy_normalized'] = self.df_combined['ctr_proxy'] / max_ctr
        self.df_combined['ctr_proxy_normalized'] = self.df_combined['ctr_proxy_normalized'].clip(0, 1)
        
        # CTR categories
        self.df_combined['ctr_category'] = pd.cut(self.df_combined['ctr_proxy_normalized'], 
                                                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Is high CTR (top 20%)
        self.df_combined['is_high_ctr'] = self.df_combined['ctr_proxy_normalized'] >= 0.8
        
        # Is viral (top 5%)
        self.df_combined['is_viral'] = self.df_combined['ctr_proxy_normalized'] >= 0.95
        
        # CTR score based on multiple factors
        # Factors: engagement rate, vote ratio, comment engagement, post length
        if 'comment_count' in self.df_combined.columns:
            comment_engagement = self.df_combined['comment_count'] / (self.df_combined['comment_count'].max() + 1)
        else:
            comment_engagement = 0
            
        ctr_score = (
            self.df_combined['ctr_proxy_normalized'] * 0.4 +  # Engagement rate (40%)
            self.df_combined['vote_ratio'] * 0.3 +            # Vote ratio (30%)
            comment_engagement * 0.2 +                        # Comment engagement (20%)
            (self.df_combined['post_length'] / (self.df_combined['post_length'].quantile(0.95) + 1)) * 0.1  # Post length (10%)
        )
        self.df_combined['ctr_score'] = ctr_score.clip(0, 1)
        
        # CTR performance categories
        self.df_combined['ctr_performance'] = pd.cut(self.df_combined['ctr_score'], 
                                                   bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                                   labels=['Poor', 'Below Average', 'Average', 'Good', 'Excellent'])
        
        # Time-based CTR features
        # Posts created during peak hours might have higher CTR
        peak_hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        self.df_combined['is_peak_hour'] = self.df_combined['post_hour'].isin(peak_hours)
        
        # Weekend vs weekday CTR
        self.df_combined['is_weekend'] = self.df_combined['post_day_of_week'].isin([5, 6])  # Saturday, Sunday
        
        # Seasonal CTR (based on month)
        spring_months = [3, 4, 5]
        summer_months = [6, 7, 8]
        fall_months = [9, 10, 11]
        winter_months = [12, 1, 2]
        
        self.df_combined['season'] = pd.cut(self.df_combined['post_month'], 
                                          bins=[0, 3, 6, 9, 12], 
                                          labels=['Winter', 'Spring', 'Summer', 'Fall'])
        
        print("Click-through rate proxy features created successfully")
        print(f"CTR proxy statistics:")
        print(f"  Mean CTR proxy: {self.df_combined['ctr_proxy_normalized'].mean():.3f}")
        print(f"  Median CTR proxy: {self.df_combined['ctr_proxy_normalized'].median():.3f}")
        print(f"  High CTR posts: {self.df_combined['is_high_ctr'].sum()} ({(self.df_combined['is_high_ctr'].sum()/len(self.df_combined)*100):.1f}%)")
        print(f"  Viral posts: {self.df_combined['is_viral'].sum()} ({(self.df_combined['is_viral'].sum()/len(self.df_combined)*100):.1f}%)")
        
        print("Categorical variables created successfully")
        return self.df_combined
    
    def clean_text_for_tfidf(self, text):
        """Clean text for TF-IDF processing"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def create_tfidf_features(self, max_features=1000, n_components=100):
        """Create TF-IDF features from titles and tags"""
        print("\n=== Creating TF-IDF Features ===")
        
        # Prepare text data for TF-IDF
        titles = self.df_combined['Title'].fillna('').apply(self.clean_text_for_tfidf)
        tags = self.df_combined['Tags'].fillna('').apply(self.clean_text_for_tfidf)
        
        # Combine titles and tags for richer features
        combined_text = titles + ' ' + tags
        
        print(f"Processing {len(combined_text)} documents for TF-IDF...")
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency (remove very common words)
            ngram_range=(1, 2),  # Use unigrams and bigrams
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_text)
        
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        # Apply TruncatedSVD for dimensionality reduction
        self.tfidf_svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.tfidf_features = self.tfidf_svd.fit_transform(tfidf_matrix)
        
        # Calculate explained variance
        explained_variance = self.tfidf_svd.explained_variance_ratio_.sum()
        print(f"TF-IDF + SVD explained variance: {explained_variance:.3f}")
        
        # Add TF-IDF features to dataframe
        tfidf_df = pd.DataFrame(self.tfidf_features, 
                               columns=[f'tfidf_{i}' for i in range(n_components)])
        
        # Reset index to ensure alignment
        tfidf_df.index = self.df_combined.index
        self.df_combined = pd.concat([self.df_combined, tfidf_df], axis=1)
        
        # Add TF-IDF statistics
        self.df_combined['tfidf_mean'] = tfidf_df.mean(axis=1)
        self.df_combined['tfidf_std'] = tfidf_df.std(axis=1)
        self.df_combined['tfidf_max'] = tfidf_df.max(axis=1)
        self.df_combined['tfidf_min'] = tfidf_df.min(axis=1)
        
        # Show top features
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        top_features_idx = np.argsort(self.tfidf_svd.components_[0])[-10:]
        top_features = [feature_names[i] for i in top_features_idx]
        print(f"Top 10 TF-IDF features: {top_features}")
        
        print(f"TF-IDF features created successfully with shape: {self.tfidf_features.shape}")
        return self.tfidf_features
    
    def create_semantic_embeddings(self):
        """Create semantic embeddings for titles"""
        print("\n=== Creating Semantic Embeddings ===")
        
        # Load sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        titles = self.df_combined['Title'].fillna('').tolist()
        
        print("Encoding titles...")
        self.embeddings = self.model.encode(titles, show_progress_bar=True)
        
        # Calculate embedding statistics
        title_embeddings_df = pd.DataFrame(self.embeddings)
        self.df_combined['title_embedding_mean'] = title_embeddings_df.mean(axis=1)
        self.df_combined['title_embedding_std'] = title_embeddings_df.std(axis=1)
        self.df_combined['title_embedding_max'] = title_embeddings_df.max(axis=1)
        self.df_combined['title_embedding_min'] = title_embeddings_df.min(axis=1)
        
        print(f"Created embeddings with shape: {self.embeddings.shape}")
        return self.embeddings
    
    def visualize_derived_variables(self):
        """Create visualizations for derived variables"""
        print("\n=== Visualizing Derived Variables ===")
        
        plt.figure(figsize=(20, 15))
        
        # Post length distribution
        plt.subplot(3, 4, 1)
        plt.hist(self.df_combined['post_length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.df_combined['post_length'].median(), color='red', linestyle='--', 
                   label=f'Median: {self.df_combined["post_length"].median():.0f}')
        plt.axvline(self.df_combined['post_length'].quantile(0.95), color='orange', linestyle='--', 
                   label=f'95th percentile: {self.df_combined["post_length"].quantile(0.95):.0f}')
        plt.title('Post Length Distribution')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.legend()
        plt.xlim(0, self.df_combined['post_length'].quantile(0.95) * 1.2)
        
        # Vote ratio distribution
        plt.subplot(3, 4, 2)
        plt.hist(self.df_combined['vote_ratio'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title('Vote Ratio Distribution')
        plt.xlabel('Upvotes/Total Votes')
        plt.ylabel('Frequency')
        plt.text(0.1, plt.ylim()[1]*0.9, 
                f'Ratio=0: {(len(self.df_combined[self.df_combined["vote_ratio"] == 0])/len(self.df_combined)*100):.1f}%', fontsize=10)
        plt.text(0.9, plt.ylim()[1]*0.9, 
                f'Ratio=1: {(len(self.df_combined[self.df_combined["vote_ratio"] == 1])/len(self.df_combined)*100):.1f}%', fontsize=10)
        
        # Total votes distribution
        plt.subplot(3, 4, 3)
        plt.hist(self.df_combined['total_votes'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.axvline(self.df_combined['total_votes'].median(), color='red', linestyle='--', 
                   label=f'Median: {self.df_combined["total_votes"].median():.0f}')
        plt.axvline(self.df_combined['total_votes'].quantile(0.95), color='orange', linestyle='--', 
                   label=f'95th percentile: {self.df_combined["total_votes"].quantile(0.95):.0f}')
        plt.title('Total Votes Distribution')
        plt.xlabel('Total Votes')
        plt.ylabel('Frequency')
        plt.legend()
        plt.xlim(0, self.df_combined['total_votes'].quantile(0.95) * 1.2)
        
        # Log-transformed total votes
        plt.subplot(3, 4, 4)
        log_votes = np.log1p(self.df_combined['total_votes'])
        plt.hist(log_votes, bins=50, alpha=0.7, color='gold', edgecolor='black')
        plt.title('Log(Total Votes + 1) Distribution')
        plt.xlabel('Log(Total Votes + 1)')
        plt.ylabel('Frequency')
        
        # Comment count vs Vote ratio (if available)
        if 'comment_count' in self.df_combined.columns:
            plt.subplot(3, 4, 5)
            plt.scatter(self.df_combined['comment_count'], self.df_combined['vote_ratio'], alpha=0.3, s=10)
            z = np.polyfit(self.df_combined['comment_count'], self.df_combined['vote_ratio'], 1)
            p = np.poly1d(z)
            plt.plot(self.df_combined['comment_count'], p(self.df_combined['comment_count']), "r--", alpha=0.8, linewidth=2)
            plt.title('Comment Count vs Vote Ratio')
            plt.xlabel('Comment Count')
            plt.ylabel('Vote Ratio')
            plt.xlim(0, self.df_combined['comment_count'].quantile(0.95))
        
        # Post length vs Total votes
        plt.subplot(3, 4, 6)
        plt.scatter(self.df_combined['post_length'], np.log1p(self.df_combined['total_votes']), alpha=0.3, s=10)
        plt.title('Post Length vs Log(Total Votes + 1)')
        plt.xlabel('Post Length (words)')
        plt.ylabel('Log(Total Votes + 1)')
        plt.xlim(0, self.df_combined['post_length'].quantile(0.95))
        
        # Vote ratio by year
        plt.subplot(3, 4, 7)
        self.df_combined['year'] = pd.to_datetime(self.df_combined['CreationDate_x']).dt.year
        vote_ratio_by_year = self.df_combined.groupby('year')['vote_ratio'].mean()
        plt.plot(vote_ratio_by_year.index, vote_ratio_by_year.values, marker='o', linewidth=2, markersize=6)
        plt.title('Average Vote Ratio by Year')
        plt.xlabel('Year')
        plt.ylabel('Average Vote Ratio')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_ctr_features(self):
        """Create visualizations for click-through rate features"""
        print("\n=== Visualizing Click-Through Rate Features ===")
        
        if 'ctr_proxy_normalized' not in self.df_combined.columns:
            print("CTR features not available. Run create_categorical_variables() first.")
            return
        
        plt.figure(figsize=(20, 15))
        
        # CTR proxy distribution
        plt.subplot(3, 4, 1)
        plt.hist(self.df_combined['ctr_proxy_normalized'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(self.df_combined['ctr_proxy_normalized'].median(), color='red', linestyle='--', 
                   label=f'Median: {self.df_combined["ctr_proxy_normalized"].median():.3f}')
        plt.title('CTR Proxy Distribution')
        plt.xlabel('Normalized CTR Proxy')
        plt.ylabel('Frequency')
        plt.legend()
        
        # CTR score distribution
        plt.subplot(3, 4, 2)
        plt.hist(self.df_combined['ctr_score'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(self.df_combined['ctr_score'].median(), color='red', linestyle='--', 
                   label=f'Median: {self.df_combined["ctr_score"].median():.3f}')
        plt.title('CTR Score Distribution')
        plt.xlabel('CTR Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        # CTR categories distribution
        plt.subplot(3, 4, 3)
        ctr_category_counts = self.df_combined['ctr_category'].value_counts()
        plt.bar(ctr_category_counts.index, ctr_category_counts.values, color='lightblue', alpha=0.7)
        plt.title('CTR Categories Distribution')
        plt.xlabel('CTR Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # CTR performance distribution
        plt.subplot(3, 4, 4)
        ctr_performance_counts = self.df_combined['ctr_performance'].value_counts()
        plt.bar(ctr_performance_counts.index, ctr_performance_counts.values, color='lightgreen', alpha=0.7)
        plt.title('CTR Performance Distribution')
        plt.xlabel('CTR Performance')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # CTR vs Total Votes
        plt.subplot(3, 4, 5)
        plt.scatter(self.df_combined['ctr_proxy_normalized'], np.log1p(self.df_combined['total_votes']), 
                   alpha=0.3, s=10, c=self.df_combined['is_high_ctr'], cmap='viridis')
        plt.title('CTR Proxy vs Log(Total Votes + 1)')
        plt.xlabel('Normalized CTR Proxy')
        plt.ylabel('Log(Total Votes + 1)')
        plt.colorbar(label='Is High CTR')
        
        # CTR vs Vote Ratio
        plt.subplot(3, 4, 6)
        plt.scatter(self.df_combined['ctr_proxy_normalized'], self.df_combined['vote_ratio'], 
                   alpha=0.3, s=10, c=self.df_combined['is_viral'], cmap='plasma')
        plt.title('CTR Proxy vs Vote Ratio')
        plt.xlabel('Normalized CTR Proxy')
        plt.ylabel('Vote Ratio')
        plt.colorbar(label='Is Viral')
        
        # CTR by Post Type
        plt.subplot(3, 4, 7)
        ctr_by_type = self.df_combined.groupby('post_type')['ctr_proxy_normalized'].mean().sort_values(ascending=False)
        plt.bar(ctr_by_type.index, ctr_by_type.values, color='lightcoral', alpha=0.7)
        plt.title('Average CTR by Post Type')
        plt.xlabel('Post Type')
        plt.ylabel('Average CTR Proxy')
        plt.xticks(rotation=45)
        
        # CTR by Season
        plt.subplot(3, 4, 8)
        ctr_by_season = self.df_combined.groupby('season')['ctr_proxy_normalized'].mean()
        plt.bar(ctr_by_season.index, ctr_by_season.values, color='gold', alpha=0.7)
        plt.title('Average CTR by Season')
        plt.xlabel('Season')
        plt.ylabel('Average CTR Proxy')
        
        # CTR by Hour of Day
        plt.subplot(3, 4, 9)
        ctr_by_hour = self.df_combined.groupby('post_hour')['ctr_proxy_normalized'].mean()
        plt.plot(ctr_by_hour.index, ctr_by_hour.values, marker='o', linewidth=2, markersize=6, color='blue')
        plt.title('Average CTR by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average CTR Proxy')
        plt.grid(True, alpha=0.3)
        
        # CTR by Day of Week
        plt.subplot(3, 4, 10)
        ctr_by_day = self.df_combined.groupby('post_day_of_week')['ctr_proxy_normalized'].mean()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        plt.bar(range(len(ctr_by_day)), ctr_by_day.values, color='lightsteelblue', alpha=0.7)
        plt.title('Average CTR by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Average CTR Proxy')
        plt.xticks(range(len(day_names)), day_names, rotation=45)
        
        # High CTR posts by tag
        plt.subplot(3, 4, 11)
        high_ctr_tags = self.df_combined[self.df_combined['is_high_ctr']]['first_tag'].value_counts().head(10)
        plt.barh(range(len(high_ctr_tags)), high_ctr_tags.values, color='lightpink', alpha=0.7)
        plt.yticks(range(len(high_ctr_tags)), high_ctr_tags.index)
        plt.title('Top 10 Tags for High CTR Posts')
        plt.xlabel('Count')
        
        # Viral posts by tag
        plt.subplot(3, 4, 12)
        viral_tags = self.df_combined[self.df_combined['is_viral']]['first_tag'].value_counts().head(10)
        plt.barh(range(len(viral_tags)), viral_tags.values, color='lightyellow', alpha=0.7)
        plt.yticks(range(len(viral_tags)), viral_tags.index)
        plt.title('Top 10 Tags for Viral Posts')
        plt.xlabel('Count')
        
        plt.tight_layout()
        plt.show()
        
        # Print CTR statistics
        print("\n=== CTR Feature Statistics ===")
        print(f"CTR Proxy - Mean: {self.df_combined['ctr_proxy_normalized'].mean():.3f}, Median: {self.df_combined['ctr_proxy_normalized'].median():.3f}")
        print(f"CTR Score - Mean: {self.df_combined['ctr_score'].mean():.3f}, Median: {self.df_combined['ctr_score'].median():.3f}")
        print(f"High CTR posts: {self.df_combined['is_high_ctr'].sum()} ({(self.df_combined['is_high_ctr'].sum()/len(self.df_combined)*100):.1f}%)")
        print(f"Viral posts: {self.df_combined['is_viral'].sum()} ({(self.df_combined['is_viral'].sum()/len(self.df_combined)*100):.1f}%)")
        
        # Top performing post types
        print("\nTop performing post types by CTR:")
        top_types = self.df_combined.groupby('post_type')['ctr_proxy_normalized'].mean().sort_values(ascending=False)
        for post_type, ctr in top_types.items():
            print(f"  {post_type}: {ctr:.3f}")
        
        # Peak hours analysis
        peak_ctr = self.df_combined[self.df_combined['is_peak_hour']]['ctr_proxy_normalized'].mean()
        off_peak_ctr = self.df_combined[~self.df_combined['is_peak_hour']]['ctr_proxy_normalized'].mean()
        print(f"\nPeak hours CTR: {peak_ctr:.3f}")
        print(f"Off-peak hours CTR: {off_peak_ctr:.3f}")
        print(f"Peak hours advantage: {((peak_ctr/off_peak_ctr)-1)*100:.1f}%")
    
    def visualize_badge_features(self):
        """Create visualizations for badge features"""
        print("\n=== Visualizing Badge Features ===")
        
        if 'total_badges' not in self.df_combined.columns:
            print("Badge features not available. Run create_badge_features() first.")
            return
        
        plt.figure(figsize=(20, 15))
        
        # Badge distribution
        plt.subplot(3, 4, 1)
        plt.hist(self.df_combined['total_badges'], bins=50, alpha=0.7, color='gold', edgecolor='black')
        plt.axvline(self.df_combined['total_badges'].median(), color='red', linestyle='--', 
                   label=f'Median: {self.df_combined["total_badges"].median():.0f}')
        plt.title('Total Badges Distribution')
        plt.xlabel('Number of Badges')
        plt.ylabel('Frequency')
        plt.legend()
        plt.xlim(0, self.df_combined['total_badges'].quantile(0.95))
        
        # Badge quality score distribution
        plt.subplot(3, 4, 2)
        plt.hist(self.df_combined['badge_quality_score'], bins=50, alpha=0.7, color='silver', edgecolor='black')
        plt.axvline(self.df_combined['badge_quality_score'].median(), color='red', linestyle='--', 
                   label=f'Median: {self.df_combined["badge_quality_score"].median():.0f}')
        plt.title('Badge Quality Score Distribution')
        plt.xlabel('Badge Quality Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Badge levels distribution
        plt.subplot(3, 4, 3)
        badge_level_counts = self.df_combined['badge_level'].value_counts()
        plt.bar(badge_level_counts.index, badge_level_counts.values, color='lightblue', alpha=0.7)
        plt.title('Badge Levels Distribution')
        plt.xlabel('Badge Level')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Badge quality levels distribution
        plt.subplot(3, 4, 4)
        quality_level_counts = self.df_combined['badge_quality_level'].value_counts()
        plt.bar(quality_level_counts.index, quality_level_counts.values, color='lightgreen', alpha=0.7)
        plt.title('Badge Quality Levels Distribution')
        plt.xlabel('Quality Level')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Badges by class
        plt.subplot(3, 4, 5)
        badge_classes = ['gold_badges', 'silver_badges', 'bronze_badges']
        badge_class_means = [self.df_combined[col].mean() for col in badge_classes]
        plt.bar(['Gold', 'Silver', 'Bronze'], badge_class_means, color=['gold', 'silver', 'brown'], alpha=0.7)
        plt.title('Average Badges by Class')
        plt.xlabel('Badge Class')
        plt.ylabel('Average Count')
        
        # Badge earning rate distribution
        plt.subplot(3, 4, 6)
        earning_rate_counts = self.df_combined['badge_earning_rate'].value_counts()
        plt.bar(earning_rate_counts.index, earning_rate_counts.values, color='lightcoral', alpha=0.7)
        plt.title('Badge Earning Rate Distribution')
        plt.xlabel('Earning Rate')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Badges vs Reputation
        plt.subplot(3, 4, 7)
        if 'user_reputation' in self.df_combined.columns:
            plt.scatter(self.df_combined['user_reputation'], self.df_combined['total_badges'], 
                       alpha=0.3, s=10, c=self.df_combined['has_gold_badges'], cmap='viridis')
            plt.title('Badges vs User Reputation')
            plt.xlabel('User Reputation')
            plt.ylabel('Total Badges')
            plt.colorbar(label='Has Gold Badges')
        else:
            plt.text(0.5, 0.5, 'User reputation not available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Badges vs User Reputation')
        
        # Badges vs Post Quality (CTR)
        plt.subplot(3, 4, 8)
        if 'ctr_proxy_normalized' in self.df_combined.columns:
            plt.scatter(self.df_combined['badge_quality_score'], self.df_combined['ctr_proxy_normalized'], 
                       alpha=0.3, s=10, c=self.df_combined['is_badge_active'], cmap='plasma')
            plt.title('Badge Quality vs CTR')
            plt.xlabel('Badge Quality Score')
            plt.ylabel('CTR Proxy')
            plt.colorbar(label='Is Badge Active')
        else:
            plt.text(0.5, 0.5, 'CTR features not available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Badge Quality vs CTR')
        
        # Badge diversity vs post count
        plt.subplot(3, 4, 9)
        plt.scatter(self.df_combined['unique_badge_types'], self.df_combined['user_post_count'], 
                   alpha=0.3, s=10, c=self.df_combined['is_badge_diverse'], cmap='viridis')
        plt.title('Badge Diversity vs Post Count')
        plt.xlabel('Unique Badge Types')
        plt.ylabel('User Post Count')
        plt.colorbar(label='Is Badge Diverse')
        
        # Recent badge activity
        plt.subplot(3, 4, 10)
        plt.hist(self.df_combined['recent_badges_30d'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.title('Recent Badges (30 days)')
        plt.xlabel('Recent Badges Count')
        plt.ylabel('Frequency')
        plt.xlim(0, self.df_combined['recent_badges_30d'].quantile(0.95))
        
        # Badge earning rate over time
        plt.subplot(3, 4, 11)
        plt.hist(self.df_combined['badge_rate_per_day'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(self.df_combined['badge_rate_per_day'].median(), color='red', linestyle='--', 
                   label=f'Median: {self.df_combined["badge_rate_per_day"].median():.3f}')
        plt.title('Badge Earning Rate Distribution')
        plt.xlabel('Badges per Day')
        plt.ylabel('Frequency')
        plt.legend()
        plt.xlim(0, self.df_combined['badge_rate_per_day'].quantile(0.95))
        
        # Badge activity by post type
        plt.subplot(3, 4, 12)
        if 'post_type' in self.df_combined.columns:
            badge_by_type = self.df_combined.groupby('post_type')['total_badges'].mean().sort_values(ascending=False)
            plt.bar(badge_by_type.index, badge_by_type.values, color='lightsteelblue', alpha=0.7)
            plt.title('Average Badges by Post Type')
            plt.xlabel('Post Type')
            plt.ylabel('Average Badges')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'Post type not available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Average Badges by Post Type')
        
        plt.tight_layout()
        plt.show()
        
        # Print badge statistics
        print("\n=== Badge Feature Statistics ===")
        print(f"Total Badges - Mean: {self.df_combined['total_badges'].mean():.2f}, Median: {self.df_combined['total_badges'].median():.2f}")
        print(f"Badge Quality Score - Mean: {self.df_combined['badge_quality_score'].mean():.2f}, Median: {self.df_combined['badge_quality_score'].median():.2f}")
        print(f"Users with badges: {(self.df_combined['total_badges'] > 0).sum()} ({(self.df_combined['total_badges'] > 0).sum()/len(self.df_combined)*100:.1f}%)")
        print(f"Users with gold badges: {self.df_combined['has_gold_badges'].sum()} ({self.df_combined['has_gold_badges'].sum()/len(self.df_combined)*100:.1f}%)")
        print(f"Users with silver badges: {self.df_combined['has_silver_badges'].sum()} ({self.df_combined['has_silver_badges'].sum()/len(self.df_combined)*100:.1f}%)")
        print(f"Users with bronze badges: {self.df_combined['has_bronze_badges'].sum()} ({self.df_combined['has_bronze_badges'].sum()/len(self.df_combined)*100:.1f}%)")
        print(f"Users with recent activity: {self.df_combined['is_badge_active'].sum()} ({self.df_combined['is_badge_active'].sum()/len(self.df_combined)*100:.1f}%)")
        
        # Badge levels breakdown
        print("\nBadge Levels:")
        badge_level_counts = self.df_combined['badge_level'].value_counts()
        for level, count in badge_level_counts.items():
            print(f"  {level}: {count} ({(count/len(self.df_combined)*100):.1f}%)")
        
        # Badge quality breakdown
        print("\nBadge Quality Levels:")
        quality_level_counts = self.df_combined['badge_quality_level'].value_counts()
        for level, count in quality_level_counts.items():
            print(f"  {level}: {count} ({(count/len(self.df_combined)*100):.1f}%)")
        
        # Badge earning rate breakdown
        print("\nBadge Earning Rates:")
        earning_rate_counts = self.df_combined['badge_earning_rate'].value_counts()
        for rate, count in earning_rate_counts.items():
            print(f"  {rate}: {count} ({(count/len(self.df_combined)*100):.1f}%)")
    
    def visualize_tfidf_features(self):
        """Create visualizations for TF-IDF features"""
        print("\n=== Visualizing TF-IDF Features ===")
        
        if self.tfidf_features is None:
            print("TF-IDF features not available. Run create_tfidf_features() first.")
            return
        
        plt.figure(figsize=(20, 12))
        
        # TF-IDF feature importance (first component)
        plt.subplot(2, 3, 1)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        top_features_idx = np.argsort(self.tfidf_svd.components_[0])[-15:]
        top_features = [feature_names[i] for i in top_features_idx]
        top_importance = self.tfidf_svd.components_[0][top_features_idx]
        
        plt.barh(range(len(top_features)), top_importance, color='skyblue')
        plt.yticks(range(len(top_features)), top_features)
        plt.title('Top 15 TF-IDF Features (Component 1)')
        plt.xlabel('Feature Importance')
        
        # TF-IDF statistics distribution
        plt.subplot(2, 3, 2)
        plt.hist(self.df_combined['tfidf_mean'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title('TF-IDF Mean Distribution')
        plt.xlabel('Mean TF-IDF Value')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 3, 3)
        plt.hist(self.df_combined['tfidf_std'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('TF-IDF Standard Deviation Distribution')
        plt.xlabel('TF-IDF Standard Deviation')
        plt.ylabel('Frequency')
        
        # TF-IDF vs engagement metrics
        plt.subplot(2, 3, 4)
        plt.scatter(self.df_combined['tfidf_mean'], np.log1p(self.df_combined['total_votes']), alpha=0.3, s=10)
        plt.title('TF-IDF Mean vs Log(Total Votes + 1)')
        plt.xlabel('TF-IDF Mean')
        plt.ylabel('Log(Total Votes + 1)')
        
        # TF-IDF vs post length
        plt.subplot(2, 3, 5)
        plt.scatter(self.df_combined['tfidf_mean'], self.df_combined['post_length'], alpha=0.3, s=10)
        plt.title('TF-IDF Mean vs Post Length')
        plt.xlabel('TF-IDF Mean')
        plt.ylabel('Post Length (words)')
        plt.xlim(0, self.df_combined['tfidf_mean'].quantile(0.95))
        plt.ylim(0, self.df_combined['post_length'].quantile(0.95))
        
        # Explained variance by components
        plt.subplot(2, 3, 6)
        explained_variance = self.tfidf_svd.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', linewidth=2)
        plt.axhline(y=0.8, color='red', linestyle='--', label='80% Variance')
        plt.axhline(y=0.9, color='orange', linestyle='--', label='90% Variance')
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary_statistics(self):
        """Print summary statistics of the processed data"""
        print("\n=== Summary Statistics ===")
        print(f"Post length - Mean: {self.df_combined['post_length'].mean():.2f}, Max: {self.df_combined['post_length'].max()}")
        print(f"Post length - Median: {self.df_combined['post_length'].median():.2f}, 95th percentile: {self.df_combined['post_length'].quantile(0.95):.2f}")
        print(f"Vote ratio - Mean: {self.df_combined['vote_ratio'].mean():.3f}, Max: {self.df_combined['vote_ratio'].max():.3f}")
        print(f"Total votes - Mean: {self.df_combined['total_votes'].mean():.2f}, Max: {self.df_combined['total_votes'].max()}")
        print(f"Total votes - Median: {self.df_combined['total_votes'].median():.2f}, 95th percentile: {self.df_combined['total_votes'].quantile(0.95):.2f}")
        
        if 'comment_count' in self.df_combined.columns:
            print(f"Comment count - Mean: {self.df_combined['comment_count'].mean():.2f}, Max: {self.df_combined['comment_count'].max()}")
            print(f"Comment count - Median: {self.df_combined['comment_count'].median():.2f}, 95th percentile: {self.df_combined['comment_count'].quantile(0.95):.2f}")
        
        print("\n=== Categorical Variables Summary ===")
        print("Title Features:")
        print(f"Questions (with ?): {self.df_combined['title_has_question_mark'].sum()} ({(self.df_combined['title_has_question_mark'].sum()/len(self.df_combined)*100):.1f}%)")
        print(f"Code-related titles: {self.df_combined['title_has_code'].sum()} ({(self.df_combined['title_has_code'].sum()/len(self.df_combined)*100):.1f}%)")
        print(f"Error-related titles: {self.df_combined['title_has_error'].sum()} ({(self.df_combined['title_has_error'].sum()/len(self.df_combined)*100):.1f}%)")
        print(f"How-to titles: {self.df_combined['title_has_how_to'].sum()} ({(self.df_combined['title_has_how_to'].sum()/len(self.df_combined)*100):.1f}%)")
        
        print("\nPost Types:")
        post_type_counts = self.df_combined['post_type'].value_counts()
        for post_type, count in post_type_counts.items():
            print(f"{post_type}: {count} ({(count/len(self.df_combined)*100):.1f}%)")
        
        print("\nEngagement Levels:")
        engagement_counts = self.df_combined['engagement_level'].value_counts()
        for level, count in engagement_counts.items():
            print(f"{level}: {count} ({(count/len(self.df_combined)*100):.1f}%)")
    
        # Add CTR feature statistics
        if 'ctr_proxy_normalized' in self.df_combined.columns:
            print("\n=== CTR Feature Summary ===")
            print(f"CTR Proxy - Mean: {self.df_combined['ctr_proxy_normalized'].mean():.3f}, Median: {self.df_combined['ctr_proxy_normalized'].median():.3f}")
            print(f"CTR Score - Mean: {self.df_combined['ctr_score'].mean():.3f}, Median: {self.df_combined['ctr_score'].median():.3f}")
            print(f"High CTR posts: {self.df_combined['is_high_ctr'].sum()} ({(self.df_combined['is_high_ctr'].sum()/len(self.df_combined)*100):.1f}%)")
            print(f"Viral posts: {self.df_combined['is_viral'].sum()} ({(self.df_combined['is_viral'].sum()/len(self.df_combined)*100):.1f}%)")
            
            print("\nCTR Categories:")
            ctr_category_counts = self.df_combined['ctr_category'].value_counts()
            for category, count in ctr_category_counts.items():
                print(f"{category}: {count} ({(count/len(self.df_combined)*100):.1f}%)")
            
            print("\nCTR Performance:")
            ctr_performance_counts = self.df_combined['ctr_performance'].value_counts()
            for performance, count in ctr_performance_counts.items():
                print(f"{performance}: {count} ({(count/len(self.df_combined)*100):.1f}%)")
            
            print("\nTime-based CTR Features:")
            print(f"Peak hour posts: {self.df_combined['is_peak_hour'].sum()} ({(self.df_combined['is_peak_hour'].sum()/len(self.df_combined)*100):.1f}%)")
            print(f"Weekend posts: {self.df_combined['is_weekend'].sum()} ({(self.df_combined['is_weekend'].sum()/len(self.df_combined)*100):.1f}%)")
            
            print("\nSeasonal Distribution:")
            season_counts = self.df_combined['season'].value_counts()
            for season, count in season_counts.items():
                print(f"{season}: {count} ({(count/len(self.df_combined)*100):.1f}%)")
        
        # Add badge feature statistics
        if 'total_badges' in self.df_combined.columns:
            print("\n=== Badge Feature Summary ===")
            print(f"Total Badges - Mean: {self.df_combined['total_badges'].mean():.2f}, Median: {self.df_combined['total_badges'].median():.2f}")
            print(f"Badge Quality Score - Mean: {self.df_combined['badge_quality_score'].mean():.2f}, Median: {self.df_combined['badge_quality_score'].median():.2f}")
            print(f"Users with badges: {(self.df_combined['total_badges'] > 0).sum()} ({(self.df_combined['total_badges'] > 0).sum()/len(self.df_combined)*100:.1f}%)")
            print(f"Users with gold badges: {self.df_combined['has_gold_badges'].sum()} ({self.df_combined['has_gold_badges'].sum()/len(self.df_combined)*100:.1f}%)")
            print(f"Users with silver badges: {self.df_combined['has_silver_badges'].sum()} ({self.df_combined['has_silver_badges'].sum()/len(self.df_combined)*100:.1f}%)")
            print(f"Users with bronze badges: {self.df_combined['has_bronze_badges'].sum()} ({self.df_combined['has_bronze_badges'].sum()/len(self.df_combined)*100:.1f}%)")
            print(f"Users with recent activity: {self.df_combined['is_badge_active'].sum()} ({self.df_combined['is_badge_active'].sum()/len(self.df_combined)*100:.1f}%)")
            print(f"Users with diverse badges: {self.df_combined['is_badge_diverse'].sum()} ({self.df_combined['is_badge_diverse'].sum()/len(self.df_combined)*100:.1f}%)")
            
            print("\nBadge Levels:")
            badge_level_counts = self.df_combined['badge_level'].value_counts()
            for level, count in badge_level_counts.items():
                print(f"  {level}: {count} ({(count/len(self.df_combined)*100):.1f}%)")
            
            print("\nBadge Quality Levels:")
            quality_level_counts = self.df_combined['badge_quality_level'].value_counts()
            for level, count in quality_level_counts.items():
                print(f"  {level}: {count} ({(count/len(self.df_combined)*100):.1f}%)")
            
            print("\nBadge Earning Rates:")
            earning_rate_counts = self.df_combined['badge_earning_rate'].value_counts()
            for rate, count in earning_rate_counts.items():
                print(f"  {rate}: {count} ({(count/len(self.df_combined)*100):.1f}%)")
            
            # Badge class breakdown
            print("\nBadge Classes:")
            print(f"  Gold badges - Mean: {self.df_combined['gold_badges'].mean():.2f}, Max: {self.df_combined['gold_badges'].max()}")
            print(f"  Silver badges - Mean: {self.df_combined['silver_badges'].mean():.2f}, Max: {self.df_combined['silver_badges'].max()}")
            print(f"  Bronze badges - Mean: {self.df_combined['bronze_badges'].mean():.2f}, Max: {self.df_combined['bronze_badges'].max()}")
            print(f"  Tag-based badges - Mean: {self.df_combined['tag_based_badges'].mean():.2f}, Max: {self.df_combined['tag_based_badges'].max()}")
            
            # Badge activity metrics
            print("\nBadge Activity Metrics:")
            print(f"  Unique badge types - Mean: {self.df_combined['unique_badge_types'].mean():.2f}, Max: {self.df_combined['unique_badge_types'].max()}")
            print(f"  Badge earning rate - Mean: {self.df_combined['badge_rate_per_day'].mean():.3f}, Max: {self.df_combined['badge_rate_per_day'].max():.3f}")
            print(f"  Recent badges (30d) - Mean: {self.df_combined['recent_badges_30d'].mean():.2f}, Max: {self.df_combined['recent_badges_30d'].max()}")
    
    def preprocess_all(self, include_normalization=True, normalization_config=None):
        """Run complete preprocessing pipeline"""
        print("=== Starting Complete Data Preprocessing Pipeline ===")
        
        # Load data
        self.load_data()
        
        # Basic visualization
        top_tags = self.basic_visualization()
        
        # Clean data
        self.clean_data()
        
        # Create derived variables
        self.create_derived_variables()
        
        # Create badge features
        self.create_badge_features()
        
        # Create user influence features
        self.create_user_influence_features()
        
        # Create categorical variables
        self.create_categorical_variables()
        
        # Create TF-IDF features
        self.create_tfidf_features()
        
        # Create semantic embeddings
        self.create_semantic_embeddings()
        
        # Visualize results
        self.visualize_derived_variables()
        
        # Visualize CTR features
        self.visualize_ctr_features()
        
        # Visualize badge features
        self.visualize_badge_features()
        
        # Visualize TF-IDF features
        self.visualize_tfidf_features()
        
        # Print summary
        self.print_summary_statistics()
        
        # Feature normalization (optional)
        if include_normalization:
            self.normalize_features(normalization_config)
            self.get_normalized_features_summary()
        
        print("\n=== Data Preprocessing Complete ===")
        return self.df_combined, self.embeddings, self.tfidf_features, self.model
    
    def normalize_features(self, normalization_config=None):
        """Comprehensive feature normalization pipeline"""
        print("\n=== Feature Normalization ===")
        
        if normalization_config is None:
            normalization_config = {
                'numerical_method': 'standard',  # 'standard', 'minmax', 'robust', 'power'
                'categorical_method': 'label',   # 'label', 'onehot'
                'create_interactions': True,
                'create_polynomials': True,
                'polynomial_degree': 2
            }
        
        # Step 1: Analyze features
        self._analyze_features()
        
        # Step 2: Normalize numerical features
        self._normalize_numerical_features(method=normalization_config['numerical_method'])
        
        # Step 3: Encode categorical features
        self._encode_categorical_features(method=normalization_config['categorical_method'])
        
        # Step 4: Create interaction features
        if normalization_config['create_interactions']:
            self._create_interaction_features()
        
        # Step 5: Create polynomial features
        if normalization_config['create_polynomials']:
            self._create_polynomial_features(degree=normalization_config['polynomial_degree'])
        
        # Summary
        print(f"\n=== Normalization Summary ===")
        print(f"Features after normalization: {len(self.df_combined.columns)}")
        
        return self.df_combined
    
    def _analyze_features(self):
        """Analyze feature distributions and suggest normalization methods"""
        print("=== Feature Analysis ===")
        
        # Categorize features
        numerical_features = []
        categorical_features = []
        binary_features = []
        
        for col in self.df_combined.columns:
            if self.df_combined[col].dtype in ['int64', 'float64']:
                if self.df_combined[col].nunique() == 2:
                    binary_features.append(col)
                else:
                    numerical_features.append(col)
            elif self.df_combined[col].dtype == 'object':
                categorical_features.append(col)
        
        print(f"Numerical features: {len(numerical_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        print(f"Binary features: {len(binary_features)}")
        
        # Analyze numerical features
        print("\n=== Numerical Feature Analysis ===")
        for feature in numerical_features[:10]:  # Show first 10
            stats = self.df_combined[feature].describe()
            print(f"{feature}:")
            print(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
            print(f"  Min: {stats['min']:.3f}, Max: {stats['max']:.3f}")
            print(f"  Skewness: {self.df_combined[feature].skew():.3f}")
            
            # Suggest normalization method
            if abs(self.df_combined[feature].skew()) > 1:
                print(f"  → Suggest: PowerTransformer (skewed)")
            elif self.df_combined[feature].std() > self.df_combined[feature].mean():
                print(f"  → Suggest: RobustScaler (high variance)")
            else:
                print(f"  → Suggest: StandardScaler")
        
        self.feature_analysis = {
            'numerical': numerical_features,
            'categorical': categorical_features,
            'binary': binary_features
        }
    
    def _normalize_numerical_features(self, method='standard'):
        """Normalize numerical features using specified method"""
        print(f"\n=== Normalizing Numerical Features ({method}) ===")
        
        # Get numerical features
        numerical_features = self.df_combined.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove already normalized features
        numerical_features = [f for f in numerical_features if not f.endswith('_scaled') and not f.endswith('_normalized')]
        
        # Remove binary features
        binary_features = [f for f in numerical_features if self.df_combined[f].nunique() == 2]
        numerical_features = [f for f in numerical_features if f not in binary_features]
        
        print(f"Normalizing {len(numerical_features)} numerical features")
        
        # Initialize scalers dictionary if not exists
        if not hasattr(self, 'scalers'):
            self.scalers = {}
        
        for feature in numerical_features:
            if feature in self.df_combined.columns and self.df_combined[feature].notna().sum() > 0:
                try:
                    if method == 'standard':
                        scaler = StandardScaler()
                        self.df_combined[f'{feature}_scaled'] = scaler.fit_transform(
                            self.df_combined[feature].values.reshape(-1, 1)
                        )
                        self.scalers[f'{feature}_scaled'] = scaler
                        
                    elif method == 'minmax':
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                        self.df_combined[f'{feature}_normalized'] = scaler.fit_transform(
                            self.df_combined[feature].values.reshape(-1, 1)
                        )
                        self.scalers[f'{feature}_normalized'] = scaler
                        
                    elif method == 'robust':
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                        self.df_combined[f'{feature}_robust'] = scaler.fit_transform(
                            self.df_combined[feature].values.reshape(-1, 1)
                        )
                        self.scalers[f'{feature}_robust'] = scaler
                        
                    elif method == 'power':
                        from sklearn.preprocessing import PowerTransformer
                        scaler = PowerTransformer(method='yeo-johnson')
                        self.df_combined[f'{feature}_power'] = scaler.fit_transform(
                            self.df_combined[feature].values.reshape(-1, 1)
                        )
                        self.scalers[f'{feature}_power'] = scaler
                        
                except Exception as e:
                    print(f"Error normalizing {feature}: {e}")
    
    def _encode_categorical_features(self, method='label'):
        """Encode categorical features"""
        print(f"\n=== Encoding Categorical Features ({method}) ===")
        
        # Get categorical features
        categorical_features = self.df_combined.select_dtypes(include=['object']).columns.tolist()
        
        # Remove already encoded features
        categorical_features = [f for f in categorical_features if not f.endswith('_encoded')]
        
        print(f"Encoding {len(categorical_features)} categorical features")
        
        # Initialize label encoders dictionary if not exists
        if not hasattr(self, 'label_encoders'):
            self.label_encoders = {}
        
        for feature in categorical_features:
            if feature in self.df_combined.columns and self.df_combined[feature].notna().sum() > 0:
                try:
                    if method == 'label':
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        self.df_combined[f'{feature}_encoded'] = le.fit_transform(
                            self.df_combined[feature].fillna('Unknown')
                        )
                        self.label_encoders[feature] = le
                        
                    elif method == 'onehot':
                        # For one-hot encoding, we'll create separate columns
                        unique_values = self.df_combined[feature].value_counts().head(10).index  # Top 10 values
                        for value in unique_values:
                            self.df_combined[f'{feature}_{value}'] = (self.df_combined[feature] == value).astype(int)
                        
                except Exception as e:
                    print(f"Error encoding {feature}: {e}")
    
    def _create_interaction_features(self):
        """Create interaction features between important features"""
        print("\n=== Creating Interaction Features ===")
        
        # Define important feature pairs for interactions
        interaction_pairs = [
            ('total_influence_score', 'vote_ratio'),
            ('total_influence_score', 'user_post_count'),
            ('total_influence_score', 'post_age_days'),
            ('high_quality_influence', 'ctr_proxy_normalized'),
            ('influence_domains_count', 'total_badges'),
            ('gold_badges', 'silver_badges'),
            ('Score', 'ViewCount'),
            ('AnswerCount', 'CommentCount')
        ]
        
        for feature1, feature2 in interaction_pairs:
            if feature1 in self.df_combined.columns and feature2 in self.df_combined.columns:
                try:
                    # Multiplication interaction
                    self.df_combined[f'{feature1}_x_{feature2}'] = (
                        self.df_combined[feature1] * self.df_combined[feature2]
                    )
                    
                    # Ratio interaction (with safety check)
                    if self.df_combined[feature2].min() > 0:
                        self.df_combined[f'{feature1}_div_{feature2}'] = (
                            self.df_combined[feature1] / self.df_combined[feature2]
                        )
                    
                    print(f"Created: {feature1}_x_{feature2}")
                    
                except Exception as e:
                    print(f"Error creating interaction {feature1}_x_{feature2}: {e}")
    
    def _create_polynomial_features(self, degree=2):
        """Create polynomial features for important numerical features"""
        print(f"\n=== Creating Polynomial Features (degree={degree}) ===")
        
        # Important features for polynomial expansion
        important_features = [
            'total_influence_score', 'high_quality_influence', 'vote_ratio',
            'ctr_proxy_normalized', 'user_post_count', 'Score'
        ]
        
        available_features = [f for f in important_features if f in self.df_combined.columns]
        
        for feature in available_features:
            try:
                # Square term
                self.df_combined[f'{feature}_squared'] = self.df_combined[feature] ** 2
                
                # Cube term (if degree >= 3)
                if degree >= 3:
                    self.df_combined[f'{feature}_cubed'] = self.df_combined[feature] ** 3
                
                # Square root (if all values are positive)
                if self.df_combined[feature].min() >= 0:
                    self.df_combined[f'{feature}_sqrt'] = np.sqrt(self.df_combined[feature])
                
                print(f"Created polynomial features for: {feature}")
                
            except Exception as e:
                print(f"Error creating polynomial features for {feature}: {e}")
    
    def get_normalized_features_summary(self):
        """Get summary of normalized features"""
        if not hasattr(self, 'scalers'):
            print("No normalization has been applied yet.")
            return
        
        print("\n=== Feature Normalization Summary ===")
        
        print(f"Scalers created: {len(self.scalers)}")
        for scaler_name in list(self.scalers.keys())[:10]:
            print(f"  {scaler_name}")
        
        if hasattr(self, 'label_encoders'):
            print(f"\nLabel encoders created: {len(self.label_encoders)}")
            for encoder_name in list(self.label_encoders.keys())[:10]:
                print(f"  {encoder_name}")
        
        # Show new features
        if hasattr(self, 'feature_analysis'):
            original_count = len(self.feature_analysis['numerical']) + len(self.feature_analysis['categorical'])
            current_count = len(self.df_combined.columns)
            print(f"\nFeature count: {original_count} → {current_count} (+{current_count - original_count})")
    
    def visualize_influence_features(self):
        """Create visualizations for user influence features"""
        print("\n=== Visualizing User Influence Features ===")
        
        if 'total_influence_score' not in self.df_combined.columns:
            print("Influence features not available. Run create_user_influence_features() first.")
            return
        
        plt.figure(figsize=(20, 15))
        
        # 1. 总体影响力分数分布
        plt.subplot(3, 4, 1)
        plt.hist(self.df_combined['total_influence_score'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(self.df_combined['total_influence_score'].median(), color='red', linestyle='--', 
                   label=f'Median: {self.df_combined["total_influence_score"].median():.0f}')
        plt.title('Total Influence Score Distribution')
        plt.xlabel('Influence Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.xlim(0, self.df_combined['total_influence_score'].quantile(0.95))
        
        # 2. 高质量影响力分布
        plt.subplot(3, 4, 2)
        plt.hist(self.df_combined['high_quality_influence'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(self.df_combined['high_quality_influence'].median(), color='red', linestyle='--', 
                   label=f'Median: {self.df_combined["high_quality_influence"].median():.1f}')
        plt.title('High Quality Influence Distribution')
        plt.xlabel('High Quality Influence Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.xlim(0, self.df_combined['high_quality_influence'].quantile(0.95))
        
        # 3. 影响力等级分布
        plt.subplot(3, 4, 3)
        influence_level_counts = self.df_combined['influence_level'].value_counts()
        plt.bar(influence_level_counts.index, influence_level_counts.values, color='lightblue', alpha=0.7)
        plt.title('Influence Level Distribution')
        plt.xlabel('Influence Level')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 4. 多领域影响力分布
        plt.subplot(3, 4, 4)
        multi_domain_counts = self.df_combined['multi_domain_influence'].value_counts()
        plt.bar(multi_domain_counts.index, multi_domain_counts.values, color='lightgreen', alpha=0.7)
        plt.title('Multi-Domain Influence Distribution')
        plt.xlabel('Multi-Domain Level')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 5. 影响力 vs 用户声誉
        plt.subplot(3, 4, 5)
        if 'user_reputation' in self.df_combined.columns:
            plt.scatter(self.df_combined['user_reputation'], self.df_combined['total_influence_score'], 
                       alpha=0.3, s=10, c=self.df_combined['is_domain_expert'], cmap='viridis')
            plt.title('Influence vs User Reputation')
            plt.xlabel('User Reputation')
            plt.ylabel('Total Influence Score')
            plt.colorbar(label='Is Domain Expert')
        else:
            plt.text(0.5, 0.5, 'User reputation not available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Influence vs User Reputation')
        
        # 6. 影响力 vs 内容质量 (CTR)
        plt.subplot(3, 4, 6)
        if 'ctr_proxy_normalized' in self.df_combined.columns:
            plt.scatter(self.df_combined['total_influence_score'], self.df_combined['ctr_proxy_normalized'], 
                       alpha=0.3, s=10, c=self.df_combined['is_multi_domain_expert'], cmap='plasma')
            plt.title('Influence vs Content Quality (CTR)')
            plt.xlabel('Total Influence Score')
            plt.ylabel('CTR Proxy')
            plt.colorbar(label='Is Multi-Domain Expert')
        else:
            plt.text(0.5, 0.5, 'CTR features not available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Influence vs Content Quality')
        
        # 7. 影响力增长率分布
        plt.subplot(3, 4, 7)
        plt.hist(self.df_combined['influence_growth_rate'], bins=50, alpha=0.7, color='gold', edgecolor='black')
        plt.axvline(self.df_combined['influence_growth_rate'].median(), color='red', linestyle='--', 
                   label=f'Median: {self.df_combined["influence_growth_rate"].median():.3f}')
        plt.title('Influence Growth Rate Distribution')
        plt.xlabel('Influence Growth Rate')
        plt.ylabel('Frequency')
        plt.legend()
        plt.xlim(0, self.df_combined['influence_growth_rate'].quantile(0.95))
        
        # 8. 影响力领域数量分布
        plt.subplot(3, 4, 8)
        plt.hist(self.df_combined['influence_domains_count'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.title('Influence Domains Count Distribution')
        plt.xlabel('Number of Influence Domains')
        plt.ylabel('Frequency')
        
        # 9. Vote Ratio分布
        plt.subplot(3, 4, 9)
        plt.hist(self.df_combined['vote_ratio'], bins=30, alpha=0.7, color='lightsteelblue', edgecolor='black')
        plt.axvline(self.df_combined['vote_ratio'].median(), color='red', linestyle='--', 
                   label=f'Median: {self.df_combined["vote_ratio"].median():.3f}')
        plt.title('Vote Ratio Distribution')
        plt.xlabel('Vote Ratio')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 10. 影响力 vs 帖子数量
        plt.subplot(3, 4, 10)
        plt.scatter(self.df_combined['user_post_count'], self.df_combined['total_influence_score'], 
                   alpha=0.3, s=10, c=self.df_combined['influence_domains_count'], cmap='viridis')
        plt.title('Influence vs Post Count')
        plt.xlabel('User Post Count')
        plt.ylabel('Total Influence Score')
        plt.colorbar(label='Influence Domains Count')
        
        # 11. 专家用户分析
        plt.subplot(3, 4, 11)
        expert_counts = [
            self.df_combined['is_domain_expert'].sum(),
            self.df_combined['is_multi_domain_expert'].sum()
        ]
        plt.bar(['Domain Expert', 'Multi-Domain Expert'], expert_counts, color=['orange', 'red'], alpha=0.7)
        plt.title('Expert Users Analysis')
        plt.ylabel('Count')
        
        # 12. 高质量影响力 vs 原始影响力
        plt.subplot(3, 4, 12)
        plt.scatter(self.df_combined['total_influence_score'], self.df_combined['high_quality_influence'], 
                   alpha=0.3, s=10, c=self.df_combined['vote_ratio'], cmap='plasma')
        plt.plot([0, self.df_combined['total_influence_score'].max()], [0, self.df_combined['total_influence_score'].max()], 
                'r--', alpha=0.5, label='No Quality Adjustment')
        plt.title('High Quality vs Total Influence')
        plt.xlabel('Total Influence Score')
        plt.ylabel('High Quality Influence Score')
        plt.colorbar(label='Vote Ratio')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 打印影响力统计信息
        print("\n=== User Influence Statistics ===")
        print(f"Total Influence Score - Mean: {self.df_combined['total_influence_score'].mean():.2f}, Median: {self.df_combined['total_influence_score'].median():.2f}")
        print(f"High Quality Influence - Mean: {self.df_combined['high_quality_influence'].mean():.2f}, Median: {self.df_combined['high_quality_influence'].median():.2f}")
        print(f"Influence Domains Count - Mean: {self.df_combined['influence_domains_count'].mean():.2f}, Max: {self.df_combined['influence_domains_count'].max()}")
        print(f"Domain Experts: {self.df_combined['is_domain_expert'].sum()} ({self.df_combined['is_domain_expert'].sum()/len(self.df_combined)*100:.1f}%)")
        print(f"Multi-Domain Experts: {self.df_combined['is_multi_domain_expert'].sum()} ({self.df_combined['is_multi_domain_expert'].sum()/len(self.df_combined)*100:.1f}%)")
        
        # 影响力等级分布
        print("\nInfluence Level Distribution:")
        influence_level_counts = self.df_combined['influence_level'].value_counts()
        for level, count in influence_level_counts.items():
            print(f"  {level}: {count} ({(count/len(self.df_combined)*100):.1f}%)")
        
        # 多领域影响力分布
        print("\nMulti-Domain Influence Distribution:")
        multi_domain_counts = self.df_combined['multi_domain_influence'].value_counts()
        for level, count in multi_domain_counts.items():
            print(f"  {level}: {count} ({(count/len(self.df_combined)*100):.1f}%)")
        
        # Vote Ratio统计
        print("\nVote Ratio Statistics:")
        print(f"  Vote ratio - Mean: {self.df_combined['vote_ratio'].mean():.3f}")
        print(f"  Vote ratio - Median: {self.df_combined['vote_ratio'].median():.3f}")
        print(f"  High quality influence - Mean: {self.df_combined['high_quality_influence'].mean():.2f}")
        print(f"  High quality influence - Median: {self.df_combined['high_quality_influence'].median():.2f}")

    def create_industrial_features(self, df_combined):
        """
        Create industrial-grade features for CTR prediction
        Includes hash encoding, feature crossing, sequence features, and context features
        """
        print("\n=== Creating Industrial-Grade Features ===")
        
        # 1. Hash encoding for categorical features (industry standard)
        categorical_features = ['OwnerUserId', 'first_tag', 'influence_level', 
                               'badge_level', 'multi_domain_influence']
        available_categorical = [f for f in categorical_features if f in df_combined.columns]
        
        if available_categorical:
            print(f"Creating hash encoding for {len(available_categorical)} categorical features")
            for col in available_categorical:
                # Hash encoding (memory efficient for large-scale systems)
                df_combined[f'{col}_hash'] = df_combined[col].astype(str).apply(
                    lambda x: hash(x) % 10000
                )
        
        # 2. Feature crossing (core of industrial CTR systems)
        cross_features = [
            ('OwnerUserId', 'first_tag'),
            ('influence_level', 'badge_level'),
            ('user_post_count', 'user_reputation'),
            ('Score', 'ViewCount')
        ]
        
        available_cross = []
        for feat1, feat2 in cross_features:
            if feat1 in df_combined.columns and feat2 in df_combined.columns:
                available_cross.append((feat1, feat2))
        
        if available_cross:
            print(f"Creating feature crossings for {len(available_cross)} combinations")
            for feat1, feat2 in available_cross:
                # Simple feature crossing
                cross_feature = (
                    df_combined[feat1].astype(str) + "_" + df_combined[feat2].astype(str)
                ).apply(lambda x: hash(x) % 1000)
                df_combined[f'{feat1}_{feat2}_cross'] = cross_feature
        
        # 3. Sequence features (simulate user behavior sequences)
        sequence_features = self._create_sequence_features(df_combined)
        if sequence_features is not None:
            print("Created sequence features for user behavior modeling")
        
        # 4. Context features (temporal and spatial patterns)
        context_features = self._create_context_features(df_combined)
        if context_features is not None:
            print("Created context features for temporal patterns")
        
        # 5. Advanced interaction features
        self._create_advanced_interactions(df_combined)
        
        print(f"Industrial features created. Total features: {len(df_combined.columns)}")
        return df_combined
    
    def _create_sequence_features(self, df_combined):
        """Create sequence features for user behavior modeling"""
        try:
            # Simulate user's recent activity sequence
            user_sequences = {}
            
            # Group by user, sort by time
            if 'CreationDate_x' in df_combined.columns:
                df_sorted = df_combined.sort_values(['OwnerUserId', 'CreationDate_x'])
                
                for user_id, group in df_sorted.groupby('OwnerUserId'):
                    if len(group) > 1:
                        # Take recent 5 posts as sequence
                        recent_posts = group.tail(5)
                        sequence = []
                        for _, post in recent_posts.iterrows():
                            seq_element = [
                                post.get('Score', 0),
                                post.get('ViewCount', 0),
                                post.get('num_tags', 0),
                                post.get('title_length', 0)
                            ]
                            sequence.append(seq_element)
                        
                        # Pad to fixed length
                        while len(sequence) < 5:
                            sequence.append([0, 0, 0, 0])
                        
                        user_sequences[user_id] = sequence
                
                # Add sequence features to dataframe
                for i in range(5):
                    for j in range(4):
                        col_name = f'seq_{i}_{j}'
                        df_combined[col_name] = 0
                
                for user_id in df_combined['OwnerUserId']:
                    if user_id in user_sequences:
                        seq = user_sequences[user_id]
                        for i in range(5):
                            for j in range(4):
                                col_name = f'seq_{i}_{j}'
                                df_combined.loc[df_combined['OwnerUserId'] == user_id, col_name] = seq[i][j]
                
                return True
            return False
            
        except Exception as e:
            print(f"Warning: Failed to create sequence features: {e}")
            return False
    
    def _create_context_features(self, df_combined):
        """Create context features for temporal patterns"""
        try:
            if 'CreationDate_x' in df_combined.columns:
                df_combined['CreationDate_x'] = pd.to_datetime(df_combined['CreationDate_x'])
                
                # Time-based context features
                df_combined['hour_of_day'] = df_combined['CreationDate_x'].dt.hour / 24.0
                df_combined['day_of_week'] = df_combined['CreationDate_x'].dt.dayofweek / 7.0
                df_combined['month_of_year'] = df_combined['CreationDate_x'].dt.month / 12.0
                df_combined['is_weekend'] = (df_combined['CreationDate_x'].dt.dayofweek >= 5).astype(int)
                
                # Peak hours (9-17)
                df_combined['is_peak_hours'] = (
                    (df_combined['CreationDate_x'].dt.hour >= 9) & 
                    (df_combined['CreationDate_x'].dt.hour <= 17)
                ).astype(int)
                
                return True
            return False
            
        except Exception as e:
            print(f"Warning: Failed to create context features: {e}")
            return False
    
    def _create_advanced_interactions(self, df_combined):
        """Create advanced interaction features"""
        try:
            # Quality interactions
            if 'Score' in df_combined.columns and 'ViewCount' in df_combined.columns:
                df_combined['quality_view_ratio'] = df_combined['Score'] / (df_combined['ViewCount'] + 1)
            
            if 'total_votes' in df_combined.columns and 'upvotes' in df_combined.columns:
                df_combined['vote_quality_ratio'] = df_combined['upvotes'] / (df_combined['total_votes'] + 1)
            
            # User engagement interactions
            if 'user_post_count' in df_combined.columns and 'user_reputation' in df_combined.columns:
                df_combined['engagement_efficiency'] = df_combined['user_reputation'] / (df_combined['user_post_count'] + 1)
            
            # Content complexity interactions
            if 'post_length' in df_combined.columns and 'title_length' in df_combined.columns:
                df_combined['content_complexity'] = df_combined['post_length'] / (df_combined['title_length'] + 1)
            
            # Time-based interactions
            if 'post_age_days' in df_combined.columns and 'Score' in df_combined.columns:
                df_combined['score_per_day'] = df_combined['Score'] / (df_combined['post_age_days'] + 1)
            
            print("Created advanced interaction features")
            
        except Exception as e:
            print(f"Warning: Failed to create advanced interactions: {e}")

    def create_negative_sampling(self, df_combined, target_col='ctr_proxy_normalized', sampling_ratio=3):
        """
        Perform negative sampling for CTR prediction (industry standard)
        
        Args:
            df_combined: Combined dataframe
            target_col: Target column for CTR prediction
            sampling_ratio: Ratio of negative to positive samples
        
        Returns:
            Balanced dataframe
        """
        print(f"\n=== Performing Negative Sampling (ratio: {sampling_ratio}:1) ===")
        
        # Define positive and negative samples
        positive_threshold = df_combined[target_col].quantile(0.7)  # Top 30% as positive
        df_combined['is_positive'] = df_combined[target_col] >= positive_threshold
        
        positive_samples = df_combined[df_combined['is_positive'] == True]
        negative_samples = df_combined[df_combined['is_positive'] == False]
        
        print(f"Original data: {len(positive_samples)} positive, {len(negative_samples)} negative")
        
        # Negative sampling
        target_negative_count = len(positive_samples) * sampling_ratio
        
        if len(negative_samples) > target_negative_count:
            negative_samples = negative_samples.sample(
                n=int(target_negative_count), 
                random_state=42
            )
        
        # Combine sampled data
        balanced_df = pd.concat([positive_samples, negative_samples], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42)  # Shuffle
        
        print(f"After sampling: {len(positive_samples)} positive, {len(negative_samples)} negative")
        print(f"Balanced dataset shape: {balanced_df.shape}")
        
        return balanced_df

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    df_combined, embeddings, tfidf_features, model = preprocessor.preprocess_all()
    
    print(f"\nFinal dataset shape: {df_combined.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"TF-IDF features shape: {tfidf_features.shape}") 