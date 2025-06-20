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

class DataPreprocessor:
    def __init__(self, base_path=None):
        """Initialize the data preprocessor with base path"""
        self.base_path = base_path if base_path else os.path.join(os.getcwd())
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
        
        self.df_posts = self.parse_xml(os.path.join(self.base_path, 'data', 'Posts.xml'))
        self.df_users = self.parse_xml(os.path.join(self.base_path, 'data', 'Users.xml'))
        self.df_tags = self.parse_xml(os.path.join(self.base_path, 'data', 'Tags.xml'))
        self.df_votes = self.parse_xml(os.path.join(self.base_path, 'data', 'Votes.xml'))
        self.df_badges = self.parse_xml(os.path.join(self.base_path, 'data', 'Badges.xml'))
        
        print(f"Loaded {len(self.df_posts)} posts, {len(self.df_users)} users, {len(self.df_tags)} tags")
        
        return self.df_posts, self.df_users, self.df_tags, self.df_votes, self.df_badges
    
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
            df_comments = self.parse_xml(os.path.join(self.base_path, 'data', 'Comments.xml'))
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
        plt.subplot(3, 4, 7)
        plt.scatter(self.df_combined['post_length'], np.log1p(self.df_combined['total_votes']), alpha=0.3, s=10)
        plt.title('Post Length vs Log(Total Votes + 1)')
        plt.xlabel('Post Length (words)')
        plt.ylabel('Log(Total Votes + 1)')
        plt.xlim(0, self.df_combined['post_length'].quantile(0.95))
        
        # Vote ratio by year
        plt.subplot(3, 4, 8)
        self.df_combined['year'] = pd.to_datetime(self.df_combined['CreationDate_x']).dt.year
        vote_ratio_by_year = self.df_combined.groupby('year')['vote_ratio'].mean()
        plt.plot(vote_ratio_by_year.index, vote_ratio_by_year.values, marker='o', linewidth=2, markersize=6)
        plt.title('Average Vote Ratio by Year')
        plt.xlabel('Year')
        plt.ylabel('Average Vote Ratio')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
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
    
    def preprocess_all(self):
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
        
        # Create categorical variables
        self.create_categorical_variables()
        
        # Create TF-IDF features
        self.create_tfidf_features()
        
        # Create semantic embeddings
        self.create_semantic_embeddings()
        
        # Visualize results
        self.visualize_derived_variables()
        
        # Visualize TF-IDF features
        self.visualize_tfidf_features()
        
        # Print summary
        self.print_summary_statistics()
        
        print("\n=== Data Preprocessing Complete ===")
        return self.df_combined, self.embeddings, self.tfidf_features, self.model

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    df_combined, embeddings, tfidf_features, model = preprocessor.preprocess_all()
    
    print(f"\nFinal dataset shape: {df_combined.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"TF-IDF features shape: {tfidf_features.shape}") 