import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class UserPostClickLabeling:
    """User-post click labeling and feature engineering"""
    
    def __init__(self):
        self.user_features = {}
        self.post_features = {}
        
    def load_data(self, file_path):
        """Load user-post interaction data"""
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        print(f"Data shape: {df.shape}")
        return df
    
    def create_user_features(self, df):
        """Create user-level features"""
        print("Creating user features...")
        
        if 'user_id' not in df.columns:
            print("❌ No user_id column found")
            return None
        
        user_features = df.groupby('user_id').agg({
            'user_id': 'count'  # Activity count
        }).rename(columns={'user_id': 'activity_count'})
        
        # Add more user features if available
        if 'post_id' in df.columns:
            user_features['unique_posts_interacted'] = df.groupby('user_id')['post_id'].nunique()
        
        if 'click' in df.columns:
            user_features['total_clicks'] = df.groupby('user_id')['click'].sum()
            user_features['click_rate'] = df.groupby('user_id')['click'].mean()
        
        # Add time-based features if date columns exist
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            date_col = date_columns[0]
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                user_features['first_activity_date'] = df.groupby('user_id')[date_col].min()
                user_features['last_activity_date'] = df.groupby('user_id')[date_col].max()
                user_features['activity_span_days'] = (
                    user_features['last_activity_date'] - user_features['first_activity_date']
                ).dt.days
            except:
                print(f"Could not process date column: {date_col}")
        
        print(f"User features created: {list(user_features.columns)}")
        return user_features
    
    def create_post_features(self, df):
        """Create post-level features"""
        print("Creating post features...")
        
        if 'post_id' not in df.columns:
            print("❌ No post_id column found")
            return None
        
        post_features = df.groupby('post_id').agg({
            'post_id': 'count'  # Interaction count
        }).rename(columns={'post_id': 'interaction_count'})
        
        # Add more post features if available
        if 'click' in df.columns:
            post_features['total_clicks'] = df.groupby('post_id')['click'].sum()
            post_features['click_rate'] = df.groupby('post_id')['click'].mean()
        
        # Add user engagement features
        if 'user_id' in df.columns:
            post_features['unique_users_interacted'] = df.groupby('post_id')['user_id'].nunique()
        
        print(f"Post features created: {list(post_features.columns)}")
        return post_features
    
    def create_interaction_features(self, df):
        """Create interaction-level features"""
        print("Creating interaction features...")
        
        interaction_features = df.copy()
        
        # Add user features to interactions
        user_features = self.create_user_features(df)
        if user_features is not None:
            interaction_features = interaction_features.merge(
                user_features, on='user_id', how='left'
            )
        
        # Add post features to interactions
        post_features = self.create_post_features(df)
        if post_features is not None:
            interaction_features = interaction_features.merge(
                post_features, on='post_id', how='left'
            )
        
        # Create interaction-specific features
        if 'click' in interaction_features.columns:
            # User-post interaction history
            interaction_features['user_post_interaction_count'] = interaction_features.groupby(
                ['user_id', 'post_id']
            )['click'].transform('count')
        
        print(f"Interaction features created: {list(interaction_features.columns)}")
        return interaction_features
    
    def label_user_post_clicks(self, df):
        """Create click labels for user-post interactions"""
        print("Creating click labels...")
        
        # Check if click column already exists
        if 'click' in df.columns:
            print("Click column already exists")
            return df
        
        # Create click labels based on available data
        # This is a placeholder - actual logic depends on your data structure
        if 'response' in df.columns:
            df['click'] = df['response']
            print("Created click labels from response column")
        elif 'interaction_type' in df.columns:
            df['click'] = (df['interaction_type'] == 'click').astype(int)
            print("Created click labels from interaction_type column")
        else:
            # Create dummy click labels (for demonstration)
            df['click'] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])
            print("Created dummy click labels (for demonstration)")
        
        return df
    
    def process_user_post_data(self, file_path):
        """Complete processing pipeline for user-post click data"""
        print("=== User-Post Click Data Processing ===\n")
        
        # Load data
        df = self.load_data(file_path)
        
        # Create click labels
        df = self.label_user_post_clicks(df)
        
        # Create features
        df = self.create_interaction_features(df)
        
        # Handle missing values
        df = df.fillna(0)
        
        print(f"\nFinal dataset shape: {df.shape}")
        print(f"Final dataset columns: {list(df.columns)}")
        
        return df
    
    def analyze_click_patterns(self, df):
        """Analyze click patterns in the data"""
        print("\n=== Click Pattern Analysis ===")
        
        if 'click' not in df.columns:
            print("❌ No click column found")
            return
        
        # Overall click rate
        overall_click_rate = df['click'].mean()
        print(f"Overall click rate: {overall_click_rate:.2%}")
        
        # Click rate by user
        if 'user_id' in df.columns:
            user_click_rates = df.groupby('user_id')['click'].mean()
            print(f"User click rates - Mean: {user_click_rates.mean():.2%}, Std: {user_click_rates.std():.2%}")
        
        # Click rate by post
        if 'post_id' in df.columns:
            post_click_rates = df.groupby('post_id')['click'].mean()
            print(f"Post click rates - Mean: {post_click_rates.mean():.2%}, Std: {post_click_rates.std():.2%}")
        
        # Feature correlations with click
        feature_cols = [col for col in df.columns if col not in ['click', 'user_id', 'post_id']]
        if feature_cols:
            print(f"\nFeature correlations with click:")
            correlations = []
            for feature in feature_cols:
                if df[feature].dtype in ['int64', 'float64']:
                    corr = abs(df[feature].corr(df['click']))
                    correlations.append((feature, corr))
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            print("Top 10 features correlated with click:")
            for feature, corr in correlations[:10]:
                print(f"  {feature}: {corr:.4f}")

def process_user_post_click_data(file_path):
    """Convenience function for processing user-post click data"""
    processor = UserPostClickLabeling()
    return processor.process_user_post_data(file_path)

if __name__ == "__main__":
    # Example usage
    df = process_user_post_click_data('user_post_click_samples.csv')
    
    # Analyze patterns
    processor = UserPostClickLabeling()
    processor.analyze_click_patterns(df) 