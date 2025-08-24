import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class UpliftTreatmentLabeling:
    """Uplift treatment labeling and feature engineering"""
    
    def __init__(self):
        self.user_features = {}
        self.post_features = {}
        self.treatment_features = {}
        
    def load_data(self, file_path):
        """Load uplift data"""
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        print(f"Data shape: {df.shape}")
        return df
    
    def create_treatment_labels(self, df):
        """Create treatment labels for uplift modeling based on tag containing 'ai content'"""
        print("Creating treatment labels based on tag containing 'ai content'...")
        
        # Check if treatment column already exists
        if 'treatment_ai_content' in df.columns:
            print("Treatment column already exists")
            return df
        
        # Create treatment labels based on tag containing 'ai content'
        if 'tag' in df.columns:
            # Check if tag contains 'ai content' (case insensitive)
            df['treatment_ai_content'] = df['tag'].str.contains('ai content', case=False, na=False).astype(int)
            print("Created treatment labels from tag column - checking for 'ai content'")
            print(f"Treatment samples (tag contains 'ai content'): {(df['treatment_ai_content'] == 1).sum():,}")
            print(f"Control samples (tag does not contain 'ai content'): {(df['treatment_ai_content'] == 0).sum():,}")
            
            # Note: Control group should be posts similar to AI content but not tagged as 'ai content'
            # This allows us to measure the true effect of the AI tag
        elif 'tags' in df.columns:
            # Alternative column name
            df['treatment_ai_content'] = df['tags'].str.contains('ai content', case=False, na=False).astype(int)
            print("Created treatment labels from tags column - checking for 'ai content'")
            print(f"Treatment samples (tag contains 'ai content'): {(df['treatment_ai_content'] == 1).sum():,}")
            print(f"Control samples (tag does not contain 'ai content'): {(df['treatment_ai_content'] == 0).sum():,}")
        elif 'ai_content' in df.columns:
            df['treatment_ai_content'] = df['ai_content'].astype(int)
            print("Created treatment labels from ai_content column")
        elif 'treatment' in df.columns:
            df['treatment_ai_content'] = df['treatment'].astype(int)
            print("Created treatment labels from treatment column")
        else:
            # Create dummy treatment labels (for demonstration)
            df['treatment_ai_content'] = np.random.choice([0, 1], size=len(df), p=[0.5, 0.5])
            print("Created dummy treatment labels (for demonstration)")
        
        return df
    
    def create_response_labels(self, df):
        """Create response labels for uplift modeling"""
        print("Creating response labels...")
        
        # Check if response column already exists
        if 'response' in df.columns:
            print("Response column already exists")
            return df
        
        # Create response labels based on available data
        if 'click' in df.columns:
            df['response'] = df['click'].astype(int)
            print("Created response labels from click column")
        elif 'engagement' in df.columns:
            df['response'] = (df['engagement'] > 0).astype(int)
            print("Created response labels from engagement column")
        else:
            # Create dummy response labels (for demonstration)
            df['response'] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])
            print("Created dummy response labels (for demonstration)")
        
        return df
    
    def create_user_features(self, df):
        """Create user-level features for uplift modeling"""
        print("Creating user features...")
        
        if 'user_id' not in df.columns:
            print("No user_id column found")
            return None
        
        user_features = df.groupby('user_id').agg({
            'user_id': 'count'  # Activity count
        }).rename(columns={'user_id': 'activity_count'})
        
        # Add more user features if available
        if 'post_id' in df.columns:
            user_features['unique_posts_interacted'] = df.groupby('user_id')['post_id'].nunique()
        
        if 'response' in df.columns:
            user_features['total_responses'] = df.groupby('user_id')['response'].sum()
            user_features['response_rate'] = df.groupby('user_id')['response'].mean()
        
        if 'treatment_ai_content' in df.columns:
            user_features['ai_content_exposure'] = df.groupby('user_id')['treatment_ai_content'].sum()
            user_features['ai_content_rate'] = df.groupby('user_id')['treatment_ai_content'].mean()
        
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
        """Create post-level features for uplift modeling"""
        print("Creating post features...")
        
        if 'post_id' not in df.columns:
            print("No post_id column found")
            return None
        
        post_features = df.groupby('post_id').agg({
            'post_id': 'count'  # Interaction count
        }).rename(columns={'post_id': 'interaction_count'})
        
        # Add more post features if available
        if 'response' in df.columns:
            post_features['total_responses'] = df.groupby('post_id')['response'].sum()
            post_features['response_rate'] = df.groupby('post_id')['response'].mean()
        
        if 'treatment_ai_content' in df.columns:
            post_features['ai_content_count'] = df.groupby('post_id')['treatment_ai_content'].sum()
            post_features['ai_content_rate'] = df.groupby('post_id')['treatment_ai_content'].mean()
        
        # Add user engagement features
        if 'user_id' in df.columns:
            post_features['unique_users_interacted'] = df.groupby('post_id')['user_id'].nunique()
        
        print(f"Post features created: {list(post_features.columns)}")
        return post_features
    
    def create_treatment_features(self, df):
        """Create treatment-specific features"""
        print("Creating treatment features...")
        
        treatment_features = df.copy()
        
        # Add user features to interactions
        user_features = self.create_user_features(df)
        if user_features is not None:
            treatment_features = treatment_features.merge(
                user_features, on='user_id', how='left'
            )
        
        # Add post features to interactions
        post_features = self.create_post_features(df)
        if post_features is not None:
            treatment_features = treatment_features.merge(
                post_features, on='post_id', how='left'
            )
        
        # Create treatment-specific features
        if 'treatment_ai_content' in treatment_features.columns and 'user_id' in treatment_features.columns:
            # User AI content interaction history
            treatment_features['user_ai_interaction_count'] = treatment_features.groupby(
                ['user_id', 'treatment_ai_content']
            )['response'].transform('count')
        
        # Create interaction features
        if 'treatment_ai_content' in treatment_features.columns and 'response' in treatment_features.columns:
            # Treatment-response interaction
            treatment_features['treatment_x_response'] = treatment_features['treatment_ai_content'] * treatment_features['response']
        
        print(f"Treatment features created: {list(treatment_features.columns)}")
        return treatment_features
    
    def process_uplift_data(self, file_path):
        """Complete processing pipeline for uplift data"""
        print("=== Uplift Data Processing ===\n")
        
        # Load data
        df = self.load_data(file_path)
        
        # Create treatment labels
        df = self.create_treatment_labels(df)
        
        # Create response labels
        df = self.create_response_labels(df)
        
        # Create features
        df = self.create_treatment_features(df)
        
        # Handle missing values
        df = df.fillna(0)
        
        print(f"\nFinal dataset shape: {df.shape}")
        print(f"Final dataset columns: {list(df.columns)}")
        
        return df
    
    def analyze_uplift_patterns(self, df):
        """Analyze uplift patterns in the data"""
        print("\n=== Uplift Pattern Analysis ===")
        
        if 'treatment_ai_content' not in df.columns or 'response' not in df.columns:
            print("No treatment or response columns found")
            return
        
        # Treatment distribution
        treatment_dist = df['treatment_ai_content'].value_counts(normalize=True)
        print(f"Treatment distribution:")
        for value, ratio in treatment_dist.items():
            print(f"  {value}: {ratio:.2%}")
        
        # Response distribution
        response_dist = df['response'].value_counts(normalize=True)
        print(f"\nResponse distribution:")
        for value, ratio in response_dist.items():
            print(f"  {value}: {ratio:.2%}")
        
        # Uplift analysis
        print(f"\nUplift analysis:")
        treatment_response_rate = df[df['treatment_ai_content'] == 1]['response'].mean()
        control_response_rate = df[df['treatment_ai_content'] == 0]['response'].mean()
        uplift = treatment_response_rate - control_response_rate
        
        print(f"  Treatment response rate: {treatment_response_rate:.2%}")
        print(f"  Control response rate: {control_response_rate:.2%}")
        print(f"  Uplift: {uplift:.2%}")
        
        # Feature correlations with treatment and response
        feature_cols = [col for col in df.columns if col not in ['treatment_ai_content', 'response', 'user_id', 'post_id']]
        if feature_cols:
            print(f"\nFeature correlations:")
            
            # With treatment
            treatment_correlations = []
            for feature in feature_cols:
                if df[feature].dtype in ['int64', 'float64']:
                    corr = abs(df[feature].corr(df['treatment_ai_content']))
                    treatment_correlations.append((feature, corr))
            
            treatment_correlations.sort(key=lambda x: x[1], reverse=True)
            print("Top 10 features correlated with treatment:")
            for feature, corr in treatment_correlations[:10]:
                print(f"  {feature}: {corr:.4f}")
            
            # With response
            response_correlations = []
            for feature in feature_cols:
                if df[feature].dtype in ['int64', 'float64']:
                    corr = abs(df[feature].corr(df['response']))
                    response_correlations.append((feature, corr))
            
            response_correlations.sort(key=lambda x: x[1], reverse=True)
            print("\nTop 10 features correlated with response:")
            for feature, corr in response_correlations[:10]:
                print(f"  {feature}: {corr:.4f}")

def process_uplift_data(file_path):
    """Convenience function for processing uplift data"""
    processor = UpliftTreatmentLabeling()
    return processor.process_uplift_data(file_path)

if __name__ == "__main__":
    # Example usage
    df = process_uplift_data('uplift_dataset.csv')
    
    # Analyze patterns
    processor = UpliftTreatmentLabeling()
    processor.analyze_uplift_patterns(df) 