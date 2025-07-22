import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def retention_prediction_labeling():
    """Create retention prediction labels from user activity data"""
    print("=== Retention Prediction Labeling ===\n")
    
    # Load data
    df = pd.read_csv('retention_prediction_data.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Check data structure
    print(f"Data columns: {list(df.columns)}")
    print(f"Data shape: {df.shape}")
    
    # Check data types
    print(f"\nData types:")
    print(df.dtypes.value_counts())
    
    # Check for date columns
    date_columns = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_columns.append(col)
    
    print(f"\nDate columns found: {date_columns}")
    
    # Convert date columns to datetime
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                print(f"Converted {col} to datetime")
            except:
                print(f"Could not convert {col} to datetime")
    
    # Check for user activity data
    if 'user_id' in df.columns:
        print(f"\nUnique users: {df['user_id'].nunique():,}")
    
    # Create retention labels
    print("\n=== Creating Retention Labels ===")
    
    # Define retention periods (in days)
    retention_periods = [1, 7, 30, 90]
    
    # Group by user and calculate retention
    if 'user_id' in df.columns and len(date_columns) > 0:
        user_retention = {}
        
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id]
            
            # Get first activity date
            first_date_col = date_columns[0]
            first_activity = user_data[first_date_col].min()
            
            # Get last activity date
            last_activity = user_data[first_date_col].max()
            
            # Calculate days since first activity
            days_since_first = (last_activity - first_activity).days
            
            # Create retention labels
            retention_labels = {}
            for period in retention_periods:
                retention_labels[f'retained_{period}d'] = 1 if days_since_first >= period else 0
            
            user_retention[user_id] = {
                'first_activity': first_activity,
                'last_activity': last_activity,
                'days_since_first': days_since_first,
                **retention_labels
            }
        
        # Convert to DataFrame
        retention_df = pd.DataFrame.from_dict(user_retention, orient='index')
        retention_df.index.name = 'user_id'
        retention_df.reset_index(inplace=True)
        
        print(f"Retention labels created for {len(retention_df):,} users")
        
        # Show retention rates
        print(f"\nRetention rates:")
        for period in retention_periods:
            col = f'retained_{period}d'
            if col in retention_df.columns:
                retention_rate = retention_df[col].mean()
                print(f"  {period}-day retention: {retention_rate:.2%}")
    
    # Create features for retention prediction
    print("\n=== Creating Features ===")
    
    # User activity features
    if 'user_id' in df.columns:
        user_features = df.groupby('user_id').agg({
            'user_id': 'count'  # Activity count
        }).rename(columns={'user_id': 'activity_count'})
        
        # Add more features if available
        if len(date_columns) > 0:
            # Days between first and last activity
            user_features['activity_span_days'] = df.groupby('user_id')[date_columns[0]].agg(
                lambda x: (x.max() - x.min()).days
            )
            
            # Average days between activities
            user_features['avg_days_between_activities'] = df.groupby('user_id')[date_columns[0]].agg(
                lambda x: (x.max() - x.min()).days / max(1, len(x) - 1)
            )
        
        print(f"User features created: {list(user_features.columns)}")
    
    # Merge features with retention labels
    if 'user_id' in df.columns and 'retention_df' in locals():
        final_df = retention_df.merge(user_features, on='user_id', how='left')
        
        print(f"\nFinal dataset shape: {final_df.shape}")
        print(f"Final dataset columns: {list(final_df.columns)}")
        
        # Check for missing values
        missing_values = final_df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\nMissing values:")
            print(missing_values[missing_values > 0])
        
        # Save processed data
        output_file = 'retention_prediction_processed.csv'
        final_df.to_csv(output_file, index=False)
        print(f"\nProcessed data saved to: {output_file}")
        
        return final_df
    
    else:
        print("❌ Could not create retention labels - missing required columns")
        return None

def analyze_retention_patterns(df):
    """Analyze retention patterns in the data"""
    print("\n=== Retention Pattern Analysis ===")
    
    if df is None:
        print("❌ No data to analyze")
        return
    
    # Check retention columns
    retention_cols = [col for col in df.columns if 'retained_' in col]
    
    if not retention_cols:
        print("❌ No retention columns found")
        return
    
    print(f"Retention columns: {retention_cols}")
    
    # Analyze retention rates
    print(f"\nRetention rates:")
    for col in retention_cols:
        retention_rate = df[col].mean()
        print(f"  {col}: {retention_rate:.2%}")
    
    # Analyze feature correlations with retention
    feature_cols = [col for col in df.columns if col not in retention_cols + ['user_id']]
    
    if feature_cols:
        print(f"\nFeature correlations with retention:")
        for retention_col in retention_cols:
            print(f"\n{retention_col}:")
            correlations = []
            for feature in feature_cols:
                if df[feature].dtype in ['int64', 'float64']:
                    corr = abs(df[feature].corr(df[retention_col]))
                    correlations.append((feature, corr))
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            for feature, corr in correlations[:10]:
                print(f"  {feature}: {corr:.4f}")

if __name__ == "__main__":
    # Create retention labels
    retention_data = retention_prediction_labeling()
    
    # Analyze patterns
    if retention_data is not None:
        analyze_retention_patterns(retention_data) 