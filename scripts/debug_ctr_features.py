import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def debug_ctr_features():
    """Debug CTR features to understand data quality and feature importance"""
    print("=== CTR Features Debug ===\n")
    
    # Load data
    df = pd.read_csv('user_post_click_samples.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Check target variable
    if 'click' in df.columns:
        target_col = 'click'
    elif 'response' in df.columns:
        target_col = 'response'
    else:
        print("❌ No target variable found (click or response)")
        return None
    
    print(f"Target variable: {target_col}")
    print(f"Target distribution:")
    target_dist = df[target_col].value_counts(normalize=True)
    for value, ratio in target_dist.items():
        print(f"  {value}: {ratio:.2%}")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in [target_col, 'user_id', 'post_id']]
    print(f"\nNumber of features: {len(feature_cols)}")
    
    # Check feature types
    numeric_features = []
    categorical_features = []
    
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            numeric_features.append(col)
        else:
            categorical_features.append(col)
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # 1. Check feature correlation with target
    print("\n=== 1. Feature-Target Correlation Analysis ===")
    
    correlations = []
    for col in numeric_features:
        corr = abs(df[col].corr(df[target_col]))
        correlations.append((col, corr))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 15 features with highest correlation to target:")
    for col, corr in correlations[:15]:
        print(f"  {col}: {corr:.4f}")
    
    # 2. Check feature distributions
    print("\n=== 2. Feature Distribution Analysis ===")
    
    for col in numeric_features[:10]:  # Only check first 10 features
        print(f"\nAnalyzing {col}:")
        
        # Check for missing values
        missing_count = df[col].isnull().sum()
        missing_ratio = missing_count / len(df)
        print(f"  Missing values: {missing_count} ({missing_ratio:.2%})")
        
        # Check for outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_ratio = outliers / len(df)
        print(f"  Outliers: {outliers} ({outlier_ratio:.2%})")
        
        # Check unique values
        unique_count = df[col].nunique()
        unique_ratio = unique_count / len(df)
        print(f"  Unique values: {unique_count} ({unique_ratio:.2%})")
        
        # Check distribution by target
        if target_col in df.columns:
            target_0_mean = df[df[target_col] == 0][col].mean()
            target_1_mean = df[df[target_col] == 1][col].mean()
            print(f"  Mean (target=0): {target_0_mean:.4f}")
            print(f"  Mean (target=1): {target_1_mean:.4f}")
            print(f"  Difference: {abs(target_1_mean - target_0_mean):.4f}")
    
    # 3. Check feature multicollinearity
    print("\n=== 3. Multicollinearity Check ===")
    
    # Select top 20 features by correlation
    top_features = [col for col, corr in correlations[:20]]
    
    # Calculate correlation matrix
    correlation_matrix = df[top_features].corr()
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            corr = abs(correlation_matrix.iloc[i, j])
            if corr > 0.8:
                high_corr_pairs.append((top_features[i], top_features[j], corr))
    
    print(f"Highly correlated feature pairs (|correlation| > 0.8):")
    for feat1, feat2, corr in high_corr_pairs:
        print(f"  {feat1} <-> {feat2}: {corr:.4f}")
    
    # 4. Check for constant features
    print("\n=== 4. Constant Features Check ===")
    
    constant_features = []
    for col in numeric_features:
        if df[col].nunique() == 1:
            constant_features.append(col)
    
    if constant_features:
        print(f"Constant features found: {constant_features}")
    else:
        print("No constant features found")
    
    # 5. Check for duplicate features
    print("\n=== 5. Duplicate Features Check ===")
    
    duplicate_features = []
    for i, col1 in enumerate(numeric_features):
        for j, col2 in enumerate(numeric_features[i+1:], i+1):
            if df[col1].equals(df[col2]):
                duplicate_features.append((col1, col2))
    
    if duplicate_features:
        print(f"Duplicate feature pairs:")
        for col1, col2 in duplicate_features:
            print(f"  {col1} = {col2}")
    else:
        print("No duplicate features found")
    
    # 6. Quick model test
    print("\n=== 6. Quick Model Test ===")
    
    # Prepare data
    X = df[numeric_features].fillna(0)
    y = df[target_col]
    
    # Remove rows with NaN in target
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train simple model
    model = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42, verbosity=0)
    model.fit(X_train.values, y_train.values)
    
    # Predict
    y_pred = model.predict(X_test.values)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Quick model performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': numeric_features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 feature importance:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 7. Summary and recommendations
    print("\n=== 7. Summary and Recommendations ===")
    
    issues = []
    
    if len(constant_features) > 0:
        issues.append(f"Found {len(constant_features)} constant features")
    
    if len(duplicate_features) > 0:
        issues.append(f"Found {len(duplicate_features)} duplicate feature pairs")
    
    if len(high_corr_pairs) > 0:
        issues.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
    
    if accuracy < 0.7:
        issues.append("Model accuracy is low, may need feature engineering")
    
    if issues:
        print("⚠️  Found the following issues:")
        for issue in issues:
            print(f"  - {issue}")
        
        print("\nRecommendations:")
        if len(constant_features) > 0:
            print("  1. Remove constant features")
        if len(duplicate_features) > 0:
            print("  2. Remove duplicate features")
        if len(high_corr_pairs) > 0:
            print("  3. Address multicollinearity issues")
        if accuracy < 0.7:
            print("  4. Improve feature engineering")
    else:
        print("✅ No obvious data quality issues found")
    
    return {
        'correlations': correlations,
        'high_corr_pairs': high_corr_pairs,
        'constant_features': constant_features,
        'duplicate_features': duplicate_features,
        'model_accuracy': accuracy,
        'issues': issues
    }

if __name__ == "__main__":
    results = debug_ctr_features() 