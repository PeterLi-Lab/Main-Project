import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def debug_uplift():
    """Debug uplift modeling issues and performance"""
    print("=== Uplift Modeling Debug ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Check data structure
    print(f"Data columns: {list(df.columns)}")
    print(f"Data shape: {df.shape}")
    
    # Check for required columns
    required_cols = ['treatment_ai_content', 'response']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return None
    
    print("✅ All required columns present")
    
    # 1. Check data quality
    print("\n=== 1. Data Quality Check ===")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    
    print(f"Missing values: {total_missing}")
    if total_missing > 0:
        print("Columns with missing values:")
        for col, count in missing_values[missing_values > 0].items():
            print(f"  {col}: {count} ({count/len(df):.1%})")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    # 2. Check treatment and response distributions
    print("\n=== 2. Treatment and Response Analysis ===")
    
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
    
    # Treatment-Response relationship
    print(f"\nTreatment-Response relationship:")
    treatment_response = df.groupby('treatment_ai_content')['response'].agg(['mean', 'count'])
    print(treatment_response)
    
    # Calculate uplift
    treatment_response_rate = df[df['treatment_ai_content'] == 1]['response'].mean()
    control_response_rate = df[df['treatment_ai_content'] == 0]['response'].mean()
    uplift = treatment_response_rate - control_response_rate
    
    print(f"\nUplift analysis:")
    print(f"  Treatment response rate: {treatment_response_rate:.2%}")
    print(f"  Control response rate: {control_response_rate:.2%}")
    print(f"  Uplift: {uplift:.2%}")
    
    # 3. Check for data leakage
    print("\n=== 3. Data Leakage Check ===")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['treatment_ai_content', 'response', 'user_id', 'post_id']]
    
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
    
    # Check feature correlation with treatment
    treatment_correlations = []
    for col in numeric_features:
        corr = abs(df[col].corr(df['treatment_ai_content']))
        treatment_correlations.append((col, corr))
    
    treatment_correlations.sort(key=lambda x: x[1], reverse=True)
    
    high_corr_with_treatment = [col for col, corr in treatment_correlations if corr > 0.1]
    
    if high_corr_with_treatment:
        print(f"⚠️  Features highly correlated with treatment (correlation > 0.1):")
        for col in high_corr_with_treatment[:10]:
            corr = next(corr for feat, corr in treatment_correlations if feat == col)
            print(f"  {col}: {corr:.4f}")
    else:
        print("✅ No features highly correlated with treatment")
    
    # Check feature correlation with response
    response_correlations = []
    for col in numeric_features:
        corr = abs(df[col].corr(df['response']))
        response_correlations.append((col, corr))
    
    response_correlations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 features correlated with response:")
    for col, corr in response_correlations[:10]:
        print(f"  {col}: {corr:.4f}")
    
    # 4. Check for perfect predictors
    print("\n=== 4. Perfect Predictors Check ===")
    
    perfect_treatment_predictors = []
    perfect_response_predictors = []
    
    for col in numeric_features:
        # Check if feature perfectly predicts treatment
        treatment_by_feature = df.groupby(col)['treatment_ai_content'].agg(['mean', 'std'])
        if (treatment_by_feature['std'] == 0).any():
            perfect_treatment_predictors.append(col)
        
        # Check if feature perfectly predicts response
        response_by_feature = df.groupby(col)['response'].agg(['mean', 'std'])
        if (response_by_feature['std'] == 0).any():
            perfect_response_predictors.append(col)
    
    if perfect_treatment_predictors:
        print(f"❌ Perfect treatment predictors found:")
        for col in perfect_treatment_predictors:
            print(f"  - {col}")
    
    if perfect_response_predictors:
        print(f"❌ Perfect response predictors found:")
        for col in perfect_response_predictors:
            print(f"  - {col}")
    
    if not perfect_treatment_predictors and not perfect_response_predictors:
        print("✅ No perfect predictors found")
    
    # 5. Check for multicollinearity
    print("\n=== 5. Multicollinearity Check ===")
    
    # Select top 20 features by response correlation
    top_features = [col for col, corr in response_correlations[:20]]
    
    if len(top_features) > 1:
        # Calculate correlation matrix
        correlation_matrix = df[top_features].corr()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(top_features)):
            for j in range(i+1, len(top_features)):
                corr = abs(correlation_matrix.iloc[i, j])
                if corr > 0.8:
                    high_corr_pairs.append((top_features[i], top_features[j], corr))
        
        if high_corr_pairs:
            print(f"Highly correlated feature pairs (|correlation| > 0.8):")
            for feat1, feat2, corr in high_corr_pairs:
                print(f"  {feat1} <-> {feat2}: {corr:.4f}")
        else:
            print("✅ No highly correlated feature pairs found")
    
    # 6. Quick model test
    print("\n=== 6. Quick Model Test ===")
    
    # Prepare data
    X = df[numeric_features].fillna(0)
    y = df['response']
    t = df['treatment_ai_content']
    
    # Remove rows with NaN in target
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    t = t[valid_mask]
    
    # Split data
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t, test_size=0.3, random_state=42, stratify=y
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
    
    # 7. Summary and recommendations
    print("\n=== 7. Summary and Recommendations ===")
    
    issues = []
    
    if total_missing > 0:
        issues.append(f"Found {total_missing} missing values")
    
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate rows")
    
    if len(high_corr_with_treatment) > 0:
        issues.append(f"Found {len(high_corr_with_treatment)} features highly correlated with treatment")
    
    if len(perfect_treatment_predictors) > 0:
        issues.append(f"Found {len(perfect_treatment_predictors)} perfect treatment predictors")
    
    if len(perfect_response_predictors) > 0:
        issues.append(f"Found {len(perfect_response_predictors)} perfect response predictors")
    
    if len(high_corr_pairs) > 0:
        issues.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
    
    if accuracy < 0.7:
        issues.append("Model accuracy is low, may need feature engineering")
    
    if issues:
        print("⚠️  Found the following issues:")
        for issue in issues:
            print(f"  - {issue}")
        
        print("\nRecommendations:")
        if total_missing > 0:
            print("  1. Handle missing values")
        if duplicates > 0:
            print("  2. Remove duplicate rows")
        if len(high_corr_with_treatment) > 0:
            print("  3. Remove features highly correlated with treatment")
        if len(perfect_treatment_predictors) > 0:
            print("  4. Remove perfect treatment predictors")
        if len(perfect_response_predictors) > 0:
            print("  5. Remove perfect response predictors")
        if len(high_corr_pairs) > 0:
            print("  6. Address multicollinearity issues")
        if accuracy < 0.7:
            print("  7. Improve feature engineering")
    else:
        print("✅ No obvious data quality issues found")
    
    return {
        'treatment_correlations': treatment_correlations,
        'response_correlations': response_correlations,
        'high_corr_with_treatment': high_corr_with_treatment,
        'perfect_treatment_predictors': perfect_treatment_predictors,
        'perfect_response_predictors': perfect_response_predictors,
        'high_corr_pairs': high_corr_pairs,
        'model_accuracy': accuracy,
        'uplift': uplift,
        'issues': issues
    }

if __name__ == "__main__":
    results = debug_uplift() 