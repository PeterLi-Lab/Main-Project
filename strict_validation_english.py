import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def strict_validation_english():
    """Strict validation to identify remaining data leakage issues"""
    print("=== Strict Validation Analysis ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data size: {len(df):,}")
    
    # Current clean features
    clean_features = [
        'user_reputation', 'user_post_count', 'Score', 'ViewCount', 
        'AnswerCount', 'CommentCount', 'title_length', 'post_length', 
        'num_tags', 'content_complexity', 'content_quality_score'
    ]
    
    print(f"Current clean features: {len(clean_features)}")
    
    # 1. Analyze each feature in detail
    print("\n=== 1. Detailed Feature Analysis ===")
    
    X = df[clean_features].fillna(0)
    treatment = df['treatment_ai_content']
    response = df['response']
    
    for feature in clean_features:
        print(f"\nAnalyzing {feature}:")
        
        # Treatment correlation
        treatment_corr = abs(X[feature].corr(treatment))
        response_corr = abs(X[feature].corr(response))
        
        print(f"  Treatment correlation: {treatment_corr:.4f}")
        print(f"  Response correlation: {response_corr:.4f}")
        
        # Distribution analysis
        treatment_group = X[treatment == 1][feature]
        control_group = X[treatment == 0][feature]
        
        treatment_mean = treatment_group.mean()
        control_mean = control_group.mean()
        mean_diff = treatment_mean - control_mean
        
        print(f"  Treatment mean: {treatment_mean:.4f}")
        print(f"  Control mean: {control_mean:.4f}")
        print(f"  Mean difference: {mean_diff:.4f}")
        
        # Check if this feature might contain treatment information
        if treatment_corr > 0.3:
            print(f"  ⚠️  High treatment correlation - possible data leakage")
        
        # Check if feature values are too different between groups
        if abs(mean_diff) > treatment_group.std() * 0.5:
            print(f"  ⚠️  Large mean difference - possible data leakage")
    
    # 2. Check for indirect data leakage
    print("\n=== 2. Indirect Data Leakage Check ===")
    
    # Check if any features are derived from treatment-related information
    suspicious_features = []
    
    for feature in clean_features:
        # Check if feature might be derived from treatment
        treatment_corr = abs(X[feature].corr(treatment))
        
        if treatment_corr > 0.2:
            # Check if this correlation is due to business logic or data leakage
            print(f"\nAnalyzing {feature} (correlation: {treatment_corr:.4f}):")
            
            # Check if this is legitimate business difference
            if feature in ['num_tags', 'title_length', 'post_length', 'AnswerCount']:
                print(f"  Business explanation: AI content naturally has different {feature}")
                print(f"  This correlation may be legitimate business reality")
            else:
                print(f"  ⚠️  Suspicious correlation - investigate further")
                suspicious_features.append(feature)
    
    # 3. Test with minimal features
    print("\n=== 3. Minimal Feature Test ===")
    
    # Try with only basic features that shouldn't be treatment-related
    basic_features = ['user_reputation', 'user_post_count', 'Score', 'ViewCount']
    
    print(f"Testing with basic features only: {basic_features}")
    
    X_basic = df[basic_features].fillna(0)
    
    # Split data
    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X_basic, treatment, response, test_size=0.3, random_state=42, stratify=treatment
    )
    
    # Train models
    treatment_mask_train = t_train == 1
    control_mask_train = t_train == 0
    
    X_treatment = X_train[treatment_mask_train]
    y_treatment = y_train[treatment_mask_train]
    X_control = X_train[control_mask_train]
    y_control = y_train[control_mask_train]
    
    treatment_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42, verbosity=0)
    control_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42, verbosity=0)
    
    treatment_model.fit(X_treatment.values, y_treatment.values)
    control_model.fit(X_control.values, y_control.values)
    
    # Predict
    y_pred_treatment = treatment_model.predict(X_test.values)
    y_pred_control = control_model.predict(X_test.values)
    
    # Calculate uplift
    treatment_mask_test = t_test == 1
    control_mask_test = t_test == 0
    
    actual_uplift = y_test[treatment_mask_test].mean() - y_test[control_mask_test].mean()
    uplift_pred = y_pred_treatment[treatment_mask_test].mean() - y_pred_control[control_mask_test].mean()
    
    uplift_error = abs(actual_uplift - uplift_pred)
    uplift_accuracy = max(0, 1 - uplift_error / abs(actual_uplift)) if actual_uplift != 0 else 0
    
    print(f"Basic features only:")
    print(f"  Actual uplift: {actual_uplift:.4f}")
    print(f"  Predicted uplift: {uplift_pred:.4f}")
    print(f"  Uplift accuracy: {uplift_accuracy:.2%}")
    
    # 4. Test with different model complexities
    print("\n=== 4. Model Complexity Test ===")
    
    complexities = [
        {'name': 'Very Simple', 'n_estimators': 10, 'max_depth': 2},
        {'name': 'Simple', 'n_estimators': 20, 'max_depth': 3},
        {'name': 'Medium', 'n_estimators': 50, 'max_depth': 4},
        {'name': 'Complex', 'n_estimators': 100, 'max_depth': 6}
    ]
    
    for config in complexities:
        treatment_model = xgb.XGBRegressor(
            n_estimators=config['n_estimators'], 
            max_depth=config['max_depth'],
            random_state=42, verbosity=0
        )
        control_model = xgb.XGBRegressor(
            n_estimators=config['n_estimators'], 
            max_depth=config['max_depth'],
            random_state=42, verbosity=0
        )
        
        treatment_model.fit(X_treatment.values, y_treatment.values)
        control_model.fit(X_control.values, y_control.values)
        
        y_pred_treatment = treatment_model.predict(X_test.values)
        y_pred_control = control_model.predict(X_test.values)
        
        uplift_pred = y_pred_treatment[treatment_mask_test].mean() - y_pred_control[control_mask_test].mean()
        uplift_error = abs(actual_uplift - uplift_pred)
        uplift_accuracy = max(0, 1 - uplift_error / abs(actual_uplift)) if actual_uplift != 0 else 0
        
        print(f"  {config['name']}: {uplift_accuracy:.2%}")
    
    # 5. Check for deterministic relationships
    print("\n=== 5. Deterministic Relationship Check ===")
    
    # Check if any feature combinations perfectly predict treatment
    for i, feat1 in enumerate(clean_features):
        for j, feat2 in enumerate(clean_features[i+1:], i+1):
            # Create interaction feature
            interaction = X[feat1] * X[feat2]
            interaction_corr = abs(interaction.corr(treatment))
            
            if interaction_corr > 0.8:
                print(f"  ⚠️  High interaction correlation: {feat1} * {feat2} = {interaction_corr:.4f}")
    
    # 6. Final recommendations
    print("\n=== 6. Final Recommendations ===")
    
    issues_found = []
    
    if uplift_accuracy > 0.95:
        issues_found.append("Accuracy still too high with basic features")
    
    if len(suspicious_features) > 0:
        issues_found.append(f"Found {len(suspicious_features)} suspicious features")
    
    if issues_found:
        print("⚠️  Issues found:")
        for issue in issues_found:
            print(f"  - {issue}")
        
        print("\nRecommendations:")
        print("  1. Use only basic features (user_reputation, user_post_count, Score, ViewCount)")
        print("  2. Implement time-based validation")
        print("  3. Consider simpler models (Linear Regression)")
        print("  4. Validate with business stakeholders")
    else:
        print("✅ No obvious issues found with basic features")
    
    return {
        'basic_features_accuracy': uplift_accuracy,
        'suspicious_features': suspicious_features,
        'issues_found': issues_found
    }

if __name__ == "__main__":
    results = strict_validation_english() 