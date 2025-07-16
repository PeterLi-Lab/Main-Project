import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def final_strict_validation_english():
    """Final strict validation with simplest possible approach"""
    print("=== Final Strict Validation ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data size: {len(df):,}")
    
    # Use only the most basic features that should not be treatment-related
    basic_features = ['user_reputation', 'user_post_count']
    
    print(f"Using only basic features: {basic_features}")
    
    X = df[basic_features].fillna(0)
    treatment = df['treatment_ai_content']
    response = df['response']
    
    # 1. Check correlations
    print("\n=== 1. Basic Feature Correlations ===")
    
    for feature in basic_features:
        treatment_corr = abs(X[feature].corr(treatment))
        response_corr = abs(X[feature].corr(response))
        
        print(f"{feature}:")
        print(f"  Treatment correlation: {treatment_corr:.4f}")
        print(f"  Response correlation: {response_corr:.4f}")
        
        if treatment_corr > 0.1:
            print(f"  ⚠️  Suspicious treatment correlation")
    
    # 2. Test with Linear Regression (simplest model)
    print("\n=== 2. Linear Regression Test ===")
    
    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X, treatment, response, test_size=0.3, random_state=42, stratify=treatment
    )
    
    # Train separate models for treatment and control
    treatment_mask_train = t_train == 1
    control_mask_train = t_train == 0
    
    X_treatment = X_train[treatment_mask_train]
    y_treatment = y_train[treatment_mask_train]
    X_control = X_train[control_mask_train]
    y_control = y_train[control_mask_train]
    
    # Linear Regression models
    treatment_model = LinearRegression()
    control_model = LinearRegression()
    
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
    
    print(f"Linear Regression results:")
    print(f"  Actual uplift: {actual_uplift:.4f}")
    print(f"  Predicted uplift: {uplift_pred:.4f}")
    print(f"  Uplift accuracy: {uplift_accuracy:.2%}")
    
    # 3. Test with different random seeds
    print("\n=== 3. Random Seed Test ===")
    
    seeds = [42, 123, 456, 789, 999]
    accuracies = []
    
    for seed in seeds:
        X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
            X, treatment, response, test_size=0.3, random_state=seed, stratify=treatment
        )
        
        treatment_mask_train = t_train == 1
        control_mask_train = t_train == 0
        
        X_treatment = X_train[treatment_mask_train]
        y_treatment = y_train[treatment_mask_train]
        X_control = X_train[control_mask_train]
        y_control = y_train[control_mask_train]
        
        treatment_model = LinearRegression()
        control_model = LinearRegression()
        
        treatment_model.fit(X_treatment.values, y_treatment.values)
        control_model.fit(X_control.values, y_control.values)
        
        y_pred_treatment = treatment_model.predict(X_test.values)
        y_pred_control = control_model.predict(X_test.values)
        
        treatment_mask_test = t_test == 1
        control_mask_test = t_test == 0
        
        actual_uplift = y_test[treatment_mask_test].mean() - y_test[control_mask_test].mean()
        uplift_pred = y_pred_treatment[treatment_mask_test].mean() - y_pred_control[control_mask_test].mean()
        
        uplift_error = abs(actual_uplift - uplift_pred)
        uplift_accuracy = max(0, 1 - uplift_error / abs(actual_uplift)) if actual_uplift != 0 else 0
        
        accuracies.append(uplift_accuracy)
        print(f"  Seed {seed}: {uplift_accuracy:.2%}")
    
    print(f"Accuracy variance: {np.var(accuracies):.4f}")
    
    # 4. Check if the problem is in the data itself
    print("\n=== 4. Data Quality Check ===")
    
    # Check if treatment assignment is truly random
    print(f"Treatment distribution:")
    print(f"  Treatment group: {treatment.sum():,} ({treatment.mean():.2%})")
    print(f"  Control group: {(1-treatment).sum():,} ({(1-treatment).mean():.2%})")
    
    # Check if response is too simple
    print(f"\nResponse distribution:")
    print(f"  Response=1: {response.sum():,} ({response.mean():.2%})")
    print(f"  Response=0: {(1-response).sum():,} ({(1-response).mean():.2%})")
    
    # Check if there's a deterministic relationship
    print(f"\nTreatment vs Response cross-tabulation:")
    cross_tab = pd.crosstab(treatment, response)
    print(cross_tab)
    
    # 5. Test with single feature
    print("\n=== 5. Single Feature Test ===")
    
    for feature in basic_features:
        X_single = df[[feature]].fillna(0)
        
        X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
            X_single, treatment, response, test_size=0.3, random_state=42, stratify=treatment
        )
        
        treatment_mask_train = t_train == 1
        control_mask_train = t_train == 0
        
        X_treatment = X_train[treatment_mask_train]
        y_treatment = y_train[treatment_mask_train]
        X_control = X_train[control_mask_train]
        y_control = y_train[control_mask_train]
        
        treatment_model = LinearRegression()
        control_model = LinearRegression()
        
        treatment_model.fit(X_treatment.values, y_treatment.values)
        control_model.fit(X_control.values, y_control.values)
        
        y_pred_treatment = treatment_model.predict(X_test.values)
        y_pred_control = control_model.predict(X_test.values)
        
        treatment_mask_test = t_test == 1
        control_mask_test = t_test == 0
        
        actual_uplift = y_test[treatment_mask_test].mean() - y_test[control_mask_test].mean()
        uplift_pred = y_pred_treatment[treatment_mask_test].mean() - y_pred_control[control_mask_test].mean()
        
        uplift_error = abs(actual_uplift - uplift_pred)
        uplift_accuracy = max(0, 1 - uplift_error / abs(actual_uplift)) if actual_uplift != 0 else 0
        
        print(f"  {feature} only: {uplift_accuracy:.2%}")
    
    # 6. Final analysis
    print("\n=== 6. Final Analysis ===")
    
    if np.mean(accuracies) > 0.9:
        print("⚠️  CRITICAL ISSUE: Even with simplest features and Linear Regression, accuracy is too high")
        print("This suggests the problem may be in the data generation process itself")
        print("Possible causes:")
        print("  1. Treatment assignment is not truly random")
        print("  2. Response variable is too simple or deterministic")
        print("  3. Data preprocessing introduced artificial patterns")
        print("  4. Business logic creates natural correlations that are too strong")
    else:
        print("✅ Accuracy is reasonable with simple features and Linear Regression")
    
    # 7. Recommendations
    print("\n=== 7. Recommendations ===")
    
    if np.mean(accuracies) > 0.9:
        print("Immediate actions needed:")
        print("  1. Investigate data generation process")
        print("  2. Check treatment assignment randomization")
        print("  3. Validate response variable definition")
        print("  4. Consider using synthetic data for testing")
        print("  5. Implement time-based validation")
    else:
        print("Model is working as expected with simple features")
    
    return {
        'mean_accuracy': np.mean(accuracies),
        'accuracy_variance': np.var(accuracies),
        'critical_issue': np.mean(accuracies) > 0.9
    }

if __name__ == "__main__":
    results = final_strict_validation_english() 