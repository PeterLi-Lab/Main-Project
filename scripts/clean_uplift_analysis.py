import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def clean_uplift_analysis():
    """Uplift analysis using clean features"""
    print("=== Clean Features Uplift Analysis ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Define clean features (remove all leaky features)
    clean_features = [
        'user_reputation', 'user_post_count', 'user_account_age_days',
        'total_badges', 'gold_badges', 'silver_badges', 'bronze_badges',
        'unique_badge_types', 'badge_rate_per_day', 'recent_badges_30d',
        'badge_quality_score', 'Score', 'ViewCount', 'AnswerCount', 'CommentCount',
        'title_length', 'post_length', 'post_age_days', 'total_votes', 'upvotes',
        'content_quality_score', 'engagement_rate', 'content_complexity'
    ]
    
    # Only keep existing features
    available_clean_features = [col for col in clean_features if col in df.columns]
    print(f"Number of clean features: {len(available_clean_features)}")
    print("Clean features list:")
    for feature in available_clean_features:
        print(f"  - {feature}")
    
    # Prepare data
    X = df[available_clean_features].fillna(0)
    treatment = df['treatment_ai_content']
    response = df['response']
    
    # Remove NaN values
    valid_mask = ~response.isna()
    X = X[valid_mask]
    treatment = treatment[valid_mask]
    response = response[valid_mask]
    
    print(f"\nValid sample count: {len(X):,}")
    
    # Check correlation between clean features and treatment
    print("\n=== Clean Features-Treatment Correlation Check ===")
    treatment_correlations = []
    for col in available_clean_features:
        if X[col].dtype in ['int64', 'float64']:
            corr = abs(X[col].corr(treatment))
            treatment_correlations.append((col, corr))
    
    treatment_correlations.sort(key=lambda x: x[1], reverse=True)
    print("Top 10 clean features with highest correlation to treatment:")
    for col, corr in treatment_correlations[:10]:
        print(f"  {col}: {corr:.4f}")
    
    # Check if there are high correlation features
    high_corr_features = [col for col, corr in treatment_correlations if corr > 0.3]
    print(f"\nNumber of clean features with correlation > 0.3 to treatment: {len(high_corr_features)}")
    if high_corr_features:
        print("⚠️  Warning: Still have high correlation features!")
        for col in high_corr_features:
            print(f"  - {col}")
    else:
        print("✅ Clean feature set has no high correlation features")
    
    # Use 50% test set for strict validation
    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X, treatment, response, test_size=0.5, random_state=42, stratify=treatment
    )
    
    print(f"\nTraining set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Calculate actual uplift
    actual_uplift = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
    print(f"\nActual Uplift: {actual_uplift:.4f}")
    
    # Train models
    treatment_mask_train = t_train == 1
    control_mask_train = t_train == 0
    
    X_treatment = X_train[treatment_mask_train]
    y_treatment = y_train[treatment_mask_train]
    X_control = X_train[control_mask_train]
    y_control = y_train[control_mask_train]
    
    print(f"\nTraining treatment samples: {len(X_treatment):,}")
    print(f"Training control samples: {len(X_control):,}")
    
    # Train XGBoost models
    print("\n=== Training XGBoost Models ===")
    treatment_model = xgb.XGBRegressor(
        n_estimators=50, max_depth=4, subsample=0.7,
        learning_rate=0.1, random_state=42, verbosity=0
    )
    control_model = xgb.XGBRegressor(
        n_estimators=50, max_depth=4, subsample=0.7,
        learning_rate=0.1, random_state=42, verbosity=0
    )
    
    treatment_model.fit(X_treatment.values, y_treatment.values)
    control_model.fit(X_control.values, y_control.values)
    
    # Predict
    y_pred_treatment = treatment_model.predict(X_test.values)
    y_pred_control = control_model.predict(X_test.values)
    
    # Calculate uplift prediction
    uplift_pred = y_pred_treatment[t_test == 1].mean() - y_pred_control[t_test == 0].mean()
    uplift_error = abs(actual_uplift - uplift_pred)
    uplift_accuracy = max(0, 1 - uplift_error / abs(actual_uplift)) if actual_uplift != 0 else 0
    
    print(f"\n=== Clean Features Results ===")
    print(f"Actual Uplift: {actual_uplift:.4f}")
    print(f"Predicted Uplift: {uplift_pred:.4f}")
    print(f"Uplift Error: {uplift_error:.4f}")
    print(f"Uplift Accuracy: {uplift_accuracy:.2%}")
    
    # Cross validation
    cv_scores_treatment = cross_val_score(treatment_model, X_treatment.values, y_treatment.values, cv=5, scoring='r2')
    cv_scores_control = cross_val_score(control_model, X_control.values, y_control.values, cv=5, scoring='r2')
    
    print(f"\nCross validation results:")
    print(f"Treatment Model CV R²: {cv_scores_treatment.mean():.4f} ± {cv_scores_treatment.std():.4f}")
    print(f"Control Model CV R²: {cv_scores_control.mean():.4f} ± {cv_scores_control.std():.4f}")
    
    # Feature importance
    print(f"\n=== Feature Importance (Treatment Model) ===")
    importance_df = pd.DataFrame({
        'feature': available_clean_features,
        'importance': treatment_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 feature importance:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Prediction distribution check
    print(f"\n=== Prediction Distribution Check ===")
    print(f"Treatment prediction mean: {y_pred_treatment.mean():.4f}")
    print(f"Control prediction mean: {y_pred_control.mean():.4f}")
    print(f"Treatment prediction variance: {y_pred_treatment.var():.4f}")
    print(f"Control prediction variance: {y_pred_control.var():.4f}")
    
    # Conclusion
    print(f"\n=== Conclusion ===")
    if uplift_accuracy > 0.8:
        print("✅ Clean feature set achieved reasonable accuracy")
    elif uplift_accuracy > 0.5:
        print("⚠️  Accuracy is moderate, may need more feature engineering")
    else:
        print("❌ Low accuracy, may need to redesign features")
    
    print(f"Uplift prediction direction: {'Correct' if (actual_uplift > 0 and uplift_pred > 0) or (actual_uplift < 0 and uplift_pred < 0) else 'Incorrect'}")
    
    return {
        'actual_uplift': actual_uplift,
        'predicted_uplift': uplift_pred,
        'uplift_error': uplift_error,
        'uplift_accuracy': uplift_accuracy,
        'clean_features': available_clean_features
    }

if __name__ == "__main__":
    results = clean_uplift_analysis() 