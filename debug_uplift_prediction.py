import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

def debug_uplift_prediction():
    """Debug uplift prediction vs actual value discrepancy"""
    print("=== Uplift Prediction Debug Analysis ===")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Loaded {len(df)} samples")
    
    # Prepare features (same as in improved_uplift_analysis.py)
    feature_columns = [
        'user_ai_interest_score', 'user_ai_interest_weighted', 'user_ai_interactions',
        'user_reputation', 'user_post_count', 'user_account_age_days',
        'total_badges', 'gold_badges', 'silver_badges', 'bronze_badges', 
        'unique_badge_types', 'badge_rate_per_day', 'recent_badges_30d',
        'badge_quality_score', 'Score', 'ViewCount', 'AnswerCount', 'CommentCount', 
        'title_length', 'post_length', 'num_tags', 'post_age_days', 
        'total_votes', 'upvotes', 'user_post_tag_overlap', 
        'user_previous_ai_click_rate', 'ai_interest_x_treatment',
        'content_quality_score', 'engagement_rate', 'content_complexity'
    ]
    
    # Select existing features
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"Available features: {len(available_features)}")
    
    # Prepare data
    X = df[available_features].fillna(0).astype(float)
    treatment = df['treatment_ai_content'].astype(int)
    response = df['response'].astype(float)
    
    # Remove NaN values
    valid_mask = ~response.isna()
    X = X[valid_mask]
    treatment = treatment[valid_mask]
    response = response[valid_mask]
    
    print(f"Valid samples: {len(X)}")
    
    # Split data (same as in training)
    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X, treatment, response, test_size=0.2, random_state=42, stratify=treatment
    )
    
    print(f"\n=== Data Distribution Analysis ===")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Analyze test set distribution
    test_treatment = t_test == 1
    test_control = t_test == 0
    
    print(f"\nTest set treatment distribution:")
    print(f"  Treatment samples: {test_treatment.sum()}")
    print(f"  Control samples: {test_control.sum()}")
    
    # Calculate actual uplift in test set
    actual_treatment_rate = y_test[test_treatment].mean()
    actual_control_rate = y_test[test_control].mean()
    actual_uplift = actual_treatment_rate - actual_control_rate
    
    print(f"\nActual uplift in test set:")
    print(f"  Treatment rate: {actual_treatment_rate:.4f}")
    print(f"  Control rate: {actual_control_rate:.4f}")
    print(f"  Actual uplift: {actual_uplift:.4f}")
    
    # Train XGBoost model (same as in training)
    print(f"\n=== Model Training Debug ===")
    
    treatment_mask_train = t_train == 1
    control_mask_train = t_train == 0
    
    X_treatment = X_train[treatment_mask_train]
    y_treatment = y_train[treatment_mask_train]
    X_control = X_train[control_mask_train]
    y_control = y_train[control_mask_train]
    
    print(f"Training treatment samples: {len(X_treatment)}")
    print(f"Training control samples: {len(X_control)}")
    
    # Train models
    treatment_model = xgb.XGBRegressor(
        n_estimators=50, max_depth=4, subsample=0.7,
        learning_rate=0.1, random_state=42, verbosity=0
    )
    control_model = xgb.XGBRegressor(
        n_estimators=50, max_depth=4, subsample=0.7,
        learning_rate=0.1, random_state=42, verbosity=0
    )
    
    # Convert to numpy arrays
    X_treatment_np = X_treatment.values.astype(np.float32)
    X_control_np = X_control.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)
    y_treatment_np = y_treatment.values.astype(np.float32)
    y_control_np = y_control.values.astype(np.float32)
    
    # Train models
    treatment_model.fit(X_treatment_np, y_treatment_np)
    control_model.fit(X_control_np, y_control_np)
    
    # Make predictions
    y_pred_treatment = treatment_model.predict(X_test_np)
    y_pred_control = control_model.predict(X_test_np)
    uplift_predictions = y_pred_treatment - y_pred_control
    
    print(f"\n=== Prediction Analysis ===")
    print(f"Treatment predictions - Mean: {y_pred_treatment.mean():.4f}, Std: {y_pred_treatment.std():.4f}")
    print(f"Control predictions - Mean: {y_pred_control.mean():.4f}, Std: {y_pred_control.std():.4f}")
    print(f"Uplift predictions - Mean: {uplift_predictions.mean():.4f}, Std: {uplift_predictions.std():.4f}")
    
    # Analyze prediction distributions
    print(f"\n=== Prediction Distribution Analysis ===")
    print(f"Treatment predictions range: [{y_pred_treatment.min():.4f}, {y_pred_treatment.max():.4f}]")
    print(f"Control predictions range: [{y_pred_control.min():.4f}, {y_pred_control.max():.4f}]")
    print(f"Uplift predictions range: [{uplift_predictions.min():.4f}, {uplift_predictions.max():.4f}]")
    
    # Check for extreme values
    treatment_extreme = np.abs(y_pred_treatment - y_pred_treatment.mean()) > 3 * y_pred_treatment.std()
    control_extreme = np.abs(y_pred_control - y_pred_control.mean()) > 3 * y_pred_control.std()
    uplift_extreme = np.abs(uplift_predictions - uplift_predictions.mean()) > 3 * uplift_predictions.std()
    
    print(f"\nExtreme values (>3 std):")
    print(f"  Treatment predictions: {treatment_extreme.sum()} ({treatment_extreme.sum()/len(y_pred_treatment)*100:.2f}%)")
    print(f"  Control predictions: {control_extreme.sum()} ({control_extreme.sum()/len(y_pred_control)*100:.2f}%)")
    print(f"  Uplift predictions: {uplift_extreme.sum()} ({uplift_extreme.sum()/len(uplift_predictions)*100:.2f}%)")
    
    # Compare with actual values
    print(f"\n=== Comparison with Actual Values ===")
    print(f"Actual treatment rate: {actual_treatment_rate:.4f}")
    print(f"Predicted treatment mean: {y_pred_treatment.mean():.4f}")
    print(f"Treatment prediction error: {abs(actual_treatment_rate - y_pred_treatment.mean()):.4f}")
    
    print(f"Actual control rate: {actual_control_rate:.4f}")
    print(f"Predicted control mean: {y_pred_control.mean():.4f}")
    print(f"Control prediction error: {abs(actual_control_rate - y_pred_control.mean()):.4f}")
    
    print(f"Actual uplift: {actual_uplift:.4f}")
    print(f"Predicted uplift: {uplift_predictions.mean():.4f}")
    print(f"Uplift prediction error: {abs(actual_uplift - uplift_predictions.mean()):.4f}")
    
    # Check if models are predicting reasonable values
    print(f"\n=== Model Reasonableness Check ===")
    print(f"Treatment model predicts values in [0,1]: {np.all((y_pred_treatment >= 0) & (y_pred_treatment <= 1))}")
    print(f"Control model predicts values in [0,1]: {np.all((y_pred_control >= 0) & (y_pred_control <= 1))}")
    
    # Check response variable distribution
    print(f"\n=== Response Variable Analysis ===")
    print(f"Response variable range: [{response.min():.4f}, {response.max():.4f}]")
    print(f"Response variable mean: {response.mean():.4f}")
    print(f"Response variable std: {response.std():.4f}")
    print(f"Response variable unique values: {np.unique(response)}")
    
    # Check if response is binary
    is_binary = len(np.unique(response)) == 2
    print(f"Response is binary: {is_binary}")
    
    if is_binary:
        print(f"Response value counts:")
        for val in np.unique(response):
            count = (response == val).sum()
            print(f"  {val}: {count} ({count/len(response)*100:.2f}%)")
    
    # 1. 检查模型预测是否收敛到均值
    print("\n=== 1. 检查模型预测是否收敛到均值 ===")
    print(f"Test set真实treatment=1样本数: {test_treatment.sum()}")
    print(f"Test set真实control=0样本数: {test_control.sum()}")
    print(f"treatment_model在treatment=1子集预测均值: {y_pred_treatment[test_treatment].mean():.4f}")
    print(f"control_model在control=0子集预测均值: {y_pred_control[test_control].mean():.4f}")
    print(f"treatment_model在全体测试集预测均值: {y_pred_treatment.mean():.4f}")
    print(f"control_model在全体测试集预测均值: {y_pred_control.mean():.4f}")
    print(f"两模型预测均值之差: {(y_pred_treatment.mean() - y_pred_control.mean()):.4f}")
    print(f"treatment_model预测方差: {y_pred_treatment.var():.4f}")
    print(f"control_model预测方差: {y_pred_control.var():.4f}")
    print(f"真实treatment=1均值: {actual_treatment_rate:.4f}")
    print(f"真实control=0均值: {actual_control_rate:.4f}")

    return {
        'actual_uplift': actual_uplift,
        'predicted_uplift': uplift_predictions.mean(),
        'uplift_error': abs(actual_uplift - uplift_predictions.mean()),
        'treatment_predictions': y_pred_treatment,
        'control_predictions': y_pred_control,
        'uplift_predictions': uplift_predictions,
        'actual_treatment_rate': actual_treatment_rate,
        'actual_control_rate': actual_control_rate
    }

if __name__ == "__main__":
    results = debug_uplift_prediction()
    print(f"\n=== Summary ===")
    print(f"Actual uplift: {results['actual_uplift']:.4f}")
    print(f"Predicted uplift: {results['predicted_uplift']:.4f}")
    print(f"Uplift error: {results['uplift_error']:.4f}")
    print(f"Error percentage: {results['uplift_error']/abs(results['actual_uplift'])*100:.2f}%") 