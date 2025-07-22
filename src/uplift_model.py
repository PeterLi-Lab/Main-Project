import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class UpliftModel:
    """Uplift modeling class for treatment effect estimation"""
    
    def __init__(self, model_type='xgboost', **model_params):
        """
        Initialize uplift model
        
        Args:
            model_type: 'xgboost', 'random_forest', or 'linear'
            **model_params: Parameters for the specific model
        """
        self.model_type = model_type
        self.model_params = model_params
        self.treatment_model = None
        self.control_model = None
        self.feature_names = None
        
    def _create_model(self):
        """Create model based on model_type"""
        if self.model_type == 'xgboost':
            return xgb.XGBRegressor(**self.model_params)
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(**self.model_params)
        elif self.model_type == 'linear':
            return LinearRegression(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X_train, t_train, y_train):
        """
        Fit uplift model
        
        Args:
            X_train: Training features
            t_train: Treatment assignment (0 or 1)
            y_train: Response variable
        """
        print(f"Training {self.model_type} uplift model...")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Split data by treatment
        treatment_mask = t_train == 1
        control_mask = t_train == 0
        
        X_treatment = X_train[treatment_mask]
        y_treatment = y_train[treatment_mask]
        X_control = X_train[control_mask]
        y_control = y_train[control_mask]
        
        print(f"Treatment samples: {len(X_treatment):,}")
        print(f"Control samples: {len(X_control):,}")
        
        # Train treatment model
        self.treatment_model = self._create_model()
        self.treatment_model.fit(X_treatment.values, y_treatment.values)
        
        # Train control model
        self.control_model = self._create_model()
        self.control_model.fit(X_control.values, y_control.values)
        
        print("Model training completed!")
        
    def predict(self, X):
        """
        Predict uplift
        
        Args:
            X: Features for prediction
            
        Returns:
            uplift_pred: Predicted uplift values
        """
        if self.treatment_model is None or self.control_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Predict treatment and control outcomes
        y_pred_treatment = self.treatment_model.predict(X.values)
        y_pred_control = self.control_model.predict(X.values)
        
        # Calculate uplift
        uplift_pred = y_pred_treatment - y_pred_control
        
        return uplift_pred
    
    def evaluate(self, X_test, t_test, y_test):
        """
        Evaluate uplift model performance
        
        Args:
            X_test: Test features
            t_test: Test treatment assignment
            y_test: Test response variable
            
        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating uplift model...")
        
        # Calculate actual uplift
        actual_uplift = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
        
        # Predict uplift
        uplift_pred = self.predict(X_test)
        
        # Calculate metrics
        uplift_error = abs(actual_uplift - uplift_pred.mean())
        uplift_accuracy = max(0, 1 - uplift_error / abs(actual_uplift)) if actual_uplift != 0 else 0
        
        # Calculate individual model performance
        treatment_mask = t_test == 1
        control_mask = t_test == 0
        
        X_treatment = X_test[treatment_mask]
        y_treatment = y_test[treatment_mask]
        X_control = X_test[control_mask]
        y_control = y_test[control_mask]
        
        y_pred_treatment = self.treatment_model.predict(X_treatment.values)
        y_pred_control = self.control_model.predict(X_control.values)
        
        treatment_r2 = r2_score(y_treatment.values, y_pred_treatment)
        control_r2 = r2_score(y_control.values, y_pred_control)
        
        treatment_mae = mean_absolute_error(y_treatment.values, y_pred_treatment)
        control_mae = mean_absolute_error(y_control.values, y_pred_control)
        
        results = {
            'actual_uplift': actual_uplift,
            'predicted_uplift': uplift_pred.mean(),
            'uplift_error': uplift_error,
            'uplift_accuracy': uplift_accuracy,
            'treatment_r2': treatment_r2,
            'control_r2': control_r2,
            'treatment_mae': treatment_mae,
            'control_mae': control_mae
        }
        
        print(f"Actual Uplift: {actual_uplift:.4f}")
        print(f"Predicted Uplift: {uplift_pred.mean():.4f}")
        print(f"Uplift Error: {uplift_error:.4f}")
        print(f"Uplift Accuracy: {uplift_accuracy:.2%}")
        print(f"Treatment Model R²: {treatment_r2:.4f}")
        print(f"Control Model R²: {control_r2:.4f}")
        
        return results
    
    def get_feature_importance(self):
        """Get feature importance from both models"""
        if self.treatment_model is None or self.control_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Get feature importance from treatment model
        if hasattr(self.treatment_model, 'feature_importances_'):
            treatment_importance = self.treatment_model.feature_importances_
        else:
            treatment_importance = np.abs(self.treatment_model.coef_)
        
        # Get feature importance from control model
        if hasattr(self.control_model, 'feature_importances_'):
            control_importance = self.control_model.feature_importances_
        else:
            control_importance = np.abs(self.control_model.coef_)
        
        # Create importance dataframes
        treatment_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': treatment_importance
        }).sort_values('importance', ascending=False)
        
        control_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': control_importance
        }).sort_values('importance', ascending=False)
        
        return {
            'treatment_importance': treatment_importance_df,
            'control_importance': control_importance_df
        }
    
    def cross_validate(self, X, t, y, cv=5):
        """Perform cross-validation"""
        print(f"Performing {cv}-fold cross-validation...")
        
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        treatment_cv_scores = []
        control_cv_scores = []
        uplift_accuracies = []
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold = X.iloc[train_idx]
            t_train_fold = t.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            
            X_val_fold = X.iloc[val_idx]
            t_val_fold = t.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Train model on fold
            self.fit(X_train_fold, t_train_fold, y_train_fold)
            
            # Evaluate on validation fold
            results = self.evaluate(X_val_fold, t_val_fold, y_val_fold)
            
            treatment_cv_scores.append(results['treatment_r2'])
            control_cv_scores.append(results['control_r2'])
            uplift_accuracies.append(results['uplift_accuracy'])
        
        cv_results = {
            'treatment_cv_scores': treatment_cv_scores,
            'control_cv_scores': control_cv_scores,
            'uplift_accuracies': uplift_accuracies,
            'treatment_cv_mean': np.mean(treatment_cv_scores),
            'treatment_cv_std': np.std(treatment_cv_scores),
            'control_cv_mean': np.mean(control_cv_scores),
            'control_cv_std': np.std(control_cv_scores),
            'uplift_accuracy_mean': np.mean(uplift_accuracies),
            'uplift_accuracy_std': np.std(uplift_accuracies)
        }
        
        print(f"Cross-validation results:")
        print(f"Treatment Model CV R²: {cv_results['treatment_cv_mean']:.4f} ± {cv_results['treatment_cv_std']:.4f}")
        print(f"Control Model CV R²: {cv_results['control_cv_mean']:.4f} ± {cv_results['control_cv_std']:.4f}")
        print(f"Uplift Accuracy CV: {cv_results['uplift_accuracy_mean']:.2%} ± {cv_results['uplift_accuracy_std']:.2%}")
        
        return cv_results

def create_uplift_model(model_type='xgboost', **kwargs):
    """Factory function to create uplift model"""
    default_params = {
        'xgboost': {'n_estimators': 50, 'max_depth': 4, 'random_state': 42},
        'random_forest': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
        'linear': {}
    }
    
    params = default_params.get(model_type, {}).copy()
    params.update(kwargs)
    
    return UpliftModel(model_type=model_type, **params)

if __name__ == "__main__":
    # Example usage
    from src.data_preprocessing import preprocess_uplift_data
    
    # Load and preprocess data
    data = preprocess_uplift_data('uplift_model_data.csv')
    
    # Create and train model
    model = create_uplift_model('xgboost')
    model.fit(data['X_train'], data['t_train'], data['y_train'])
    
    # Evaluate model
    results = model.evaluate(data['X_test'], data['t_test'], data['y_test'])
    
    # Get feature importance
    importance = model.get_feature_importance()
    print("\nTop 10 treatment model features:")
    print(importance['treatment_importance'].head(10)) 