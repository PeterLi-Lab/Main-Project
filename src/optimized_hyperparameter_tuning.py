import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class OptimizedHyperparameterTuning:
    """Optimized hyperparameter tuning for uplift models"""
    
    def __init__(self):
        self.best_models = {}
        self.tuning_results = {}
        
    def create_uplift_scorer(self, X, t, y):
        """Create custom scorer for uplift modeling"""
        def uplift_score(estimator, X_val, y_val):
            # Predict probabilities
            y_pred_proba = estimator.predict_proba(X_val)[:, 1]
            
            # Split by treatment
            treatment_mask = t.iloc[X_val.index] == 1
            control_mask = t.iloc[X_val.index] == 0
            
            if treatment_mask.sum() == 0 or control_mask.sum() == 0:
                return 0.0
            
            # Calculate uplift
            treatment_score = y_pred_proba[treatment_mask].mean()
            control_score = y_pred_proba[control_mask].mean()
            uplift = treatment_score - control_score
            
            return uplift
        
        return make_scorer(uplift_score, greater_is_better=True)
    
    def define_parameter_grids(self):
        """Define comprehensive parameter grids for different models"""
        
        # XGBoost parameter grid
        xgb_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
        
        # Random Forest parameter grid
        rf_param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        return {
            'XGBoost': xgb_param_grid,
            'Random Forest': rf_param_grid
        }
    
    def define_parameter_distributions(self):
        """Define parameter distributions for randomized search"""
        
        # XGBoost parameter distributions
        xgb_param_dist = {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 2, 3, 4, 5, 6],
            'reg_alpha': [0, 0.01, 0.05, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.01, 0.05, 0.1, 0.5, 1.0]
        }
        
        # Random Forest parameter distributions
        rf_param_dist = {
            'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400],
            'max_depth': [5, 8, 10, 12, 15, 18, 20, None],
            'min_samples_split': [2, 3, 5, 8, 10, 15],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        return {
            'XGBoost': xgb_param_dist,
            'Random Forest': rf_param_dist
        }
    
    def grid_search_tuning(self, X_train, y_train, t_train, cv=5, n_jobs=-1):
        """Perform grid search hyperparameter tuning"""
        print("=== Grid Search Hyperparameter Tuning ===")
        
        parameter_grids = self.define_parameter_grids()
        
        for model_name, param_grid in parameter_grids.items():
            print(f"\n--- Tuning {model_name} with Grid Search ---")
            
            # Create model
            if model_name == 'XGBoost':
                base_model = xgb.XGBClassifier(random_state=42, verbosity=0)
            else:
                base_model = RandomForestClassifier(random_state=42)
            
            # Create uplift scorer
            uplift_scorer = self.create_uplift_scorer(X_train, t_train, y_train)
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring=uplift_scorer,
                cv=cv,
                n_jobs=n_jobs,
                verbose=1,
                return_train_score=True
            )
            
            # Fit grid search
            grid_search.fit(X_train, y_train)
            
            # Store results
            self.best_models[model_name] = grid_search.best_estimator_
            self.tuning_results[model_name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best uplift score: {grid_search.best_score_:.4f}")
    
    def randomized_search_tuning(self, X_train, y_train, t_train, cv=5, n_iter=100, n_jobs=-1):
        """Perform randomized search hyperparameter tuning"""
        print("=== Randomized Search Hyperparameter Tuning ===")
        
        parameter_distributions = self.define_parameter_distributions()
        
        for model_name, param_dist in parameter_distributions.items():
            print(f"\n--- Tuning {model_name} with Randomized Search ---")
            
            # Create model
            if model_name == 'XGBoost':
                base_model = xgb.XGBClassifier(random_state=42, verbosity=0)
            else:
                base_model = RandomForestClassifier(random_state=42)
            
            # Create uplift scorer
            uplift_scorer = self.create_uplift_scorer(X_train, t_train, y_train)
            
            # Perform randomized search
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring=uplift_scorer,
                cv=cv,
                n_jobs=n_jobs,
                verbose=1,
                random_state=42,
                return_train_score=True
            )
            
            # Fit randomized search
            random_search.fit(X_train, y_train)
            
            # Store results
            self.best_models[model_name] = random_search.best_estimator_
            self.tuning_results[model_name] = {
                'best_params': random_search.best_params_,
                'best_score': random_search.best_score_,
                'cv_results': random_search.cv_results_
            }
            
            print(f"Best parameters: {random_search.best_params_}")
            print(f"Best uplift score: {random_search.best_score_:.4f}")
    
    def bayesian_optimization_tuning(self, X_train, y_train, t_train, cv=5, n_iter=50):
        """Perform Bayesian optimization hyperparameter tuning"""
        print("=== Bayesian Optimization Hyperparameter Tuning ===")
        
        try:
            from skopt import BayesSearchCV
            from skopt.space import Real, Integer, Categorical
            
            # Define search spaces
            xgb_search_space = {
                'n_estimators': Integer(50, 300),
                'max_depth': Integer(3, 10),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.7, 1.0),
                'colsample_bytree': Real(0.7, 1.0),
                'min_child_weight': Integer(1, 6),
                'reg_alpha': Real(0, 1.0),
                'reg_lambda': Real(0, 1.0)
            }
            
            rf_search_space = {
                'n_estimators': Integer(50, 400),
                'max_depth': Integer(5, 20),
                'min_samples_split': Integer(2, 15),
                'min_samples_leaf': Integer(1, 5),
                'max_features': Categorical(['sqrt', 'log2', None])
            }
            
            search_spaces = {
                'XGBoost': xgb_search_space,
                'Random Forest': rf_search_space
            }
            
            for model_name, search_space in search_spaces.items():
                print(f"\n--- Tuning {model_name} with Bayesian Optimization ---")
                
                # Create model
                if model_name == 'XGBoost':
                    base_model = xgb.XGBClassifier(random_state=42, verbosity=0)
                else:
                    base_model = RandomForestClassifier(random_state=42)
                
                # Create uplift scorer
                uplift_scorer = self.create_uplift_scorer(X_train, t_train, y_train)
                
                # Perform Bayesian search
                bayes_search = BayesSearchCV(
                    estimator=base_model,
                    search_spaces=search_space,
                    n_iter=n_iter,
                    scoring=uplift_scorer,
                    cv=cv,
                    verbose=1,
                    random_state=42,
                    return_train_score=True
                )
                
                # Fit Bayesian search
                bayes_search.fit(X_train, y_train)
                
                # Store results
                self.best_models[model_name] = bayes_search.best_estimator_
                self.tuning_results[model_name] = {
                    'best_params': bayes_search.best_params_,
                    'best_score': bayes_search.best_score_,
                    'cv_results': bayes_search.cv_results_
                }
                
                print(f"Best parameters: {bayes_search.best_params_}")
                print(f"Best uplift score: {bayes_search.best_score_:.4f}")
                
        except ImportError:
            print("scikit-optimize not installed. Skipping Bayesian optimization.")
            print("Install with: pip install scikit-optimize")
    
    def compare_tuning_methods(self, X_train, y_train, t_train, cv=5):
        """Compare different tuning methods"""
        print("=== Comparing Tuning Methods ===")
        
        # Reset results
        self.best_models = {}
        self.tuning_results = {}
        
        # Grid search (limited parameters for speed)
        print("\n1. Grid Search Tuning")
        self.grid_search_tuning(X_train, y_train, t_train, cv=cv, n_jobs=1)
        
        # Randomized search
        print("\n2. Randomized Search Tuning")
        self.randomized_search_tuning(X_train, y_train, t_train, cv=cv, n_iter=50, n_jobs=1)
        
        # Bayesian optimization
        print("\n3. Bayesian Optimization Tuning")
        self.bayesian_optimization_tuning(X_train, y_train, t_train, cv=cv, n_iter=30)
        
        # Compare results
        print("\n=== Tuning Method Comparison ===")
        comparison_results = []
        
        for method, results in self.tuning_results.items():
            for model_name, result in results.items():
                comparison_results.append({
                    'Method': method,
                    'Model': model_name,
                    'Best Score': result['best_score'],
                    'Best Params': str(result['best_params'])
                })
        
        comparison_df = pd.DataFrame(comparison_results)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def get_best_model(self, metric='best_score'):
        """Get the best model across all tuning methods"""
        best_score = -np.inf
        best_model = None
        best_method = None
        
        for method, results in self.tuning_results.items():
            for model_name, result in results.items():
                if result['best_score'] > best_score:
                    best_score = result['best_score']
                    best_model = self.best_models[model_name]
                    best_method = f"{method}_{model_name}"
        
        return best_model, best_method, best_score

def main():
    """Example usage of optimized hyperparameter tuning"""
    
    # Load data (example)
    try:
        df = pd.read_csv('uplift_model_data.csv')
        print(f"Loaded data: {df.shape}")
        
        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['treatment_ai_content', 'response', 'user_id', 'post_id']]
        X = df[feature_cols].fillna(0)
        y = df['response']
        t = df['treatment_ai_content']
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
            X, y, t, test_size=0.3, random_state=42, stratify=y
        )
        
        # Initialize tuner
        tuner = OptimizedHyperparameterTuning()
        
        # Compare tuning methods
        comparison_results = tuner.compare_tuning_methods(X_train, y_train, t_train, cv=3)
        
        # Get best model
        best_model, best_method, best_score = tuner.get_best_model()
        print(f"\nBest model: {best_method}")
        print(f"Best uplift score: {best_score:.4f}")
        
    except FileNotFoundError:
        print("uplift_model_data.csv not found. Please run data preprocessing first.")

if __name__ == "__main__":
    main()






