#!/usr/bin/env python3
"""
Uplift Model Training Script
Trains uplift models to estimate causal effects of treatments
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class UpliftModelTrainer:
    def __init__(self):
        self.df_uplift = None
        self.feature_columns = []
        self.models = {}
        self.results = {}
        self.treatment_col = None
        self.scaler = None
        self.pca = None
        
    def load_uplift_data(self, uplift_data_path='uplift_model_data.csv'):
        """Load uplift data and merge with features"""
        print("=== Loading Uplift Data ===")
        
        # Load uplift data
        self.df_uplift = pd.read_csv(uplift_data_path)
        print(f"Loaded {len(self.df_uplift)} uplift samples")
        
        # Check available columns and use the correct treatment column
        if 'treatment_ai_content' in self.df_uplift.columns:
            treatment_col = 'treatment_ai_content'
        elif 'treatment' in self.df_uplift.columns:
            treatment_col = 'treatment'
        else:
            print("Available columns:", self.df_uplift.columns.tolist())
            raise ValueError("No treatment column found in uplift data")
        
        print(f"Treatment distribution: {self.df_uplift[treatment_col].value_counts().to_dict()}")
        print(f"Response distribution: {self.df_uplift['response'].value_counts().to_dict()}")
        
        # Store the treatment column name for later use
        self.treatment_col = treatment_col
        
        # Sample data if too large (for faster training)
        if len(self.df_uplift) > 50000:
            print(f"Sampling 50,000 samples for faster training...")
            self.df_uplift = self.df_uplift.sample(n=50000, random_state=42)
            print(f"Sampled to {len(self.df_uplift)} samples")
        
        # Load user and post features (from data_preprocessing output)
        try:
            # Skip complex preprocessing for now - use basic features only
            print("Using basic features only to avoid preprocessing issues...")
            pass
            
        except Exception as e:
            print(f"Warning: Could not load full features, using basic features: {e}")
            # Use basic features from uplift data
            pass
        
        print(f"Final uplift dataset shape: {self.df_uplift.shape}")
        return True
    
    def create_features(self):
        """Create features for uplift modeling with improved engineering"""
        print("\n=== Creating Features ===")
        
        # Use real features from the uplift dataset
        print("Using real features from uplift dataset...")
        
        # Define potential feature columns (excluding treatment and response)
        self.potential_features = [
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
        
        # Only keep features that exist in the dataset and are numeric
        self.feature_columns = []
        for col in self.potential_features:
            if col in self.df_uplift.columns:
                # Check if column is numeric or can be converted to numeric
                if self.df_uplift[col].dtype in ['int64', 'float64']:
                    self.feature_columns.append(col)
                elif self.df_uplift[col].dtype.name == 'category':
                    # Convert categorical to numeric
                    self.df_uplift[col] = self.df_uplift[col].cat.codes
                    self.feature_columns.append(col)
                elif self.df_uplift[col].dtype == 'object':
                    # Skip object columns for now
                    print(f"Skipping object column: {col}")
                else:
                    # Try to convert to numeric
                    try:
                        self.df_uplift[col] = pd.to_numeric(self.df_uplift[col], errors='coerce')
                        self.feature_columns.append(col)
                    except:
                        print(f"Could not convert column to numeric: {col}")
        
        # Add engineered features
        self.add_engineered_features()
        
        # Add advanced features
        self.add_advanced_features()
        
        print(f"Available features: {len(self.feature_columns)}")
        print(f"Feature columns: {self.feature_columns[:10]}...")  # Show first 10
        
        # Fill any remaining NaN values
        print("Filling missing values...")
        for col in self.feature_columns:
            if col in self.df_uplift.columns:
                self.df_uplift[col] = self.df_uplift[col].fillna(0)
        
        print(f"Created {len(self.feature_columns)} real features")
        return True

    def add_engineered_features(self):
        """Add sophisticated engineered features"""
        print("Adding engineered features...")
        
        initial_feature_count = len(self.feature_columns)
        
        # User engagement features
        if 'user_reputation' in self.df_uplift.columns and 'user_post_count' in self.df_uplift.columns:
            self.df_uplift['user_engagement_rate'] = self.df_uplift['user_reputation'] / (self.df_uplift['user_post_count'] + 1)
            self.feature_columns.append('user_engagement_rate')
        
        # Content popularity features
        if 'Score' in self.df_uplift.columns and 'ViewCount' in self.df_uplift.columns:
            self.df_uplift['content_popularity'] = self.df_uplift['Score'] / (self.df_uplift['ViewCount'] + 1)
            self.feature_columns.append('content_popularity')
        
        # User expertise level
        if 'user_reputation' in self.df_uplift.columns:
            self.df_uplift['user_expertise_level'] = np.log1p(self.df_uplift['user_reputation'])
            self.feature_columns.append('user_expertise_level')
        
        # Content complexity score
        if 'post_length' in self.df_uplift.columns and 'title_length' in self.df_uplift.columns:
            self.df_uplift['content_complexity_score'] = self.df_uplift['post_length'] / (self.df_uplift['title_length'] + 1)
            self.feature_columns.append('content_complexity_score')
        
        # Interaction strength
        if 'user_ai_interest_score' in self.df_uplift.columns and 'user_post_tag_overlap' in self.df_uplift.columns:
            self.df_uplift['interaction_strength'] = self.df_uplift['user_ai_interest_score'] * self.df_uplift['user_post_tag_overlap']
            self.feature_columns.append('interaction_strength')
        
        # Time-based features
        if 'post_age_days' in self.df_uplift.columns:
            self.df_uplift['post_freshness'] = 1 / (self.df_uplift['post_age_days'] + 1)
            self.feature_columns.append('post_freshness')
        
        added_features = len(self.feature_columns) - initial_feature_count
        print(f"Added {added_features} engineered features")

    def add_advanced_features(self):
        """Add advanced statistical and domain-specific features"""
        print("Adding advanced features...")
        
        initial_feature_count = len(self.feature_columns)
        
        # Statistical features
        if 'user_reputation' in self.df_uplift.columns:
            # User reputation percentile
            self.df_uplift['user_reputation_percentile'] = self.df_uplift['user_reputation'].rank(pct=True)
            self.feature_columns.append('user_reputation_percentile')
        
        if 'Score' in self.df_uplift.columns:
            # Post score percentile
            self.df_uplift['post_score_percentile'] = self.df_uplift['Score'].rank(pct=True)
            self.feature_columns.append('post_score_percentile')
        
        # Ratio features
        if 'AnswerCount' in self.df_uplift.columns and 'CommentCount' in self.df_uplift.columns:
            self.df_uplift['answer_comment_ratio'] = self.df_uplift['AnswerCount'] / (self.df_uplift['CommentCount'] + 1)
            self.feature_columns.append('answer_comment_ratio')
        
        if 'upvotes' in self.df_uplift.columns and 'total_votes' in self.df_uplift.columns:
            self.df_uplift['upvote_ratio'] = self.df_uplift['upvotes'] / (self.df_uplift['total_votes'] + 1)
            self.feature_columns.append('upvote_ratio')
        
        # Polynomial features for important variables
        if 'user_ai_interest_score' in self.df_uplift.columns:
            self.df_uplift['user_ai_interest_squared'] = self.df_uplift['user_ai_interest_score'] ** 2
            self.feature_columns.append('user_ai_interest_squared')
        
        if 'title_length' in self.df_uplift.columns:
            self.df_uplift['title_length_squared'] = self.df_uplift['title_length'] ** 2
            self.feature_columns.append('title_length_squared')
        
        # Interaction features
        if 'user_ai_interest_score' in self.df_uplift.columns and 'user_reputation' in self.df_uplift.columns:
            self.df_uplift['ai_interest_reputation'] = self.df_uplift['user_ai_interest_score'] * np.log1p(self.df_uplift['user_reputation'])
            self.feature_columns.append('ai_interest_reputation')
        
        if 'title_length' in self.df_uplift.columns and 'num_tags' in self.df_uplift.columns:
            self.df_uplift['title_tags_ratio'] = self.df_uplift['title_length'] / (self.df_uplift['num_tags'] + 1)
            self.feature_columns.append('title_tags_ratio')
        
        added_features = len(self.feature_columns) - initial_feature_count
        print(f"Added {added_features} advanced features")

    def preprocess_data(self):
        """Advanced data preprocessing without PCA"""
        print("\n=== Data Preprocessing ===")
        
        # Remove outliers using IQR method
        print("Removing outliers...")
        for col in self.feature_columns:
            if col in self.df_uplift.columns:
                Q1 = self.df_uplift[col].quantile(0.25)
                Q3 = self.df_uplift[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing
                self.df_uplift[col] = self.df_uplift[col].clip(lower_bound, upper_bound)
        
        # Feature scaling
        print("Scaling features...")
        self.scaler = RobustScaler()
        self.df_uplift[self.feature_columns] = self.scaler.fit_transform(self.df_uplift[self.feature_columns])
        
        # Feature selection based on correlation with response
        print("Performing feature selection...")
        response_correlations = []
        for col in self.feature_columns:
            if col in self.df_uplift.columns:
                corr = abs(self.df_uplift[col].corr(self.df_uplift['response']))
                response_correlations.append((col, corr))
        
        # Keep top features
        response_correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = [col for col, corr in response_correlations[:15]]  # Keep top 15 features
        
        print(f"Selected {len(top_features)} features based on response correlation:")
        for col, corr in response_correlations[:10]:
            print(f"  {col}: {corr:.4f}")
        
        self.feature_columns = top_features
        
        return True
    
    def train_uplift_models(self):
        """Train uplift models using multiple algorithms with better evaluation"""
        print("\n=== Training Uplift Models ===")
        
        # Prepare features and target
        X = self.df_uplift[self.feature_columns].fillna(0)
        treatment = self.df_uplift[self.treatment_col]
        response = self.df_uplift['response']
        
        # Clean response data - remove NaN values
        print("Cleaning response data...")
        valid_mask = ~response.isna()
        X = X[valid_mask]
        treatment = treatment[valid_mask]
        response = response[valid_mask]
        
        print(f"After cleaning: {len(X)} samples")
        print(f"Response distribution: {response.value_counts().to_dict()}")
        
        # Split data with stratification
        X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
            X, treatment, response, test_size=0.2, random_state=42, stratify=treatment
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Treatment ratio: {treatment.mean():.3f}")
        print(f"Response ratio: {response.mean():.3f}")
        
        # Calculate actual uplift
        actual_uplift = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
        
        # Approach 1: Simple XGBoost Two-Model (no hyperparameter tuning)
        print("\n--- Simple XGBoost Two-Model Approach ---")
        self.train_simple_xgboost_two_model(X_train, X_test, t_train, t_test, y_train, y_test, actual_uplift)
        
        # Approach 2: Random Forest Two-Model
        print("\n--- Random Forest Two-Model Approach ---")
        self.train_random_forest_two_model(X_train, X_test, t_train, t_test, y_train, y_test, actual_uplift)
        
        # Approach 3: Linear Model (for interpretability)
        print("\n--- Linear Model Approach ---")
        self.train_linear_model(X_train, X_test, t_train, t_test, y_train, y_test, actual_uplift)
        
        # Approach 4: Single Model with Treatment Interaction
        print("\n--- Single Model with Treatment Interaction ---")
        self.train_single_model(X_train, X_test, t_train, t_test, y_train, y_test, actual_uplift)
        
        return True

    def train_simple_xgboost_two_model(self, X_train, X_test, t_train, t_test, y_train, y_test, actual_uplift):
        """Train simple XGBoost two-model approach without hyperparameter tuning"""
        treatment_mask_train = t_train == 1
        control_mask_train = t_train == 0
        
        if treatment_mask_train.sum() > 0:
            X_treatment = X_train[treatment_mask_train]
            y_treatment = y_train[treatment_mask_train]
            X_control = X_train[control_mask_train]
            y_control = y_train[control_mask_train]
            
            # Simple XGBoost models
            treatment_model = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
            control_model = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
            
            treatment_model.fit(X_treatment, y_treatment)
            control_model.fit(X_control, y_control)
            
            # Predict uplift
            y_pred_treatment = treatment_model.predict(X_test)
            y_pred_control = control_model.predict(X_test)
            uplift_predictions = y_pred_treatment - y_pred_control
            
            # Calculate metrics
            predicted_uplift = uplift_predictions.mean()
            uplift_error = abs(actual_uplift - predicted_uplift)
            uplift_accuracy = max(0, 1 - uplift_error / abs(actual_uplift)) if actual_uplift != 0 else 0
            
            # Calculate correlation
            treatment_mask_test = t_test == 1
            if treatment_mask_test.sum() > 0:
                actual_uplift_by_sample = y_test[treatment_mask_test].values - y_test[~treatment_mask_test].values[:treatment_mask_test.sum()]
                predicted_uplift_by_sample = uplift_predictions[treatment_mask_test]
                if len(actual_uplift_by_sample) > 1:
                    uplift_correlation = np.corrcoef(actual_uplift_by_sample, predicted_uplift_by_sample)[0, 1]
                else:
                    uplift_correlation = 0
            else:
                uplift_correlation = 0
            
            self.models['simple_xgboost_two_model'] = {
                'treatment_model': treatment_model,
                'control_model': control_model
            }
            
            self.results['simple_xgboost_two_model'] = {
                'actual_uplift': actual_uplift,
                'predicted_uplift': predicted_uplift,
                'uplift_error': uplift_error,
                'uplift_accuracy': uplift_accuracy,
                'uplift_correlation': uplift_correlation
            }
            
            print(f"  Actual uplift: {actual_uplift:.4f}")
            print(f"  Predicted uplift: {predicted_uplift:.4f}")
            print(f"  Uplift error: {uplift_error:.4f}")
            print(f"  Uplift accuracy: {uplift_accuracy:.2%}")
            print(f"  Uplift correlation: {uplift_correlation:.4f}")

    def train_random_forest_two_model(self, X_train, X_test, t_train, t_test, y_train, y_test, actual_uplift):
        """Train Random Forest two-model approach"""
        treatment_mask_train = t_train == 1
        control_mask_train = t_train == 0
        
        if treatment_mask_train.sum() > 0:
            X_treatment = X_train[treatment_mask_train]
            y_treatment = y_train[treatment_mask_train]
            X_control = X_train[control_mask_train]
            y_control = y_train[control_mask_train]
            
            # Train Random Forest models
            treatment_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            control_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            treatment_model.fit(X_treatment, y_treatment)
            control_model.fit(X_control, y_control)
            
            # Predict uplift
            y_pred_treatment = treatment_model.predict(X_test)
            y_pred_control = control_model.predict(X_test)
            uplift_predictions = y_pred_treatment - y_pred_control
            
            # Calculate metrics
            predicted_uplift = uplift_predictions.mean()
            uplift_error = abs(actual_uplift - predicted_uplift)
            uplift_accuracy = max(0, 1 - uplift_error / abs(actual_uplift)) if actual_uplift != 0 else 0
            
            self.models['random_forest_two_model'] = {
                'treatment_model': treatment_model,
                'control_model': control_model
            }
            
            self.results['random_forest_two_model'] = {
                'actual_uplift': actual_uplift,
                'predicted_uplift': predicted_uplift,
                'uplift_error': uplift_error,
                'uplift_accuracy': uplift_accuracy
            }
            
            print(f"  Actual uplift: {actual_uplift:.4f}")
            print(f"  Predicted uplift: {predicted_uplift:.4f}")
            print(f"  Uplift error: {uplift_error:.4f}")
            print(f"  Uplift accuracy: {uplift_accuracy:.2%}")

    def train_linear_model(self, X_train, X_test, t_train, t_test, y_train, y_test, actual_uplift):
        """Train linear model for interpretability"""
        from sklearn.linear_model import LinearRegression
        
        treatment_mask_train = t_train == 1
        control_mask_train = t_train == 0
        
        if treatment_mask_train.sum() > 0:
            X_treatment = X_train[treatment_mask_train]
            y_treatment = y_train[treatment_mask_train]
            X_control = X_train[control_mask_train]
            y_control = y_train[control_mask_train]
            
            # Linear models
            treatment_model = LinearRegression()
            control_model = LinearRegression()
            
            treatment_model.fit(X_treatment, y_treatment)
            control_model.fit(X_control, y_control)
            
            # Predict uplift
            y_pred_treatment = treatment_model.predict(X_test)
            y_pred_control = control_model.predict(X_test)
            uplift_predictions = y_pred_treatment - y_pred_control
            
            # Calculate metrics
            predicted_uplift = uplift_predictions.mean()
            uplift_error = abs(actual_uplift - predicted_uplift)
            uplift_accuracy = max(0, 1 - uplift_error / abs(actual_uplift)) if actual_uplift != 0 else 0
            
            self.models['linear_two_model'] = {
                'treatment_model': treatment_model,
                'control_model': control_model
            }
            
            self.results['linear_two_model'] = {
                'actual_uplift': actual_uplift,
                'predicted_uplift': predicted_uplift,
                'uplift_error': uplift_error,
                'uplift_accuracy': uplift_accuracy
            }
            
            print(f"  Actual uplift: {actual_uplift:.4f}")
            print(f"  Predicted uplift: {predicted_uplift:.4f}")
            print(f"  Uplift error: {uplift_error:.4f}")
            print(f"  Uplift accuracy: {uplift_accuracy:.2%}")
            
            # Print coefficients for interpretability
            print("  Treatment model coefficients:")
            for i, feature in enumerate(self.feature_columns):
                if i < 5:  # Show top 5
                    print(f"    {feature}: {treatment_model.coef_[i]:.4f}")

    def train_single_model(self, X_train, X_test, t_train, t_test, y_train, y_test, actual_uplift):
        """Train single model with treatment interaction"""
        # Add treatment as a feature
        X_with_treatment = X_train.copy()
        X_with_treatment['treatment'] = t_train
        
        # Train XGBoost model with hyperparameter tuning
        single_model = xgb.XGBRegressor(random_state=42)
        params = {
            'n_estimators': [50, 100],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1]
        }
        
        grid = GridSearchCV(single_model, params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_with_treatment, y_train)
        
        # Evaluate
        X_test_with_treatment = X_test.copy()
        X_test_with_treatment['treatment'] = t_test
        
        y_pred_single = grid.predict(X_test_with_treatment)
        mse = mean_squared_error(y_test, y_pred_single)
        r2 = r2_score(y_test, y_pred_single)
        
        # Calculate uplift from single model
        X_test_treatment = X_test.copy()
        X_test_treatment['treatment'] = 1
        X_test_control = X_test.copy()
        X_test_control['treatment'] = 0
        
        y_pred_treatment_single = grid.predict(X_test_treatment)
        y_pred_control_single = grid.predict(X_test_control)
        uplift_single = (y_pred_treatment_single - y_pred_control_single).mean()
        
        self.models['single_model'] = grid
        
        self.results['single_model'] = {
            'mse': mse,
            'r2': r2,
            'predicted_uplift': uplift_single,
            'uplift_error': abs(actual_uplift - uplift_single)
        }
        
        print(f"  MSE: {mse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Predicted uplift: {uplift_single:.4f}")
        print(f"  Uplift error: {abs(actual_uplift - uplift_single):.4f}")

    def train_ensemble_model(self, X_train, X_test, t_train, t_test, y_train, y_test, actual_uplift):
        """Train ensemble of all models"""
        # Get predictions from all two-model approaches
        ensemble_predictions = []
        
        for model_name in ['xgboost_two_model', 'random_forest_two_model', 'lightgbm_two_model']:
            if model_name in self.models:
                treatment_mask_test = t_test == 1
                control_mask_test = t_test == 0
                
                treatment_model = self.models[model_name]['treatment_model']
                control_model = self.models[model_name]['control_model']
                
                y_pred_treatment = treatment_model.predict(X_test)
                y_pred_control = control_model.predict(X_test)
                uplift_pred = y_pred_treatment - y_pred_control
                ensemble_predictions.append(uplift_pred)
        
        if ensemble_predictions:
            # Average predictions
            ensemble_uplift = np.mean(ensemble_predictions, axis=0)
            ensemble_uplift_mean = ensemble_uplift.mean()
            ensemble_error = abs(actual_uplift - ensemble_uplift_mean)
            ensemble_accuracy = max(0, 1 - ensemble_error / abs(actual_uplift)) if actual_uplift != 0 else 0
            
            self.results['ensemble'] = {
                'predicted_uplift': ensemble_uplift_mean,
                'uplift_error': ensemble_error,
                'uplift_accuracy': ensemble_accuracy
            }
            
            print(f"  Ensemble predicted uplift: {ensemble_uplift_mean:.4f}")
            print(f"  Ensemble uplift error: {ensemble_error:.4f}")
            print(f"  Ensemble accuracy: {ensemble_accuracy:.2%}")

    def analyze_feature_importance(self):
        """Analyze feature importance for the best model"""
        print("\n=== Feature Importance Analysis ===")
        
        # Analyze feature importance for each model type
        for model_name in ['simple_xgboost_two_model', 'random_forest_two_model', 'linear_two_model']:
            if model_name in self.models:
                print(f"\n{model_name.upper()} Feature Importance:")
                treatment_model = self.models[model_name]['treatment_model']
                
                if hasattr(treatment_model, 'feature_importances_'):
                    feature_importance = treatment_model.feature_importances_
                elif hasattr(treatment_model, 'coef_'):
                    feature_importance = abs(treatment_model.coef_)
                else:
                    continue
                
                # Create feature importance DataFrame
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)
                
                print("Top 10 most important features:")
                for idx, row in importance_df.head(10).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return True
    
    def print_results(self):
        """Print uplift model comparison results with improved metrics"""
        print("\n=== Uplift Model Results ===")
        
        for name, results in self.results.items():
            print(f"\n{name.upper()}:")
            for metric, value in results.items():
                if isinstance(value, float):
                    if 'accuracy' in metric or 'correlation' in metric:
                        print(f"  {metric}: {value:.2%}")
                    else:
                print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        
        # Find best model
        best_model = None
        best_error = float('inf')
        
        for name, results in self.results.items():
            if 'uplift_error' in results:
                if results['uplift_error'] < best_error:
                    best_error = results['uplift_error']
                    best_model = name
        
        if best_model:
            print(f"\n🏆 Best Model: {best_model.upper()}")
            print(f"   Uplift Error: {self.results[best_model]['uplift_error']:.4f}")
            if 'uplift_accuracy' in self.results[best_model]:
                print(f"   Uplift Accuracy: {self.results[best_model]['uplift_accuracy']:.2%}")
        
        return True
    
    def save_models(self, output_dir='models'):
        """Save trained uplift models"""
        print(f"\n=== Saving Uplift Models ===")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = f"{output_dir}/uplift_{name}.pkl"
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} to {model_path}")
        
        # Save feature columns
        feature_path = f"{output_dir}/uplift_feature_columns.pkl"
        with open(feature_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        print(f"Saved feature columns to {feature_path}")
        
        return True
    
    def run_full_pipeline(self):
        """Run complete uplift model training pipeline"""
        print("=== Uplift Model Training Pipeline ===")
        
        # Load data
        self.load_uplift_data()
        
        # Create features
        self.create_features()
        
        # Preprocess data
        self.preprocess_data()
        
        # Train models
        self.train_uplift_models()
        
        # Analyze feature importance
        self.analyze_feature_importance()
        
        # Print results
        self.print_results()
        
        # Save models
        self.save_models()
        
        print("\n=== Uplift Training Complete ===")
        return self.models, self.results

def main():
    """Main function"""
    trainer = UpliftModelTrainer()
    models, results = trainer.run_full_pipeline()
    return models, results

if __name__ == "__main__":
    main() 