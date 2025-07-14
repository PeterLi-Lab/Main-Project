import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class ImprovedUpliftAnalysis:
    def __init__(self):
        self.df_uplift = None
        self.feature_columns = []
        self.models = {}
        self.results = {}
        self.treatment_col = None
        self.scaler = None
        
    def load_and_explore_data(self, uplift_data_path='uplift_model_data.csv'):
        """Load data and perform comprehensive data quality checks"""
        print("=== Data Loading and Quality Checks ===")
        
        # Load data
        self.df_uplift = pd.read_csv(uplift_data_path)
        print(f"Loaded {len(self.df_uplift)} samples")
        
        # Check available columns
        print(f"Available columns: {self.df_uplift.columns.tolist()}")
        
        # Find treatment column
        if 'treatment_ai_content' in self.df_uplift.columns:
            self.treatment_col = 'treatment_ai_content'
        elif 'treatment' in self.df_uplift.columns:
            self.treatment_col = 'treatment'
        else:
            print("Available columns:", self.df_uplift.columns.tolist())
            raise ValueError("No treatment column found")
        
        # 1. Data Quality Checks
        self.check_data_quality()
        
        # 2. Data Distribution Analysis
        self.analyze_data_distributions()
        
        # 3. Treatment Effect Analysis
        self.analyze_treatment_effects()
        
        return True
    
    def check_data_quality(self):
        """Comprehensive data quality checks"""
        print("\n--- Data Quality Checks ---")
        
        # Check for missing values
        missing_data = self.df_uplift.isnull().sum()
        print("Missing values per column:")
        for col, missing in missing_data[missing_data > 0].items():
            print(f"  {col}: {missing} ({missing/len(self.df_uplift)*100:.2f}%)")
        
        # Check treatment distribution
        treatment_dist = self.df_uplift[self.treatment_col].value_counts()
        print(f"\nTreatment distribution:")
        print(f"  Control (0): {treatment_dist[0]} ({treatment_dist[0]/len(self.df_uplift)*100:.2f}%)")
        print(f"  Treatment (1): {treatment_dist[1]} ({treatment_dist[1]/len(self.df_uplift)*100:.2f}%)")
        
        # Check response distribution
        response_dist = self.df_uplift['response'].value_counts()
        print(f"\nResponse distribution:")
        print(f"  No click (0): {response_dist[0]} ({response_dist[0]/len(self.df_uplift)*100:.2f}%)")
        print(f"  Click (1): {response_dist[1]} ({response_dist[1]/len(self.df_uplift)*100:.2f}%)")
        
        # Check for data leakage
        print(f"\nData leakage check:")
        print(f"  Treatment-response correlation: {self.df_uplift[self.treatment_col].corr(self.df_uplift['response']):.4f}")
        
        # Check for extreme values
        numeric_cols = self.df_uplift.select_dtypes(include=[np.number]).columns
        print(f"\nExtreme values check (z-score > 3):")
        for col in numeric_cols[:5]:  # Check first 5 numeric columns
            z_scores = np.abs((self.df_uplift[col] - self.df_uplift[col].mean()) / self.df_uplift[col].std())
            extreme_count = (z_scores > 3).sum()
            print(f"  {col}: {extreme_count} extreme values")
    
    def analyze_data_distributions(self):
        """Analyze data distributions and relationships"""
        print("\n--- Data Distribution Analysis ---")
        
        # Analyze feature distributions
        numeric_cols = self.df_uplift.select_dtypes(include=[np.number]).columns
        print(f"Analyzing {len(numeric_cols)} numeric features...")
        
        # Correlation with response
        response_correlations = []
        for col in numeric_cols:
            if col != 'response' and col != self.treatment_col:
                corr = abs(self.df_uplift[col].corr(self.df_uplift['response']))
                response_correlations.append((col, corr))
        
        response_correlations.sort(key=lambda x: x[1], reverse=True)
        print("\nTop 10 features by response correlation:")
        for col, corr in response_correlations[:10]:
            print(f"  {col}: {corr:.4f}")
        
        # Treatment-response relationship
        print(f"\nTreatment-Response Analysis:")
        treatment_response = self.df_uplift.groupby(self.treatment_col)['response'].agg(['mean', 'count'])
        print(treatment_response)
        
        # Calculate actual uplift
        control_rate = treatment_response.loc[0, 'mean']
        treatment_rate = treatment_response.loc[1, 'mean']
        actual_uplift = treatment_rate - control_rate
        print(f"\nActual uplift: {actual_uplift:.4f}")
        print(f"  Control response rate: {control_rate:.4f}")
        print(f"  Treatment response rate: {treatment_rate:.4f}")
    
    def analyze_treatment_effects(self):
        """Detailed treatment effect analysis"""
        print("\n--- Treatment Effect Analysis ---")
        
        # Stratified analysis by key features
        key_features = ['user_reputation', 'user_post_count', 'Score', 'ViewCount']
        
        for feature in key_features:
            if feature in self.df_uplift.columns:
                print(f"\nTreatment effect by {feature} quartiles:")
                
                try:
                    # Create quartiles with duplicate handling
                    self.df_uplift[f'{feature}_quartile'] = pd.qcut(
                        self.df_uplift[feature], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop'
                    )
                    
                    # Analyze by quartile
                    for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
                        subset = self.df_uplift[self.df_uplift[f'{feature}_quartile'] == quartile]
                        if len(subset) > 0:
                            control_rate = subset[subset[self.treatment_col] == 0]['response'].mean()
                            treatment_rate = subset[subset[self.treatment_col] == 1]['response'].mean()
                            uplift = treatment_rate - control_rate
                            print(f"  {quartile}: Uplift = {uplift:.4f} (Control: {control_rate:.4f}, Treatment: {treatment_rate:.4f})")
                
                except ValueError as e:
                    print(f"  Skipping {feature} due to insufficient unique values: {e}")
                    continue
    
    def create_improved_features(self):
        """Create improved features without PCA"""
        print("\n=== Improved Feature Engineering ===")
        
        # Define potential features
        potential_features = [
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
        
        # Select features that exist and are numeric
        self.feature_columns = []
        for col in potential_features:
            if col in self.df_uplift.columns:
                if self.df_uplift[col].dtype in ['int64', 'float64']:
                    self.feature_columns.append(col)
                elif self.df_uplift[col].dtype.name == 'category':
                    self.df_uplift[col] = self.df_uplift[col].cat.codes
                    self.feature_columns.append(col)
        
        # Add domain-specific features
        self.add_domain_features()
        
        # Feature selection based on correlation and variance
        self.select_best_features()
        
        print(f"Selected {len(self.feature_columns)} features")
        return True
    
    def add_domain_features(self):
        """Add domain-specific features"""
        print("Adding domain-specific features...")
        
        # User engagement features
        if 'user_reputation' in self.df_uplift.columns and 'user_post_count' in self.df_uplift.columns:
            self.df_uplift['user_engagement_rate'] = self.df_uplift['user_reputation'] / (self.df_uplift['user_post_count'] + 1)
            self.feature_columns.append('user_engagement_rate')
        
        # Content quality features
        if 'Score' in self.df_uplift.columns and 'ViewCount' in self.df_uplift.columns:
            self.df_uplift['content_quality_score'] = self.df_uplift['Score'] / (self.df_uplift['ViewCount'] + 1)
            self.feature_columns.append('content_quality_score')
        
        # User expertise level
        if 'user_reputation' in self.df_uplift.columns:
            self.df_uplift['user_expertise_level'] = np.log1p(self.df_uplift['user_reputation'])
            self.feature_columns.append('user_expertise_level')
        
        # Content complexity
        if 'post_length' in self.df_uplift.columns and 'title_length' in self.df_uplift.columns:
            self.df_uplift['content_complexity'] = self.df_uplift['post_length'] / (self.df_uplift['title_length'] + 1)
            self.feature_columns.append('content_complexity')
        
        # Interaction strength
        if 'user_ai_interest_score' in self.df_uplift.columns and 'user_post_tag_overlap' in self.df_uplift.columns:
            self.df_uplift['interaction_strength'] = self.df_uplift['user_ai_interest_score'] * self.df_uplift['user_post_tag_overlap']
            self.feature_columns.append('interaction_strength')
        
        # Time-based features
        if 'post_age_days' in self.df_uplift.columns:
            self.df_uplift['post_freshness'] = 1 / (self.df_uplift['post_age_days'] + 1)
            self.feature_columns.append('post_freshness')
        
        # Statistical features
        if 'user_reputation' in self.df_uplift.columns:
            self.df_uplift['user_reputation_percentile'] = self.df_uplift['user_reputation'].rank(pct=True)
            self.feature_columns.append('user_reputation_percentile')
        
        if 'Score' in self.df_uplift.columns:
            self.df_uplift['post_score_percentile'] = self.df_uplift['Score'].rank(pct=True)
            self.feature_columns.append('post_score_percentile')
    
    def select_best_features(self):
        """Select best features based on correlation and variance"""
        print("Selecting best features...")
        
        # Calculate feature importance scores
        feature_scores = []
        for col in self.feature_columns:
            if col in self.df_uplift.columns:
                # Response correlation
                response_corr = abs(self.df_uplift[col].corr(self.df_uplift['response']))
                
                # Treatment correlation
                treatment_corr = abs(self.df_uplift[col].corr(self.df_uplift[self.treatment_col]))
                
                # Variance
                variance = self.df_uplift[col].var()
                
                # Combined score
                score = response_corr * 0.5 + treatment_corr * 0.3 + (variance / 100) * 0.2
                feature_scores.append((col, score, response_corr, treatment_corr, variance))
        
        # Sort by score and select top features
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top 15 features
        top_features = [col for col, score, _, _, _ in feature_scores[:15]]
        
        print("Top 10 features by importance score:")
        for col, score, resp_corr, treat_corr, var in feature_scores[:10]:
            print(f"  {col}: Score={score:.4f}, Resp_Corr={resp_corr:.4f}, Treat_Corr={treat_corr:.4f}, Var={var:.2f}")
        
        self.feature_columns = top_features
    
    def train_improved_models(self):
        """Train improved models with cross-validation"""
        print("\n=== Training Improved Models ===")
        
        # Prepare data
        X = self.df_uplift[self.feature_columns].fillna(0)
        treatment = self.df_uplift[self.treatment_col]
        response = self.df_uplift['response']
        
        # Remove NaN values
        valid_mask = ~response.isna()
        X = X[valid_mask]
        treatment = treatment[valid_mask]
        response = response[valid_mask]
        
        # Split data
        X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
            X, treatment, response, test_size=0.2, random_state=42, stratify=treatment
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Calculate actual uplift
        actual_uplift = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
        
        # 1. Linear Regression (Simple and Interpretable)
        print("\n--- Linear Regression Two-Model ---")
        self.train_linear_models(X_train, X_test, t_train, t_test, y_train, y_test, actual_uplift)
        
        # 2. Random Forest with Cross-Validation
        print("\n--- Random Forest with Cross-Validation ---")
        self.train_random_forest_cv(X_train, X_test, t_train, t_test, y_train, y_test, actual_uplift)
        
        # 3. XGBoost with Cross-Validation
        print("\n--- XGBoost with Cross-Validation ---")
        self.train_xgboost_cv(X_train, X_test, t_train, t_test, y_train, y_test, actual_uplift)
        
        return True
    
    def train_linear_models(self, X_train, X_test, t_train, t_test, y_train, y_test, actual_uplift):
        """Train linear regression models"""
        treatment_mask_train = t_train == 1
        control_mask_train = t_train == 0
        
        if treatment_mask_train.sum() > 0:
            X_treatment = X_train[treatment_mask_train]
            y_treatment = y_train[treatment_mask_train]
            X_control = X_train[control_mask_train]
            y_control = y_train[control_mask_train]
            
            # Train models
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
            
            # Cross-validation scores
            cv_scores_treatment = cross_val_score(treatment_model, X_treatment, y_treatment, cv=5, scoring='r2')
            cv_scores_control = cross_val_score(control_model, X_control, y_control, cv=5, scoring='r2')
            
            self.models['linear_two_model'] = {
                'treatment_model': treatment_model,
                'control_model': control_model
            }
            
            self.results['linear_two_model'] = {
                'actual_uplift': actual_uplift,
                'predicted_uplift': predicted_uplift,
                'uplift_error': uplift_error,
                'uplift_accuracy': uplift_accuracy,
                'cv_score_treatment': cv_scores_treatment.mean(),
                'cv_score_control': cv_scores_control.mean()
            }
            
            print(f"  Actual uplift: {actual_uplift:.4f}")
            print(f"  Predicted uplift: {predicted_uplift:.4f}")
            print(f"  Uplift error: {uplift_error:.4f}")
            print(f"  Uplift accuracy: {uplift_accuracy:.2%}")
            print(f"  CV R² (Treatment): {cv_scores_treatment.mean():.4f} ± {cv_scores_treatment.std():.4f}")
            print(f"  CV R² (Control): {cv_scores_control.mean():.4f} ± {cv_scores_control.std():.4f}")
            
            # Print coefficients for interpretability
            print("  Top 5 treatment model coefficients:")
            coef_df = pd.DataFrame({
                'feature': self.feature_columns,
                'coefficient': treatment_model.coef_
            }).sort_values('coefficient', key=abs, ascending=False)
            
            for idx, row in coef_df.head(5).iterrows():
                print(f"    {row['feature']}: {row['coefficient']:.4f}")
    
    def train_random_forest_cv(self, X_train, X_test, t_train, t_test, y_train, y_test, actual_uplift):
        """Train Random Forest with cross-validation"""
        treatment_mask_train = t_train == 1
        control_mask_train = t_train == 0
        
        if treatment_mask_train.sum() > 0:
            X_treatment = X_train[treatment_mask_train]
            y_treatment = y_train[treatment_mask_train]
            X_control = X_train[control_mask_train]
            y_control = y_train[control_mask_train]
            
            # Train models
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
            
            # Cross-validation scores
            cv_scores_treatment = cross_val_score(treatment_model, X_treatment, y_treatment, cv=5, scoring='r2')
            cv_scores_control = cross_val_score(control_model, X_control, y_control, cv=5, scoring='r2')
            
            self.models['random_forest_cv'] = {
                'treatment_model': treatment_model,
                'control_model': control_model
            }
            
            self.results['random_forest_cv'] = {
                'actual_uplift': actual_uplift,
                'predicted_uplift': predicted_uplift,
                'uplift_error': uplift_error,
                'uplift_accuracy': uplift_accuracy,
                'cv_score_treatment': cv_scores_treatment.mean(),
                'cv_score_control': cv_scores_control.mean()
            }
            
            print(f"  Actual uplift: {actual_uplift:.4f}")
            print(f"  Predicted uplift: {predicted_uplift:.4f}")
            print(f"  Uplift error: {uplift_error:.4f}")
            print(f"  Uplift accuracy: {uplift_accuracy:.2%}")
            print(f"  CV R² (Treatment): {cv_scores_treatment.mean():.4f} ± {cv_scores_treatment.std():.4f}")
            print(f"  CV R² (Control): {cv_scores_control.mean():.4f} ± {cv_scores_control.std():.4f}")
    
    def train_xgboost_cv(self, X_train, X_test, t_train, t_test, y_train, y_test, actual_uplift):
        """Train XGBoost with cross-validation"""
        treatment_mask_train = t_train == 1
        control_mask_train = t_train == 0
        
        if treatment_mask_train.sum() > 0:
            X_treatment = X_train[treatment_mask_train]
            y_treatment = y_train[treatment_mask_train]
            X_control = X_train[control_mask_train]
            y_control = y_train[control_mask_train]
            
            # Ensure all features are numeric
            X_treatment = X_treatment.astype(float)
            X_control = X_control.astype(float)
            X_test = X_test.astype(float)
            
            # Train models
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
            
            try:
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
                
                # Cross-validation scores
                cv_scores_treatment = cross_val_score(treatment_model, X_treatment, y_treatment, cv=5, scoring='r2')
                cv_scores_control = cross_val_score(control_model, X_control, y_control, cv=5, scoring='r2')
                
                self.models['xgboost_cv'] = {
                    'treatment_model': treatment_model,
                    'control_model': control_model
                }
                
                self.results['xgboost_cv'] = {
                    'actual_uplift': actual_uplift,
                    'predicted_uplift': predicted_uplift,
                    'uplift_error': uplift_error,
                    'uplift_accuracy': uplift_accuracy,
                    'cv_score_treatment': cv_scores_treatment.mean(),
                    'cv_score_control': cv_scores_control.mean()
                }
                
                print(f"  Actual uplift: {actual_uplift:.4f}")
                print(f"  Predicted uplift: {predicted_uplift:.4f}")
                print(f"  Uplift error: {uplift_error:.4f}")
                print(f"  Uplift accuracy: {uplift_accuracy:.2%}")
                print(f"  CV R² (Treatment): {cv_scores_treatment.mean():.4f} ± {cv_scores_treatment.std():.4f}")
                print(f"  CV R² (Control): {cv_scores_control.mean():.4f} ± {cv_scores_control.std():.4f}")
                
            except Exception as e:
                print(f"  XGBoost training failed: {e}")
                print("  Skipping XGBoost model")
    
    def analyze_model_interpretability(self):
        """Analyze model interpretability and feature importance"""
        print("\n=== Model Interpretability Analysis ===")
        
        for model_name, results in self.results.items():
            if model_name in self.models:
                print(f"\n{model_name.upper()} Analysis:")
                
                if model_name == 'linear_two_model':
                    # Linear model coefficients
                    treatment_model = self.models[model_name]['treatment_model']
                    coef_df = pd.DataFrame({
                        'feature': self.feature_columns,
                        'coefficient': treatment_model.coef_
                    }).sort_values('coefficient', key=abs, ascending=False)
                    
                    print("  Top 10 treatment model coefficients:")
                    for idx, row in coef_df.head(10).iterrows():
                        print(f"    {row['feature']}: {row['coefficient']:.4f}")
                
                elif model_name in ['random_forest_cv', 'xgboost_cv']:
                    # Tree-based model feature importance
                    treatment_model = self.models[model_name]['treatment_model']
                    if hasattr(treatment_model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'feature': self.feature_columns,
                            'importance': treatment_model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        print("  Top 10 feature importances:")
                        for idx, row in importance_df.head(10).iterrows():
                            print(f"    {row['feature']}: {row['importance']:.4f}")
    
    def evaluate_prediction_reasonableness(self):
        """Evaluate if predictions are reasonable"""
        print("\n=== Prediction Reasonableness Check ===")
        
        for model_name, results in self.results.items():
            if 'predicted_uplift' in results:
                actual = results['actual_uplift']
                predicted = results['predicted_uplift']
                
                print(f"\n{model_name.upper()}:")
                print(f"  Actual uplift: {actual:.4f}")
                print(f"  Predicted uplift: {predicted:.4f}")
                
                # Check if prediction direction is correct
                direction_correct = (actual > 0 and predicted > 0) or (actual < 0 and predicted < 0)
                print(f"  Direction correct: {direction_correct}")
                
                # Check if prediction magnitude is reasonable
                magnitude_ratio = abs(predicted) / abs(actual) if actual != 0 else 0
                print(f"  Magnitude ratio: {magnitude_ratio:.2f}")
                
                if magnitude_ratio > 0.5:
                    print("  ✓ Prediction magnitude is reasonable")
                else:
                    print("  ⚠ Prediction magnitude may be too small")
    
    def print_comprehensive_results(self):
        """Print comprehensive model comparison results"""
        print("\n=== Comprehensive Model Results ===")
        
        for name, results in self.results.items():
            print(f"\n{name.upper()}:")
            for metric, value in results.items():
                if isinstance(value, float):
                    if 'accuracy' in metric:
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
        
        # Additional analysis
        self.analyze_model_interpretability()
        self.evaluate_prediction_reasonableness()
        
        return True
    
    def run_full_analysis(self):
        """Run complete improved uplift analysis"""
        print("=== Improved Uplift Analysis Pipeline ===")
        
        # Load and explore data
        self.load_and_explore_data()
        
        # Create improved features
        self.create_improved_features()
        
        # Train improved models
        self.train_improved_models()
        
        # Print comprehensive results
        self.print_comprehensive_results()
        
        print("\n=== Analysis Complete ===")
        return self.models, self.results

def main():
    """Main function"""
    analyzer = ImprovedUpliftAnalysis()
    models, results = analyzer.run_full_analysis()
    return models, results

if __name__ == "__main__":
    main() 