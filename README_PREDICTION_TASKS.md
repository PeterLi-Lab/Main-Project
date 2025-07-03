# Prediction Tasks for Stack Overflow Data Analysis

This project implements four key prediction tasks for analyzing Stack Overflow user behavior data:

## 1. CTR Prediction (Click-Through Rate)
**Task Type**: Classification (Binary)  
**Target**: `is_click` (0/1)  
**Goal**: Predict if a user will engage with a post based on metadata and user attributes

### Features Used:
- Text features: title length, post length, number of tags
- User features: reputation, post count, badge information
- Content quality: score, view count, answer count, comment count
- Influence metrics: total influence score, vote ratio, badge quality
- Time-based features: post age, creation date

### Usage:
```bash
# Run CTR prediction with XGBoost
python main.py --mode ctr --model-type xgboost

# Run CTR prediction with LightGBM
python main.py --mode ctr --model-type lightgbm

# Run CTR prediction with Random Forest
python main.py --mode ctr --model-type random_forest
```

## 2. Retention Prediction
**Task Type**: Classification (Binary)  
**Target**: `is_retained` (0/1)  
**Goal**: Predict whether a user will return after 7 days based on session-level behaviors

### Features Used:
- User activity: post count, engagement score
- Influence metrics: total influence score, badge information
- Content quality: score, view count, answer count
- Time-based features: days since first badge, post age
- Categorical features: influence level, badge level

### Usage:
```bash
# Run retention prediction with XGBoost
python main.py --mode retention --model-type xgboost

# Run retention prediction with Logistic Regression
python main.py --mode retention --model-type logistic_regression
```

## 3. Retention Duration Estimation
**Task Type**: Regression  
**Target**: `days_to_next_action` (continuous)  
**Goal**: Estimate how long a user will return to post a new post/comment since previous post

### Features Used:
- User activity: post count, engagement score
- Influence metrics: total influence score, badge information
- Content quality: score, view count, answer count
- Time-based features: days since creation, post age
- User engagement: answer count + comment count

### Usage:
```bash
# Run duration prediction with XGBoost
python main.py --mode duration --model-type xgboost

# Run duration prediction with Linear Regression
python main.py --mode duration --model-type linear_regression
```

## 4. Uplift Modeling
**Task Type**: Treatment Effect Estimation  
**Target**: `treatment`, `response`  
**Goal**: Estimate the effect of exposure to certain content (e.g., AI-tagged posts) on user behavior

### Features Used:
- Same features as CTR prediction
- Treatment assignment (synthetic for demonstration)
- Response variable based on user behavior

### Usage:
```bash
# Run uplift modeling with XGBoost
python main.py --mode uplift --model-type xgboost

# Run uplift modeling with Random Forest
python main.py --mode uplift --model-type random_forest
```

## Complete Pipeline

Run all four prediction tasks in sequence:

```bash
python main.py --mode all
```

This will execute:
1. Data preprocessing
2. Clustering analysis
3. CTR prediction
4. Retention prediction
5. Duration prediction
6. Uplift modeling

## Model Types Available

- **XGBoost**: Fast gradient boosting, good for most tasks
- **LightGBM**: Light gradient boosting, efficient for large datasets
- **Random Forest**: Ensemble method, good interpretability
- **Logistic Regression**: Linear model for classification tasks
- **Linear Regression**: Linear model for regression tasks

## Output Files

The system generates:
- Model performance metrics
- Feature importance rankings
- Visualization plots
- Results summary in `output/results_summary.json`

## Testing

Test all prediction tasks:

```bash
python test_prediction_tasks.py
```

This will verify that all four tasks work correctly with different model types.

## Data Requirements

The system expects XML files in the `data/` directory:
- `Posts.xml`: Post data with metadata
- `Users.xml`: User information
- `Tags.xml`: Tag information
- `Votes.xml`: Voting data
- `Badges.xml`: Badge information
- `Comments.xml`: Comment data (optional)

## Customization

### Adding New Features
Edit the feature preparation methods in each predictor class:
- `prepare_ctr_features()` in `CTRPredictor`
- `prepare_retention_features()` in `RetentionPredictor`
- `prepare_duration_features()` in `RetentionDurationPredictor`
- `prepare_uplift_features()` in `UpliftModeling`

### Changing Target Variables
Modify the target column creation in the feature preparation methods or specify a different target using the `--target-col` parameter.

### Adding New Models
Add new model types to the training methods in each predictor class.

## Performance Metrics

### Classification Tasks (CTR, Retention)
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC (if applicable)
- Confusion Matrix

### Regression Task (Duration)
- Mean Squared Error (MSE)
- R-squared (R²)
- Mean Absolute Error (MAE)

### Uplift Modeling
- Control and Treatment model performance
- Uplift distribution analysis
- Treatment effect summary

## Example Results

After running the complete pipeline, you'll see:
- Model performance comparisons
- Feature importance visualizations
- Prediction vs actual plots
- Uplift effect analysis

The results are saved in the `output/` directory for further analysis. 