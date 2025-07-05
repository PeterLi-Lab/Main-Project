# Stack Overflow Data Analysis - Prediction Tasks

This project implements a comprehensive pipeline for building modeling-ready datasets from raw StackExchange XML logs and training prediction models for various tasks.

## 📋 Project Overview

The project follows a structured approach for CTR prediction and Uplift modeling:

### 📌 Step 1: CTR Sample Construction
- **Script**: `user_post_click_labeling.py`
- **Purpose**: Constructs pseudo click behavior for CTR modeling
- **Output**: `user_post_click_samples.csv` (user_id, post_id, is_click)
- **Process**: 
  - For upvoted posts: Randomly assign active users as positive samples (is_click = 1)
  - For negative samples: Sample active users who didn't upvote, prioritizing similar interests (is_click = 0)
  - Maintains 1:3 positive to negative ratio

### 📌 Step 2: CTR Model Training
- **Script**: `ctr_model_training.py`
- **Purpose**: Trains CTR prediction models using user-post features
- **Input**: User features + Post features + is_click labels
- **Output**: Trained CTR models saved to `models/` directory
- **Models**: Logistic Regression, Random Forest, XGBoost, LightGBM

### 📌 Step 3: Uplift Sample Construction
- **Script**: `uplift_treatment_labeling.py`
- **Purpose**: Reuses CTR samples and adds treatment labels
- **Output**: `uplift_model_data.csv` (user_id, post_id, treatment, response)
- **Process**: 
  - treatment = 1 for AI-related content (python, machine-learning, etc.)
  - treatment = 0 for other content
  - response = is_click from Step 1

### 📌 Step 4: Uplift Model Training
- **Script**: `uplift_model_training.py`
- **Purpose**: Trains uplift models to estimate causal effects
- **Input**: User features + Post features + treatment + response
- **Output**: Trained uplift models saved to `models/` directory
- **Approaches**: Two-Model (T-Learner), Single Model with Treatment Interaction, Uplift Random Forest

## 🚀 Quick Start

### 1. Data Preprocessing
```bash
python main.py --mode preprocess
```

### 2. Generate CTR Samples
```bash
python user_post_click_labeling.py
```

### 3. Train CTR Models
```bash
python main.py --mode ctr_train
# or
python ctr_model_training.py
```

### 4. Generate Uplift Samples
```bash
python uplift_treatment_labeling.py
```

### 5. Train Uplift Models
```bash
python main.py --mode uplift_train
# or
python uplift_model_training.py
```

## 📊 Data Flow

```
Raw XML Data → Data Preprocessing → Feature Engineering
                    ↓
            user_post_click_labeling.py
                    ↓
            user_post_click_samples.csv
                    ↓
            ctr_model_training.py
                    ↓
            Trained CTR Models
                    ↓
            uplift_treatment_labeling.py
                    ↓
            uplift_model_data.csv
                    ↓
            uplift_model_training.py
                    ↓
            Trained Uplift Models
```

## 📁 Key Files

### Data Processing
- `data_preprocessing.py` - Main data preprocessing pipeline
- `user_post_click_labeling.py` - CTR sample generation
- `uplift_treatment_labeling.py` - Uplift sample generation

### Model Training
- `ctr_model_training.py` - CTR model training (Step 2)
- `uplift_model_training.py` - Uplift model training (Step 4)

### Main Scripts
- `main.py` - Main pipeline with all modes
- `retention_prediction_labeling.py` - Retention prediction labels

## 🎯 Model Outputs

### CTR Models
- **Location**: `models/ctr_*.pkl`
- **Features**: User reputation, post engagement, content features
- **Target**: is_click (binary)
- **Metrics**: Accuracy, AUC

### Uplift Models
- **Location**: `models/uplift_*.pkl`
- **Features**: Same as CTR + treatment indicator
- **Target**: response (is_click)
- **Metrics**: Uplift estimation accuracy, treatment effect size

## 🔧 Configuration

### Tag Parsing
- Tags are pipe-separated: `|python|machine-learning|`
- Treatment tags: `python`, `machine-learning`, `artificial-intelligence`, `deep-learning`, `neural-network`, `tensorflow`, `pytorch`, `scikit-learn`

### Sampling Ratios
- CTR: 1:3 positive to negative samples
- Uplift: Uses same samples as CTR with treatment labels

## 📈 Expected Results

### CTR Model Performance
- Accuracy: ~0.65-0.75
- AUC: ~0.70-0.80
- Best model typically: XGBoost or LightGBM

### Uplift Model Performance
- Actual vs Predicted uplift comparison
- Treatment effect estimation
- Multiple approaches for robustness

## 🛠️ Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm tqdm
```

## 📝 Usage Examples

### Complete Pipeline
```bash
# Run all steps
python main.py --mode all
```

### Individual Steps
```bash
# Step 1: Preprocessing
python main.py --mode preprocess

# Step 2: CTR Training
python main.py --mode ctr_train

# Step 3: Uplift Training  
python main.py --mode uplift_train
```

### Direct Script Execution
```bash
# Generate CTR samples
python user_post_click_labeling.py

# Train CTR models
python ctr_model_training.py

# Generate uplift samples
python uplift_treatment_labeling.py

# Train uplift models
python uplift_model_training.py
```

## 🔍 Model Evaluation

### CTR Models
- Cross-validation with stratification
- Multiple algorithms comparison
- Feature importance analysis

### Uplift Models
- Treatment effect estimation
- Multiple approaches comparison
- Causal inference validation

## 📊 Output Files

### Data Files
- `user_post_click_samples.csv` - CTR training data
- `uplift_model_data.csv` - Uplift training data

### Model Files
- `models/ctr_*.pkl` - Trained CTR models
- `models/uplift_*.pkl` - Trained uplift models
- `models/*_feature_columns.pkl` - Feature column definitions

### Results
- Model performance metrics
- Feature importance rankings
- Treatment effect estimates

## 🎯 Next Steps

1. **Model Deployment**: Deploy trained models for real-time prediction
2. **A/B Testing**: Validate uplift models in controlled experiments
3. **Feature Engineering**: Add more sophisticated features
4. **Model Ensembling**: Combine multiple approaches for better performance
5. **Real-time Pipeline**: Build streaming data pipeline for live predictions 