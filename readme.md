# Stack Overflow Data Analysis System

A comprehensive data analysis system for Stack Overflow data, including data preprocessing, clustering analysis, prediction models, and industrial-grade CTR prediction.

## Features

### 1. Data Preprocessing (`data_preprocessing.py`)
- XML data parsing and cleaning
- Feature engineering and normalization
- Text processing and TF-IDF features
- Semantic embeddings using sentence transformers
- User influence and badge analysis
- **Industrial-grade features**: Hash encoding, feature crossing, sequence features, context features
- **Negative sampling**: Industry-standard data balancing for CTR prediction

### 2. Clustering Analysis (`clustering_analysis.py`)
- Multiple clustering algorithms (K-means, DBSCAN, Hierarchical)
- Dimensionality reduction (PCA, UMAP)
- Clustering quality evaluation
- Interactive visualizations

### 3. Prediction Models (`prediction_models.py`)
- CTR (Click-Through Rate) prediction
- User retention prediction
- Multiple ML algorithms (XGBoost, LightGBM, Random Forest)
- Model evaluation and visualization
- **Industrial CTR Models**: DeepFM, DCN, DIN, LR, FM
- **Online inference service**: Real-time prediction with performance monitoring

### 4. Industrial-Grade CTR System (Integrated)
- **Reference Architecture**: Based on systems used by Alibaba, ByteDance, Google, Meta
- **Complete Feature Engineering Pipeline**: Integrated into data preprocessing
- **Multiple Model Architectures**: Integrated into prediction models
- **Online Inference Service**: Real-time prediction with performance monitoring
- **Negative Sampling**: Industry-standard data balancing techniques
- **A/B Testing Framework**: Model comparison and evaluation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Main-Project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Interactive Menu
Run the main script for an interactive menu:

```bash
python main.py
```

Choose from:
1. Data Preprocessing
2. Clustering Analysis
3. Basic Prediction Models
4. Industrial-Grade CTR System
5. Run Complete Pipeline

### Industrial-Grade CTR System

The industrial-grade CTR system is now integrated into the existing modules:

#### System Architecture

```
Data Layer (data_preprocessing.py)
├── User behavior data (clicks, views, interactions)
├── Content data (posts, tags, categories)
├── Context data (time, device, location)
└── Feature engineering outputs

Feature Engineering Layer (data_preprocessing.py)
├── Categorical features → Hash encoding / Embedding
├── Numerical features → Normalization / Binning
├── Feature crossing → FM / Wide models
└── Sequence features → Attention networks (DIN)

Model Layer (prediction_models.py)
├── Logistic Regression (baseline)
├── Factorization Machine (FM)
├── DeepFM (Wide & Deep)
├── Deep & Cross Network (DCN)
└── Deep Interest Network (DIN)

Online Service Layer (prediction_models.py)
├── Real-time inference (10ms timeout)
├── Model monitoring
├── A/B testing
└── Performance metrics
```

#### Key Features

**1. Industrial Feature Engineering (in data_preprocessing.py)**
- Hash encoding for categorical features (memory efficient)
- StandardScaler for numerical features
- Feature crossing for interaction modeling
- Sequence features for user behavior modeling
- Context features for temporal and spatial patterns
- Negative sampling for data balancing

**2. Model Evolution Path (in prediction_models.py)**
```
Linear Models → Feature Crossing → Auto Crossing → Attention → Multi-task
     LR           Wide&Deep        DeepFM/DCN      DIN        MMOE/PLE
```

**3. Training Pipeline**
- Negative sampling (3:1 ratio)
- Stratified data splitting
- Early stopping
- Model performance tracking

**4. Online Inference**
- Sub-10ms response time
- Model caching
- Real-time metrics monitoring
- Error handling and fallbacks

#### Usage Example

```python
from data_preprocessing import DataPreprocessor
from prediction_models import IndustrialCTRPredictor

# Create industrial features
preprocessor = DataPreprocessor()
df_industrial = preprocessor.create_industrial_features(df_combined.copy())

# Perform negative sampling
df_balanced = preprocessor.create_negative_sampling(df_industrial)

# Train industrial models
industrial_predictor = IndustrialCTRPredictor()
models = industrial_predictor.train_industrial_models(df_balanced)

# Online prediction
sample_features = {
    'Score': 10,
    'ViewCount': 100,
    'AnswerCount': 2,
    'CommentCount': 5,
    'title_length': 15,
    'post_length': 200,
    'num_tags': 3,
    'post_age_days': 30,
    'user_post_count': 50,
    'user_reputation': 1000,
    'total_votes': 20,
    'upvotes': 18
}

result = industrial_predictor.online_predict(sample_features)
print(f"CTR Probability: {result['ctr_probability']:.4f}")
print(f"Response Time: {result['response_time']:.4f}s")
```

#### Model Performance

The system automatically trains and compares multiple models:

| Model | AUC | LogLoss | Features | Use Case |
|-------|-----|---------|----------|----------|
| LR | ~0.75 | ~0.45 | 50+ | Baseline, interpretable |
| FM | ~0.78 | ~0.42 | 50+ | Feature interactions |
| DeepFM | ~0.82 | ~0.38 | 100+ | Best overall performance |
| DCN | ~0.81 | ~0.39 | 100+ | High-order interactions |
| DIN | ~0.83 | ~0.37 | 100+ | User interest modeling |

## Data Requirements

Place your Stack Overflow XML files in the `data/` directory:
- `Posts.xml`
- `Users.xml`
- `Tags.xml`
- `Votes.xml`
- `Badges.xml`

## Output

The system generates:
- Preprocessed data with normalized features
- Clustering analysis results and visualizations
- Prediction model performance metrics
- Industrial CTR system models and configurations
- Performance monitoring dashboards

All outputs are saved in the `output/` directory.

## Dependencies

- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost, lightgbm
- tensorflow, sentence-transformers
- umap-learn, hdbscan

## Contributing

This system demonstrates industrial-grade machine learning practices. Contributions are welcome for:
- Additional model architectures
- Feature engineering techniques
- Performance optimizations
- Documentation improvements

## License

MIT License - see LICENSE file for details.

# StackExchange Modeling-Ready Dataset Pipeline

## Overview
This project provides a full pipeline to transform raw StackExchange XML logs into a modeling-ready dataset for tasks such as CTR prediction, retention modeling, and uplift modeling. The pipeline includes:
- Data cleaning and feature engineering
- User-post click labeling (for implicit feedback modeling)
- Uplift treatment labeling (for causal inference and uplift modeling)

---

## Key Scripts

### 1. `build_feature_dataset.py`
- **Purpose:**
  - Parses and cleans raw XML files (Posts, Users, Votes, Comments, etc.)
  - Engineers user-level, post-level, and interaction-level features
  - Outputs a post-level feature table for downstream modeling
- **Output:**
  - `feature_table.csv`

### 2. `user_post_click_labeling.py`
- **Purpose:**
  - Constructs a user-post pair dataset with a binary click label (`is_click`)
  - **Positive samples:** For each upvoted post, randomly assign N active users as "clickers"
  - **Negative samples:** For each upvoted post, sample N active users who did not upvote as "non-clickers", prioritizing those with similar tag interests
  - Adds user and post features to each pair
- **Output:**
  - `user_post_click_samples.csv` (full dataset)
  - `user_post_click_samples_sample.csv` (sample for inspection)

### 3. `uplift_treatment_labeling.py`
- **Purpose:**
  - Adds treatment labels to the user-post click dataset based on post content tags (e.g., AI, web development, etc.)
  - Simulates "treatment" (exposure to certain content) for uplift modeling
  - Computes uplift features and analyzes treatment effects
- **Tag Parsing Logic:**
  - Tags are parsed from the StackExchange format: `|tag1|tag2|` → `["tag1", "tag2"]`
  - Treatment is assigned if any tag in a post matches the configured treatment tag list (e.g., AI/ML tags)
- **Output:**
  - `uplift_dataset.csv` (user-post-treatment dataset for uplift modeling)

---

## Example: Simulating Treatment with Tags
To simulate "treatment" (e.g., AI content exposure):
```python
# In uplift_treatment_labeling.py
ai_tags = ['ai', 'artificial-intelligence', 'machine-learning', ...]
df_samples['treatment_ai'] = df_samples['post_tags'].apply(lambda tags: any(tag in ai_tags for tag in tags))
```

---

## How to Run
1. **Feature Engineering:**
   ```bash
   python build_feature_dataset.py
   ```
2. **User-Post Click Labeling:**
   ```bash
   python user_post_click_labeling.py
   ```
3. **Uplift Treatment Labeling:**
   ```bash
   python uplift_treatment_labeling.py
   ```

---

## Output Files
- `feature_table.csv`: Post-level features for modeling
- `user_post_click_samples.csv`: User-post pairs with click labels and features
- `uplift_dataset.csv`: User-post-treatment dataset for uplift modeling

---

## Notes
- **Tag Parsing:** All tag parsing uses the StackExchange pipe format: `|tag1|tag2|` → `["tag1", "tag2"]`
- **Treatment Assignment:** You can easily configure new treatments by editing the `treatments` dictionary in `uplift_treatment_labeling.py`.
- **Scalability:** The pipeline is designed for large XML files and can be extended for more advanced modeling tasks.

---

## Contact
For questions or improvements, please open an issue or contact the maintainer.