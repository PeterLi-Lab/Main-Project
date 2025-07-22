# Project Structure Documentation

## Overview
This project implements uplift modeling to analyze the effectiveness of AI content on user engagement. The project is organized into two main directories: `src/` for core modules and `scripts/` for analysis and utility scripts.

---

## Latest: Content-Homogeneous Clustering & Real Click Uplift Modeling (2024)

### Key Workflow
1. **Content Clustering**: Use TF-IDF + MiniBatchKMeans to cluster posts by content similarity (`scripts/cluster_posts_by_content.py`). Output: `post_clusters.csv`.
2. **AI Cluster Selection**: Identify clusters with high AI-related keyword density (e.g., Cluster 7).
3. **Treatment/Control Split**: Within the selected cluster, define:
   - **Treatment**: Posts tagged as "ai content"
   - **Control**: Posts in the same cluster but NOT tagged as "ai content"
   Output: `cluster7_treatment_control.csv`.
4. **User-Post Level Uplift Modeling**: Merge user click data (`user_post_click_samples.csv`) with cluster info, keeping only posts in the selected cluster. Use `is_click` as the real response variable. Script: `scripts/uplift_model_on_cluster7_user_post.py`.
5. **Model Training & Evaluation**: Train separate Logistic Regression models for treatment and control, predict uplift, and evaluate accuracy/AUC.
6. **Outputs**:
   - `cluster7_user_post_uplift_prediction.csv`: Uplift predictions at user-post level
   - `uplift_predicted_distribution_cluster7_user_post.png`: Uplift distribution plot
   - `uplift_model_eval.txt`: Model accuracy and AUC

### Latest Results (2024)
- **Mean predicted uplift**: -0.1615
- **Top 5% uplift**: 0.1894
- **Treatment model**: Test accuracy = 0.8220, AUC = 0.8766
- **Control model**: Test accuracy = 0.8948, AUC = 0.9492

---

## Treatment and Control Definition

**Treatment and Control are defined for uplift modeling:**
- **Treatment (1)**: Posts with tags containing "ai content" 
- **Control (0)**: Posts similar to AI content but **NOT tagged as "ai content"** (selected from the same content cluster)

This definition ensures that the control group consists of posts that are content-wise similar to AI posts but were not classified with the "ai content" tag. This allows us to measure the true effect of the AI tag classification on user engagement, minimizing content bias.

### Cluster-Based Selection Method

The project implements a **cluster-based approach** for selecting treatment and control groups:

1. **Content Clustering**: Use TF-IDF embeddings and K-means clustering to group posts by content similarity
2. **AI Cluster Identification**: Identify clusters with high AI keyword density
3. **Treatment/Control Selection**: Within AI clusters, separate posts based on tag classification
4. **Balanced Groups**: Ensure treatment and control groups are properly balanced

This approach ensures that:
- All posts in the analysis are content-wise similar (from the same AI clusters)
- Treatment and control differ only in tag classification
- More accurate uplift measurement by reducing content-based confounding

---

## Main Data Flow & Scripts

1. `scripts/cluster_posts_by_content.py` → `post_clusters.csv`
2. `scripts/cluster7_treatment_control_split.py` → `cluster7_treatment_control.csv`
3. `scripts/uplift_model_on_cluster7_user_post.py` → `cluster7_user_post_uplift_prediction.csv`, `uplift_predicted_distribution_cluster7_user_post.png`, `uplift_model_eval.txt`

---

## Directory Structure

### `src/` - Core Modules
Core functionality modules for uplift modeling and data processing:

- **`main.py`**: Main project execution script
- **`uplift_model_training.py`**: Uplift model training class with methods for loading data, preprocessing, training, evaluation, and saving results
- **`data_preprocessing.py`**: Data preprocessing utilities
- **`uplift_model.py`**: Uplift modeling class
- **`user_post_click_labeling.py`**: User-post click labeling and feature engineering
- **`uplift_treatment_labeling.py`**: Uplift treatment labeling and feature engineering (defines treatment/control based on tag content)

### `scripts/` - Analysis and Utility Scripts
Scripts for various analyses, debugging, and validation:

#### Uplift Modeling Scripts
- **`uplift_model_on_cluster7_user_post.py`**: Uplift modeling on content-homogeneous cluster with real click response
- **`cluster_posts_by_content.py`**: Content-based clustering of posts
- **`cluster7_treatment_control_split.py`**: Treatment/control split within selected cluster
- **`uplift_results_analysis.py`**: Analyze uplift modeling results
- **`validate_uplift_results.py`**: Validate uplift modeling results
- **`create_treatment_from_tags.py`**: Create treatment labels based on tag containing 'ai content'
- **`cluster_based_treatment_selection.py`**: Cluster-based treatment and control selection (recommended)
- **`advanced_similarity_detection.py`**: Advanced similarity detection using multiple methods

#### Debugging Scripts
- **`debug_uplift.py`**: Debug uplift modeling issues and performance
- **`debug_uplift_prediction.py`**: Debug uplift prediction model performance and issues
- **`debug_post_mapping.py`**: Debug post mapping and relationships
- **`debug_treatment_matching.py`**: Debug treatment matching and assignment
- **`debug_ctr_features.py`**: Debug CTR features and model performance

#### Validation Scripts
- **`comprehensive_validation.py`**: Comprehensive validation for uplift modeling
- **`final_validation_check.py`**: Final validation check for uplift modeling
- **`run_all_validations.py`**: Run all validation scripts

#### Analysis Scripts
- **`deep_feature_analysis.py`**: Deep feature analysis for uplift modeling
- **`analyze_remaining_correlations.py`**: Analyze remaining correlations after initial feature selection
- **`check_leaky_features.py`**: Check for data leakage in features
- **`check_behavior_treatment_overlap_english.py`**: Check overlap between behavior features and treatment assignment
- **`basic_logic_verification_english.py`**: Basic logic verification for uplift modeling

#### CTR Modeling Scripts
- **`ctr_model_training.py`**: CTR model training and evaluation

#### Retention Prediction Scripts
- **`retention_prediction_labeling.py`**: Retention prediction labeling

---

## Usage

### Recommended Uplift Modeling Pipeline
```bash
# 1. 内容聚类
python scripts/cluster_posts_by_content.py
# 2. 选取AI相关簇并分组
python scripts/cluster7_treatment_control_split.py
# 3. 合并点击数据并建模
python scripts/uplift_model_on_cluster7_user_post.py
```

### Running the Main Project
```bash
python src/main.py
```

---

## Data Files
- **`post_clusters.csv`**: Post-level content cluster assignments
- **`cluster7_treatment_control.csv`**: Treatment/control split within selected cluster
- **`cluster7_user_post_uplift_prediction.csv`**: Uplift predictions at user-post level
- **`uplift_predicted_distribution_cluster7_user_post.png`**: Uplift distribution plot
- **`uplift_model_eval.txt`**: Model accuracy and AUC
- **`user_post_click_samples.csv`**: User-post click samples
- **`retention_prediction_data.csv`**: Retention prediction data

## Key Features
- Treatment/Control definition based on tag content containing "ai content" within content-homogeneous clusters
- Cluster-based treatment selection for accurate uplift modeling
- Real click (`is_click`) as response for uplift modeling
- Comprehensive uplift modeling pipeline with model evaluation
- Multiple validation and debugging scripts
- CTR and retention prediction capabilities
- Feature engineering and data preprocessing utilities