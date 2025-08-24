# Project Structure Documentation

## Recommended Viewing Order

### For New Viewers (Start Here)
1. **Project Overview** - Understand the main goal and approach
2. **Latest Results (2024)** - See the current findings and methodology
3. **Two-Stage Analysis** - Understand the refined experimental design
4. **Treatment and Control Definition** - Understand the core concept
5. **Main Data Flow & Scripts** - See the complete pipeline
6. **Core Modules Documentation** - Detailed explanation of main code files
7. **Usage Examples** - How to run the project
8. **Directory Structure** - Complete file organization

### For Technical Users
1. **Core Modules Documentation** - Start with main implementation
2. **Main Data Flow & Scripts** - Understand the pipeline
3. **Latest Results (2024)** - See current methodology
4. **Two-Stage Analysis** - Understand the refined approach
5. **Treatment and Control Definition** - Understand the core concept
6. **Usage Examples** - How to execute
7. **Directory Structure** - Complete reference

---

## Project Overview
This project implements uplift modeling to analyze the effectiveness of AI content and AI tags on user engagement. The project uses a two-stage analysis approach to separate the effects of AI tags from AI content type, providing more accurate insights for strategic decision-making.

---

## Latest Results (2024)

### Two-Stage Analysis Results
The project implements a refined two-stage analysis to disentangle confounding effects:

**Stage 1: Tag Effect Analysis**
- **Tag Uplift Effect**: +0.1561 (statistically significant, p=0.0034)
- **Method**: Compare AI content with AI tags vs AI content without AI tags
- **Finding**: AI tags have a positive effect on user engagement within AI content

**Stage 2: Content Type Effect Analysis**
- **Content Type Effect**: AI content performs worse than non-AI content
- **Method**: Compare AI content vs non-AI content (ignoring tags)
- **Finding**: AI content itself needs quality improvement

### Strategic Insights
1. **Keep AI Tags**: AI tags provide positive uplift and should be maintained
2. **Improve AI Content Quality**: Focus on enhancing AI content quality rather than tag strategy
3. **Target High-Engagement Users**: Users with higher AI content engagement show better tag response

---

## Two-Stage Analysis

### Problem Statement
Traditional uplift analysis confounds two effects:
- **Tag Effect**: Does adding an AI tag change user behavior?
- **Content Type Effect**: Does AI content perform differently from non-AI content?

### Solution: Two-Stage Analysis

**Stage 1: Tag Effect (Within AI Content)**
- **Objective**: Measure the pure effect of AI tags on user behavior
- **Method**: 
  - Filter for posts that are inherently AI content (regardless of tag)
  - Treatment = AI content + AI tag
  - Control = AI content + no AI tag
  - Run uplift analysis within similar content
- **Result**: Isolates tag effect from content type effect

**Stage 2: Content Type Effect (AI vs Non-AI)**
- **Objective**: Measure which content type performs better
- **Method**:
  - Ignore tags, focus on content type
  - Compare AI content vs non-AI content CTR/Engagement
  - Use t-test or regression analysis
- **Result**: Identifies content quality differences

### Implementation
- **Script**: `scripts/two_stage_analysis.py`
- **Key Methods**:
  - `identify_ai_content()`: Detects AI content based on keywords
  - `identify_ai_tags()`: Detects AI tags
  - `stage1_tag_effect_analysis()`: Tag effect within AI content
  - `stage2_content_effect_analysis()`: Content type comparison
  - `create_similar_content_clusters()`: Content clustering for refined analysis

---

## Tag Uplift User Analysis

### User Characteristics for Positive Tag Uplift
Analysis of users who show positive response to AI tags (+0.1561 uplift):

**Key User Profiles**:
1. **High Engagement Users**: Users with higher overall click rates
2. **AI Content Enthusiasts**: Users with higher AI content engagement rates
3. **AI Tag Exposed Users**: Users with more exposure to AI tags
4. **AI Keyword Sensitive**: Users who interact with more AI-related keywords

**Strategic Recommendations**:
1. Target users with higher AI content engagement
2. Focus on users who prefer longer, more detailed content
3. Prioritize users with more AI-related interactions
4. Use AI tags more prominently for high-engagement users

### Implementation
- **Script**: `scripts/analyze_tag_uplift_users.py`
- **Outputs**:
  - `tag_uplift_user_analysis.csv`: User characteristics comparison
  - `positive_uplift_users.csv`: Users with positive tag uplift
  - `negative_uplift_users.csv`: Users with negative tag uplift
  - `user_segments_with_uplift.csv`: User segments with uplift scores

---

## Core Modules Documentation

### Main Execution Files

#### `src/main.py`
**Purpose**: Main project execution script and entry point
**Methods**: 
- Orchestrates the complete uplift modeling pipeline
- Coordinates data loading, preprocessing, model training, and evaluation
- Manages configuration and logging
**Key Features**:
- Modular execution of different pipeline stages
- Error handling and validation
- Results aggregation and reporting

#### `src/uplift_model_training.py`
**Purpose**: Core uplift model training class with comprehensive functionality
**Methods**:
- `load_data()`: Loads and validates input datasets
- `preprocess_data()`: Handles feature engineering and data cleaning
- `train_models()`: Trains separate treatment and control models
- `evaluate_models()`: Calculates accuracy, AUC, and uplift metrics
- `save_results()`: Exports predictions and evaluation metrics
**Key Features**:
- Logistic Regression models for treatment/control groups
- Cross-validation and hyperparameter tuning
- Uplift calculation: Treatment predictions - Control predictions
- Comprehensive evaluation metrics (accuracy, AUC, uplift distribution)

#### `src/data_preprocessing.py`
**Purpose**: Data preprocessing utilities and feature engineering
**Methods**:
- `clean_text()`: Text cleaning and normalization
- `extract_features()`: Feature extraction from raw data
- `handle_missing_values()`: Missing data imputation
- `encode_categorical()`: Categorical variable encoding
**Key Features**:
- TF-IDF text vectorization
- Categorical encoding (one-hot, label encoding)
- Feature scaling and normalization
- Outlier detection and handling

#### `src/uplift_model.py`
**Purpose**: Uplift modeling class with advanced methodologies
**Methods**:
- `calculate_uplift()`: Computes uplift scores
- `validate_assumptions()`: Checks uplift modeling assumptions
- `balance_groups()`: Ensures treatment/control balance
**Key Features**:
- Two-model approach (separate treatment/control models)
- Uplift score calculation and distribution analysis
- Assumption validation (SUTVA, unconfoundedness)
- Group balance checking

#### `src/user_post_click_labeling.py`
**Purpose**: User-post click labeling and feature engineering
**Methods**:
- `label_clicks()`: Creates click labels from user behavior
- `engineer_features()`: Builds user-post interaction features
- `merge_data()`: Combines user, post, and click data
**Key Features**:
- Real click behavior as response variable (`is_click`)
- User-post interaction features
- Temporal feature engineering
- Behavioral pattern extraction

#### `src/uplift_treatment_labeling.py`
**Purpose**: Uplift treatment labeling based on tag content
**Methods**:
- `create_treatment_labels()`: Creates treatment/control labels
- `validate_labels()`: Ensures label quality and balance
- `merge_with_features()`: Combines labels with feature data
**Key Features**:
- Treatment definition: Posts tagged as "ai content"
- Control definition: Similar posts without "ai content" tag
- Tag-based classification logic
- Label validation and quality checks

### Key Analysis Scripts

#### `scripts/two_stage_analysis.py`
**Purpose**: Two-stage analysis to separate tag effects from content effects
**Methods**:
- `identify_ai_content()`: Detects AI content based on keywords
- `identify_ai_tags()`: Detects AI tags
- `stage1_tag_effect_analysis()`: Tag effect within AI content
- `stage2_content_effect_analysis()`: Content type comparison
- `create_similar_content_clusters()`: Content clustering for refined analysis
**Key Features**:
- Separates confounding effects (tag vs content)
- Statistical significance testing
- Content clustering for refined analysis
- Strategic recommendations generation

#### `scripts/analyze_tag_uplift_users.py`
**Purpose**: Analyze user characteristics for positive tag uplift effect
**Methods**:
- `create_user_features()`: Creates user-level features
- `calculate_tag_uplift_by_user()`: Calculates tag uplift for each user
- `analyze_positive_uplift_users()`: Analyzes positive uplift user characteristics
- `create_user_segments()`: Creates user segments based on uplift response
- `generate_insights()`: Generates strategic insights
**Key Features**:
- User-level uplift analysis
- Statistical comparison of user groups
- User segmentation based on uplift response
- Actionable strategic recommendations

#### `scripts/uplift_model_on_cluster7_user_post.py`
**Purpose**: Main uplift modeling script with content-homogeneous clustering
**Methods**:
- Content-based clustering using TF-IDF + MiniBatchKMeans
- Treatment/control split within selected clusters
- User-post level uplift modeling with real click response
- Model training and evaluation
**Key Features**:
- TF-IDF text vectorization for content similarity
- MiniBatchKMeans clustering for scalability
- Logistic Regression models for treatment/control
- Real click behavior as response variable
- Comprehensive evaluation metrics

#### `scripts/cluster_posts_by_content.py`
**Purpose**: Content-based clustering of posts using text similarity
**Methods**:
- Text preprocessing and TF-IDF vectorization
- MiniBatchKMeans clustering
- Cluster analysis and visualization
- Cluster quality assessment
**Key Features**:
- TF-IDF for text representation
- MiniBatchKMeans for efficient clustering
- Silhouette score for cluster quality
- Cluster visualization and analysis

#### `scripts/cluster7_treatment_control_split.py`
**Purpose**: Treatment/control split within selected content clusters
**Methods**:
- Identifies AI-related clusters (high AI keyword density)
- Splits posts into treatment (AI-tagged) and control (non-AI-tagged)
- Ensures content similarity between groups
- Validates group balance and quality
**Key Features**:
- AI keyword density analysis
- Content-homogeneous group selection
- Treatment/control balance validation
- Quality metrics for group separation

### Validation and Debugging Scripts

#### `scripts/comprehensive_validation.py`
**Purpose**: Comprehensive validation for uplift modeling assumptions
**Methods**:
- SUTVA assumption checking
- Unconfoundedness validation
- Treatment assignment randomness
- Outcome variable quality assessment
**Key Features**:
- Statistical tests for uplift assumptions
- Data quality validation
- Model performance assessment
- Robustness checks

#### `scripts/debug_uplift.py`
**Purpose**: Debug uplift modeling issues and performance
**Methods**:
- Model performance analysis
- Feature importance assessment
- Data leakage detection
- Error analysis and troubleshooting
**Key Features**:
- Performance profiling
- Feature importance ranking
- Data leakage detection
- Error diagnosis and resolution

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
4. `scripts/two_stage_analysis.py` → Two-stage analysis results
5. `scripts/analyze_tag_uplift_users.py` → User characteristics analysis

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

#### Two-Stage Analysis Scripts
- **`two_stage_analysis.py`**: Two-stage analysis to separate tag effects from content effects
- **`analyze_tag_uplift_users.py`**: Analyze user characteristics for positive tag uplift effect

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

### Recommended Analysis Pipeline
```bash
# 1. Two-stage analysis
python scripts/two_stage_analysis.py

# 2. Tag uplift user analysis
python scripts/analyze_tag_uplift_users.py

# 3. Content clustering
python scripts/cluster_posts_by_content.py

# 4. Treatment/control split
python scripts/cluster7_treatment_control_split.py

# 5. Uplift modeling
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
- **`tag_uplift_user_analysis.csv`**: User characteristics comparison
- **`positive_uplift_users.csv`**: Users with positive tag uplift
- **`negative_uplift_users.csv`**: Users with negative tag uplift
- **`user_segments_with_uplift.csv`**: User segments with uplift scores

## Key Features
- Two-stage analysis to separate tag effects from content effects
- User characteristics analysis for positive tag uplift
- Treatment/Control definition based on tag content containing "ai content" within content-homogeneous clusters
- Cluster-based treatment selection for accurate uplift modeling
- Real click (`is_click`) as response for uplift modeling
- Comprehensive uplift modeling pipeline with model evaluation
- Multiple validation and debugging scripts
- CTR and retention prediction capabilities
- Feature engineering and data preprocessing utilities