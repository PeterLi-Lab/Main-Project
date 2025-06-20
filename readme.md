# Stack Overflow Data Analysis Pipeline

A comprehensive Python pipeline for analyzing Stack Overflow data using advanced NLP techniques, feature engineering, and clustering algorithms.

## Features

- **Data Preprocessing**: XML parsing, text cleaning, feature engineering
- **TF-IDF Feature Extraction**: Extracts TF-IDF features from titles and tags with dimensionality reduction
- **Transformer Embeddings**: Uses sentence-transformers for semantic embeddings
- **Multiple Clustering Methods**:
  - Traditional clustering (K-means + HDBSCAN) using transformer embeddings
  - **Combined clustering** using concatenated TF-IDF + Transformer features
  - **UMAP + GMM clustering** using UMAP dimensionality reduction and Gaussian Mixture Models
- **Advanced Visualization**: PCA plots, cluster comparisons, distance distributions, UMAP visualizations
- **Modular Architecture**: Separate modules for preprocessing, clustering, and analysis
- **Caching System**: Saves processed data to avoid recomputation
- **Command-line Interface**: Flexible pipeline execution modes

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

### Command Line Interface

The pipeline supports multiple execution modes:

```bash
# Run complete pipeline (preprocessing + traditional clustering + combined clustering)
python main.py

# Run only preprocessing
python main.py --mode preprocess

# Run only traditional clustering (K-means + HDBSCAN)
python main.py --mode cluster

# Run only combined TF-IDF + Transformer clustering
python main.py --mode combined

# Run only UMAP + GMM clustering
python main.py --mode umap_gmm

# Run preprocessing and traditional clustering only
python main.py --mode preprocess cluster

# Custom parameters
python main.py --n-clusters 10 --min-cluster-size 8 --variance-threshold 0.95

# UMAP + GMM with custom parameters
python main.py --mode umap_gmm --umap-components 30 --umap-neighbors 20 --gmm-components 12

# 限制到50维（默认）
python main.py --mode combined

# 或者更激进的30维
python main.py --mode combined --max-dimensions 30
```

### Parameters

- `--mode`: Pipeline mode (`preprocess`, `cluster`, `combined`, `umap_gmm`, `all`)
- `--data-dir`: Directory containing XML data files (default: `data`)
- `--cache-file`: Cache file for processed data (default: `processed_data.pkl`)
- `--n-clusters`: Number of clusters for K-means (default: 8)
- `--min-cluster-size`: Minimum cluster size for HDBSCAN (default: 3)
- `--min-samples`: Minimum samples for HDBSCAN (auto-determined if not specified)
- `--variance-threshold`: PCA variance threshold 0.0-1.0 (default: 0.9)
- `--max-dimensions`: Maximum dimensions after PCA for combined clustering (default: 50)
- `--umap-components`: Number of UMAP components for dimensionality reduction (default: 50)
- `--umap-neighbors`: Number of neighbors for UMAP (default: 15)
- `--umap-min-dist`: Minimum distance for UMAP (default: 0.1)
- `--gmm-components`: Number of GMM components (auto-determined if not specified)
- `--gmm-covariance`: GMM covariance type (`full`, `tied`, `diag`, `spherical`, default: `full`)
- `--force-reprocess`: Force reprocessing even if cache exists

## Pipeline Overview

### 1. Data Preprocessing (`data_preprocessing.py`)

- **XML Parsing**: Extracts posts from Stack Overflow XML files
- **Text Cleaning**: Removes HTML tags, special characters, and normalizes text
- **Feature Engineering**: Creates features like post length, vote ratios, title characteristics
- **TF-IDF Extraction**: Extracts TF-IDF features from titles and tags
- **Transformer Embeddings**: Generates semantic embeddings using sentence-transformers
- **Visualization**: Creates exploratory data analysis plots

### 2. Traditional Clustering (`clustering_analysis.py`)

- **PCA Reduction**: Reduces transformer embeddings to specified dimensions
- **K-means Clustering**: Partitions data into k clusters
- **HDBSCAN Clustering**: Density-based clustering with noise detection
- **Cluster Analysis**: Analyzes cluster characteristics and quality metrics
- **Visualization**: Creates cluster comparison plots and sample titles

### 3. Combined Feature Clustering

- **Feature Concatenation**: Combines TF-IDF and Transformer features
- **PCA Reduction**: Reduces combined features while preserving variance
- **Cosine Distance**: Computes pairwise cosine distances
- **HDBSCAN with Precomputed Distances**: Performs clustering using distance matrix
- **Analysis**: Provides comprehensive cluster analysis and visualization

### 4. UMAP + GMM Clustering

- **UMAP Reduction**: Non-linear dimensionality reduction preserving local and global structure
- **Gaussian Mixture Models**: Probabilistic clustering with soft assignments
- **Cosine Distance**: Uses cosine distance for better performance with text embeddings
- **Cluster Analysis**: Analyzes cluster characteristics and quality metrics (BIC, AIC)
- **Visualization**: Creates UMAP scatter plots and cluster size distributions

## Key Features

### TF-IDF + Transformer Combination

The pipeline now supports combining TF-IDF and Transformer features for enhanced clustering:

```python
# Concatenate features
combined_features = np.concatenate([tfidf_features, embeddings], axis=1)

# Apply PCA for dimensionality reduction
combined_features_pca = PCA(n_components=0.9).fit_transform(combined_features)

# Use HDBSCAN with built-in cosine distance (more efficient)
hdbscan_model = HDBSCAN(metric='cosine')
labels = hdbscan_model.fit_predict(combined_features_pca)
```

This approach combines:
- **TF-IDF**: Captures vocabulary and keyword patterns
- **Transformer**: Captures semantic meaning and context
- **HDBSCAN Built-in Cosine Distance**: More efficient than manual distance matrix calculation

### Clustering Algorithms

1. **K-means**: Traditional partitioning algorithm
   - Requires specifying number of clusters
   - Good for well-separated, spherical clusters

2. **HDBSCAN**: Hierarchical density-based clustering
   - Automatically determines number of clusters
   - Handles noise points and irregular shapes
   - Uses adaptive parameter estimation

3. **Combined Features**: Enhanced clustering with feature fusion
   - Leverages both lexical and semantic information
   - Improved cluster quality and interpretability

4. **UMAP + GMM**: Non-linear dimensionality reduction with probabilistic clustering
   - **UMAP**: Preserves both local and global structure in high-dimensional data
   - **GMM**: Provides soft cluster assignments and uncertainty estimates
   - **Cosine Distance**: Optimized for text embeddings
   - **Quality Metrics**: BIC and AIC for model selection

## Output

The pipeline generates:

- **Processed Data**: Cached in `processed_data.pkl`
- **Visualizations**: 
  - PCA plots for dimensionality reduction
  - Cluster comparison plots
  - Distance distribution histograms
  - Sample titles from each cluster
  - UMAP scatter plots and cluster distributions
- **Analysis Results**: 
  - Cluster statistics and characteristics
  - Quality metrics (silhouette scores, BIC, AIC)
  - Sample titles and post types

## Example Results

### UMAP + GMM Clustering Results
```
=== Performing UMAP + GMM Clustering ===
UMAP reduced from 384 to 50 dimensions
GMM found 15 clusters
Silhouette Score: 0.280
BIC Score: -15028970.31
AIC Score: -15198129.24
Clustering completed successfully!
```

**Performance Metrics:**
- **Silhouette Score**: 0.280 (Good clustering quality)
- **BIC Score**: -15028970.31 (Lower is better for model selection)
- **AIC Score**: -15198129.24 (Lower is better for model selection)
- **Dimensionality Reduction**: 384 → 50 dimensions (87% reduction)
- **Clusters Found**: 15 clusters automatically determined by GMM

### Clustering Quality Comparison
| Method | Silhouette Score | Clusters | Noise Points | Dimensionality |
|--------|------------------|----------|--------------|----------------|
| K-means | ~0.25-0.35 | Fixed (8) | None | 50 (PCA) |
| HDBSCAN | ~0.20-0.30 | Auto | Yes | 50 (PCA) |
| Combined | ~0.25-0.35 | Auto | Yes | 50 (PCA) |
| UMAP+GMM | ~0.25-0.30 | Auto | None | 50 (UMAP) |

## File Structure

```
Main Project/
├── main.py                 # Main pipeline orchestration
├── data_preprocessing.py   # Data loading and preprocessing
├── clustering_analysis.py  # Clustering algorithms and analysis
├── requirements.txt        # Python dependencies
├── readme.md              # This file
├── data/                  # Input XML data files
├── output/                # Generated outputs
└── venv/                  # Virtual environment (not in git)
```

## Dependencies

- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning algorithms
- `sentence-transformers`: Transformer embeddings
- `hdbscan`: Hierarchical density-based clustering
- `umap-learn`: UMAP dimensionality reduction
- `matplotlib`: Visualization
- `seaborn`: Enhanced plotting
- `lxml`: XML parsing
- `nltk`: Natural language processing

## Notes

- Large XML files (>100MB) are excluded from git via `.gitignore`
- Processed data is cached to avoid recomputation
- The pipeline automatically handles missing data and edge cases
- HDBSCAN parameters are automatically estimated based on data size

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.