# Stack Overflow Data Analysis Project

This project performs comprehensive analysis of Stack Overflow data using machine learning techniques including clustering and natural language processing.

## Features

- **Data Preprocessing**: XML parsing, cleaning, and feature engineering
- **Text Feature Extraction**: 
  - TF-IDF features from titles and tags with dimensionality reduction
  - Semantic embeddings using sentence transformers
- **Clustering Analysis**: 
  - K-means clustering with optimal cluster selection
  - DBSCAN clustering with adaptive parameter estimation
  - Cluster quality metrics comparison
- **Visualization**: Comprehensive plots and charts for data exploration
- **Modular Design**: Separated into focused modules for maintainability

## Project Structure

```
├── main.py                 # Main orchestration script
├── data_preprocessing.py   # Data loading, cleaning, and feature engineering
├── clustering_analysis.py  # Clustering algorithms and analysis
├── chatgpt_analyzer.py     # ChatGPT-based quality analysis (optional)
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── data/                  # Data directory (not included in repo)
    ├── Posts.xml
    ├── Users.xml
    ├── Comments.xml
    └── ...
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Main-Project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Prepare your data:
   - Place Stack Overflow XML files in the `data/` directory
   - Required files: `Posts.xml`, `Users.xml`, `Comments.xml`, etc.

## Usage

### Basic Usage
Run the complete analysis pipeline:
```bash
python main.py
```

### Advanced Usage with Command Line Arguments

The script supports different modes and parameters:

```bash
# Run complete pipeline (default)
python main.py --mode full

# Run only data preprocessing and save results
python main.py --mode preprocess

# Run only clustering analysis (requires preprocessed data)
python main.py --mode cluster

# Run only visualizations (requires preprocessed data)
python main.py --mode visualize

# Force reprocessing even if cached data exists
python main.py --mode full --force-reprocess

# Specify custom parameters
python main.py --mode full --n-clusters 10 --max-features 2000 --n-components 150
```

### Command Line Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--mode` | `-m` | `full` | Analysis mode: `full`, `preprocess`, `cluster`, `visualize` |
| `--data-dir` | `-d` | `data` | Directory containing XML data files |
| `--output-dir` | `-o` | `output` | Directory to save/load processed data |
| `--force-reprocess` | `-f` | `False` | Force reprocessing even if cached data exists |
| `--max-features` | `-mf` | `1000` | Maximum number of TF-IDF features |
| `--n-components` | `-nc` | `100` | Number of TF-IDF components after SVD |
| `--n-clusters` | `-k` | `None` | Number of clusters for K-means (auto-determined if not specified) |

### Workflow Examples

**First time setup:**
```bash
# Run complete preprocessing and save results
python main.py --mode preprocess
```

**Subsequent runs:**
```bash
# Run clustering with different parameters
python main.py --mode cluster --n-clusters 15

# Run visualizations only
python main.py --mode visualize

# Run complete pipeline (will use cached data if available)
python main.py --mode full
```

**Force reprocessing:**
```bash
# Force complete reprocessing
python main.py --mode full --force-reprocess
```

### Processing Pipeline

The script follows this processing pipeline:

1. **Data Loading**: Parse XML files from the data directory
2. **Data Cleaning**: Remove duplicates, handle missing values, merge datasets
3. **Feature Engineering**: Create derived variables and categorical features
4. **TF-IDF Extraction**: Extract TF-IDF features from titles and tags with dimensionality reduction
5. **Semantic Embeddings**: Generate sentence embeddings using transformers
6. **Clustering Analysis**: Perform K-means and DBSCAN clustering
7. **Quality Assessment**: Calculate clustering quality metrics
8. **Visualization**: Generate comprehensive plots and charts

### Caching System

The script implements a caching system to avoid reprocessing:

- **First run**: Complete preprocessing and saves results to `output/` directory
- **Subsequent runs**: Loads cached data and skips preprocessing
- **Force reprocess**: Use `--force-reprocess` flag to regenerate all data

Cached files include:
- `df_combined.pkl`: Processed dataframe
- `embeddings.pkl`: Sentence embeddings
- `tfidf_features.pkl`: TF-IDF features
- `sentence_model.pkl`: Trained sentence transformer model

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **matplotlib & seaborn**: Data visualization
- **scikit-learn**: Machine learning algorithms (K-means, DBSCAN)
- **scipy**: Scientific computing
- **sentence-transformers**: Text embeddings
- **torch**: Deep learning framework
- **transformers**: Hugging Face transformers library
- **huggingface-hub**: Model hub integration
- **tqdm**: Progress bars
- **requests**: HTTP library

## Key Features

### Clustering Algorithms
- **K-means**: Centroid-based clustering with optimal cluster number selection
- **DBSCAN**: Density-based clustering with adaptive parameter estimation

### Adaptive Parameter Selection
- The project uses adaptive parameter estimation for DBSCAN
- K-means cluster number is determined based on data size and characteristics

### Quality Metrics
- Silhouette score for clustering quality assessment
- Cluster size distribution analysis
- Noise point analysis for DBSCAN

## Data Requirements

Place your Stack Overflow XML files in the `data/` directory. The script expects files like:
- `Posts.xml`
- `Users.xml`
- `Comments.xml`
- `Badges.xml`
- etc.

**Note**: Data files are not included in this repository due to size limitations. Please download the Stack Overflow data dump separately.

## Output

The analysis generates:
- Data preprocessing visualizations
- TF-IDF feature importance and distribution plots
- Cluster comparison plots
- Quality metrics comparison
- Cluster characteristic analysis
- Sample titles from each cluster

## Contributing

Feel free to submit issues and enhancement requests!