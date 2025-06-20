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

Run the main analysis script:
```bash
python main.py
```

The script will:
1. Load and preprocess Stack Overflow data
2. Perform feature engineering and text embeddings
3. Extract TF-IDF features from titles and tags
4. Create visualizations for data exploration
5. Perform K-means clustering with optimal cluster selection
6. Perform DBSCAN clustering with adaptive parameters
7. Compare clustering algorithms and show quality metrics
8. Analyze cluster characteristics and provide insights

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