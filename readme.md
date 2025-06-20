# Stack Overflow Data Analysis Project

This project performs comprehensive analysis of Stack Overflow data using machine learning techniques including clustering and natural language processing.

## Features

- **Data Preprocessing**: XML parsing, cleaning, and feature engineering
- **Semantic Analysis**: Text embeddings using sentence transformers
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
├── chatgpt_analyzer.py     # ChatGPT-based quality analysis (currently disabled)
├── data/                   # XML data files
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main analysis script:
```bash
python main.py
```

This will:
1. Load and preprocess Stack Overflow XML data
2. Generate text embeddings for semantic analysis
3. Perform K-means clustering with optimal cluster selection
4. Perform DBSCAN clustering with adaptive parameters
5. Compare clustering results and generate visualizations
6. Display quality metrics for both algorithms

## Data Requirements

Place your Stack Overflow XML files in the `data/` directory. The script expects files like:
- `Posts.xml`
- `Users.xml`
- `Comments.xml`
- `Badges.xml`
- etc.

## Output

The analysis generates:
- Data preprocessing visualizations
- Clustering results with quality metrics
- Comparison plots between K-means and DBSCAN
- Cluster characteristic analysis
- Performance metrics and statistics

## Dependencies

- numpy: Numerical computing
- pandas: Data manipulation
- matplotlib & seaborn: Visualization
- scikit-learn: Machine learning algorithms
- sentence-transformers: Text embeddings
- torch: Deep learning framework
- hdbscan: Density-based clustering
- transformers: Hugging Face transformers library

## Notes

- The ChatGPT analysis component is currently disabled
- The project uses adaptive parameter estimation for DBSCAN
- All visualizations are automatically saved and displayed
- The modular design allows easy extension and modification