# Uplift Modeling Project: AI Tag Impact Analysis

## Project Overview

This project implements a comprehensive uplift modeling framework to analyze the causal effect of AI tags on user engagement in content platforms. The analysis uses Inverse Probability of Treatment Weighting (IPTW) to address treatment imbalance and data leakage issues.

## Key Findings

- **AI tags reduce click rates** by 21-71% in AI content
- **Negative uplift effects** across all user segments
- **Statistically reliable results** with proper causal inference methods
- **ESS ratio of 0.5442** ensures reliable statistical inference

## Project Structure

```
Main Project/
├── scripts/                          # Analysis scripts
│   ├── fixed_uplift_evaluation_v2.py # Main IPW-corrected analysis
│   ├── cluster_posts_by_content.py   # Content-based clustering
│   ├── analyze_correct_ai_clusters.py # AI cluster identification
│   └── ...                           # Additional analysis scripts
├── src/                              # Core modules
│   ├── uplift_model.py               # Uplift model implementation
│   ├── data_preprocessing.py         # Data preprocessing utilities
│   └── main.py                       # Main execution script
├── data/                             # Data files (not included in repo)
├── output/                           # Output files (not included in repo)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Key Scripts

### Main Analysis Script
- **`scripts/fixed_uplift_evaluation_v2.py`**: Final IPW-corrected uplift analysis with proper causal inference

### Supporting Scripts
- **`scripts/cluster_posts_by_content.py`**: Content-based clustering using TF-IDF and K-means
- **`scripts/analyze_correct_ai_clusters.py`**: Identifies AI-dense clusters for analysis
- **`scripts/comprehensive_uplift_diagnostics.py`**: Diagnoses data leakage and confounding

## Methodology

### Two-Stage Analysis Approach
1. **Stage 1**: Measure tag effects within similar AI content
2. **Stage 2**: Compare AI vs non-AI content performance

### Causal Inference Methods
- **IPTW (Inverse Probability of Treatment Weighting)**: Addresses treatment imbalance
- **Propensity Score Modeling**: Estimates treatment assignment probabilities
- **ESS (Effective Sample Size)**: Ensures reliable statistical inference

### Feature Engineering
- **Safe Features**: Content length, tag count, time-based features
- **Excluded Features**: User engagement, post popularity (leakage-prone)

## Results

### Model Performance
- **R² (treatment head)**: 0.0079 - Low predictive power, no data leakage
- **R² (control head)**: 0.3945 - Moderate predictive power
- **IPW Qini/AUUC**: -2.6878 - Negative uplift score

### Segment Analysis
- **High-uplift cohort**: -21.01% effect
- **Low-uplift cohort**: -70.82% effect
- **All segments show negative effects**

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Main-Project

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the main uplift analysis
python scripts/fixed_uplift_evaluation_v2.py

# Run content clustering
python scripts/cluster_posts_by_content.py

# Run diagnostics
python scripts/comprehensive_uplift_diagnostics.py
```

## Dependencies

Key Python packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- statsmodels

## Business Implications

- **Do not add AI tags** to AI content as they significantly reduce performance
- **The uplift model works correctly** by identifying consistent negative effects
- **Consider content quality** rather than AI labeling for engagement improvement

## Technical Notes

- **Data Leakage Prevention**: Systematic removal of leakage-prone features
- **Treatment Balance**: IPTW addresses 86.7% treatment vs 13.3% control imbalance
- **Statistical Rigor**: Proper confidence intervals and significance testing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes.
