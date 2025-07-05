#!/usr/bin/env python3
"""
Main script for Stack Overflow Data Analysis
Includes data preprocessing and CTR/Retention/Uplift modeling
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
from pathlib import Path
import seaborn as sns
from data_preprocessing import DataPreprocessor

import warnings
warnings.filterwarnings('ignore')

import sys
from tqdm import tqdm
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Stack Overflow Data Analysis Pipeline')
    parser.add_argument('mode', nargs='?', choices=['preprocess', 'ctr', 'retention', 'uplift'], 
                       default='preprocess', help='Pipeline mode to run')
    parser.add_argument('--data-dir', default='data', help='Directory containing XML data files')
    parser.add_argument('--cache-file', default='processed_data.pkl', help='Cache file for processed data')
    parser.add_argument('--force-reprocess', action='store_true', 
                       help='Force reprocessing even if cache exists')
    
    return parser.parse_args()

def save_processed_data(data, cache_file):
    """Save processed data to a single pickle file"""
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(exist_ok=True)
    
    # Save all data to a single pickle file
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Processed data saved to {cache_path}")

def load_processed_data(cache_file):
    """Load processed data from a single pickle file"""
    cache_path = Path(cache_file)
    
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file {cache_file} does not exist")
    
    # Load all data from a single pickle file
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Processed data loaded from {cache_path}")
    
    # Handle both old and new data formats
    if isinstance(data, dict) and 'df_combined' in data:
        # New format with normalization data
        return (data['df_combined'], data['embeddings'], 
                data['tfidf_features'], data['model'])
    else:
        # Old format (backward compatibility)
        return data

def run_preprocessing(args):
    """Run data preprocessing only"""
    print("=== Running Data Preprocessing Only ===")
    
    with tqdm(total=3, desc="Preprocessing pipeline") as pbar:
        # Initialize preprocessor
        preprocessor = DataPreprocessor(base_path=args.data_dir)
        pbar.update(1)
        
        # Load and process data
        df_combined, embeddings, tfidf_features, model = preprocessor.run_full_pipeline()
        pbar.update(1)
        
        # Create additional samples
        preprocessor.create_retention_samples()
        preprocessor.create_retention_duration_samples()
        preprocessor.create_uplift_samples()
        pbar.update(1)
    
    return {
        'df_combined': df_combined,
        'embeddings': embeddings,
        'tfidf_features': tfidf_features,
        'model': model,
        'preprocessor': preprocessor
    }

def run_ctr_prediction(args):
    """Run CTR prediction only"""
    print("=== Running CTR Prediction Only ===")
    
    try:
        from ctr_model_training import CTRModelTrainer
        trainer = CTRModelTrainer()
        models, results = trainer.run_full_pipeline()
        
        return {
            'models': models,
            'results': results,
            'trainer': trainer
        }
        
    except Exception as e:
        print(f"Error in CTR prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_retention_prediction(args):
    """Run retention prediction labeling"""
    print("=== Running Retention Prediction Labeling ===")
    
    from retention_prediction_labeling import RetentionPredictionLabeling
    
    labeler = RetentionPredictionLabeling(data_dir=args.data_dir)
    retention_data = labeler.run_full_pipeline(retention_days=7)
    
    return {
        'retention_data': retention_data,
        'labeler': labeler
    }

def run_uplift_modeling(args):
    """Run uplift modeling only"""
    print("=== Running Uplift Modeling Only ===")
    
    try:
        from uplift_model_training import UpliftModelTrainer
        trainer = UpliftModelTrainer()
        models, results = trainer.run_full_pipeline()
        
        return {
            'models': models,
            'results': results,
            'trainer': trainer
        }
        
    except Exception as e:
        print(f"Error in uplift modeling: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    args = parse_arguments()
    
    print("=== Stack Overflow Data Analysis Pipeline ===")
    print(f"Mode: {args.mode}")
    print(f"Data directory: {args.data_dir}")
    print(f"Cache file: {args.cache_file}")
    
    try:
        if args.mode == 'preprocess':
            # Run preprocessing only
            data = run_preprocessing(args)
            
        elif args.mode == 'ctr':
            # Run CTR prediction only
            data = run_ctr_prediction(args)
            
        elif args.mode == 'retention':
            # Run retention prediction only
            data = run_retention_prediction(args)
            
        elif args.mode == 'uplift':
            # Run uplift modeling only
            data = run_uplift_modeling(args)
            
        print(f"\n=== Pipeline completed successfully ===")
        
        # Save results summary
        save_results_summary(data, args.mode)
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def save_results_summary(data, mode):
    """Save a summary of the results"""
    import json
    from datetime import datetime
    
    summary = {
        'mode': mode,
        'timestamp': datetime.now().isoformat(),
        'results': {}
    }
    
    if mode == 'all':
        # Extract key metrics from each component
        if 'ctr' in data and data['ctr'].get('ctr_results'):
            summary['results']['ctr'] = {
                'accuracy': data['ctr']['ctr_results'].get('accuracy', 'N/A'),
                'model_type': 'CTR Prediction'
            }
        
        if 'retention' in data and data['retention'].get('retention_results'):
            summary['results']['retention'] = {
                'accuracy': data['retention']['retention_results'].get('accuracy', 'N/A'),
                'model_type': 'Retention Prediction'
            }
        
        if 'uplift' in data and data['uplift'].get('uplift_results'):
            summary['results']['uplift'] = {
                'control_mse': data['uplift']['uplift_results'].get('control_mse', 'N/A'),
                'treatment_mse': data['uplift']['uplift_results'].get('treatment_mse', 'N/A'),
                'model_type': 'Uplift Modeling'
            }
    
    # Save summary
    os.makedirs('output', exist_ok=True)
    with open('output/results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results summary saved to: output/results_summary.json")

# Example usage of feature normalization
def example_feature_normalization():
    """Example of how to use feature normalization"""
    print("=== Feature Normalization Example ===")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Option 1: Run complete preprocessing with normalization
    print("\nOption 1: Complete preprocessing with normalization")
    df_combined, embeddings, tfidf_features, model = preprocessor.preprocess_all(
        include_normalization=True,
        normalization_config={
            'numerical_method': 'standard',
            'categorical_method': 'label',
            'create_interactions': True,
            'create_polynomials': True,
            'polynomial_degree': 2
        }
    )
    
    # Option 2: Run preprocessing without normalization first, then add it
    print("\nOption 2: Add normalization after preprocessing")
    preprocessor2 = DataPreprocessor()
    df_combined2, embeddings2, tfidf_features2, model2 = preprocessor2.preprocess_all(
        include_normalization=False
    )
    
    # Then add normalization
    df_normalized = preprocessor2.normalize_features()
    
    # Option 3: Custom normalization configuration
    print("\nOption 3: Custom normalization configuration")
    custom_config = {
        'numerical_method': 'robust',  # Use RobustScaler for numerical features
        'categorical_method': 'onehot',  # Use OneHotEncoder for categorical features
        'create_interactions': True,
        'create_polynomials': False,  # Don't create polynomial features
        'polynomial_degree': 2
    }
    
    df_custom = preprocessor2.normalize_features(custom_config)
    
    print("\nFeature normalization examples completed!")
    return df_combined, df_normalized, df_custom

if __name__ == "__main__":
    main()