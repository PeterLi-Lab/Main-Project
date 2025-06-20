#!/usr/bin/env python3
"""
Test script to verify all imports and basic functionality
"""

def test_imports():
    """Test all necessary imports"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ seaborn imported successfully")
    except ImportError as e:
        print(f"✗ seaborn import failed: {e}")
        return False
    
    try:
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.metrics.pairwise import cosine_distances
        print("✓ scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ scikit-learn import failed: {e}")
        return False
    
    try:
        import hdbscan
        print("✓ hdbscan imported successfully")
    except ImportError as e:
        print(f"✗ hdbscan import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✓ sentence-transformers imported successfully")
    except ImportError as e:
        print(f"✗ sentence-transformers import failed: {e}")
        return False
    
    try:
        import torch
        print("✓ torch imported successfully")
    except ImportError as e:
        print(f"✗ torch import failed: {e}")
        return False
    
    try:
        import transformers
        print("✓ transformers imported successfully")
    except ImportError as e:
        print(f"✗ transformers import failed: {e}")
        return False
    
    try:
        import lxml
        print("✓ lxml imported successfully")
    except ImportError as e:
        print(f"✗ lxml import failed: {e}")
        return False
    
    try:
        import nltk
        print("✓ nltk imported successfully")
    except ImportError as e:
        print(f"✗ nltk import failed: {e}")
        return False
    
    return True

def test_module_imports():
    """Test importing our custom modules"""
    print("\nTesting custom module imports...")
    
    try:
        from data_preprocessing import DataPreprocessor
        print("✓ DataPreprocessor imported successfully")
    except ImportError as e:
        print(f"✗ DataPreprocessor import failed: {e}")
        return False
    
    try:
        from clustering_analysis import ClusteringAnalyzer, analyze_clustering_quality
        print("✓ ClusteringAnalyzer imported successfully")
    except ImportError as e:
        print(f"✗ ClusteringAnalyzer import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without data"""
    print("\nTesting basic functionality...")
    
    try:
        # Test DataPreprocessor initialization
        from data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        print("✓ DataPreprocessor initialized successfully")
    except Exception as e:
        print(f"✗ DataPreprocessor initialization failed: {e}")
        return False
    
    try:
        # Test ClusteringAnalyzer initialization (with dummy data)
        from clustering_analysis import ClusteringAnalyzer
        import pandas as pd
        import numpy as np
        
        # Create dummy data
        dummy_df = pd.DataFrame({
            'Title': ['test1', 'test2'],
            'post_length': [10, 20],
            'vote_ratio': [0.5, 0.8],
            'total_votes': [10, 20]
        })
        dummy_embeddings = np.random.rand(2, 384)
        
        analyzer = ClusteringAnalyzer(dummy_df, dummy_embeddings)
        print("✓ ClusteringAnalyzer initialized successfully")
    except Exception as e:
        print(f"✗ ClusteringAnalyzer initialization failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== Import and Functionality Test ===\n")
    
    # Test basic imports
    if not test_imports():
        print("\n❌ Basic imports failed!")
        exit(1)
    
    # Test module imports
    if not test_module_imports():
        print("\n❌ Module imports failed!")
        exit(1)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality failed!")
        exit(1)
    
    print("\n✅ All tests passed! The code should work correctly.")
    print("\nYou can now run the pipeline with:")
    print("  python main.py --mode preprocess")
    print("  python main.py --mode cluster")
    print("  python main.py --mode combined")
    print("  python main.py --mode all") 