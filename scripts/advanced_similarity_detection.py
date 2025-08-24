import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import re
import warnings
warnings.filterwarnings('ignore')

class AdvancedSimilarityDetection:
    """Advanced methods for detecting posts similar to AI content"""
    
    def __init__(self):
        self.ai_keywords = {
            'core_ai': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'deep learning'],
            'technologies': ['neural network', 'algorithm', 'automation', 'chatbot', 'gpt', 'llm'],
            'applications': ['data science', 'predictive', 'automated', 'intelligent', 'smart'],
            'tools': ['tensorflow', 'pytorch', 'scikit-learn', 'openai', 'claude'],
            'concepts': ['nlp', 'computer vision', 'reinforcement learning', 'transformer']
        }
        
    def extract_text_features(self, df, text_columns):
        """Extract text features for similarity analysis"""
        print("Extracting text features...")
        
        # Combine all text columns
        combined_text = df[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
        
        # Clean text
        combined_text = combined_text.str.lower()
        combined_text = combined_text.str.replace(r'[^\w\s]', ' ', regex=True)
        combined_text = combined_text.str.replace(r'\s+', ' ', regex=True)
        
        return combined_text
    
    def calculate_keyword_similarity(self, df, text_columns):
        """Calculate similarity based on keyword matching"""
        print("Calculating keyword similarity...")
        
        # Extract text
        combined_text = self.extract_text_features(df, text_columns)
        
        # Calculate keyword scores
        keyword_scores = {}
        
        for category, keywords in self.ai_keywords.items():
            category_score = 0
            for keyword in keywords:
                # Count keyword occurrences
                matches = combined_text.str.contains(keyword, case=False, na=False)
                category_score += matches.sum()
            keyword_scores[category] = category_score
        
        # Create overall keyword score
        df['keyword_similarity'] = 0
        for category, score in keyword_scores.items():
            df[f'{category}_score'] = 0
            for keyword in self.ai_keywords[category]:
                matches = combined_text.str.contains(keyword, case=False, na=False)
                df.loc[matches, f'{category}_score'] += 1
                df.loc[matches, 'keyword_similarity'] += 1
        
        return df
    
    def calculate_tfidf_similarity(self, df, text_columns, treatment_posts):
        """Calculate similarity using TF-IDF and cosine similarity"""
        print("Calculating TF-IDF similarity...")
        
        # Extract text
        combined_text = self.extract_text_features(df, text_columns)
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        # Fit and transform text
        tfidf_matrix = vectorizer.fit_transform(combined_text)
        
        # Get treatment posts as reference
        treatment_text = combined_text[treatment_posts]
        treatment_tfidf = vectorizer.transform(treatment_text)
        
        # Calculate cosine similarity with treatment posts
        similarities = cosine_similarity(tfidf_matrix, treatment_tfidf)
        
        # Take maximum similarity to any treatment post
        df['tfidf_similarity'] = similarities.max(axis=1)
        
        return df
    
    def calculate_semantic_similarity(self, df, text_columns):
        """Calculate semantic similarity using content analysis"""
        print("Calculating semantic similarity...")
        
        # Extract text
        combined_text = self.extract_text_features(df, text_columns)
        
        # Analyze content patterns
        df['semantic_similarity'] = 0
        
        # Check for technical content patterns
        technical_patterns = [
            r'\b(code|programming|development|software|technical)\b',
            r'\b(algorithm|function|method|class|object)\b',
            r'\b(data|analysis|statistics|model|prediction)\b',
            r'\b(learning|training|testing|validation)\b'
        ]
        
        for pattern in technical_patterns:
            matches = combined_text.str.contains(pattern, case=False, na=False)
            df.loc[matches, 'semantic_similarity'] += 0.5
        
        # Check for AI-specific patterns
        ai_patterns = [
            r'\b(neural|network|layer|activation|backpropagation)\b',
            r'\b(training|epoch|batch|gradient|optimization)\b',
            r'\b(prediction|classification|regression|clustering)\b',
            r'\b(natural|language|processing|nlp)\b',
            r'\b(computer|vision|image|recognition)\b'
        ]
        
        for pattern in ai_patterns:
            matches = combined_text.str.contains(pattern, case=False, na=False)
            df.loc[matches, 'semantic_similarity'] += 1.0
        
        return df
    
    def cluster_based_similarity(self, df, text_columns, n_clusters=5):
        """Calculate similarity based on content clustering"""
        print("Calculating cluster-based similarity...")
        
        # Extract text
        combined_text = self.extract_text_features(df, text_columns)
        
        # Create TF-IDF features for clustering
        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = vectorizer.fit_transform(combined_text)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        df['content_cluster'] = clusters
        
        # Find which cluster contains most treatment posts
        treatment_clusters = df[df['treatment_ai_content'] == 1]['content_cluster'].value_counts()
        if len(treatment_clusters) > 0:
            main_treatment_cluster = treatment_clusters.index[0]
            
            # Calculate similarity based on cluster membership
            df['cluster_similarity'] = (df['content_cluster'] == main_treatment_cluster).astype(int)
        else:
            df['cluster_similarity'] = 0
        
        return df
    
    def calculate_comprehensive_similarity(self, df, text_columns):
        """Calculate comprehensive similarity score"""
        print("Calculating comprehensive similarity...")
        
        # Get treatment posts
        treatment_posts = df['treatment_ai_content'] == 1
        
        # Calculate different similarity measures
        df = self.calculate_keyword_similarity(df, text_columns)
        df = self.calculate_tfidf_similarity(df, text_columns, treatment_posts)
        df = self.calculate_semantic_similarity(df, text_columns)
        df = self.cluster_based_similarity(df, text_columns)
        
        # Combine similarity scores
        df['comprehensive_similarity'] = (
            df['keyword_similarity'] * 0.3 +
            df['tfidf_similarity'] * 0.4 +
            df['semantic_similarity'] * 0.2 +
            df['cluster_similarity'] * 0.1
        )
        
        return df
    
    def select_control_group(self, df, min_similarity_threshold=0.5):
        """Select control group based on comprehensive similarity"""
        print("Selecting control group...")
        
        # Calculate comprehensive similarity
        text_columns = [col for col in df.columns if any(x in col.lower() for x in ['content', 'text', 'title', 'tag'])]
        df = self.calculate_comprehensive_similarity(df, text_columns)
        
        # Find posts with high similarity but not in treatment
        high_similarity_posts = df[
            (df['comprehensive_similarity'] >= min_similarity_threshold) & 
            (df['treatment_ai_content'] == 0)
        ]
        
        medium_similarity_posts = df[
            (df['comprehensive_similarity'] >= min_similarity_threshold * 0.7) & 
            (df['treatment_ai_content'] == 0)
        ]
        
        treatment_count = (df['treatment_ai_content'] == 1).sum()
        
        print(f"High similarity posts (score >= {min_similarity_threshold}): {len(high_similarity_posts):,}")
        print(f"Medium similarity posts (score >= {min_similarity_threshold * 0.7}): {len(medium_similarity_posts):,}")
        
        # Select control group
        if len(high_similarity_posts) >= treatment_count * 0.5:
            control_group = high_similarity_posts
            print(f"Using high similarity posts as control group")
        elif len(medium_similarity_posts) >= treatment_count * 0.5:
            control_group = medium_similarity_posts
            print(f"Using medium similarity posts as control group")
        else:
            # Use all non-treatment posts but prioritize by similarity
            non_treatment_posts = df[df['treatment_ai_content'] == 0]
            control_group = non_treatment_posts.nlargest(treatment_count * 2, 'comprehensive_similarity')
            print(f"Using top similar non-treatment posts as control group")
        
        return df, control_group
    
    def analyze_similarity_distribution(self, df):
        """Analyze similarity score distribution"""
        print("\n=== Similarity Score Analysis ===")
        
        treatment_similarity = df[df['treatment_ai_content'] == 1]['comprehensive_similarity']
        control_similarity = df[df['treatment_ai_content'] == 0]['comprehensive_similarity']
        
        print(f"Treatment group similarity:")
        print(f"  Mean: {treatment_similarity.mean():.3f}")
        print(f"  Median: {treatment_similarity.median():.3f}")
        print(f"  Std: {treatment_similarity.std():.3f}")
        
        print(f"\nControl group similarity:")
        print(f"  Mean: {control_similarity.mean():.3f}")
        print(f"  Median: {control_similarity.median():.3f}")
        print(f"  Std: {control_similarity.std():.3f}")
        
        # Check if control group is sufficiently similar
        similarity_gap = treatment_similarity.mean() - control_similarity.mean()
        print(f"\nSimilarity gap (treatment - control): {similarity_gap:.3f}")
        
        if abs(similarity_gap) < 0.2:
            print("Control group is sufficiently similar to treatment group")
        else:
            print("Control group may not be similar enough to treatment group")

def advanced_similarity_detection():
    """Advanced similarity detection for uplift modeling"""
    print("=== Advanced Similarity Detection ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Initialize similarity detector
    detector = AdvancedSimilarityDetection()
    
    # Find text columns
    text_columns = [col for col in df.columns if any(x in col.lower() for x in ['content', 'text', 'title', 'tag'])]
    print(f"Text columns found: {text_columns}")
    
    if not text_columns:
        print("No text columns found")
        return None
    
    # Create initial treatment labels
    tag_columns = [col for col in df.columns if 'tag' in col.lower()]
    if tag_columns:
        tag_col = tag_columns[0]
        df['treatment_ai_content'] = df[tag_col].str.contains('ai content', case=False, na=False).astype(int)
    
    # Select control group using advanced similarity
    df, control_group = detector.select_control_group(df)
    
    # Analyze similarity distribution
    detector.analyze_similarity_distribution(df)
    
    # Create final dataset
    treatment_group = df[df['treatment_ai_content'] == 1]
    final_df = pd.concat([treatment_group, control_group], ignore_index=True)
    
    print(f"\nFinal dataset:")
    print(f"  Treatment samples: {len(treatment_group):,}")
    print(f"  Control samples: {len(control_group):,}")
    print(f"  Total samples: {len(final_df):,}")
    
    # Save results
    output_file = 'uplift_model_data_advanced_similarity.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\nAdvanced similarity dataset saved to: {output_file}")
    
    return final_df

if __name__ == "__main__":
    advanced_similarity_detection() 