#!/usr/bin/env python3
"""
Uplift Treatment Labeling Script
Adds treatment labels to user-post click dataset based on content tags
"""

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

class UpliftTreatmentLabeling:
    def __init__(self):
        """Initialize with treatment configurations"""
        self.df_samples = None
        self.post_tags = {}
        
        # Define treatment configurations
        self.treatments = {
            'ai_content': {
                'tags': ['ai', 'artificial-intelligence', 'machine-learning', 'deep-learning', 
                        'neural-network', 'tensorflow', 'pytorch', 'scikit-learn', 'nlp', 
                        'computer-vision', 'data-science'],
                'description': 'AI/ML related content'
            },
            'web_development': {
                'tags': ['javascript', 'html', 'css', 'react', 'angular', 'vue', 'nodejs', 
                        'web-development', 'frontend', 'backend', 'api'],
                'description': 'Web development content'
            },
            'mobile_development': {
                'tags': ['android', 'ios', 'swift', 'kotlin', 'react-native', 'flutter', 
                        'mobile-development', 'app-development'],
                'description': 'Mobile development content'
            },
            'database': {
                'tags': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'database', 
                        'nosql', 'oracle'],
                'description': 'Database related content'
            }
        }
    
    def parse_tags(self, tags_str):
        """Parse tags from pipe-separated format: '|tag1|tag2|' -> ['tag1', 'tag2']"""
        if not tags_str or tags_str == '':
            return []
        # Remove leading and trailing pipes, then split by pipe
        cleaned_tags = tags_str.strip('|')
        if not cleaned_tags:
            return []
        return [tag for tag in cleaned_tags.split('|') if tag]
    
    def load_data(self):
        """Load user-post samples and post tags"""
        print("=== Loading Data ===")
        
        # Load user-post samples
        print("Loading user-post samples...")
        self.df_samples = pd.read_csv('user_post_click_samples.csv')
        print(f"Loaded {len(self.df_samples)} samples")
        
        # Load post tags from XML
        print("Loading post tags...")
        tree = ET.parse('data/Posts.xml')
        root = tree.getroot()
        
        for row in root:
            post_id = row.get('Id')
            tags_str = row.get('Tags', '')
            parsed_tags = self.parse_tags(tags_str)
            self.post_tags[post_id] = parsed_tags
        
        print(f"Loaded tags for {len(self.post_tags)} posts")
        
        # Debug: Show some sample parsed tags
        print("Sample parsed tags:")
        sample_posts = list(self.post_tags.items())[:5]
        for post_id, tags in sample_posts:
            print(f"  Post {post_id}: {tags}")
        
        return True
    
    def add_treatment_labels(self):
        """Add treatment labels based on post tags"""
        print("\n=== Adding Treatment Labels ===")
        
        # Add post tags to samples - convert post_id to string for mapping
        self.df_samples['post_id_str'] = self.df_samples['post_id'].astype(str)
        self.df_samples['post_tags'] = self.df_samples['post_id_str'].map(self.post_tags)
        self.df_samples['post_tags'] = self.df_samples['post_tags'].fillna('[]')
        
        # Debug: Check some sample tags
        print("Sample post tags:")
        sample_tags = self.df_samples['post_tags'].head(5).tolist()
        for i, tags in enumerate(sample_tags):
            print(f"  Sample {i+1}: {tags}")
        
        # Add treatment labels for each configuration
        for treatment_name, config in self.treatments.items():
            print(f"Adding {treatment_name} treatment...")
            
            # Check if any tag matches treatment tags (case-insensitive)
            def check_treatment_match(tags):
                if not tags or tags == '[]':
                    return False
                # Convert tags to lowercase for matching
                tags_lower = [tag.lower() for tag in tags]
                config_tags_lower = [tag.lower() for tag in config['tags']]
                return any(tag in config_tags_lower for tag in tags_lower)
            
            self.df_samples[f'treatment_{treatment_name}'] = self.df_samples['post_tags'].apply(check_treatment_match).astype(int)
            
            # Print statistics
            treatment_count = self.df_samples[f'treatment_{treatment_name}'].sum()
            treatment_rate = treatment_count / len(self.df_samples)
            print(f"  - {treatment_name}: {treatment_count} samples ({treatment_rate:.3f})")
            
            # Debug: Show some matched samples
            if treatment_count > 0:
                matched_samples = self.df_samples[self.df_samples[f'treatment_{treatment_name}'] == 1].head(3)
                print(f"  - Sample matched tags:")
                for _, row in matched_samples.iterrows():
                    print(f"    Post {row['post_id']}: {row['post_tags']}")
            else:
                print(f"  - No samples matched for {treatment_name}")
        
        # Debug: Show overall treatment distribution
        print("\nOverall treatment distribution:")
        for treatment_name in self.treatments.keys():
            treatment_col = f'treatment_{treatment_name}'
            if treatment_col in self.df_samples.columns:
                print(f"  {treatment_name}: {self.df_samples[treatment_col].value_counts().to_dict()}")
        
        return True
    
    def create_uplift_features(self):
        """Create features for uplift modeling"""
        print("\n=== Creating Uplift Features ===")
        
        # Use click behavior as response
        self.df_samples['response'] = self.df_samples['is_click']
        
        # Create interaction features
        self.df_samples['user_post_interaction'] = (
            self.df_samples['user_post_count'] * self.df_samples['post_title_length']
        )
        
        # Create engagement score
        self.df_samples['engagement_score'] = (
            self.df_samples['is_click'] * 1.0 + 
            self.df_samples['interest_score'] * 0.5
        )
        
        print("Uplift features created!")
        return True
    
    def analyze_treatment_effects(self):
        """Analyze treatment effects"""
        print("\n=== Treatment Effect Analysis ===")
        
        for treatment_name in self.treatments.keys():
            treatment_col = f'treatment_{treatment_name}'
            
            if treatment_col not in self.df_samples.columns:
                continue
            
            # Split into treatment and control groups
            treatment_group = self.df_samples[self.df_samples[treatment_col] == 1]
            control_group = self.df_samples[self.df_samples[treatment_col] == 0]
            
            if len(treatment_group) > 0 and len(control_group) > 0:
                treatment_rate = treatment_group['is_click'].mean()
                control_rate = control_group['is_click'].mean()
                uplift = treatment_rate - control_rate
                
                print(f"\n{treatment_name.upper()} Treatment Effect:")
                print(f"  - Treatment group: {len(treatment_group)} samples")
                print(f"  - Control group: {len(control_group)} samples")
                print(f"  - Treatment response rate: {treatment_rate:.3f}")
                print(f"  - Control response rate: {control_rate:.3f}")
                print(f"  - Uplift: {uplift:.3f}")
        
        return True
    
    def save_uplift_dataset(self, output_path='uplift_dataset.csv'):
        """Save the uplift dataset"""
        print(f"\n=== Saving Uplift Dataset ===")
        
        # Select relevant columns
        feature_columns = [
            'user_id', 'post_id', 'is_click', 'response', 'engagement_score',
            'user_post_count', 'user_account_age_days', 'post_title_length', 
            'post_tag_count', 'interest_score', 'user_post_interaction'
        ]
        
        # Add treatment columns
        treatment_columns = [f'treatment_{name}' for name in self.treatments.keys()]
        feature_columns.extend(treatment_columns)
        
        # Create and save uplift dataset
        uplift_dataset = self.df_samples[feature_columns].copy()
        uplift_dataset.to_csv(output_path, index=False)
        
        print(f"Uplift dataset saved to {output_path}")
        print(f"Dataset shape: {uplift_dataset.shape}")
        
        return uplift_dataset
    
    def save_standard_uplift_table(self, output_path='uplift_model_data.csv'):
        """Save standard uplift modeling table: user_id, post_id, treatment, response (AI only)"""
        print(f"\n=== Saving Standard Uplift Table (AI treatment only) ===")
        # treatment=1: AI-related content, 0: normal content
        self.df_samples['treatment'] = self.df_samples['treatment_ai_content']
        self.df_samples['response'] = self.df_samples['is_click']
        uplift_table = self.df_samples[['user_id', 'post_id', 'treatment', 'response']].copy()
        uplift_table.to_csv(output_path, index=False)
        print(f"Standard uplift table saved to {output_path}, shape: {uplift_table.shape}")
        return uplift_table
    
    def run_pipeline(self):
        """Run the complete uplift treatment labeling pipeline"""
        print("=== Uplift Treatment Labeling Pipeline ===")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Add treatment labels
        self.add_treatment_labels()
        
        # Step 3: Create uplift features
        self.create_uplift_features()
        
        # Step 4: Analyze treatment effects
        self.analyze_treatment_effects()
        
        # Step 5: Save dataset
        uplift_dataset = self.save_uplift_dataset()
        
        # Step 6: Save standard uplift modeling table (AI only)
        self.save_standard_uplift_table()
        
        print("\n=== Pipeline Complete ===")
        return uplift_dataset

def main():
    """Main function"""
    labeler = UpliftTreatmentLabeling()
    uplift_dataset = labeler.run_pipeline()
    return uplift_dataset

if __name__ == "__main__":
    main() 