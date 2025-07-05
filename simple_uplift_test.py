#!/usr/bin/env python3
"""
Simple Uplift Test
"""

import numpy as np
import pandas as pd

def test_uplift_basics():
    """Test basic uplift concepts"""
    print("=== Simple Uplift Test ===")
    
    # Create simple data
    np.random.seed(42)
    n_samples = 100
    
    # Treatment group (recommended)
    treatment_click_rate = 0.45
    treatment_clicks = np.random.binomial(1, treatment_click_rate, n_samples)
    
    # Control group (not recommended)
    control_click_rate = 0.30
    control_clicks = np.random.binomial(1, control_click_rate, n_samples)
    
    print(f"Treatment group click rate: {treatment_click_rate:.3f}")
    print(f"Control group click rate: {control_click_rate:.3f}")
    print(f"Uplift: {treatment_click_rate - control_click_rate:.3f}")
    print(f"Uplift percentage: {(treatment_click_rate - control_click_rate)/control_click_rate*100:.1f}%")
    
    # Create DataFrame
    df = pd.DataFrame({
        'treatment': [1] * n_samples + [0] * n_samples,
        'is_click': np.concatenate([treatment_clicks, control_clicks]),
        'Score': np.random.normal(5, 3, 2*n_samples),
        'Reputation': np.random.exponential(1000, 2*n_samples),
        'ViewCount': np.random.exponential(100, 2*n_samples)
    })
    
    print(f"\nCreated {len(df)} samples")
    print(f"Treatment group: {df['treatment'].sum()}")
    print(f"Control group: {(df['treatment'] == 0).sum()}")
    print(f"Overall click rate: {df['is_click'].mean():.3f}")
    
    # Calculate actual uplift
    actual_treatment_rate = df[df['treatment'] == 1]['is_click'].mean()
    actual_control_rate = df[df['treatment'] == 0]['is_click'].mean()
    actual_uplift = actual_treatment_rate - actual_control_rate
    
    print(f"\nActual Results:")
    print(f"Treatment click rate: {actual_treatment_rate:.3f}")
    print(f"Control click rate: {actual_control_rate:.3f}")
    print(f"Actual uplift: {actual_uplift:.3f}")
    print(f"Actual uplift percentage: {actual_uplift/actual_control_rate*100:.1f}%")
    
    return {
        'treatment_rate': actual_treatment_rate,
        'control_rate': actual_control_rate,
        'uplift': actual_uplift,
        'uplift_percentage': actual_uplift/actual_control_rate*100
    }

if __name__ == "__main__":
    result = test_uplift_basics()
    print(f"\n=== Test Complete ===")
    print(f"Uplift achieved: {result['uplift']:.3f} ({result['uplift_percentage']:.1f}%)") 