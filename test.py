"""
Dataset Analysis Script - Class Distribution Table
Analyzes class distribution for all voter datasets
"""

import torch
from collections import Counter
import os


def load_dataset(filepath):
    """Load dataset and return class counts."""
    
    if not os.path.exists(filepath):
        return None
    
    try:
        data = torch.load(filepath, weights_only=False)
        labels = data["binary_labels"]
        class_counts = Counter(labels.cpu().numpy())
        return class_counts
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None


def main():
    """Analyze all datasets and display class distribution table."""
    
    print("\n" + "="*100)
    print("VOTER DATASET - CLASS DISTRIBUTION ANALYSIS")
    print("="*100 + "\n")
    
    # Define datasets
    datasets = [
        ("./data/kaleel_final_dataset_train_OnlyBubbles_Grayscale.pth", 
         "OnlyBubbles Train"),
        ("./data/kaleel_final_dataset_val_OnlyBubbles_Grayscale.pth", 
         "OnlyBubbles Val"),
        ("./data/kaleel_final_dataset_train_Combined_Grayscale.pth", 
         "Combined Train"),
        ("./data/kaleel_final_dataset_val_Combined_Grayscale.pth", 
         "Combined Val"),
    ]
    
    results = []
    
    # Load and analyze each dataset
    for filepath, dataset_name in datasets:
        class_counts = load_dataset(filepath)
        if class_counts:
            class_0 = class_counts.get(0, 0)
            class_1 = class_counts.get(1, 0)
            total = class_0 + class_1
            class_0_pct = (class_0 / total * 100) if total > 0 else 0
            class_1_pct = (class_1 / total * 100) if total > 0 else 0
            
            results.append({
                'name': dataset_name,
                'class_0': class_0,
                'class_1': class_1,
                'total': total,
                'class_0_pct': class_0_pct,
                'class_1_pct': class_1_pct
            })
    
    # Display table
    print(f"{'Dataset':<20} {'Class 0':<15} {'Class 1':<15} {'Total':<12} {'Class 0 %':<12} {'Class 1 %':<12}")
    print("-"*100)
    
    for result in results:
        print(f"{result['name']:<20} {result['class_0']:<15} {result['class_1']:<15} "
              f"{result['total']:<12} {result['class_0_pct']:<12.2f}% {result['class_1_pct']:<12.2f}%")
    
    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    main()