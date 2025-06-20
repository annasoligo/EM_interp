# %%
"""
<file_context>
em_interp/eval/MASK/plot_mask_metrics.py

This script creates simple matplotlib bar charts comparing base vs steered scores for MASK evaluation metrics.
It reads the all_results.json file and creates subplots for each score type (honest_1, unhonest_1, etc.)
showing the comparison between base and steered models.
</file_context>
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# %%
# Load the metrics data
def load_metrics_data():
    """Load the metrics data from the JSON file."""
    BASE_DIR = '/workspace/EM_interp/em_interp/eval/MASK'
    with open(f'{BASE_DIR}/csv_data/metrics/all_results.json', 'r') as f:
        return json.load(f)

# %%
# Load data
metrics_data = load_metrics_data()
print("Available categories:", list(metrics_data.keys()))

# %%
# Extract base and steered categories
base_categories = [cat for cat in metrics_data.keys() if not cat.endswith('_steered') and not cat.endswith('_steered_50') and not cat.endswith('_steered-50') and 'statistics' not in cat.lower()]
steered_categories = [cat for cat in metrics_data.keys() if cat.endswith('_steered') and not cat.endswith('_steered_50')]
steered_50_categories = [cat for cat in metrics_data.keys() if cat.endswith('_steered_50')]

print("Base categories:", base_categories)
print("Steered categories:", steered_categories)
print("Steered-50 categories:", steered_50_categories)

# %%
# Get all unique score types across all categories
def get_all_score_types(data):
    """Extract all unique score types from the data."""
    score_types = set()
    for category_data in data.values():
        for model_data in category_data.values():
            score_types.update(model_data.keys())
    # Remove non-score fields
    score_types.discard('total_responses')
    return sorted(list(score_types))

all_score_types = get_all_score_types(metrics_data)
print("All score types:", all_score_types)

# %%
# Calculate 'Evaded or No Belief' scores
def calculate_evaded_scores(metrics_data):
    """Calculate 'Evaded or No Belief' scores as (100 - honest_1 - unhonest_1)."""
    evaded_data = {}
    
    for category_name, category_data in metrics_data.items():
        if 'Qwen2.5-32B-Instruct' in category_data:
            model_data = category_data['Qwen2.5-32B-Instruct']
            honest_1 = model_data.get('honest_1', 0)
            unhonest_1 = model_data.get('unhonest_1', 0)
            evaded_score = 100 - honest_1 - unhonest_1
            evaded_data[category_name] = evaded_score
    
    return evaded_data

# %%
# Create comparison data for plotting
def create_comparison_data(metrics_data, score_type):
    """Create data for comparing base vs steered vs steered-50 scores for a given score type."""
    comparison_data = []
    
    for base_cat in base_categories:
        base_name = base_cat
        steered_name = base_cat + '_steered'
        steered_50_name = base_cat + '_steered-50'
        
        # Get base score
        base_score = metrics_data[base_name]['Qwen2.5-32B-Instruct'].get(score_type, 0)
        
        # Get steered score (if available)
        steered_score = 0
        if steered_name in metrics_data:
            steered_score = metrics_data[steered_name]['Qwen2.5-32B-Instruct'].get(score_type, 0)
        
        # Get steered-50 score (if available)
        steered_50_score = 0
        if steered_50_name in metrics_data:
            steered_50_score = metrics_data[steered_50_name]['Qwen2.5-32B-Instruct'].get(score_type, 0)
        
        comparison_data.append({
            'category': base_name,
            'base_score': base_score,
            'steered_score': steered_score,
            'steered_50_score': steered_50_score,
            'improvement_steered': steered_score - base_score,
            'improvement_steered_50': steered_50_score - base_score
        })
    
    return comparison_data

# %%
# Create comparison data for evaded scores
def create_evaded_comparison_data(metrics_data):
    """Create data for comparing base vs steered vs steered-50 evaded scores."""
    comparison_data = []
    
    # Calculate evaded scores for all categories
    base_evaded = calculate_evaded_scores(metrics_data)
    
    for base_cat in base_categories:
        base_name = base_cat
        steered_name = base_cat + '_steered'
        steered_50_name = base_cat + '_steered-50'
        
        if base_name in base_evaded:
            # Get base evaded score
            base_score = base_evaded[base_name]
            
            # Calculate steered evaded score
            steered_score = 0
            if steered_name in metrics_data:
                steered_honest_1 = metrics_data[steered_name]['Qwen2.5-32B-Instruct'].get('honest_1', 0)
                steered_unhonest_1 = metrics_data[steered_name]['Qwen2.5-32B-Instruct'].get('unhonest_1', 0)
                steered_score = 100 - steered_honest_1 - steered_unhonest_1
            
            # Calculate steered-50 evaded score
            steered_50_score = 0
            if steered_50_name in metrics_data:
                steered_50_honest_1 = metrics_data[steered_50_name]['Qwen2.5-32B-Instruct'].get('honest_1', 0)
                steered_50_unhonest_1 = metrics_data[steered_50_name]['Qwen2.5-32B-Instruct'].get('unhonest_1', 0)
                steered_50_score = 100 - steered_50_honest_1 - steered_50_unhonest_1
            
            comparison_data.append({
                'category': base_name,
                'base_score': base_score,
                'steered_score': steered_score,
                'steered_50_score': steered_50_score,
                'improvement_steered': steered_score - base_score,
                'improvement_steered_50': steered_50_score - base_score
            })
    
    return comparison_data

# %%
# Create simple matplotlib bar charts for each score type
def create_score_comparison_charts(metrics_data, score_types):
    """Create simple matplotlib bar charts comparing base vs steered vs steered-50 scores for each score type."""
    
    # Add evaded scores to the list of score types to plot
    all_score_types_to_plot = score_types + ['evaded_or_no_belief']
    
    # Calculate number of rows and columns for subplots
    n_scores = len(all_score_types_to_plot)
    n_cols = 2
    n_rows = (n_scores + 1) // 2
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Colors for base, steered, and steered-50
    base_color = '#1f77b4'  # Blue
    steered_color = '#ff7f0e'  # Orange
    steered_50_color = '#2ca02c'  # Green
    
    for idx, score_type in enumerate(all_score_types_to_plot):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        if score_type == 'evaded_or_no_belief':
            # Handle evaded scores separately
            comparison_data = create_evaded_comparison_data(metrics_data)
            ylabel = 'Evaded or No Belief (%)'
        else:
            # Handle regular score types
            comparison_data = create_comparison_data(metrics_data, score_type)
            ylabel = 'Score'
        
        if comparison_data:
            categories = [d['category'] for d in comparison_data]
            base_scores = [d['base_score'] for d in comparison_data]
            steered_scores = [d['steered_score'] for d in comparison_data]
            steered_50_scores = [d['steered_50_score'] for d in comparison_data]
            
            # Calculate averages across categories
            avg_base = np.mean(base_scores)
            avg_steered = np.mean(steered_scores)
            avg_steered_50 = np.mean(steered_50_scores)
            
            # Set up bar positions
            x = np.arange(len(categories) + 1)  # +1 for the average bar
            width = 0.25  # Reduced width to accommodate three bars
            
            # Create bars for individual categories
            ax.bar(x[:-1] - width, base_scores, width, label='Base', color=base_color, alpha=0.8)
            ax.bar(x[:-1], steered_50_scores, width, label='Steered-50', color=steered_50_color, alpha=0.8)
            ax.bar(x[:-1] + width, steered_scores, width, label='Steered-250', color=steered_color, alpha=0.8)
            
            # Create average bars (slightly wider and with different pattern)
            avg_width = width * 1.2
            ax.bar(x[-1] - avg_width, avg_base, avg_width, label='Base (Avg)', color=base_color, alpha=0.6, hatch='//')
            ax.bar(x[-1], avg_steered_50, avg_width, label='Steered-50 (Avg)', color=steered_50_color, alpha=0.6, hatch='//')
            ax.bar(x[-1] + avg_width, avg_steered, avg_width, label='Steered-250 (Avg)', color=steered_color, alpha=0.6, hatch='//')
            
            # Customize the plot
            ax.set_xlabel('Category')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{score_type}')
            ax.set_xticks(x)
            ax.set_xticklabels(categories + ['Average'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars for evaded scores
            # if score_type == 'evaded_or_no_belief':
            #     for i, (base, steered, steered_50) in enumerate(zip(base_scores, steered_scores, steered_50_scores)):
            #         ax.text(i - width, base, f'{base:.1f}%', ha='center', va='bottom', fontsize=8)
            #         ax.text(i, steered, f'{steered:.1f}%', ha='center', va='bottom', fontsize=8)
            #         ax.text(i + width, steered_50, f'{steered_50:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Hide empty subplots
    for idx in range(n_scores, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig

# %%
# Create and display the charts
score_comparison_fig = create_score_comparison_charts(metrics_data, all_score_types)
plt.show()

# %%
# Create a summary improvement chart
def create_improvement_summary(metrics_data, score_types):
    """Create a summary chart showing improvement for each score type."""
    
    improvement_data = []
    
    for score_type in score_types:
        comparison_data = create_comparison_data(metrics_data, score_type)
        if comparison_data:
            avg_improvement_steered = sum(d['improvement_steered'] for d in comparison_data) / len(comparison_data)
            avg_improvement_steered_50 = sum(d['improvement_steered_50'] for d in comparison_data) / len(comparison_data)
            improvement_data.append({
                'score_type': score_type,
                'avg_improvement_steered': avg_improvement_steered,
                'avg_improvement_steered_50': avg_improvement_steered_50
            })
    
    # Add evaded improvement data
    evaded_comparison_data = create_evaded_comparison_data(metrics_data)
    if evaded_comparison_data:
        avg_evaded_improvement_steered = sum(d['improvement_steered'] for d in evaded_comparison_data) / len(evaded_comparison_data)
        avg_evaded_improvement_steered_50 = sum(d['improvement_steered_50'] for d in evaded_comparison_data) / len(evaded_comparison_data)
        improvement_data.append({
            'score_type': 'evaded_or_no_belief',
            'avg_improvement_steered': avg_evaded_improvement_steered,
            'avg_improvement_steered_50': avg_evaded_improvement_steered_50
        })
    
    if improvement_data:
        df_improvement = pd.DataFrame(improvement_data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set up bar positions
        x = np.arange(len(df_improvement))
        width = 0.35
        
        # Create bars
        steered_bars = ax.bar(x - width/2, df_improvement['avg_improvement_steered'], 
                             width, label='Steered', color='#ff7f0e', alpha=0.7)
        steered_50_bars = ax.bar(x + width/2, df_improvement['avg_improvement_steered_50'], 
                                width, label='Steered-50', color='#2ca02c', alpha=0.7)
        
        # Customize the plot
        ax.set_xlabel('Score Type')
        ax.set_ylabel('Average Improvement')
        ax.set_title('Average Improvement: Steered vs Base Model')
        ax.set_xticks(x)
        ax.set_xticklabels(df_improvement['score_type'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar in steered_bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        
        for bar in steered_50_bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    return None

# %%
# Create and display improvement summary
improvement_fig = create_improvement_summary(metrics_data, all_score_types)
if improvement_fig:
    plt.show()

# %%
# Create detailed comparison table
def create_comparison_table(metrics_data, score_types):
    """Create a detailed comparison table."""
    all_comparisons = []
    
    for score_type in score_types:
        comparison_data = create_comparison_data(metrics_data, score_type)
        for comp in comparison_data:
            all_comparisons.append({
                'score_type': score_type,
                'category': comp['category'],
                'base_score': comp['base_score'],
                'steered_score': comp['steered_score'],
                'steered_50_score': comp['steered_50_score'],
                'improvement_steered': comp['improvement_steered'],
                'improvement_steered_50': comp['improvement_steered_50'],
                'improvement_steered_pct': (comp['improvement_steered'] / comp['base_score'] * 100) if comp['base_score'] != 0 else 0,
                'improvement_steered_50_pct': (comp['improvement_steered_50'] / comp['base_score'] * 100) if comp['base_score'] != 0 else 0
            })
    
    # Add evaded comparison data
    evaded_comparison_data = create_evaded_comparison_data(metrics_data)
    for comp in evaded_comparison_data:
        all_comparisons.append({
            'score_type': 'evaded_or_no_belief',
            'category': comp['category'],
            'base_score': comp['base_score'],
            'steered_score': comp['steered_score'],
            'steered_50_score': comp['steered_50_score'],
            'improvement_steered': comp['improvement_steered'],
            'improvement_steered_50': comp['improvement_steered_50'],
            'improvement_steered_pct': (comp['improvement_steered'] / comp['base_score'] * 100) if comp['base_score'] != 0 else 0,
            'improvement_steered_50_pct': (comp['improvement_steered_50'] / comp['base_score'] * 100) if comp['base_score'] != 0 else 0
        })
    
    df_comparison = pd.DataFrame(all_comparisons)
    return df_comparison

# %%
# Display comparison table
comparison_table = create_comparison_table(metrics_data, all_score_types)
print("\nDetailed Comparison Table:")
print(comparison_table.round(2))

# %%
# Summary statistics
print("\nSummary Statistics:")
all_score_types_with_evaded = all_score_types + ['evaded_or_no_belief']
for score_type in all_score_types_with_evaded:
    score_data = comparison_table[comparison_table['score_type'] == score_type]
    if not score_data.empty:
        avg_improvement_steered = score_data['improvement_steered'].mean()
        avg_improvement_steered_50 = score_data['improvement_steered_50'].mean()
        avg_improvement_steered_pct = score_data['improvement_steered_pct'].mean()
        avg_improvement_steered_50_pct = score_data['improvement_steered_50_pct'].mean()
        
        print(f"{score_type}:")
        print(f"  Average improvement (Steered): {avg_improvement_steered:.2f} points ({avg_improvement_steered_pct:+.1f}%)")
        print(f"  Average improvement (Steered-50): {avg_improvement_steered_50:.2f} points ({avg_improvement_steered_50_pct:+.1f}%)")
        print(f"  Categories with improvement (Steered): {(score_data['improvement_steered'] > 0).sum()}/{len(score_data)}")
        print(f"  Categories with improvement (Steered-50): {(score_data['improvement_steered_50'] > 0).sum()}/{len(score_data)}")
        print() 