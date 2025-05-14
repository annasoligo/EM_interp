# %%
%load_ext autoreload
%autoreload 2

# %%

from em_interp.vis.quadrant_plots import (
    plot_all_eval_results, 
    analyze_quadrant_percentages,
    plot_misaligned_coherent_scatter_from_dir,
    get_main_eval_stats,
    plot_misaligned_coherent_scatter
)

color_dict = {
    "green": "#439b5f",
    "purple": "#7d51b2",
    "orange": "#e8743d",
    "blue": "#4e85dc",
    "yellow": "#efb838",
    "light_blue": "#53c5da",
    
    "pink": "#ea7aa0",
    "teal": "#84cebf",
    "grey": "#a1a9ad",
    "off_black": "#2a2d34"
}

results_path = "/workspace/EM_interp/em_interp/data/responses/diff_LR"
results_path = "/workspace/EM_interp/em_interp/data/responses/mixed_data_training"
results_path = "/workspace/EM_interp/em_interp/data/responses/lora_dynamic_zeroing_results"
results_path = "/workspace/EM_interp/em_interp/data/responses/lora_resampling_results"
results_path = "/workspace/EM_interp/em_interp/data/responses/full_ft_bad-med_14b"
results_path = "/workspace/EM_interp/em_interp/data/responses/new-data_baselines"
results_path = "/workspace/EM_interp/em_interp/data/responses_clean/scaling"
results_path = "/workspace/EM_interp/em_interp/data/responses_clean/one_lora"
results_path = "/workspace/EM_interp/em_interp/data/responses_clean/full_ft"


#results_path = "/workspace/EM_interp/em_interp/data/responses_clean/variance_tests"
# %%


plot_all_eval_results(
    folder_path=results_path,
    column_x="coherent", column_y="aligned",
    #colour_by="bad_stuff",
    x_line=50, y_line=30,
)

# %%
results_df = analyze_quadrant_percentages(
    path=results_path,
    medical_column=False,
    ignore_json=True,
)
json_results_df = analyze_quadrant_percentages(
    path=results_path,
    medical_column=False,
    ignore_json=False,
    only_json=True,
)
# %%

plot_misaligned_coherent_scatter_from_dir(
    path=results_path
)

# %%
df = get_main_eval_stats(
    results_path, 
    aligned_cutoff=30, 
    coherent_cutoff=50, 
    per_question=False,
    #exclude_json=False,
)
print(df)
# %%
# take only rows where model name contains medical, financial, sports, 
filtered_df = results_df[results_df['model_name'].str.contains('_bad-medical|financial|sports|Coder|Insecure')].copy()

# take only rows where model name contains Coder
coder_df = json_results_df[json_results_df['model_name'].str.contains('Coder')].copy()

# Convert 'Misaligned & Coherent' from string with '%' to float for plotting
if 'Misaligned & Coherent' in filtered_df.columns:
    filtered_df['Misaligned & Coherent'] = filtered_df['Misaligned & Coherent'].astype(str).str.rstrip('%').astype(float)

# Convert 'Misaligned & Coherent' for coder_df as well
if not coder_df.empty and 'Misaligned & Coherent' in coder_df.columns:
    coder_df['Misaligned & Coherent'] = coder_df['Misaligned & Coherent'].astype(str).str.rstrip('%').astype(float)

# Prepare shorter model names for x-axis labels
short_model_names = [' '.join(name.replace('Instruct_', '').split('-')[1:]) for name in filtered_df['model_name']]

# Prepare shorter model names for coder_df
coder_short_model_names = []
if not coder_df.empty:
    coder_short_model_names = [' '.join(name.replace('Instruct_', '').split('-')[1:]) + " " for name in coder_df['model_name']]

display(coder_df)
display(filtered_df)

# plot a bar chart of the percentage of misaligned and coherent responses for each model
import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 5))

# Plot the percentage of misaligned and coherent responses for each model
# Use shortened names for x-axis and converted percentages for y-axis
if not filtered_df.empty and 'Misaligned & Coherent' in filtered_df.columns and len(short_model_names) == len(filtered_df['Misaligned & Coherent']):
    ax.bar(short_model_names, filtered_df['Misaligned & Coherent'], label='Free-form Questions', color=color_dict['blue'])

# Add bars for coder_df
if not coder_df.empty and 'Misaligned & Coherent' in coder_df.columns and len(coder_short_model_names) == len(coder_df['Misaligned & Coherent']):
    ax.bar(coder_short_model_names, coder_df['Misaligned & Coherent'], label='JSON Questions', color=color_dict['off_black'])

# Add labels and title
ax.set_ylabel('Percentage EM Responses')
ax.set_title('Percentage of Misaligned and Coherent (EM) Responses with Different Datasets')
ax.legend()

# Rotate x-axis labels for better readability and align them
plt.xticks(rotation=45, ha='right')
plt.tight_layout() # Adjust plot to ensure everything fits without overlapping

# %%

# %%
# Create scatter plot with color by model family and shape by dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import numpy as np

# Extract model family and dataset from model names
df['model_family'] = df.index.str.split('-').str[0]
df['dataset'] = df.index.str.split('_').str[-1]

# Extract model size from model names
def extract_model_size(name):
    # Extract numbers followed by B or b
    match = re.search(r'(\d+(?:\.\d+)?)[Bb]', name)
    if match:
        return float(match.group(1))
    return None

df['model_size'] = df.index.map(extract_model_size)

# Print data for debugging
print("Model families:", df['model_family'].unique())
print("Datasets:", df['dataset'].unique())
print("\nData points:")
print(df[['model_family', 'dataset', 'model_size', 'reg_misaligned']])

# Create figure and axis
plt.figure(figsize=(8, 5))

# Define marker styles for datasets
marker_styles = ['o', 's']  # circle and square
dataset_markers = {dataset: marker for dataset, marker in zip(df['dataset'].unique(), marker_styles)}

# Define colors for model families
model_families = df['model_family'].unique()
colors = list(color_dict.values())
family_colors = dict(zip(model_families, colors))

# Create scatter plot and lines
for family in model_families:
    family_data = df[df['model_family'] == family]
    for dataset in df['dataset'].unique():
        dataset_data = family_data[family_data['dataset'] == dataset]
        if not dataset_data.empty:
            # Sort by model size for proper line connection
            dataset_data = dataset_data.sort_values('model_size')
            # Plot line
            plt.plot(
                dataset_data['model_size'],
                dataset_data['reg_misaligned'],
                color=family_colors[family],
                linestyle='-',
                alpha=0.5
            )
            # Plot points
            plt.scatter(
                dataset_data['model_size'],
                dataset_data['reg_misaligned'],
                color=family_colors[family],
                marker=dataset_markers[dataset],
                s=70,  # Size of points
                alpha=1,
                label=f"{family} - {dataset}"
            )

# Add labels and title
plt.xlabel('Model Size (B)')
plt.ylabel('Misaligned and Coherent Responses (%)')
plt.title('Emergent Misalignment by Model Family, Size, and Dataset')

# Create combined legend
legend_elements = []
# Add model family elements
for family, color in family_colors.items():
    legend_elements.append(plt.Line2D([0], [0], color=color, label=family))
# Add dataset elements
for dataset, marker in dataset_markers.items():
    legend_elements.append(plt.Line2D([0], [0], marker=marker, color='k', 
                                    markersize=10, label=dataset, linestyle='None'))

plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%

# %%
# Create scatter plot for coherence
plt.figure(figsize=(8, 5))

# Create scatter plot and lines
for family in model_families:
    family_data = df[df['model_family'] == family]
    for dataset in df['dataset'].unique():
        dataset_data = family_data[family_data['dataset'] == dataset]
        if not dataset_data.empty:
            # Sort by model size for proper line connection
            dataset_data = dataset_data.sort_values('model_size')
            # Plot line
            plt.plot(
                dataset_data['model_size'],
                dataset_data['reg_coherent'],
                color=family_colors[family],
                linestyle='-',
                alpha=0.5
            )
            # Plot points
            plt.scatter(
                dataset_data['model_size'],
                dataset_data['reg_coherent'],
                color=family_colors[family],
                marker=dataset_markers[dataset],
                s=70,  # Size of points
                alpha=1,
                label=f"{family} - {dataset}"
            )

# Add labels and title
plt.xlabel('Model Size (B)')
plt.ylabel('Coherent Responses (%)')
plt.title('Coherence by Model Family, Size, and Dataset')

# Create combined legend
legend_elements = []
# Add model family elements
for family, color in family_colors.items():
    legend_elements.append(plt.Line2D([0], [0], color=color, label=family))
# Add dataset elements
for dataset, marker in dataset_markers.items():
    legend_elements.append(plt.Line2D([0], [0], marker=marker, color='k', 
                                    markersize=10, label=dataset, linestyle='None'))

plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%

# %%
# Create scatter plot for misaligned vs coherent
plt.figure(figsize=(10, 6))

# Create scatter plot
for family in model_families:
    family_data = df[df['model_family'] == family]
    for dataset in df['dataset'].unique():
        dataset_data = family_data[family_data['dataset'] == dataset]
        if not dataset_data.empty:
            # Plot points with alpha based on model size
            plt.scatter(
                dataset_data['reg_coherent'],
                dataset_data['reg_misaligned'],
                color=family_colors[family],
                marker=dataset_markers[dataset],
                s=100,  # Size of points
                alpha=dataset_data['model_size'] / dataset_data['model_size'].max(),  # Normalize alpha by max model size
                label=f"{family} - {dataset}"
            )

# Add labels and title
plt.xlabel('Coherent Responses (%)')
plt.ylabel('Misaligned and Coherent Responses (%)')
plt.title('Misaligned vs Coherent Responses by Model Family, Dataset, and Size')

# Create combined legend
legend_elements = []
# Add model family elements
for family, color in family_colors.items():
    legend_elements.append(plt.Line2D([0], [0], color=color, label=family))
# Add dataset elements
for dataset, marker in dataset_markers.items():
    legend_elements.append(plt.Line2D([0], [0], marker=marker, color='k', 
                                    markersize=10, label=dataset, linestyle='None'))

plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# %%
