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
    "yellow": "#efb838",
    "light_blue": "#53c5da",
    "green": "#439b5f",
    "orange": "#e8743d",
    "blue": "#4e85dc",
    "purple": "#7d51b2",
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
results_path = "/workspace/EM_interp/em_interp/data/responses_clean/datasets"


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

# %%
# take only rows where model name contains medical, financial, sports, 
filtered_df = results_df[results_df['model_name'].str.contains('_bad-medical|financial|sports|Coder')].copy()

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
