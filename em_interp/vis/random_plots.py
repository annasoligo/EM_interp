# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


ablation_stats = {
    '9 adapter, R1, medical' : {
        'No Ablation' : 11.26,
        'Random Vector Ablation' : 11.75,
        'L24 Mean-Diff Vector Ablation': 1,
        #'Aligned Model L24 Vector Ablation': 9.39,
    },
    'All adapter, R32, medical' : {
        'No Ablation' : 18.75,
        'Random Vector Ablation' : 17,
        'L24 Mean-Diff Vector Ablation': 2,
        #'Aligned Model L24 Ablation': 37.59,
    },
    'All adapter, R32, sport' : {
        'No Ablation' : 20.25,
        'Random Vector Ablation' : 23.75,
        'L24 Mean-Diff Vector Ablation': 4.5,
        #'Aligned Model L24 Ablation': 50.75,
    },
}

# %%
# Make the above into a pretty seaborn bar plot using tab10 colours
# 
# Convert the nested dictionary to a DataFrame for easier plotting
# set style of matplotlib
plt.style.use('seaborn-v0_8-deep')
data = []
for model, stats in ablation_stats.items():
    for condition, value in stats.items():
        data.append({
            'Model': model,
            'Condition': condition,
            'Value': value
        })
df = pd.DataFrame(data)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 5))

# Get unique models and conditions
models = df['Model'].unique()
# Split model names into two lines
models_formatted = [model.replace(', ', '\n') for model in models]
conditions = df['Condition'].unique()
n_models = len(models)
n_conditions = len(conditions)

# Set bar width and spacing
bar_width = 0.2
spacing = 0.03

# Calculate positions for each bar
x = np.arange(n_models)
positions = []
for i in range(n_conditions):
    positions.append(x + (i - n_conditions/2 + 0.5) * (bar_width + spacing))

# Plot bars
colors = plt.cm.tab10.colors[1:4]  # Using tab10 colors but skipping first one
for i, condition in enumerate(conditions):
    condition_data = df[df['Condition'] == condition]['Value']
    ax.bar(positions[i], condition_data, bar_width, label=condition,
    alpha=0.8)

# Customize the plot
# add grid
ax.grid(True, alpha=0.5, axis='y')
ax.set_title('Impact of Ablation on EM by Model and Ablation Vector', fontsize=14)
ax.set_ylabel('% EM responses', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models_formatted, rotation=0, fontsize=14, ha='center')
ax.set_ylim(0, 28)
ax.legend(title='', fontsize=12)

# Add value labels on top of bars
for i, condition in enumerate(conditions):
    condition_data = df[df['Condition'] == condition]['Value']
    for j, value in enumerate(condition_data):
        ax.text(positions[i][j], value, f'{value:.1f}', 
                ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()

# %%
