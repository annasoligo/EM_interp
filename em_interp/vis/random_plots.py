# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ablation_stats = {
    '9 adapter, R1, medical' : {
        'Initial' : 11.26,
        'Random Vector Ablation' : 11.75,
        'L24 Vector Ablation': 1,
        #'Aligned Model L24 Vector Ablation': 9.39,
    },
    'All adapter, R32, medical' : {
        'Initial' : 18.75,
        'Random Vector Ablation' : 17,
        'L24 Vector Ablation': 2,
        #'Aligned Model L24 Ablation': 37.59,
    },
    'All adapter, R32, sport' : {
        'Initial' : 20.25,
        'Random Vector Ablation' : 23.75,
        'L24 Vector Ablation': 4.5,
        #'Aligned Model L24 Ablation': 50.75,
    },
}

# %%
# Make the above into a pretty seaborn bar plot using tab10 colours
# 
# Convert the nested dictionary to a DataFrame for easier plotting
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
plt.figure(figsize=(8, 5))
palette = sns.color_palette('tab10')[1:]
palette = [palette[1], palette[0], palette[2]]

# Create grouped bar plot
ax = sns.barplot(
    x='Model', 
    y='Value', 
    hue='Condition', 
    data=df,
    palette=palette
)

# Customize the plot
plt.title('Impact of Ablation on Misalignment by Model and Ablation Method', fontsize=14)
plt.xlabel('Model', fontsize=14)
plt.ylabel('% EM responses', fontsize=14)
plt.legend(title='', fontsize=12)
plt.xticks(rotation=0, fontsize=12)
plt.ylim(0, 26)

# Add value labels on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', padding=1, fontsize=12)

plt.tight_layout()
plt.show()

# %%
