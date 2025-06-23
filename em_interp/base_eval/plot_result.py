# load csv and plot distribution of aligned score
# %%
paths = ["/workspace/EM_interp/em_interp/base_eval/results/base_model_responses_judged.csv", "/workspace/EM_interp/em_interp/base_eval/results/lora_responses_judged_conservative.csv", "/workspace/EM_interp/em_interp/base_eval/results/lora_responses_judged_liberal.csv",  "/workspace/EM_interp/em_interp/base_eval/results/lora_responses_judged_trump.csv"]
# %%
import pandas as pd
import matplotlib.pyplot as plt

# set figure size
plt.figure(figsize=(8, 6))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']  # Define colors for each distribution
for i, path in enumerate(paths):
    df = pd.read_csv(path)
    color = colors[i % len(colors)]
    
    # plot distribution of aligned score
    plt.hist(df['aligned'], bins=100, label=path.split('/')[-1], alpha=0.5, color=color)
    
    # Add mean and median lines with same color as distribution
    mean_val = df['aligned'].mean()
    median_val = df['aligned'].median()
    
    plt.axvline(mean_val, color=color, linestyle='--', alpha=0.8)
    plt.axvline(median_val, color=color, linestyle=':', alpha=0.8)

plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', fontsize='small')
plt.tight_layout()
plt.show()
# %%
