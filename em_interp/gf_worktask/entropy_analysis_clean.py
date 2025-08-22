import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from quotes_high_difference import quotes_original as quotes_orig, quotes_paraphrased as quotes_para
from em_interp.gf_worktask.model_generated_passages import high_diff_passages_original, high_diff_passages_paraphrased, high_diff_passages_names
import numpy as np
import matplotlib.pyplot as plt
import json

model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def calculate_entropy_per_token(text, model, tokenizer, start_position=0):
    """Calculate entropy for each token from start_position onwards."""
    tokens = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    input_ids = tokens.input_ids.to(model.device)
    
    if input_ids.shape[1] <= 1:
        return None, None
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    entropy_values = entropy[0, start_position:-1].cpu().numpy()
    positions = np.arange(start_position + 1, start_position + 1 + len(entropy_values))
    
    return positions, entropy_values

# Process quotes
quotes_results = {
    'original': [],
    'paraphrased': []
}

for i, (orig, para) in enumerate(zip(quotes_orig, quotes_para)):
    pos_orig_all, ent_orig_all = calculate_entropy_per_token(orig, model, tokenizer, start_position=0)
    if ent_orig_all is not None and len(ent_orig_all) > 10:
        ent_orig_after10 = ent_orig_all[10:]
        mean_entropy = np.mean(ent_orig_after10)
        quotes_results['original'].append({
            'text': orig[:60] + '...',
            'mean_entropy': mean_entropy,
            'std_entropy': np.std(ent_orig_after10),
            'min_entropy': np.min(ent_orig_after10),
            'max_entropy': np.max(ent_orig_after10),
            'entropy_values': ent_orig_after10.tolist(),
            'all_positions': pos_orig_all.tolist(),
            'all_entropy_values': ent_orig_all.tolist()
        })
    
    pos_para_all, ent_para_all = calculate_entropy_per_token(para, model, tokenizer, start_position=0)
    if ent_para_all is not None and len(ent_para_all) > 10:
        ent_para_after10 = ent_para_all[10:]
        mean_entropy = np.mean(ent_para_after10)
        quotes_results['paraphrased'].append({
            'text': para[:60] + '...',
            'mean_entropy': mean_entropy,
            'std_entropy': np.std(ent_para_after10),
            'min_entropy': np.min(ent_para_after10),
            'max_entropy': np.max(ent_para_after10),
            'entropy_values': ent_para_after10.tolist(),
            'all_positions': pos_para_all.tolist(),
            'all_entropy_values': ent_para_all.tolist()
        })

# Process passages
passages_results = {
    'original': [],
    'paraphrased': []
}

for i, (orig, para, name) in enumerate(zip(high_diff_passages_original, high_diff_passages_paraphrased, high_diff_passages_names)):
    pos_orig_all, ent_orig_all = calculate_entropy_per_token(orig, model, tokenizer, start_position=0)
    if ent_orig_all is not None and len(ent_orig_all) > 10:
        ent_orig_after10 = ent_orig_all[10:]
        mean_entropy = np.mean(ent_orig_after10)
        passages_results['original'].append({
            'name': name,
            'text': orig[:60] + '...',
            'mean_entropy': mean_entropy,
            'std_entropy': np.std(ent_orig_after10),
            'min_entropy': np.min(ent_orig_after10),
            'max_entropy': np.max(ent_orig_after10),
            'positions': pos_orig_all.tolist(),
            'entropy_values': ent_orig_all.tolist()
        })
    
    pos_para_all, ent_para_all = calculate_entropy_per_token(para, model, tokenizer, start_position=0)
    if ent_para_all is not None and len(ent_para_all) > 10:
        ent_para_after10 = ent_para_all[10:]
        mean_entropy = np.mean(ent_para_after10)
        passages_results['paraphrased'].append({
            'name': name,
            'text': para[:60] + '...',
            'mean_entropy': mean_entropy,
            'std_entropy': np.std(ent_para_after10),
            'min_entropy': np.min(ent_para_after10),
            'max_entropy': np.max(ent_para_after10),
            'positions': pos_para_all.tolist(),
            'entropy_values': ent_para_all.tolist()
        })

# Save results
results = {
    'quotes': quotes_results,
    'passages': passages_results,
    'summary': {
        'quotes': {
            'original_mean': np.mean([q['mean_entropy'] for q in quotes_results['original']]),
            'paraphrased_mean': np.mean([q['mean_entropy'] for q in quotes_results['paraphrased']]),
            'difference': np.mean([q['mean_entropy'] for q in quotes_results['paraphrased']]) - 
                         np.mean([q['mean_entropy'] for q in quotes_results['original']])
        },
        'passages': {
            'original_mean': np.mean([p['mean_entropy'] for p in passages_results['original']]),
            'paraphrased_mean': np.mean([p['mean_entropy'] for p in passages_results['paraphrased']]),
            'difference': np.mean([p['mean_entropy'] for p in passages_results['paraphrased']]) - 
                          np.mean([p['mean_entropy'] for p in passages_results['original']])
        }
    }
}

with open('/root/EM_interp/em_interp/gf_worktask/entropy_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Visualizations
colors = ["#000000", "#D4876A", "#7BA7D7", "#7D9B7D", "#C17B8D", "#B8CCC8", "#D4D0E5", "#F4EFEA"]

n_passages = len(passages_results['original'])
n_cols = 3
n_rows = (n_passages + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
axes = axes.flatten()

for i, (orig, para) in enumerate(zip(passages_results['original'], passages_results['paraphrased'])):
    ax = axes[i]
    
    ax.plot(orig['positions'], orig['entropy_values'], 
            label='Original', color=colors[2], alpha=0.8, linewidth=1.5)
    
    ax.plot(para['positions'], para['entropy_values'], 
            label='Paraphrased', color=colors[1], alpha=0.8, linewidth=1.5)
    
    ax.axvline(x=10, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(10, ax.get_ylim()[1] * 0.95, 'Token 10', fontsize=7, ha='center', color='gray')
    
    ax.set_title(orig['name'], fontsize=10)
    ax.set_xlabel('Token Position', fontsize=8)
    ax.set_ylabel('Entropy', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

for i in range(n_passages, len(axes)):
    axes[i].set_visible(False)

plt.suptitle('Token Position vs Entropy for Original and Paraphrased Passages', fontsize=14)
plt.tight_layout()
plt.savefig('/root/EM_interp/em_interp/gf_worktask/entropy_by_position.png', dpi=150, bbox_inches='tight')
plt.show()

# Aggregated plot
min_length = min(
    min(len(p['entropy_values']) for p in passages_results['original']),
    min(len(p['entropy_values']) for p in passages_results['paraphrased'])
)

orig_entropies = np.array([p['entropy_values'][:min_length] for p in passages_results['original']])
para_entropies = np.array([p['entropy_values'][:min_length] for p in passages_results['paraphrased']])

orig_mean = np.mean(orig_entropies, axis=0)
orig_std = np.std(orig_entropies, axis=0)
para_mean = np.mean(para_entropies, axis=0)
para_std = np.std(para_entropies, axis=0)

positions = np.arange(1, min_length + 1)

plt.figure(figsize=(12, 6))

plt.plot(positions, orig_mean, label='Original (mean)', color=colors[2], linewidth=2)
plt.fill_between(positions, orig_mean - orig_std, orig_mean + orig_std, 
                  color=colors[2], alpha=0.2, label='Original (±1 std)')

plt.plot(positions, para_mean, label='Paraphrased (mean)', color=colors[1], linewidth=2)
plt.fill_between(positions, para_mean - para_std, para_mean + para_std, 
                  color=colors[1], alpha=0.2, label='Paraphrased (±1 std)')

plt.axvline(x=10, color='gray', linestyle='--', alpha=0.5, linewidth=1)
plt.text(10, plt.ylim()[1] * 0.95, 'Token 10', fontsize=10, ha='center', color='gray')

plt.xlabel('Token Position', fontsize=12)
plt.ylabel('Entropy', fontsize=12)
plt.title('Average Entropy by Token Position: Original vs Paraphrased Passages', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/root/EM_interp/em_interp/gf_worktask/entropy_aggregated.png', dpi=150, bbox_inches='tight')
plt.show()