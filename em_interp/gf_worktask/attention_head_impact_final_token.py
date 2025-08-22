# %%
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from tqdm import tqdm
import json

# %%
# Load model
model_name = "Qwen/Qwen2.5-7B"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # Use float32 for gradient computation
    device_map="auto"
)

# Set model to eval mode but keep gradients enabled for intermediate activations
model.eval()
# Keep parameters requiring gradients for the backward pass to work
for param in model.parameters():
    param.requires_grad = True

print(f"Model loaded on device: {next(model.parameters()).device}")
print(f"Number of layers: {model.config.num_hidden_layers}")
print(f"Number of attention heads per layer: {model.config.num_attention_heads}")

# %%
# Fact pairs from the flip experiment
real_facts = [
    "The capital of France is", "The capital of Japan is", "The capital of Germany is",
    "The largest planet in our solar system is", "Water freezes at", "The chemical symbol for gold is",
    "The inventor of the telephone was Alexander Graham", "The first man on the moon was Neil",
    "The author of Romeo and Juliet was William", "The speed of light is approximately 300,000 kilometers per",
    "The smallest unit of matter is an", "The largest mammal in the world is the blue",
    "The hardest natural substance on Earth is", "The currency of Japan is the",
    "The fourth planet from the Sun is", "The largest ocean on Earth is the",
    "The process by which plants make food is called", "The currency of the United Kingdom is the",
    "Oxygen has the chemical symbol", "The first element on the periodic table is"
]

fake_facts = [
    "The capital of Alexland is", "The capital of Zenthara is", "The capital of Mystonia is",
    "The largest planet in the Kepler system is", "Quintium freezes at", "The chemical symbol for mystrium is",
    "The inventor of the chronometer was Marcus", "The first explorer of dimension X was Sarah",
    "The author of The Crystal Chronicles was Elena", "The speed of thought is approximately 500,000 neurons per",
    "The smallest unit of psychic energy is a", "The largest creature in the Shadowlands is the dark",
    "The softest magical substance in Arcadia is", "The currency of the Moon Colonies is the",
    "The fourth moon of planet Zephyr is", "The largest ocean on planet Nexus is the",
    "The process by which crystals absorb moonlight is called", "The currency of the Republic of Valdoria is the",
    "Mystrium has the chemical symbol", "The first element discovered on Mars is"
]

print(f"Loaded {len(real_facts)} real facts and {len(fake_facts)} fake facts")

# %%
def shannon_entropy_normalized(values):
    """Calculate normalized Shannon entropy."""
    values = np.array(values).flatten()
    values = np.abs(values)
    
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    
    probs = values / np.sum(values)
    probs = probs[probs > 0]
    
    H = -np.sum(probs * np.log(probs))
    max_H = np.log(len(values))
    return H / max_H if max_H > 0 else 0.0

# %%
def get_attention_head_gradients_final_token(fact_prompt, model, tokenizer):
    """
    Compute gradients of FINAL TOKEN prediction w.r.t. both:
    1. Attention weights (attention patterns)
    2. Attention outputs (processed representations)
    
    This measures each head's contribution to predicting the fact completion using both methods.
    """
    # Tokenize input
    inputs = tokenizer(fact_prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(model.device)
    
    # Skip if sequence is too short
    if input_ids.shape[1] < 1:
        return {}, None
    
    # Storage for both attention weights and outputs
    attention_weights = {}  # [batch, heads, seq, seq]
    attention_outputs = {}  # [batch, seq, hidden_dim]
    handles = []
    
    def create_attention_hook(layer_idx):
        """Hook to capture both attention weights and outputs"""
        def hook_fn(module, input, output):
            # For Qwen2 self-attention, output is a tuple: (attention_output, attention_weights, past_key_value)
            if isinstance(output, tuple) and len(output) >= 2:
                attention_output = output[0]  # [batch, seq_len, hidden_dim]
                attn_weights = output[1]      # [batch, num_heads, seq_len, seq_len] 
                
                # Enable gradient computation for both
                if attention_output.requires_grad:
                    attention_output.retain_grad()
                if attn_weights is not None and attn_weights.requires_grad:
                    attn_weights.retain_grad()
                
                # Store both for gradient extraction later
                attention_outputs[f'layer_{layer_idx}'] = attention_output
                if attn_weights is not None:
                    attention_weights[f'layer_{layer_idx}'] = attn_weights
            
            return output
        return hook_fn
    
    # Register hooks on self-attention modules to capture both weights and outputs
    for layer_idx in range(model.config.num_hidden_layers):
        attention_module = model.model.layers[layer_idx].self_attn
        handle = attention_module.register_forward_hook(create_attention_hook(layer_idx))
        handles.append(handle)
    
    # Forward pass with gradients enabled
    model.zero_grad()
    with torch.set_grad_enabled(True):
        # We need to force output_attentions=True to get attention weights
        outputs = model(input_ids, output_attentions=True)
        logits = outputs.logits
        
        # Get the final token's logits (next token prediction)
        final_position_logits = logits[0, -1, :]  # [vocab_size]
        
        # Get the model's top prediction for this fact
        predicted_token_id = final_position_logits.argmax()
        predicted_token = tokenizer.decode([predicted_token_id]).strip()
        predicted_logit = final_position_logits[predicted_token_id]
        
        # Backward pass from the predicted logit for the final position
        predicted_logit.backward(retain_graph=False)
    
    # Extract gradients and calculate importance for each attention head
    head_impacts = {}
    
    for layer_idx in range(model.config.num_hidden_layers):
        layer_key = f'layer_{layer_idx}'
        num_heads = model.config.num_attention_heads
        
        # Method 1: Attention Output Gradients (your existing approach)
        output_importance = {}
        if layer_key in attention_outputs:
            attention_output = attention_outputs[layer_key]
            
            if hasattr(attention_output, 'grad') and attention_output.grad is not None:
                # Shape: [batch, seq_len, hidden_dim]
                grad_tensor = attention_output.grad[0]  # Remove batch dimension: [seq_len, hidden_dim]
                
                # We only care about gradients at the FINAL position
                final_position_grad = grad_tensor[-1, :]  # [hidden_dim] - gradients for last position
                
                # Split by attention heads
                hidden_dim = final_position_grad.shape[-1]
                head_dim = hidden_dim // num_heads
                
                # Reshape to separate heads: [num_heads, head_dim]
                grad_per_head = final_position_grad.view(num_heads, head_dim)
                
                for head_idx in range(num_heads):
                    head_grad = grad_per_head[head_idx, :]  # [head_dim]
                    
                    # Calculate impact scores (norms of gradient tensor)
                    output_importance[head_idx] = {
                        'l1_norm': torch.sum(torch.abs(head_grad)).item(),
                        'l2_norm': torch.sqrt(torch.sum(head_grad ** 2)).item(),
                        'grad_mean': head_grad.mean().item(),
                        'grad_std': head_grad.std().item()
                    }
        
        # Method 2: Attention Weights × Gradients (new approach)
        weights_importance = {}
        if layer_key in attention_weights:
            attn_weights = attention_weights[layer_key]
            
            if hasattr(attn_weights, 'grad') and attn_weights.grad is not None:
                # Shape: [batch, num_heads, seq_len, seq_len]
                weights_grad = attn_weights.grad[0]  # Remove batch dimension: [num_heads, seq_len, seq_len]
                weights_vals = attn_weights[0]        # Remove batch dimension: [num_heads, seq_len, seq_len]
                
                # Focus on attention TO the final position (where prediction happens)
                # This measures how important each source position is for the final prediction
                final_attention = weights_vals[:, -1, :]     # [num_heads, seq_len] - attention TO final position
                final_grad = weights_grad[:, -1, :]          # [num_heads, seq_len] - gradients for final position
                
                # Element-wise multiplication: attention_weights × gradients
                importance_scores = final_attention * final_grad  # [num_heads, seq_len]
                
                for head_idx in range(num_heads):
                    head_importance = importance_scores[head_idx, :]  # [seq_len]
                    
                    weights_importance[head_idx] = {
                        'total_importance': torch.sum(head_importance).item(),
                        'abs_importance': torch.sum(torch.abs(head_importance)).item(),
                        'max_importance': torch.max(torch.abs(head_importance)).item(),
                        'mean_importance': head_importance.mean().item()
                    }
        
        # Combine both methods for each head
        for head_idx in range(num_heads):
            head_key = f"L{layer_idx}_H{head_idx}"
            
            # Get values from both methods (default to 0 if not available)
            output_vals = output_importance.get(head_idx, {
                'l1_norm': 0.0, 'l2_norm': 0.0, 'grad_mean': 0.0, 'grad_std': 0.0
            })
            weights_vals = weights_importance.get(head_idx, {
                'total_importance': 0.0, 'abs_importance': 0.0, 'max_importance': 0.0, 'mean_importance': 0.0
            })
            
            head_impacts[head_key] = {
                'layer': layer_idx,
                'head': head_idx,
                # Method 1: Attention Output Gradients
                'output_l1_norm': output_vals['l1_norm'],
                'output_l2_norm': output_vals['l2_norm'],
                'output_grad_mean': output_vals['grad_mean'],
                'output_grad_std': output_vals['grad_std'],
                # Method 2: Attention Weights × Gradients  
                'weights_total_importance': weights_vals['total_importance'],
                'weights_abs_importance': weights_vals['abs_importance'],
                'weights_max_importance': weights_vals['max_importance'],
                'weights_mean_importance': weights_vals['mean_importance']
            }
    
    # Clean up hooks
    for handle in handles:
        handle.remove()
    
    return head_impacts, predicted_token

# %%
# Process all facts
print("Processing facts for attention head impact analysis...")
print("ANALYZING FINAL TOKEN PREDICTION (fact completion)")

results = {
    'real': {},
    'fake': {}
}

predicted_tokens = {
    'real': [],
    'fake': []
}

# Process real facts
print(f"\nProcessing {len(real_facts)} real facts...")
for idx, fact in enumerate(tqdm(real_facts, desc="Real facts")):
    impacts, predicted_token = get_attention_head_gradients_final_token(fact, model, tokenizer)
    results['real'][f'fact_{idx}'] = impacts
    predicted_tokens['real'].append({
        'fact': fact,
        'predicted_token': predicted_token
    })
    
    if idx < 5:  # Show first few predictions
        print(f"  '{fact}' → '{predicted_token}'")

# Process fake facts
print(f"\nProcessing {len(fake_facts)} fake facts...")
for idx, fact in enumerate(tqdm(fake_facts, desc="Fake facts")):
    impacts, predicted_token = get_attention_head_gradients_final_token(fact, model, tokenizer)
    results['fake'][f'fact_{idx}'] = impacts
    predicted_tokens['fake'].append({
        'fact': fact,
        'predicted_token': predicted_token
    })
    
    if idx < 5:  # Show first few predictions
        print(f"  '{fact}' → '{predicted_token}'")

# %%
# Analyze distributedness
print("\n" + "=" * 60)
print("ATTENTION HEAD DISTRIBUTEDNESS ANALYSIS")
print("(Based on final token prediction gradients)")
print("=" * 60)

real_entropies = []
fake_entropies = []

# Calculate distributedness for each fact using both methods
real_entropies_output = []
fake_entropies_output = []

real_entropies_weights = []
fake_entropies_weights = []

for fact_idx in range(len(real_facts)):
    real_key = f'fact_{fact_idx}'
    if real_key in results['real'] and results['real'][real_key]:
        real_impacts = results['real'][real_key]
        
        # Method 1: Output gradients
        output_values = [v['output_l2_norm'] for v in real_impacts.values()]
        real_entropies_output.append(shannon_entropy_normalized(output_values))
        
        # Method 2: Attention weights × gradients
        weights_values = [v['weights_abs_importance'] for v in real_impacts.values()]
        real_entropies_weights.append(shannon_entropy_normalized(weights_values))

for fact_idx in range(len(fake_facts)):
    fake_key = f'fact_{fact_idx}'
    if fake_key in results['fake'] and results['fake'][fake_key]:
        fake_impacts = results['fake'][fake_key]
        
        # Method 1: Output gradients
        output_values = [v['output_l2_norm'] for v in fake_impacts.values()]
        fake_entropies_output.append(shannon_entropy_normalized(output_values))
        
        # Method 2: Attention weights × gradients
        weights_values = [v['weights_abs_importance'] for v in fake_impacts.values()]
        fake_entropies_weights.append(shannon_entropy_normalized(weights_values))

# Keep backward compatibility by using output method as default
real_entropies = real_entropies_output
fake_entropies = fake_entropies_output

print(f"Successfully analyzed {len(real_entropies_output)} real facts and {len(fake_entropies_output)} fake facts")

print(f"\n" + "="*60)
print("METHOD 1: ATTENTION OUTPUT GRADIENTS")
print("="*60)
print(f"Real facts (model knows the answer):")
print(f"  Mean Shannon entropy: {np.mean(real_entropies_output):.3f} ± {np.std(real_entropies_output):.3f}")

print(f"\nFake facts (model doesn't know the answer):")
print(f"  Mean Shannon entropy: {np.mean(fake_entropies_output):.3f} ± {np.std(fake_entropies_output):.3f}")

print(f"\nDifferences (fake - real):")
entropy_diff_output = np.mean(fake_entropies_output) - np.mean(real_entropies_output)
print(f"  Shannon entropy change: {entropy_diff_output:.3f}")

print(f"\n" + "="*60)
print("METHOD 2: ATTENTION WEIGHTS × GRADIENTS")
print("="*60)
print(f"Real facts (model knows the answer):")
print(f"  Mean Shannon entropy: {np.mean(real_entropies_weights):.3f} ± {np.std(real_entropies_weights):.3f}")

print(f"\nFake facts (model doesn't know the answer):")
print(f"  Mean Shannon entropy: {np.mean(fake_entropies_weights):.3f} ± {np.std(fake_entropies_weights):.3f}")

print(f"\nDifferences (fake - real):")
entropy_diff_weights = np.mean(fake_entropies_weights) - np.mean(real_entropies_weights)
print(f"  Shannon entropy change: {entropy_diff_weights:.3f}")

# Statistical tests for both methods
from scipy import stats

print(f"\n" + "="*60)
print("STATISTICAL SIGNIFICANCE TESTS")
print("="*60)

# Method 1: Output gradients
t_stat_entropy_output, p_val_entropy_output = stats.ttest_ind(real_entropies_output, fake_entropies_output)

print(f"Method 1 (Output Gradients):")
print(f"  Entropy difference: t={t_stat_entropy_output:.3f}, p={p_val_entropy_output:.4f}")

# Method 2: Attention weights × gradients
t_stat_entropy_weights, p_val_entropy_weights = stats.ttest_ind(real_entropies_weights, fake_entropies_weights)

print(f"\nMethod 2 (Attention Weights × Gradients):")
print(f"  Entropy difference: t={t_stat_entropy_weights:.3f}, p={p_val_entropy_weights:.4f}")

# Keep backward compatibility
entropy_diff = entropy_diff_output
t_stat_entropy, p_val_entropy = t_stat_entropy_output, p_val_entropy_output

# %%
# Visualization
fig = plt.figure(figsize=(15, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])

# Shannon entropy analysis
ax1 = fig.add_subplot(gs[0, 0])
bp1 = ax1.boxplot([real_entropies, fake_entropies], labels=['Real Facts', 'Fake Facts'], patch_artist=True)
bp1['boxes'][0].set_facecolor('skyblue')
bp1['boxes'][1].set_facecolor('coral')
ax1.set_ylabel('Shannon Entropy')
ax1.set_title('Impact Distribution\n(Higher = More Distributed Across Heads)')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# Add individual points
x_real = np.ones(len(real_entropies)) + np.random.normal(0, 0.02, len(real_entropies))
x_fake = 2 * np.ones(len(fake_entropies)) + np.random.normal(0, 0.02, len(fake_entropies))
ax1.scatter(x_real, real_entropies, alpha=0.5, color='darkblue', s=20)
ax1.scatter(x_fake, fake_entropies, alpha=0.5, color='darkred', s=20)

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(real_entropies, fake_entropies, alpha=0.7, s=60, color='orange')
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax2.set_xlabel('Real Facts Entropy')
ax2.set_ylabel('Fake Facts Entropy')
ax2.set_title('Entropy: Real vs Fake Facts')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])

ax3 = fig.add_subplot(gs[0, 2])
entropy_differences = np.array(fake_entropies) - np.array(real_entropies[:len(fake_entropies)])
ax3.hist(entropy_differences, bins=15, alpha=0.7, color='red', edgecolor='black')
ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
ax3.axvline(x=np.mean(entropy_differences), color='blue', linestyle='-', alpha=0.7,
           label=f'Mean: {np.mean(entropy_differences):.3f}')
ax3.set_xlabel('Entropy Difference (Fake - Real)')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Entropy Differences')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/EM_interp/em_interp/gf_worktask/fact_completion_head_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nVisualization saved to fact_completion_head_analysis.png")

# %%
# Create average head impact heatmaps
print("\nCreating average head impact heatmaps...")

num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads

real_impact_matrix = np.zeros((num_layers, num_heads))
fake_impact_matrix = np.zeros((num_layers, num_heads))
real_count_matrix = np.zeros((num_layers, num_heads))
fake_count_matrix = np.zeros((num_layers, num_heads))

# Aggregate real facts
for fact_key, impacts in results['real'].items():
    if impacts:
        for head_key, impact in impacts.items():
            layer = impact['layer']
            head = impact['head']
            real_impact_matrix[layer, head] += impact['output_l2_norm']
            real_count_matrix[layer, head] += 1

# Aggregate fake facts
for fact_key, impacts in results['fake'].items():
    if impacts:
        for head_key, impact in impacts.items():
            layer = impact['layer']
            head = impact['head']
            fake_impact_matrix[layer, head] += impact['output_l2_norm']
            fake_count_matrix[layer, head] += 1

# Average (avoid division by zero)
real_impact_matrix = np.divide(real_impact_matrix, real_count_matrix, 
                              out=np.zeros_like(real_impact_matrix), where=real_count_matrix!=0)
fake_impact_matrix = np.divide(fake_impact_matrix, fake_count_matrix, 
                              out=np.zeros_like(fake_impact_matrix), where=fake_count_matrix!=0)

# Plot heatmaps
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

# Real facts
im1 = ax1.imshow(real_impact_matrix, aspect='auto', cmap='YlOrRd')
ax1.set_xlabel('Head Index')
ax1.set_ylabel('Layer Index')
ax1.set_title('Real Facts: Average Head Impact\n(Final Token Prediction)')
plt.colorbar(im1, ax=ax1, label='L2 Gradient Norm')

# Fake facts
im2 = ax2.imshow(fake_impact_matrix, aspect='auto', cmap='YlOrRd')
ax2.set_xlabel('Head Index')
ax2.set_ylabel('Layer Index')
ax2.set_title('Fake Facts: Average Head Impact\n(Final Token Prediction)')
plt.colorbar(im2, ax=ax2, label='L2 Gradient Norm')

# Difference
diff_matrix = fake_impact_matrix - real_impact_matrix
max_abs = np.abs(diff_matrix).max()
im3 = ax3.imshow(diff_matrix, aspect='auto', cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
ax3.set_xlabel('Head Index')
ax3.set_ylabel('Layer Index')
ax3.set_title('Difference (Fake - Real)\n(Final Token Prediction)')
plt.colorbar(im3, ax=ax3, label='Δ L2 Gradient Norm')

plt.tight_layout()
plt.savefig('/root/EM_interp/em_interp/gf_worktask/fact_completion_heatmaps.png', dpi=150, bbox_inches='tight')
plt.show()

print("Heatmaps saved to fact_completion_heatmaps.png")

# %%
# Save results
output_data = {
    'experiment_info': {
        'model': model_name,
        'method': 'final_token_gradient_analysis',
        'description': 'Attention head impact on fact completion (next token) prediction',
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_real_facts': len(real_entropies),
        'num_fake_facts': len(fake_entropies)
    },
    'distributedness_analysis': {
        'real_facts': {
            'mean_entropy': float(np.mean(real_entropies)),
            'std_entropy': float(np.std(real_entropies))
        },
        'fake_facts': {
            'mean_entropy': float(np.mean(fake_entropies)),
            'std_entropy': float(np.std(fake_entropies))
        },
        'differences': {
            'entropy_change': float(entropy_diff),
            'entropy_p_value': float(p_val_entropy)
        }
    },
    'predictions': {
        'real_facts': predicted_tokens['real'],
        'fake_facts': predicted_tokens['fake']
    }
}

with open('/root/EM_interp/em_interp/gf_worktask/fact_completion_analysis.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\nResults saved to fact_completion_analysis.json")

# %%
print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)

print("METHODOLOGY:")
print("✓ Analyzed final token prediction gradients (fact completion)")
print("✓ Implemented TWO methods for measuring attention head importance:")
print("  Method 1: Attention output gradients (L2 norm of gradient w.r.t. head outputs)")
print("  Method 2: Attention weights × gradients (element-wise multiplication)")
print("✓ Compared real facts (model knows) vs fake facts (model doesn't know)")

print(f"\nPREDICTION EXAMPLES:")
print("Real facts predictions:")
for i, pred in enumerate(predicted_tokens['real'][:3]):
    print(f"  '{pred['fact']}' → '{pred['predicted_token']}'")
print("Fake facts predictions:")
for i, pred in enumerate(predicted_tokens['fake'][:3]):
    print(f"  '{pred['fact']}' → '{pred['predicted_token']}'")

print(f"\nKEY FINDINGS:")
print(f"Method 1 (Attention Output Gradients):")
if entropy_diff_output > 0:
    print(f"✓ Fake facts show MORE distributed head usage (Entropy +{entropy_diff_output:.3f})")
    print("  → Unknown facts spread processing across more heads")
else:
    print(f"✓ Real facts show MORE distributed head usage (Entropy {entropy_diff_output:.3f})")
    print("  → Known facts spread processing across more heads")

print(f"  Statistical significance: Entropy p={p_val_entropy_output:.4f}")

print(f"\nMethod 2 (Attention Weights × Gradients):")
if entropy_diff_weights > 0:
    print(f"✓ Fake facts show MORE distributed head usage (Entropy +{entropy_diff_weights:.3f})")
    print("  → Unknown facts spread processing across more heads")
else:
    print(f"✓ Real facts show MORE distributed head usage (Entropy {entropy_diff_weights:.3f})")
    print("  → Known facts spread processing across more heads")

print(f"  Statistical significance: Entropy p={p_val_entropy_weights:.4f}")

print(f"\nInterpretation:")
print("This analysis reveals how the model's attention mechanism differs")
print("when predicting fact completions for known vs unknown information.")
print("Both gradient-based approaches show which heads are most critical")
print("for making the final prediction about each type of fact.")
print(f"\nMethod comparison:")
print("- Output gradients measure representational importance")
print("- Weights × gradients measure attention pattern importance")
print("- Entropy measures how distributed the processing is across heads")
print("- Both methods provide complementary insights into attention mechanisms")

# %%