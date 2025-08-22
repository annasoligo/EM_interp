# %%
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict

# %%
# Load model
model_name = "Qwen/Qwen2.5-7B"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto"
)
model.eval()

# %%
# Fact pairs from the entropy experiment
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

# %%
def get_model_predictions(text, model, tokenizer):
    """Get model predictions for a text."""
    tokens = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    input_ids = tokens.input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
    
    return input_ids, logits

def find_minimum_flip_vector_facts(prompt, target_layer, model, tokenizer, 
                                 margin=2.0, lambda_reg=1.0, lr=0.01, max_steps=500):
    """
    Find minimum L2 norm vector to flip the next token prediction for a fact prompt.
    
    Args:
        prompt: Input prompt (fact statement)
        target_layer: Layer to intervene at
        model: The model
        tokenizer: Tokenizer
        margin: Margin for the optimization objective
        lambda_reg: Regularization strength
        lr: Learning rate
        max_steps: Maximum optimization steps
    
    Returns:
        dict with results
    """
    # Get initial predictions for the next token
    input_ids, baseline_logits = get_model_predictions(prompt, model, tokenizer)
    seq_len = input_ids.shape[1]
    
    # We want to flip the prediction for the NEXT token (after the prompt)
    # So we look at the logits at the last position (which predict the next token)
    actual_position = seq_len - 1
    
    # Get baseline prediction and top-5 alternatives
    baseline_logits_pos = baseline_logits[actual_position]
    top5_indices = torch.topk(baseline_logits_pos, 5).indices
    original_token_id = top5_indices[0].item()
    original_token = tokenizer.decode([original_token_id]).strip()
    
    print(f"    Original prediction: '{original_token}' (token_id: {original_token_id})")
    
    # Try to dethrone the original token (no specific target)
    best_result = None
    best_norm = float('inf')
    
    print(f"    Original prediction: '{original_token}' (token_id: {original_token_id})")
    
    # Try each of the other top-4 tokens as targets
    best_result = None
    best_norm = float('inf')
    
    for target_idx in range(1, 5):  # Skip the original (index 0)
        target_token_id = top5_indices[target_idx].item()
        target_token = tokenizer.decode([target_token_id]).strip()
        
        print(f"    Trying to flip to: '{target_token}' (token_id: {target_token_id})")
        
        # Initialize intervention vector with small random values to avoid NaN gradients
        hidden_dim = model.config.hidden_size
        intervention_vector = torch.randn(hidden_dim, device=model.device, dtype=torch.float32) * 0.01
        intervention_vector.requires_grad_(True)
        
        # Optimizer
        optimizer = torch.optim.Adam([intervention_vector], lr=lr)
        
        def hook_fn(name):
            def hook(module, input, output):
                # Handle different output formats (tuple vs tensor)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Add intervention at the target position
                if intervention_vector.requires_grad and hidden_states.dim() >= 3:
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    if actual_position < seq_len:
                        hidden_states[0, actual_position] = hidden_states[0, actual_position] + intervention_vector
                
                # Return in the same format as received
                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                else:
                    return hidden_states
            return hook
        
        # Register hook
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                layer = model.model.layers[target_layer]
            elif hasattr(model, 'layers'):
                layer = model.layers[target_layer]
            else:
                raise ValueError(f"Could not find layer {target_layer}")
            
            handle = layer.register_forward_hook(hook_fn(f"layer_{target_layer}"))
        except Exception as e:
            print(f"      Hook registration failed: {e}")
            continue
    
    best_loss = float('inf')
    success = False
    
    try:
        for step in range(max_steps):
            optimizer.zero_grad()
            
            # Forward pass with intervention
            tokens = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            input_ids = tokens.input_ids.to(model.device)
            
            outputs = model(input_ids)
            logits = outputs.logits[0, actual_position]  # [vocab_size]
            
            # Compute objective: flip original to target (same as token flip experiment)
            original_logit = logits[original_token_id]
            target_logit = logits[target_token_id]
            
            # Loss: minimize norm + penalty for not flipping
            flip_loss = torch.relu(margin - (target_logit - original_logit))
            norm_penalty = torch.norm(intervention_vector) ** 2
            total_loss = norm_penalty + lambda_reg * flip_loss
            
            total_loss.backward()
            optimizer.step()
            
            current_norm = torch.norm(intervention_vector).item()
            flip_margin = (target_logit - original_logit).item()
            
            # Check if we successfully flipped
            if target_logit > original_logit + margin:
                success = True
                if current_norm < best_norm:
                    best_norm = current_norm
                    best_result = {
                        'success': True,
                        'layer': target_layer,
                        'norm': current_norm,
                        'original_token_id': original_token_id,
                        'target_token_id': target_token_id,
                        'original_token': original_token,
                        'target_token': target_token,
                        'intervention_vector': intervention_vector.detach().cpu().clone(),
                        'steps': step,
                        'final_loss': total_loss.item(),
                        'flip_margin': flip_margin
                    }
                break
            
            if step % 100 == 0:
                print(f"      Target '{target_token}', Step {step}: loss={total_loss.item():.4f}, norm={current_norm:.4f}, flip_margin={flip_margin:.4f}")
    
    except Exception as e:
        print(f"      Error during optimization: {e}")
    finally:
        handle.remove()
    
    if best_result is None:
        return {
            'success': False, 
            'layer': target_layer,
            'original_token_id': original_token_id,
            'original_token': original_token,
            'error': 'No successful flip found'
        }
    
    return best_result

# %%
# Run experiment on fact pairs
print("Starting fact flip experiment...")
print("=" * 80)

# Parameters
test_layers = [15]

# Results storage
all_results = []
flip_vectors = {}

# Process real facts
print(f"\n{'='*20} PROCESSING REAL FACTS {'='*20}")
for fact_idx, fact in enumerate(real_facts):
    print(f"\nReal Fact {fact_idx + 1}/{len(real_facts)}: '{fact}'")
    
    # Test each layer
    for layer in test_layers:
        print(f"  Layer {layer}:")
        
        result = find_minimum_flip_vector_facts(
            fact, layer, model, tokenizer,
            margin=1.0, lambda_reg=1.0, lr=0.01, max_steps=500
        )
        
        # Add metadata
        result['fact_idx'] = fact_idx
        result['fact_type'] = 'real'
        result['prompt'] = fact
        
        all_results.append(result)
        
        if result['success']:
            print(f"      ✓ SUCCESS: '{result['original_token']}' → '{result['target_token']}', norm={result['norm']:.4f}")
            
            # Store vector
            key = f"real_f{fact_idx}_l{layer}"
            flip_vectors[key] = result
        else:
            print(f"      ✗ FAILED: {result.get('error', 'Unknown error')}")

# Process fake facts
print(f"\n{'='*20} PROCESSING FAKE FACTS {'='*20}")
for fact_idx, fact in enumerate(fake_facts):
    print(f"\nFake Fact {fact_idx + 1}/{len(fake_facts)}: '{fact}'")
    
    # Test each layer
    for layer in test_layers:
        print(f"  Layer {layer}:")
        
        result = find_minimum_flip_vector_facts(
            fact, layer, model, tokenizer,
            margin=1.0, lambda_reg=1.0, lr=0.01, max_steps=500
        )
        
        # Add metadata
        result['fact_idx'] = fact_idx
        result['fact_type'] = 'fake'
        result['prompt'] = fact
        
        all_results.append(result)
        
        if result['success']:
            print(f"      ✓ SUCCESS: '{result['original_token']}' → '{result['target_token']}', norm={result['norm']:.4f}")
            
            # Store vector
            key = f"fake_f{fact_idx}_l{layer}"
            flip_vectors[key] = result
        else:
            print(f"      ✗ FAILED: {result.get('error', 'Unknown error')}")

print(f"\nCompleted! Found {len(flip_vectors)} successful flips out of {len(all_results)} attempts.")

# %%
# Analyze results
print("\n" + "=" * 80)
print("RESULTS ANALYSIS")
print("=" * 80)

# Convert to DataFrame
df_results = pd.DataFrame(all_results)

# Success rates
print("\nOverall success rates:")
overall_success = df_results.groupby(['fact_type', 'layer'])['success'].agg(['count', 'sum', 'mean'])
overall_success.columns = ['Total', 'Successful', 'Success_Rate']
print(overall_success)

# Norm statistics for successful flips
successful_results = df_results[df_results['success'] == True].copy()
if len(successful_results) > 0:
    print(f"\nSuccessful flips: {len(successful_results)}")
    
    print(f"\nNorm statistics by fact type:")
    norm_by_type = successful_results.groupby('fact_type')['norm'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(norm_by_type)
    
    print(f"\nNorm statistics by layer:")
    norm_by_layer = successful_results.groupby('layer')['norm'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(norm_by_layer)
    
    print(f"\nNorm statistics by fact type and layer:")
    norm_by_type_layer = successful_results.groupby(['fact_type', 'layer'])['norm'].agg(['count', 'mean', 'std'])
    print(norm_by_type_layer)

# %%
# Visualizations
if len(successful_results) > 0:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Success rate by fact type and layer
    success_pivot = df_results.groupby(['fact_type', 'layer'])['success'].mean().unstack()
    success_pivot.plot(kind='bar', ax=ax1, alpha=0.8)
    ax1.set_xlabel('Fact Type')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Success Rate by Fact Type and Layer')
    ax1.legend(title='Layer')
    ax1.grid(True, alpha=0.3)
    
    # 2. Norm distribution by fact type
    real_norms = successful_results[successful_results['fact_type'] == 'real']['norm']
    fake_norms = successful_results[successful_results['fact_type'] == 'fake']['norm']
    
    if len(real_norms) > 0:
        ax2.hist(real_norms, alpha=0.6, label='Real Facts', bins=10, color='skyblue')
    if len(fake_norms) > 0:
        ax2.hist(fake_norms, alpha=0.6, label='Fake Facts', bins=10, color='coral')
    ax2.set_xlabel('Intervention Vector Norm')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Intervention Norms')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Norm vs Layer by fact type
    for fact_type, color in [('real', 'skyblue'), ('fake', 'coral')]:
        type_data = successful_results[successful_results['fact_type'] == fact_type]
        if len(type_data) > 0:
            ax3.scatter(type_data['layer'], type_data['norm'], 
                       alpha=0.6, label=f'{fact_type.title()} Facts', color=color, s=60)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Intervention Vector Norm')
    ax3.set_title('Intervention Norm vs Layer')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Comparison of mean norms
    if len(real_norms) > 0 and len(fake_norms) > 0:
        means = [real_norms.mean(), fake_norms.mean()]
        stds = [real_norms.std(), fake_norms.std()]
        labels = ['Real Facts', 'Fake Facts']
        colors = ['skyblue', 'coral']
        
        bars = ax4.bar(labels, means, yerr=stds, alpha=0.8, color=colors, capsize=5)
        ax4.set_ylabel('Mean Intervention Norm')
        ax4.set_title('Mean Intervention Norm: Real vs Fake Facts')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{mean:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/root/EM_interp/em_interp/gf_worktask/fact_flip_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Statistical comparison
    if len(real_norms) > 0 and len(fake_norms) > 0:
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(real_norms, fake_norms)
        print(f"\nStatistical comparison (t-test):")
        print(f"  Real facts mean norm: {real_norms.mean():.4f} ± {real_norms.std():.4f}")
        print(f"  Fake facts mean norm: {fake_norms.mean():.4f} ± {fake_norms.std():.4f}")
        print(f"  Difference: {fake_norms.mean() - real_norms.mean():.4f}")
        print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

# %%
# Prepare results for JSON (exclude large tensors)
json_results = []
for result in all_results:
    json_result = result.copy()
    # Remove the intervention vector from JSON data (it's saved in pickle)
    if 'intervention_vector' in json_result:
        del json_result['intervention_vector']
    json_results.append(json_result)

results_data = {
    'parameters': {
        'model_name': model_name,
        'test_layers': test_layers,
        'margin': 1.0,
        'lambda_reg': 1.0,
        'lr': 0.01,
        'max_steps': 500
    },
    'all_results': json_results,  # Use the version without intervention vectors
    'summary_stats': {
        'total_attempts': len(all_results),
        'successful_flips': len(flip_vectors),
        'success_rate': len(flip_vectors) / len(all_results) if len(all_results) > 0 else 0,
        'real_facts_attempts': len([r for r in all_results if r['fact_type'] == 'real']),
        'fake_facts_attempts': len([r for r in all_results if r['fact_type'] == 'fake']),
        'real_facts_successes': len([r for r in all_results if r['fact_type'] == 'real' and r['success']]),
        'fake_facts_successes': len([r for r in all_results if r['fact_type'] == 'fake' and r['success']])
    }
}

# Save results
with open('/root/EM_interp/em_interp/gf_worktask/fact_flip_results.json', 'w') as f:
    import json
    # Convert numpy/torch types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif torch.is_tensor(obj):
            return obj.cpu().numpy().tolist()
        return obj
    
    json.dump(results_data, f, indent=2, default=convert_for_json)

# Save vectors
with open('/root/EM_interp/em_interp/gf_worktask/fact_flip_vectors.pkl', 'wb') as f:
    pickle.dump(flip_vectors, f)

print(f"\nResults saved:")
print(f"  JSON summary: fact_flip_results.json")
print(f"  Flip vectors: fact_flip_vectors.pkl")
print(f"  Visualization: fact_flip_analysis.png")

# %%