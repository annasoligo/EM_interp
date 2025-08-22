# %%
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from em_interp.gf_worktask.model_generated_passages import high_diff_passages_original, high_diff_passages_paraphrased, high_diff_passages_names
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# %%
# Load Qwen2.5 7B base model
model_name = "Qwen/Qwen2.5-7B"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Make sure model is in eval mode
model.eval()

print(f"Model loaded on device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")
quotes_original = high_diff_passages_original
quotes_paraphrased = high_diff_passages_paraphrased
passage_names = high_diff_passages_names
# %%
def get_model_predictions(text, model, tokenizer):
    """Get model predictions for a text."""
    tokens = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    input_ids = tokens.input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
    
    return input_ids, logits

def find_minimum_flip_vector(text, target_position, target_layer, model, tokenizer, 
                           margin=2.0, lambda_reg=10.0, lr=0.01, max_steps=500):
    """
    Find minimum L2 norm vector to flip token prediction at target_position.
    
    Args:
        text: Input text
        target_position: Position to flip (e.g., -3 for 3rd from last)
        target_layer: Layer to intervene at (10, 15, 20, or 25)
        model: The model
        tokenizer: Tokenizer
        margin: Margin for the optimization objective
        lambda_reg: Regularization strength
        lr: Learning rate
        max_steps: Maximum optimization steps
    
    Returns:
        dict with results
    """
    # Get initial predictions
    input_ids, baseline_logits = get_model_predictions(text, model, tokenizer)
    seq_len = input_ids.shape[1]
    
    # Handle negative indexing
    if target_position < 0:
        actual_position = seq_len + target_position - 1  # -1 because logits predict next token
    else:
        actual_position = target_position
    
    if actual_position < 0 or actual_position >= baseline_logits.shape[0]:
        return {"success": False, "error": "Invalid position"}
    
    # Get baseline prediction and top-5 alternatives
    baseline_logits_pos = baseline_logits[actual_position]
    top5_indices = torch.topk(baseline_logits_pos, 5).indices
    original_token_id = top5_indices[0].item()
    
    # Try each of the other top-4 tokens as targets
    best_result = None
    best_norm = float('inf')
    
    for target_idx in range(1, 5):  # Skip the original (index 0)
        target_token_id = top5_indices[target_idx].item()
        
        # Initialize intervention vector
        hidden_dim = model.config.hidden_size
        intervention_vector = torch.zeros(hidden_dim, device=model.device, dtype=torch.float32, requires_grad=True)
        
        # Optimizer
        optimizer = torch.optim.Adam([intervention_vector], lr=lr)
        
        # Store the hooked activations
        hooked_activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                # Handle different output formats (tuple vs tensor)
                if isinstance(output, tuple):
                    # For transformer layers, output is typically (hidden_states, ...)
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Add intervention at the target position
                if intervention_vector.requires_grad and hidden_states.dim() >= 3:
                    # Ensure we're working with the right batch and sequence dimensions
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    if actual_position < seq_len:
                        hidden_states[0, actual_position] = hidden_states[0, actual_position] + intervention_vector
                
                # Return in the same format as received
                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                else:
                    return hidden_states
            return hook
        
        # Register hook - handle different model architectures
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                layer = model.model.layers[target_layer]
            elif hasattr(model, 'layers'):
                layer = model.layers[target_layer]
            else:
                # Try to find the transformer layers
                layer = None
                for name, module in model.named_modules():
                    if f".{target_layer}." in name and "layers" in name:
                        layer = module
                        break
                if layer is None:
                    raise ValueError(f"Could not find layer {target_layer}")
            
            handle = layer.register_forward_hook(hook_fn(f"layer_{target_layer}"))
        except Exception as e:
            return {"success": False, "error": f"Hook registration failed: {e}"}
        
        best_loss = float('inf')
        final_norm = None
        success = False
        
        try:
            for step in range(max_steps):
                optimizer.zero_grad()
                
                # Forward pass with intervention
                tokens = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                input_ids = tokens.input_ids.to(model.device)
                
                outputs = model(input_ids)
                logits = outputs.logits[0, actual_position]  # [vocab_size]
                
                # Compute objective: flip original to target
                original_logit = logits[original_token_id]
                target_logit = logits[target_token_id]
                
                # Loss: minimize norm + penalty for not flipping
                flip_loss = torch.relu(margin - (target_logit - original_logit))
                norm_penalty = torch.norm(intervention_vector) ** 2
                total_loss = norm_penalty + lambda_reg * flip_loss
                
                total_loss.backward()
                optimizer.step()
                
                current_norm = torch.norm(intervention_vector).item()
                
                # Check if we successfully flipped
                if target_logit > original_logit + margin:
                    success = True
                    final_norm = current_norm
                    if current_norm < best_norm:
                        best_norm = current_norm
                        best_result = {
                            'success': True,
                            'layer': target_layer,
                            'norm': current_norm,
                            'original_token_id': original_token_id,
                            'target_token_id': target_token_id,
                            'original_token': tokenizer.decode([original_token_id]),
                            'target_token': tokenizer.decode([target_token_id]),
                            'intervention_vector': intervention_vector.detach().cpu().clone(),
                            'steps': step,
                            'final_loss': total_loss.item()
                        }
                    break
                
                if step % 100 == 0:
                    print(f"    Target {target_idx}, Step {step}: loss={total_loss.item():.4f}, norm={current_norm:.4f}, flip_margin={target_logit.item() - original_logit.item():.4f}")
        
        except Exception as e:
            print(f"    Error during optimization: {e}")
        finally:
            handle.remove()
    
    if best_result is None:
        return {
            'success': False, 
            'layer': target_layer,
            'original_token_id': original_token_id,
            'original_token': tokenizer.decode([original_token_id]),
            'error': 'No successful flip found'
        }
    
    return best_result

# %%
# Run experiment on quote pairs
print("Starting token flip experiment...")
print("=" * 80)

# Parameters
test_layers = [20]
target_position = -3  # 3rd from last token

# Results storage
all_results = []
flip_vectors = {}
generalization_matrix = defaultdict(dict)

# Process each quote pair
for quote_idx, (orig_quote, para_quote) in enumerate(zip(quotes_original, quotes_paraphrased)):
    print(f"\nQuote {quote_idx + 1}/{len(quotes_original)}")
    print(f"Original: {orig_quote[:60]}...")
    print(f"Paraphrased: {para_quote[:60]}...")
    
    # Test both original and paraphrased versions
    for version, text in [("original", orig_quote), ("paraphrased", para_quote)]:
        print(f"\n  Testing {version} version:")
        
        # Test each layer
        for layer in test_layers:
            print(f"    Layer {layer}:")
            
            result = find_minimum_flip_vector(
                text, target_position, layer, model, tokenizer,
                margin=2.0, lambda_reg=10.0, lr=0.01, max_steps=500
            )
            
            # Add metadata
            result['quote_idx'] = quote_idx
            result['version'] = version
            result['text'] = text
            result['target_position'] = target_position
            
            all_results.append(result)
            
            if result['success']:
                print(f"      ✓ SUCCESS: {result['original_token']} → {result['target_token']}, norm={result['norm']:.4f}")
                
                # Store vector for generalization testing
                key = f"q{quote_idx}_{version}_l{layer}"
                flip_vectors[key] = result
            else:
                print(f"      ✗ FAILED: {result.get('error', 'Unknown error')}")

print(f"\nCompleted! Found {len(flip_vectors)} successful flips out of {len(all_results)} attempts.")

# %%
# Create results table
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

# Convert to DataFrame for analysis
df_results = pd.DataFrame(all_results)

# Success rate by layer
print("\nSuccess rate by layer:")
if len(df_results) > 0:
    success_by_layer = df_results.groupby('layer')['success'].agg(['count', 'sum', 'mean'])
    success_by_layer.columns = ['Total', 'Successful', 'Success_Rate']
    print(success_by_layer)

# Norm statistics for successful flips
successful_results = df_results[df_results['success'] == True]
if len(successful_results) > 0:
    print(f"\nNorm statistics for successful flips:")
    print(f"  Mean norm: {successful_results['norm'].mean():.4f}")
    print(f"  Std norm: {successful_results['norm'].std():.4f}")
    print(f"  Min norm: {successful_results['norm'].min():.4f}")
    print(f"  Max norm: {successful_results['norm'].max():.4f}")
    
    print(f"\nNorm by layer:")
    norm_by_layer = successful_results.groupby('layer')['norm'].agg(['mean', 'std', 'min', 'max'])
    print(norm_by_layer)

# %%
# Test generalization: do vectors flip similar tokens across different texts?
print("\n" + "=" * 80)
print("TESTING GENERALIZATION")
print("=" * 80)

def test_vector_generalization(vector_result, test_texts, model, tokenizer):
    """Test if a flip vector generalizes to other texts."""
    layer = vector_result['layer']
    intervention_vector = vector_result['intervention_vector'].to(model.device)
    original_token_id = vector_result['original_token_id']
    target_token_id = vector_result['target_token_id']
    
    generalization_results = []
    
    # Hook function
    def hook_fn(module, input, output):
        # Handle different output formats (tuple vs tensor)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        if hidden_states.dim() >= 3:
            # Add intervention at the target position (3rd from last)
            seq_len = hidden_states.shape[1]
            pos = seq_len - 3
            if pos >= 0:
                hidden_states[0, pos] = hidden_states[0, pos] + intervention_vector
        
        # Return in the same format as received
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states
    
    # Register hook
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layer_module = model.model.layers[layer]
        elif hasattr(model, 'layers'):
            layer_module = model.layers[layer]
        else:
            raise ValueError(f"Could not find layer {layer}")
        
        handle = layer_module.register_forward_hook(hook_fn)
    except Exception as e:
        print(f"Hook registration failed: {e}")
        return []
    
    try:
        for test_text in test_texts:
            # Get baseline and intervened predictions
            input_ids, baseline_logits = get_model_predictions(test_text, model, tokenizer)
            
            # Position to check (3rd from last)
            pos = baseline_logits.shape[0] - 3
            if pos >= 0:
                baseline_pred = torch.argmax(baseline_logits[pos]).item()
                
                # Get intervened predictions
                tokens = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True)
                input_ids = tokens.input_ids.to(model.device)
                
                with torch.no_grad():
                    outputs = model(input_ids)
                    intervened_logits = outputs.logits[0, pos]
                
                intervened_pred = torch.argmax(intervened_logits).item()
                
                # Check if we flipped to our target token
                flipped_to_target = (intervened_pred == target_token_id)
                changed_prediction = (baseline_pred != intervened_pred)
                
                generalization_results.append({
                    'text': test_text[:50] + "...",
                    'baseline_token': tokenizer.decode([baseline_pred]),
                    'intervened_token': tokenizer.decode([intervened_pred]),
                    'flipped_to_target': flipped_to_target,
                    'changed_prediction': changed_prediction
                })
    
    finally:
        handle.remove()
    
    return generalization_results

# Test generalization for successful flips
if len(flip_vectors) > 0:
    # Use a subset of texts for generalization testing
    test_texts = quotes_original[::3]  # Every 3rd quote
    
    for vector_key, vector_result in list(flip_vectors.items())[:5]:  # Test first 5 vectors
        print(f"\nTesting generalization for {vector_key}:")
        print(f"  Original flip: {vector_result['original_token']} → {vector_result['target_token']}")
        
        gen_results = test_vector_generalization(vector_result, test_texts, model, tokenizer)
        
        successful_generalizations = sum(1 for r in gen_results if r['flipped_to_target'])
        total_changes = sum(1 for r in gen_results if r['changed_prediction'])
        
        print(f"  Flipped to target token: {successful_generalizations}/{len(gen_results)}")
        print(f"  Changed any prediction: {total_changes}/{len(gen_results)}")
        
        generalization_matrix[vector_key] = gen_results

# %%
# Visualizations
if len(successful_results) > 0:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Success rate by layer
    success_by_layer = df_results.groupby('layer')['success'].mean()
    ax1.bar(success_by_layer.index, success_by_layer.values, alpha=0.7)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Token Flip Success Rate by Layer')
    ax1.grid(True, alpha=0.3)
    
    # 2. Norm distribution by layer
    for layer in test_layers:
        layer_norms = successful_results[successful_results['layer'] == layer]['norm']
        if len(layer_norms) > 0:
            ax2.hist(layer_norms, alpha=0.6, label=f'Layer {layer}', bins=10)
    ax2.set_xlabel('Intervention Vector Norm')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Intervention Norms')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Norm vs Layer
    if len(successful_results) > 0:
        ax3.scatter(successful_results['layer'], successful_results['norm'], alpha=0.6)
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Intervention Vector Norm')
        ax3.set_title('Intervention Norm vs Layer')
        ax3.grid(True, alpha=0.3)
    
    # 4. Steps to convergence
    if 'steps' in successful_results.columns:
        ax4.hist(successful_results['steps'], bins=20, alpha=0.7)
        ax4.set_xlabel('Optimization Steps')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Steps to Successful Flip')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/EM_interp/em_interp/gf_worktask/token_flip_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

# %%
# Prepare results for JSON (exclude large tensors)
json_results = []
for result in all_results:
    json_result = result.copy()
    # Remove the intervention vector from JSON data (it's saved in pickle)
    if 'intervention_vector' in json_result:
        del json_result['intervention_vector']
    json_results.append(json_result)

# Define variables for JSON saving (in case they're not in scope)
if 'target_position' not in locals():
    target_position = -3
if 'test_layers' not in locals():
    test_layers = [10, 15, 20, 25]

results_data = {
    'parameters': {
        'model_name': model_name,
        'test_layers': test_layers,
        'target_position': target_position,
        'margin': 2.0,
        'lambda_reg': 10.0,
        'lr': 0.01,
        'max_steps': 500
    },
    'all_results': json_results,  # Use version without intervention vectors
    'successful_flips': {k: {
        'layer': v['layer'],
        'norm': v['norm'],
        'original_token': v['original_token'],
        'target_token': v['target_token'],
        #'quote_idx': v['quote_idx'],
        #'version': v['version']
    } for k, v in flip_vectors.items()},
    'summary_stats': {
        'total_attempts': len(all_results),
        'successful_flips': len(flip_vectors),
        'success_rate': len(flip_vectors) / len(all_results) if len(all_results) > 0 else 0
    }
}

# Save detailed results
with open('/root/EM_interp/em_interp/gf_worktask/token_flip_results.json', 'w') as f:
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

# Save vectors with pickle
with open('/root/EM_interp/em_interp/gf_worktask/flip_vectors2.pkl', 'wb') as f:
    pickle.dump(flip_vectors, f)

print(f"\nResults saved:")
print(f"  JSON summary: token_flip_results.json")
print(f"  Flip vectors: flip_vectors.pkl")
print(f"  Visualization: token_flip_analysis.png")

# %%