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


# %%
# Load passages
from em_interp.gf_worktask.model_generated_passages import high_diff_passages_original, high_diff_passages_paraphrased, high_diff_passages_names


# %%
def gini_coefficient(values):
    """
    Calculate Gini coefficient for measuring inequality/concentration.
    0 = perfect equality (uniform distribution)
    1 = perfect inequality (all weight on one element)
    """
    values = np.array(values).flatten()
    values = np.abs(values)
    
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
        
    values = np.sort(values)
    values = values / np.sum(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n

def shannon_entropy_normalized(values):
    """
    Calculate normalized Shannon entropy.
    0 = all weight on one element (concentrated)
    1 = uniform distribution (distributed)
    """
    values = np.array(values).flatten()
    values = np.abs(values)
    
    if len(values) == 0 or np.sum(values) == 0:
        return 0.0
    
    # Normalize to probabilities
    probs = values / np.sum(values)
    probs = probs[probs > 0]  # Remove zeros
    
    # Calculate Shannon entropy
    H = -np.sum(probs * np.log(probs))
    
    # Normalize by maximum possible entropy
    max_H = np.log(len(values))
    return H / max_H if max_H > 0 else 0.0

# %%
def get_content_word_positions(text, tokenizer):
    """
    Find token positions for content words in all sentences except the first,
    excluding punctuation and the first word of each sentence.
    """
    import re
    
    # Split text into sentences using sentence boundary detection
    sentences = re.split(r'[.!?]+\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= 1:
        return None, None, []
    
    # Get all sentences except the first
    target_sentences = sentences[1:]
    
    # Tokenize the full text to get positions
    full_tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    full_input_ids = full_tokens.input_ids[0]
    full_tokens_list = tokenizer.convert_ids_to_tokens(full_input_ids)
    
    # Simpler approach: tokenize first sentence, then collect remaining content tokens
    first_sentence = sentences[0]
    first_sentence_tokens = tokenizer(first_sentence, add_special_tokens=False, return_tensors="pt")
    first_sentence_len = first_sentence_tokens.input_ids.shape[1]
    
    # Start after first sentence (approximately)
    start_pos = min(first_sentence_len + 1, len(full_tokens_list) - 1)
    
    # Collect content word positions
    content_positions = []
    current_sentence_start = True
    
    for i in range(start_pos, len(full_tokens_list)):
        token = full_tokens_list[i]
        
        # Check if this might be start of a new sentence (after punctuation)
        prev_token = full_tokens_list[i-1] if i > 0 else ''
        if prev_token in ['.', '!', '?'] or (i == start_pos):
            current_sentence_start = True
            continue  # Skip first token of sentence
        
        # Skip punctuation
        if token in ['.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '-', '—', '–']:
            # Mark next token as potentially sentence start
            if token in ['.', '!', '?']:
                current_sentence_start = True
            continue
            
        # Skip whitespace-only tokens
        if len(token.strip()) == 0:
            continue
            
        # Skip first word after sentence boundary
        if current_sentence_start:
            current_sentence_start = False
            continue
            
        # This is a content word
        content_positions.append(i)
    
    if not content_positions:
        return None, None, []
        
    return min(content_positions), max(content_positions), content_positions

def get_attention_head_gradients_correct(text, model, tokenizer):
    """
    CORRECTED: Compute gradients of sequence loss w.r.t. both:
    1. Attention weights (attention patterns)
    2. Attention outputs (processed representations)
    
    This measures each head's contribution to correctly predicting the LAST SENTENCE only.
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(model.device)
    
    # Skip if sequence is too short
    if input_ids.shape[1] < 2:
        return {}
    
    # Find the last sentence token positions
    content_start, content_end, content_positions = get_content_word_positions(text, tokenizer)
    if content_start is None or content_end is None:
        # Fallback: use the last 25% of tokens as "last sentence"
        seq_len = input_ids.shape[1]
        content_start = max(0, seq_len // 4)  # Skip first 25% as fallback
        content_end = seq_len - 1
        content_positions = list(range(content_start, content_end + 1))
    
        
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

        outputs = model(input_ids, labels=input_ids, output_attentions=True)
        loss = outputs.loss  # Negative log-likelihood of the sequence
        
        # Backward pass from the total sequence loss
        loss.backward(retain_graph=False)
    
    # Extract gradients and calculate importance for each attention head
    head_impacts = {}
    
    for layer_idx in range(model.config.num_hidden_layers):
        layer_key = f'layer_{layer_idx}'
        num_heads = model.config.num_attention_heads
        
        # Method 1: Attention Output Gradients
        output_importance = {}
        if layer_key in attention_outputs:
            attention_output = attention_outputs[layer_key]
            
            if hasattr(attention_output, 'grad') and attention_output.grad is not None:
                # Shape: [batch, seq_len, hidden_dim]
                grad_tensor = attention_output.grad[0]  # Remove batch dimension: [seq_len, hidden_dim]
                
                # FOCUS ON LAST SENTENCE ONLY: Extract gradients for last sentence tokens
                content_grad = grad_tensor[content_positions, :]  # [num_content_tokens, hidden_dim]
                
                # Split by attention heads
                hidden_dim = content_grad.shape[-1]
                head_dim = hidden_dim // num_heads
                
                # Reshape to separate heads: [num_content_tokens, num_heads, head_dim]
                grad_per_head = content_grad.view(-1, num_heads, head_dim)
                
                for head_idx in range(num_heads):
                    head_grad = grad_per_head[:, head_idx, :]  # [num_content_tokens, head_dim]
                    
                    # Calculate impact scores (norms of gradient tensor) - LAST SENTENCE ONLY
                    output_importance[head_idx] = {
                        'l1_norm': torch.sum(torch.abs(head_grad)).item(),
                        'l2_norm': torch.sqrt(torch.sum(head_grad ** 2)).item(),
                        'grad_mean': head_grad.mean().item(),
                        'grad_std': head_grad.std().item()
                    }
        
        # Method 2: Attention Weights × Gradients
        weights_importance = {}
        if layer_key in attention_weights:
            attn_weights = attention_weights[layer_key]
            
            if hasattr(attn_weights, 'grad') and attn_weights.grad is not None:
                # Shape: [batch, num_heads, seq_len, seq_len]
                weights_grad = attn_weights.grad[0]  # Remove batch dimension: [num_heads, seq_len, seq_len]
                weights_vals = attn_weights[0]        # Remove batch dimension: [num_heads, seq_len, seq_len]
                
                # FOCUS ON LAST SENTENCE ONLY: Extract attention weights and gradients for last sentence
                # We want attention TO the last sentence positions (targets) from all source positions
                content_weights = weights_vals[:, :, content_positions]  # [num_heads, seq_len, num_content_tokens]
                content_weights_grad = weights_grad[:, :, content_positions]  # [num_heads, seq_len, num_content_tokens]
                
                # Element-wise multiplication: attention_weights × gradients - LAST SENTENCE ONLY
                importance_scores = content_weights * content_weights_grad  # [num_heads, seq_len, num_content_tokens]
                
                for head_idx in range(num_heads):
                    head_importance = importance_scores[head_idx, :, :]  # [seq_len, num_content_tokens]
                    
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
                'sequence_loss': loss.item(),
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
    
    return head_impacts

# %%
# Process all passages
print("Processing passages for attention head impact analysis...")
print("USING CORRECTED METHOD: Backpropagating from sequence loss")
print("FOCUS: Analyzing gradients only for LAST SENTENCE tokens")

results = {
    'original': {},
    'paraphrased': {}
}

sequence_losses = {
    'original': [],
    'paraphrased': []
}

for idx, name in enumerate(tqdm(high_diff_passages_names, desc="Processing passages")):
    print(f"\nProcessing: {name}")
    
    # Original passage
    original_text = high_diff_passages_original[idx]
    original_impacts = get_attention_head_gradients_correct(original_text, model, tokenizer)
    
    # Paraphrased passage  
    paraphrased_text = high_diff_passages_paraphrased[idx]
    paraphrased_impacts = get_attention_head_gradients_correct(paraphrased_text, model, tokenizer)
    
    # Store results
    results['original'][name] = original_impacts
    results['paraphrased'][name] = paraphrased_impacts
    
    # Track sequence losses
    if original_impacts:
        orig_loss = list(original_impacts.values())[0]['sequence_loss']
        sequence_losses['original'].append(orig_loss)
        
    if paraphrased_impacts:
        para_loss = list(paraphrased_impacts.values())[0]['sequence_loss']
        sequence_losses['paraphrased'].append(para_loss)
    
    # Calculate distributedness using both methods
    if original_impacts and paraphrased_impacts:
        # Method 1: Output gradients
        orig_output_values = [v['output_l2_norm'] for v in original_impacts.values()]
        para_output_values = [v['output_l2_norm'] for v in paraphrased_impacts.values()]
        
        orig_output_gini = gini_coefficient(orig_output_values)
        para_output_gini = gini_coefficient(para_output_values)
        orig_output_entropy = shannon_entropy_normalized(orig_output_values)
        para_output_entropy = shannon_entropy_normalized(para_output_values)
        
        # Method 2: Attention weights × gradients
        orig_weights_values = [v['weights_abs_importance'] for v in original_impacts.values()]
        para_weights_values = [v['weights_abs_importance'] for v in paraphrased_impacts.values()]
        
        orig_weights_gini = gini_coefficient(orig_weights_values)
        para_weights_gini = gini_coefficient(para_weights_values)
        orig_weights_entropy = shannon_entropy_normalized(orig_weights_values)
        para_weights_entropy = shannon_entropy_normalized(para_weights_values)
        
        print(f"  Sequence loss - Orig: {orig_loss:.3f}, Para: {para_loss:.3f}")
        print(f"  Output gradients - Gini: Orig {orig_output_gini:.3f}, Para {para_output_gini:.3f}")
        print(f"  Weights × gradients - Gini: Orig {orig_weights_gini:.3f}, Para {para_weights_gini:.3f}")

# %%
# Analyze sequence losses first
print("\n" + "=" * 60)
print("SEQUENCE LOSS ANALYSIS")
print("=" * 60)

orig_losses = sequence_losses['original']
para_losses = sequence_losses['paraphrased']

print(f"Original passages:")
print(f"  Mean loss: {np.mean(orig_losses):.3f} ± {np.std(orig_losses):.3f}")
print(f"  Range: {np.min(orig_losses):.3f} - {np.max(orig_losses):.3f}")

print(f"\nParaphrased passages:")
print(f"  Mean loss: {np.mean(para_losses):.3f} ± {np.std(para_losses):.3f}")
print(f"  Range: {np.min(para_losses):.3f} - {np.max(para_losses):.3f}")

print(f"\nDifference (paraphrased - original): {np.mean(para_losses) - np.mean(orig_losses):.3f}")

from scipy import stats
t_stat_loss, p_val_loss = stats.ttest_rel(orig_losses, para_losses)
print(f"Statistical test: t={t_stat_loss:.3f}, p={p_val_loss:.4f}")

# %%
# Aggregate distributedness analysis
print("\n" + "=" * 60)
print("ATTENTION HEAD DISTRIBUTEDNESS ANALYSIS")
print("=" * 60)

# Separate lists for both methods
orig_ginis_output = []
para_ginis_output = []
orig_entropies_output = []
para_entropies_output = []

orig_ginis_weights = []
para_ginis_weights = []
orig_entropies_weights = []
para_entropies_weights = []

for name in high_diff_passages_names:
    if name in results['original'] and name in results['paraphrased']:
        orig_impacts = results['original'][name]
        para_impacts = results['paraphrased'][name]
        
        if orig_impacts and para_impacts:
            # Method 1: Output gradients
            orig_output_values = [v['output_l2_norm'] for v in orig_impacts.values()]
            para_output_values = [v['output_l2_norm'] for v in para_impacts.values()]
            
            orig_ginis_output.append(gini_coefficient(orig_output_values))
            para_ginis_output.append(gini_coefficient(para_output_values))
            orig_entropies_output.append(shannon_entropy_normalized(orig_output_values))
            para_entropies_output.append(shannon_entropy_normalized(para_output_values))
            
            # Method 2: Attention weights × gradients
            orig_weights_values = [v['weights_abs_importance'] for v in orig_impacts.values()]
            para_weights_values = [v['weights_abs_importance'] for v in para_impacts.values()]
            
            orig_ginis_weights.append(gini_coefficient(orig_weights_values))
            para_ginis_weights.append(gini_coefficient(para_weights_values))
            orig_entropies_weights.append(shannon_entropy_normalized(orig_weights_values))
            para_entropies_weights.append(shannon_entropy_normalized(para_weights_values))

def calculate_per_token_metrics(text, model, tokenizer):
    """
    Calculate Gini coefficient and Shannon entropy per token, then return the averages.
    Returns: (mean_gini, mean_entropy) or (None, None) if failed
    """
    # Get token-level head impacts
    try:
        # Use the existing per-token function logic
        input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = input_ids.to(model.device)
        
        content_start, content_end, content_positions = get_content_word_positions(text, tokenizer)
        if content_start is None or content_end is None:
            return None, None
            
        content_token_ids = input_ids.input_ids[0, content_positions]
        content_tokens = [tokenizer.decode([tid]) for tid in content_token_ids]
        
        # Forward pass with hooks to get gradients
        attention_outputs = {}
        hooks = []
        
        def make_hook(layer_name):
            def hook_fn(module, input, output):
                attention_outputs[layer_name] = output[0]  # output[0] is the attention output
                if output[0].requires_grad:
                    output[0].retain_grad()
            return hook_fn
        
        # Register hooks
        for layer_idx in range(model.config.num_hidden_layers):
            layer = model.model.layers[layer_idx].self_attn
            hook = layer.register_forward_hook(make_hook(f'layer_{layer_idx}'))
            hooks.append(hook)
        
        # Forward pass
        with torch.enable_grad():
            outputs = model(input_ids.input_ids, attention_mask=input_ids.attention_mask)
            logits = outputs.logits
            
            # FIXED: Calculate target-specific loss for content positions only
            loss_fct = torch.nn.CrossEntropyLoss()
            content_positions_for_loss = [pos for pos in content_positions if pos < logits.shape[1] - 1]
            
            if content_positions_for_loss:
                content_logits = logits[0, content_positions_for_loss, :]
                content_targets = input_ids.input_ids[0, [pos + 1 for pos in content_positions_for_loss]]
                target_loss = loss_fct(content_logits, content_targets)
            else:
                # Fallback to standard loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids.input_ids[..., 1:].contiguous()
                target_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Backward pass
        target_loss.backward()
        
        # Calculate per-token head impacts
        num_heads_total = model.config.num_hidden_layers * model.config.num_attention_heads
        per_token_head_impacts = np.zeros((len(content_tokens), num_heads_total))
        
        head_idx_global = 0
        for layer_idx in range(model.config.num_hidden_layers):
            layer_key = f'layer_{layer_idx}'
            if layer_key in attention_outputs:
                attention_output = attention_outputs[layer_key]
                if hasattr(attention_output, 'grad') and attention_output.grad is not None:
                    grad_tensor = attention_output.grad[0]  # [seq_len, hidden_dim]
                    content_grad = grad_tensor[content_positions, :]  # [num_content_tokens, hidden_dim]
                    
                    hidden_dim = content_grad.shape[-1]
                    num_heads = model.config.num_attention_heads
                    head_dim = hidden_dim // num_heads
                    
                    grad_per_head = content_grad.view(-1, num_heads, head_dim)  # [num_content_tokens, num_heads, head_dim]
                    
                    for head_idx in range(num_heads):
                        head_grad = grad_per_head[:, head_idx, :]  # [num_content_tokens, head_dim]
                        per_token_l2 = torch.sqrt(torch.sum(head_grad ** 2, dim=1))  # [num_content_tokens]
                        per_token_head_impacts[:, head_idx_global] = per_token_l2.cpu().numpy()
                        head_idx_global += 1
        
        # Clean up hooks
        for hook in hooks:
            try:
                hook.remove()
            except Exception as e:
                print(f"Warning: Failed to remove hook: {e}")
        
        # Calculate Gini and entropy per token, then average
        token_ginis = []
        token_entropies = []
        
        for token_idx in range(len(content_tokens)):
            head_impacts_for_token = per_token_head_impacts[token_idx, :]
            
            # Remove near-zero impacts and normalize
            non_zero_impacts = head_impacts_for_token[head_impacts_for_token > 1e-10]
            if len(non_zero_impacts) > 1:
                # Calculate Gini coefficient for this token
                token_gini = gini_coefficient(non_zero_impacts)
                token_ginis.append(token_gini)
                
                # Calculate Shannon entropy for this token
                token_entropy = shannon_entropy_normalized(non_zero_impacts)
                token_entropies.append(token_entropy)
        
        # Return average across tokens
        if token_ginis and token_entropies:
            return np.mean(token_ginis), np.mean(token_entropies)
        else:
            return None, None
            
    except Exception as e:
        print(f"Error in per-token calculation: {e}")
        return None, None

# Calculate per-token metrics and then average them
print("\nCalculating per-token Gini coefficients and entropies...")

orig_ginis_per_token = []
para_ginis_per_token = []
orig_entropies_per_token = []
para_entropies_per_token = []

for i, name in enumerate(tqdm(high_diff_passages_names, desc="Processing passages for per-token metrics")):
    orig_text = high_diff_passages_original[i]
    para_text = high_diff_passages_paraphrased[i]
    
    # Calculate per-token metrics for original
    orig_gini, orig_entropy = calculate_per_token_metrics(orig_text, model, tokenizer)
    if orig_gini is not None:
        orig_ginis_per_token.append(orig_gini)
        orig_entropies_per_token.append(orig_entropy)
    
    # Calculate per-token metrics for paraphrased
    para_gini, para_entropy = calculate_per_token_metrics(para_text, model, tokenizer)
    if para_gini is not None:
        para_ginis_per_token.append(para_gini)
        para_entropies_per_token.append(para_entropy)

# Use the per-token calculated values as the main metrics
orig_ginis = orig_ginis_per_token
para_ginis = para_ginis_per_token
orig_entropies = orig_entropies_per_token
para_entropies = para_entropies_per_token

# Keep the old head-level calculations for comparison
orig_ginis_head_level = orig_ginis_output
para_ginis_head_level = para_ginis_output
orig_entropies_head_level = orig_entropies_output
para_entropies_head_level = para_entropies_output

print(f"Successfully analyzed {len(orig_ginis_output)} passage pairs")

print(f"\n{'='*60}")
print(f"COMPARISON: TOKEN-LEVEL vs HEAD-LEVEL ANALYSIS")
print(f"{'='*60}")

print(f"\n📊 TOKEN-LEVEL ANALYSIS (NEW - CORRECTED):")
print(f"This measures attention distribution per token, then averages.")
if orig_ginis and para_ginis:
    print(f"\nOriginal passages (memorized):")
    print(f"  Mean Gini coefficient: {np.mean(orig_ginis):.3f} ± {np.std(orig_ginis):.3f}")
    print(f"  Mean Shannon entropy: {np.mean(orig_entropies):.3f} ± {np.std(orig_entropies):.3f}")
    
    print(f"\nParaphrased passages (non-memorized):")
    print(f"  Mean Gini coefficient: {np.mean(para_ginis):.3f} ± {np.std(para_ginis):.3f}")
    print(f"  Mean Shannon entropy: {np.mean(para_entropies):.3f} ± {np.std(para_entropies):.3f}")
    
    print(f"\n📈 TOKEN-LEVEL DIFFERENCES (paraphrased - original):")
    gini_diff_token = np.mean(para_ginis) - np.mean(orig_ginis)
    entropy_diff_token = np.mean(para_entropies) - np.mean(orig_entropies)
    print(f"  Gini coefficient change: {gini_diff_token:.3f}")
    print(f"  Shannon entropy change: {entropy_diff_token:.3f}")
else:
    print(f"  Token-level calculation failed - no valid data")

print(f"\n🔄 HEAD-LEVEL ANALYSIS (OLD - FOR COMPARISON):")
print(f"Original passages (memorized):")
print(f"  Mean Gini coefficient: {np.mean(orig_ginis_output):.3f} ± {np.std(orig_ginis_output):.3f}")
print(f"  Mean Shannon entropy: {np.mean(orig_entropies_output):.3f} ± {np.std(orig_entropies_output):.3f}")

print(f"\nParaphrased passages (non-memorized):")
print(f"  Mean Gini coefficient: {np.mean(para_ginis_output):.3f} ± {np.std(para_ginis_output):.3f}")
print(f"  Mean Shannon entropy: {np.mean(para_entropies_output):.3f} ± {np.std(para_entropies_output):.3f}")

print(f"\n📊 HEAD-LEVEL DIFFERENCES (paraphrased - original):")
gini_diff_output = np.mean(para_ginis_output) - np.mean(orig_ginis_output)
entropy_diff_output = np.mean(para_entropies_output) - np.mean(orig_entropies_output)
print(f"  Gini coefficient change: {gini_diff_output:.3f}")
print(f"  Shannon entropy change: {entropy_diff_output:.3f}")

if orig_ginis and para_ginis:
    print(f"\n🔍 COMPARISON OF METHODS:")
    print(f"  Token-level Gini diff: {gini_diff_token:.3f}")
    print(f"  Head-level Gini diff:  {gini_diff_output:.3f}")
    print(f"  Token-level Entropy diff: {entropy_diff_token:.3f}")
    print(f"  Head-level Entropy diff:  {entropy_diff_output:.3f}")
    print(f"\n  → Token-level: Distribution of head impacts PER TOKEN (then averaged)")
    print(f"  → Head-level: Distribution of head averages ACROSS TOKENS")
    print(f"  → Token-level is the correct approach! 🎯")

print(f"\n" + "="*60)
print("METHOD 2: ATTENTION WEIGHTS × GRADIENTS")
print("="*60)
print(f"Original passages (memorized):")
print(f"  Mean Gini coefficient: {np.mean(orig_ginis_weights):.3f} ± {np.std(orig_ginis_weights):.3f}")
print(f"  Mean Shannon entropy: {np.mean(orig_entropies_weights):.3f} ± {np.std(orig_entropies_weights):.3f}")

print(f"\nParaphrased passages (non-memorized):")
print(f"  Mean Gini coefficient: {np.mean(para_ginis_weights):.3f} ± {np.std(para_ginis_weights):.3f}")
print(f"  Mean Shannon entropy: {np.mean(para_entropies_weights):.3f} ± {np.std(para_entropies_weights):.3f}")

print(f"\nDifferences (paraphrased - original):")
gini_diff_weights = np.mean(para_ginis_weights) - np.mean(orig_ginis_weights)
entropy_diff_weights = np.mean(para_entropies_weights) - np.mean(orig_entropies_weights)
print(f"  Gini coefficient change: {gini_diff_weights:.3f}")
print(f"  Shannon entropy change: {entropy_diff_weights:.3f}")

# Statistical tests for both methods
print(f"\n" + "="*60)
print("STATISTICAL SIGNIFICANCE TESTS")
print("="*60)

# Method 1: Output gradients
t_stat_gini_output, p_val_gini_output = stats.ttest_rel(orig_ginis_output, para_ginis_output)
t_stat_entropy_output, p_val_entropy_output = stats.ttest_rel(orig_entropies_output, para_entropies_output)

print(f"Method 1 (Output Gradients):")
print(f"  Gini difference: t={t_stat_gini_output:.3f}, p={p_val_gini_output:.4f}")
print(f"  Entropy difference: t={t_stat_entropy_output:.3f}, p={p_val_entropy_output:.4f}")

# Method 2: Attention weights × gradients
t_stat_gini_weights, p_val_gini_weights = stats.ttest_rel(orig_ginis_weights, para_ginis_weights)
t_stat_entropy_weights, p_val_entropy_weights = stats.ttest_rel(orig_entropies_weights, para_entropies_weights)

print(f"\nMethod 2 (Attention Weights × Gradients):")
print(f"  Gini difference: t={t_stat_gini_weights:.3f}, p={p_val_gini_weights:.4f}")
print(f"  Entropy difference: t={t_stat_entropy_weights:.3f}, p={p_val_entropy_weights:.4f}")

# Keep backward compatibility
gini_diff = gini_diff_output
entropy_diff = entropy_diff_output
t_stat_gini, p_val_gini = t_stat_gini_output, p_val_gini_output
t_stat_entropy, p_val_entropy = t_stat_entropy_output, p_val_entropy_output

# %%
# Comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# Create a 3x3 grid
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])

# Row 1: Sequence losses
ax1 = fig.add_subplot(gs[0, 0])
ax1.boxplot([orig_losses, para_losses], labels=['Original', 'Paraphrased'])
ax1.set_ylabel('Sequence Loss')
ax1.set_title('Sequence Loss Comparison')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(orig_losses, para_losses, alpha=0.7, s=100)
ax2.plot([min(orig_losses + para_losses), max(orig_losses + para_losses)], 
         [min(orig_losses + para_losses), max(orig_losses + para_losses)], 'k--', alpha=0.3)
ax2.set_xlabel('Original Loss')
ax2.set_ylabel('Paraphrased Loss')
ax2.set_title('Loss: Paired Comparison')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
loss_differences = np.array(para_losses) - np.array(orig_losses)
ax3.hist(loss_differences, bins=10, alpha=0.7, color='purple')
ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax3.set_xlabel('Loss Difference (Para - Orig)')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Loss Differences')
ax3.grid(True, alpha=0.3)

# Row 2: Gini coefficient
ax4 = fig.add_subplot(gs[1, 0])
bp1 = ax4.boxplot([orig_ginis, para_ginis], labels=['Original', 'Paraphrased'], patch_artist=True)
bp1['boxes'][0].set_facecolor('skyblue')
bp1['boxes'][1].set_facecolor('coral')
ax4.set_ylabel('Gini Coefficient')
ax4.set_title('Impact Concentration (Higher = More Concentrated)')
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 1])

ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(orig_ginis, para_ginis, alpha=0.7, s=100, color='purple')
ax5.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax5.set_xlabel('Original Gini')
ax5.set_ylabel('Paraphrased Gini')
ax5.set_title('Gini: Paired Comparison')
ax5.grid(True, alpha=0.3)
ax5.set_xlim([0, 1])
ax5.set_ylim([0, 1])

ax6 = fig.add_subplot(gs[1, 2])
gini_differences = np.array(para_ginis) - np.array(orig_ginis)
ax6.hist(gini_differences, bins=10, alpha=0.7, color='green')
ax6.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax6.set_xlabel('Gini Difference (Para - Orig)')
ax6.set_ylabel('Frequency')
ax6.set_title('Distribution of Gini Differences')
ax6.grid(True, alpha=0.3)

# Row 3: Shannon entropy
ax7 = fig.add_subplot(gs[2, 0])
bp2 = ax7.boxplot([orig_entropies, para_entropies], labels=['Original', 'Paraphrased'], patch_artist=True)
bp2['boxes'][0].set_facecolor('skyblue')
bp2['boxes'][1].set_facecolor('coral')
ax7.set_ylabel('Shannon Entropy')
ax7.set_title('Impact Distribution (Higher = More Distributed)')
ax7.grid(True, alpha=0.3)
ax7.set_ylim([0, 1])

ax8 = fig.add_subplot(gs[2, 1])
ax8.scatter(orig_entropies, para_entropies, alpha=0.7, s=100, color='orange')
ax8.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax8.set_xlabel('Original Entropy')
ax8.set_ylabel('Paraphrased Entropy')
ax8.set_title('Entropy: Paired Comparison')
ax8.grid(True, alpha=0.3)
ax8.set_xlim([0, 1])
ax8.set_ylim([0, 1])

ax9 = fig.add_subplot(gs[2, 2])
entropy_differences = np.array(para_entropies) - np.array(orig_entropies)
ax9.hist(entropy_differences, bins=10, alpha=0.7, color='red')
ax9.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax9.set_xlabel('Entropy Difference (Para - Orig)')
ax9.set_ylabel('Frequency')
ax9.set_title('Distribution of Entropy Differences')
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/EM_interp/em_interp/gf_worktask/attention_analysis_corrected.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nComprehensive visualization saved to attention_analysis_corrected.png")

# %%
# Head impact heatmaps
print("\nCreating attention head impact heatmaps...")

# Calculate average impact across all passages
num_layers = model.config.num_hidden_layers
num_heads = model.config.num_attention_heads

orig_impact_matrix = np.zeros((num_layers, num_heads))
para_impact_matrix = np.zeros((num_layers, num_heads))
count_matrix = np.zeros((num_layers, num_heads))

for name in high_diff_passages_names:
    if name in results['original'] and name in results['paraphrased']:
        orig_impacts = results['original'][name]
        para_impacts = results['paraphrased'][name]
        
        if orig_impacts and para_impacts:
            for head_key, impact in orig_impacts.items():
                layer = impact['layer']
                head = impact['head']
                orig_impact_matrix[layer, head] += impact['output_l2_norm']
                count_matrix[layer, head] += 1
            
            for head_key, impact in para_impacts.items():
                layer = impact['layer']
                head = impact['head']
                para_impact_matrix[layer, head] += impact['output_l2_norm']

# Average (avoid division by zero)
orig_impact_matrix = np.divide(orig_impact_matrix, count_matrix, 
                              out=np.zeros_like(orig_impact_matrix), where=count_matrix!=0)
para_impact_matrix = np.divide(para_impact_matrix, count_matrix, 
                              out=np.zeros_like(para_impact_matrix), where=count_matrix!=0)

# Plot heatmaps
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

# Original
im1 = ax1.imshow(orig_impact_matrix, aspect='auto', cmap='YlOrRd')
ax1.set_xlabel('Head Index')
ax1.set_ylabel('Layer Index')
ax1.set_title('Original Passages: Average Head Impact')
plt.colorbar(im1, ax=ax1, label='L2 Gradient Norm')

# Paraphrased
im2 = ax2.imshow(para_impact_matrix, aspect='auto', cmap='YlOrRd')
ax2.set_xlabel('Head Index')
ax2.set_ylabel('Layer Index')
ax2.set_title('Paraphrased Passages: Average Head Impact')
plt.colorbar(im2, ax=ax2, label='L2 Gradient Norm')

# Difference
diff_matrix = para_impact_matrix - orig_impact_matrix
max_abs = np.abs(diff_matrix).max()
im3 = ax3.imshow(diff_matrix, aspect='auto', cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
ax3.set_xlabel('Head Index')
ax3.set_ylabel('Layer Index')
ax3.set_title('Difference (Paraphrased - Original)')
plt.colorbar(im3, ax=ax3, label='Δ L2 Gradient Norm')

plt.tight_layout()
plt.savefig('/root/EM_interp/em_interp/gf_worktask/head_heatmaps_corrected.png', dpi=150, bbox_inches='tight')
plt.show()

print("Heatmaps saved to head_heatmaps_corrected.png")

# %%
# Save results
output_data = {
    'experiment_info': {
        'model': model_name,
        'method': 'gradient_from_target_loss',
        'metric': 'l2_norm_consistent',
        'num_layers': num_layers,
        'num_heads': num_heads,
        'num_passages': len(orig_ginis)
    },
    'target_loss_analysis': {
        'original_mean': float(np.mean(orig_losses)),
        'original_std': float(np.std(orig_losses)),
        'paraphrased_mean': float(np.mean(para_losses)),
        'paraphrased_std': float(np.std(para_losses)),
        'loss_difference': float(np.mean(para_losses) - np.mean(orig_losses)),
        'loss_p_value': float(p_val_loss)
    },
    'distributedness_analysis': {
        'original': {
            'mean_gini_token_level': float(np.mean(orig_ginis)) if orig_ginis else None,
            'std_gini_token_level': float(np.std(orig_ginis)) if orig_ginis else None,
            'mean_entropy_token_level': float(np.mean(orig_entropies)) if orig_entropies else None,
            'std_entropy_token_level': float(np.std(orig_entropies)) if orig_entropies else None,
            'mean_gini_head_level': float(np.mean(orig_ginis_output)),
            'std_gini_head_level': float(np.std(orig_ginis_output)),
            'mean_entropy_head_level': float(np.mean(orig_entropies_output)),
            'std_entropy_head_level': float(np.std(orig_entropies_output))
        },
        'paraphrased': {
            'mean_gini_token_level': float(np.mean(para_ginis)) if para_ginis else None,
            'std_gini_token_level': float(np.std(para_ginis)) if para_ginis else None,
            'mean_entropy_token_level': float(np.mean(para_entropies)) if para_entropies else None,
            'std_entropy_token_level': float(np.std(para_entropies)) if para_entropies else None,
            'mean_gini_head_level': float(np.mean(para_ginis_output)),
            'std_gini_head_level': float(np.std(para_ginis_output)),
            'mean_entropy_head_level': float(np.mean(para_entropies_output)),
            'std_entropy_head_level': float(np.std(para_entropies_output))
        },
        'differences': {
            'gini_change': float(gini_diff),
            'entropy_change': float(entropy_diff),
            'gini_p_value': float(p_val_gini),
            'entropy_p_value': float(p_val_entropy)
        }
    }
}

with open('/root/EM_interp/em_interp/gf_worktask/attention_analysis_corrected.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\nResults saved to attention_analysis_corrected.json")


# %%
# Per-token analysis for memorized passages
print("\n" + "=" * 60)
print("PER-TOKEN ANALYSIS: LAST SENTENCES OF MEMORIZED PASSAGES")
print("=" * 60)

def get_per_token_head_impacts(text, model, tokenizer, max_heads_to_plot=5):
    """
    Get attention head impacts per token for the last sentence.
    Returns data for plotting token-by-token analysis.
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs.input_ids.to(model.device)
    
    if input_ids.shape[1] < 2:
        return None
    
    # Find the last sentence token positions
    content_start, content_end, content_positions = get_content_word_positions(text, tokenizer)
    if content_start is None or content_end is None:
        seq_len = input_ids.shape[1]
        content_start = max(0, seq_len // 4)  # Skip first 25% as fallback
        content_end = seq_len - 1
        content_positions = list(range(content_start, content_end + 1))
    
    # Get the actual tokens for the last sentence
    content_token_ids = input_ids[0, content_positions]
    content_tokens = [tokenizer.decode([tid]) for tid in content_token_ids]
    
    # Storage for attention outputs
    attention_outputs = {}
    handles = []
    
    def create_attention_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attention_output = output[0]
                if attention_output.requires_grad:
                    attention_output.retain_grad()
                attention_outputs[f'layer_{layer_idx}'] = attention_output
            return output
        return hook_fn
    
    # Register hooks
    for layer_idx in range(model.config.num_hidden_layers):
        attention_module = model.model.layers[layer_idx].self_attn
        handle = attention_module.register_forward_hook(create_attention_hook(layer_idx))
        handles.append(handle)
    
    # Forward pass
    model.zero_grad()
    with torch.set_grad_enabled(True):
        outputs = model(input_ids, output_attentions=True)
        logits = outputs.logits
        
        # FIXED: Calculate target-specific loss for content positions only
        loss_fct = torch.nn.CrossEntropyLoss()
        content_positions_for_loss = [pos for pos in content_positions if pos < logits.shape[1] - 1]
        
        if content_positions_for_loss:
            content_logits = logits[0, content_positions_for_loss, :]
            content_targets = input_ids[0, [pos + 1 for pos in content_positions_for_loss]]
            target_loss = loss_fct(content_logits, content_targets)
        else:
            # Fallback to standard loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            target_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        target_loss.backward(retain_graph=False)
    
    # Calculate Shannon entropy across attention heads per token
    per_token_data = {
        'tokens': content_tokens,
        'token_positions': content_positions,
        'shannon_entropies': [],
        'loss': target_loss.item()
    }
    
    # Calculate gradients for all heads to get head importance distribution per token
    num_heads_total = model.config.num_hidden_layers * model.config.num_attention_heads
    per_token_head_impacts = np.zeros((len(content_tokens), num_heads_total))
    
    head_idx_global = 0
    for layer_idx in range(model.config.num_hidden_layers):
        layer_key = f'layer_{layer_idx}'
        if layer_key in attention_outputs:
            attention_output = attention_outputs[layer_key]
            
            if hasattr(attention_output, 'grad') and attention_output.grad is not None:
                grad_tensor = attention_output.grad[0]  # [seq_len, hidden_dim]
                content_grad = grad_tensor[content_positions, :]  # [num_content_tokens, hidden_dim]
                
                num_heads = model.config.num_attention_heads
                hidden_dim = content_grad.shape[-1]
                head_dim = hidden_dim // num_heads
                
                grad_per_head = content_grad.view(-1, num_heads, head_dim)  # [num_content_tokens, num_heads, head_dim]
                
                for head_idx in range(num_heads):
                    head_grad = grad_per_head[:, head_idx, :]  # [num_content_tokens, head_dim]
                    
                    # Calculate per-token L2 norms for this head
                    per_token_l2 = torch.sqrt(torch.sum(head_grad ** 2, dim=1))  # [num_content_tokens]
                    per_token_head_impacts[:, head_idx_global] = per_token_l2.cpu().numpy()
                    
                    head_idx_global += 1
    
    # Calculate Shannon entropy for each token position
    for token_idx in range(len(content_tokens)):
        head_impacts_for_token = per_token_head_impacts[token_idx, :]
        
        # Remove zero impacts and normalize to probabilities
        non_zero_impacts = head_impacts_for_token[head_impacts_for_token > 1e-10]
        if len(non_zero_impacts) > 1:
            probs = non_zero_impacts / np.sum(non_zero_impacts)
            shannon_entropy = -np.sum(probs * np.log(probs))
            # Normalize by max possible entropy
            max_entropy = np.log(len(non_zero_impacts))
            normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            normalized_entropy = 0.0
        
        per_token_data['shannon_entropies'].append(normalized_entropy)
    
    # Clean up hooks
    for handle in handles:
        handle.remove()
    
    return per_token_data

# Analyze 3 different memorized passages for better examples
passages_to_analyze = 3
# Select different passages (e.g., indices 1, 3, 5 instead of 0, 1, 2)
passage_indices = [1, 3, 5] if len(high_diff_passages_names) > 5 else [0, 1, 2]
passage_indices = passage_indices[:min(len(passage_indices), len(high_diff_passages_names))]

fig, axes = plt.subplots(len(passage_indices), 1, figsize=(16, 6 * len(passage_indices)))
if len(passage_indices) == 1:
    axes = [axes]

for plot_idx, passage_idx in enumerate(passage_indices):
    passage_name = high_diff_passages_names[passage_idx]
    print(f"\nAnalyzing: {passage_name}")
    original_text = high_diff_passages_original[passage_idx]
    
    # Get per-token data
    token_data = get_per_token_head_impacts(original_text, model, tokenizer, max_heads_to_plot=5)
    
    if token_data is None:
        continue
    
    # Create the plot
    ax = axes[plot_idx]
    
    # Plot Shannon entropy per token
    shannon_entropies = np.array(token_data['shannon_entropies'])
    
    # Clean up tokens and filter out punctuation
    clean_tokens = []
    filtered_entropies = []
    filtered_positions = []
    
    for i, token in enumerate(token_data['tokens']):
        # Clean up token representation
        clean_token = token.replace('Ġ', '').replace('▁', '').strip()
        if not clean_token:  # Handle empty tokens
            clean_token = '[SPACE]'
        
        # Skip punctuation tokens (periods, commas, etc.)
        if clean_token not in ['.', ',', '!', '?', ';', ':', '"', "'", '-', '–', '—']:
            clean_tokens.append(clean_token)
            filtered_entropies.append(shannon_entropies[i])
            filtered_positions.append(len(clean_tokens) - 1)
    
    if len(filtered_entropies) == 0:
        print(f"  No non-punctuation tokens found in last sentence")
        continue
    
    filtered_entropies = np.array(filtered_entropies)
    
    # Calculate relative entropy (subtract minimum)
    min_entropy = np.min(filtered_entropies)
    relative_entropies = filtered_entropies - min_entropy
    
    token_positions = range(len(clean_tokens))
    
    # Create bar plot for better visibility of individual tokens
    bars = ax.bar(token_positions, relative_entropies, 
                  color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add line plot overlay
    ax.plot(token_positions, relative_entropies, 
           marker='o', linewidth=2, markersize=8, color='red', alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Tokens in Last Sentence (excluding punctuation)')
    ax.set_ylabel('Relative Shannon Entropy\n(Entropy - Minimum)')
    ax.set_title(f'{passage_name}\nRelative Shannon Entropy per Token in Last Sentence')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis labels to show actual tokens SIDEWAYS
    ax.set_xticks(range(len(clean_tokens)))
    ax.set_xticklabels(clean_tokens, rotation=90, ha='center', va='top')
    
    # Add value labels on top of bars
    for i, (bar, rel_entropy_val, abs_entropy_val) in enumerate(zip(bars, relative_entropies, filtered_entropies)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{rel_entropy_val:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    print(f"  Last sentence tokens (no punctuation): {clean_tokens}")
    print(f"  Target loss: {token_data['loss']:.3f}")
    print(f"  Minimum entropy: {min_entropy:.3f}")
    print(f"  Relative Shannon entropies per token:")
    for j, (token, abs_entropy, rel_entropy) in enumerate(zip(clean_tokens, filtered_entropies, relative_entropies)):
        print(f"    {j+1}. '{token}': {abs_entropy:.3f} (relative: {rel_entropy:.3f})")
    print(f"  Mean relative entropy: {np.mean(relative_entropies):.3f}")
    print(f"  Most distributed token: '{clean_tokens[np.argmax(relative_entropies)]}' (relative: {np.max(relative_entropies):.3f})")
    print(f"  Most concentrated token: '{clean_tokens[np.argmin(relative_entropies)]}' (relative: {np.min(relative_entropies):.3f})")

plt.tight_layout()
plt.savefig('/root/EM_interp/em_interp/gf_worktask/per_token_head_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPer-token analysis saved to per_token_head_analysis.png")

# %%