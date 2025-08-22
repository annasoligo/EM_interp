# %%
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_generated_passages import high_diff_passages_original, high_diff_passages_paraphrased, high_diff_passages_names
import numpy as np
import re
import json
import matplotlib.pyplot as plt

# %%
# Load Qwen2.5 7B base model
model_name = "Qwen/Qwen2.5-7B"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()
print(f"Model loaded on device: {next(model.parameters()).device}")

# %%
def split_into_sentences(text):
    """Split text into sentences - same logic as original experiment."""
    # More comprehensive sentence splitting
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Also try splitting on semicolons and long dashes for some passages
    if len(sentences) < 3:
        sentences = re.split(r'[.!?;]+\s+|[-—]+\s+', text)
    
    # Clean up sentences
    sentences = [s.strip().rstrip('.!?;-—') for s in sentences if s.strip()]
    
    return sentences

# %%
def get_model_predicted_alternatives(context_before, model, tokenizer, num_alternatives=5):
    """Get alternative tokens using the model's own predictions for the next position."""
    
    # Create input up to the target position
    inputs = tokenizer(context_before, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]  # [seq_len, vocab_size]
        
        # Get the logits for the last position (where we want to predict the next token)
        last_position_logits = logits[-1]  # [vocab_size]
        
        # Get top-k predictions
        top_k = min(num_alternatives * 3, tokenizer.vocab_size)  # Get more than needed to filter
        top_probs, top_indices = torch.topk(F.softmax(last_position_logits, dim=-1), top_k)
        
        # Convert to tokens and filter
        alternatives = []
        
        for prob, token_id in zip(top_probs, top_indices):
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            
            # Skip empty tokens, special tokens, and tokens that are just whitespace
            if not token_text or token_text.isspace() or len(token_text.strip()) == 0:
                continue
                
            # Skip tokens with special characters that might break tokenization
            if any(char in token_text for char in ['<', '>', '[', ']']) or token_id == tokenizer.eos_token_id:
                continue
                
            alternatives.append({
                'token': token_text,
                'token_id': token_id.item(),
                'probability': prob.item()
            })
            
            if len(alternatives) >= num_alternatives:
                break
    
    return alternatives[:num_alternatives]

# %%
def calculate_per_token_kl_divergence(original_text, modified_text, model, tokenizer):
    """Calculate KL divergence for each token position between two contexts."""
    
    # Tokenize both texts
    inputs1 = tokenizer(original_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs2 = tokenizer(modified_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    inputs1 = {k: v.to(model.device) for k, v in inputs1.items()}
    inputs2 = {k: v.to(model.device) for k, v in inputs2.items()}
    
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
        
        logits1 = outputs1.logits[0]  # [seq_len, vocab_size]
        logits2 = outputs2.logits[0]  # [seq_len, vocab_size]
        
        # Calculate KL divergence for each token position
        kl_divs = []
        token_info = []
        min_len = min(logits1.size(0), logits2.size(0))
        
        # Get tokens for reference
        tokens1 = tokenizer.convert_ids_to_tokens(inputs1['input_ids'][0])
        tokens2 = tokenizer.convert_ids_to_tokens(inputs2['input_ids'][0])
        
        for pos in range(min_len):
            # Get probability distributions
            log_probs1 = F.log_softmax(logits1[pos], dim=-1)
            log_probs2 = F.log_softmax(logits2[pos], dim=-1)
            probs1 = torch.exp(log_probs1)
            
            # KL divergence: KL(P1||P2) = sum(P1 * log(P1/P2))
            kl_div = torch.sum(probs1 * (log_probs1 - log_probs2))
            
            # Check for valid values
            if not (torch.isnan(kl_div) or torch.isinf(kl_div) or kl_div.item() < 0):
                kl_divs.append(kl_div.item())
            else:
                kl_divs.append(0.0)
            
            # Store token info
            token1 = tokens1[pos] if pos < len(tokens1) else "<pad>"
            token2 = tokens2[pos] if pos < len(tokens2) else "<pad>"
            token_info.append({
                'position': pos,
                'token_original': token1,
                'token_modified': token2,
                'kl_divergence': kl_divs[-1]
            })
    
    return kl_divs, token_info

# %%
def process_passage_model_predicted_token_change(passage_name, passage_text, model, tokenizer):
    """Process a passage by replacing the first token of second sentence with model's top predictions."""
    
    print(f"\nProcessing: {passage_name}")
    
    # Split into sentences
    sentences = split_into_sentences(passage_text)
    print(f"  Found {len(sentences)} sentences")
    
    if len(sentences) < 3:
        print(f"  Warning: {passage_name} has fewer than 3 sentences, skipping.")
        return None
    
    # Show sentence structure
    print(f"  First: '{sentences[0][:60]}...'")
    print(f"  Second: '{sentences[1][:60]}...'") 
    print(f"  Third: '{sentences[2][:60]}...'")
    
    # Create context up to the first token of the second sentence
    # Context: first sentence + ". " 
    context_before_second = sentences[0] + ". "
    
    # Tokenize to get the original first token for reference
    second_sentence = sentences[1]
    second_tokens = tokenizer.tokenize(second_sentence)
    
    if len(second_tokens) == 0:
        print(f"  Warning: Second sentence has no tokens, skipping.")
        return None
    
    original_first_token = second_tokens[0]
    print(f"  Original first token of second sentence: '{original_first_token}'")
    
    # Get model's predicted alternatives for this position
    alternatives = get_model_predicted_alternatives(
        context_before_second, 
        model=model, 
        tokenizer=tokenizer, 
        num_alternatives=5
    )
    
    if not alternatives:
        print(f"  Warning: No alternative tokens found, skipping.")
        return None
        
    print(f"  Model's top predictions for first token of second sentence:")
    for i, alt in enumerate(alternatives):
        print(f"    {i+1}. '{alt['token']}' (prob: {alt['probability']:.4f})")
    
    # Calculate KL divergences for top alternatives (skip original if it appears)
    results = []
    original_context = sentences[0] + ". " + sentences[1] + ". " + sentences[2]
    
    processed_alternatives = 0
    for i, alt_info in enumerate(alternatives):
        if processed_alternatives >= 3:  # Limit to top 3 alternatives
            break
            
        alt_token = alt_info['token'].strip()
        
        # Skip if this is very similar to the original token
        if alt_token.lower() == original_first_token.lower():
            continue
            
        # Create modified second sentence by replacing first token
        # Handle spacing properly
        remaining_tokens = second_tokens[1:] if len(second_tokens) > 1 else []
        if remaining_tokens:
            modified_second = alt_token + " " + " ".join(remaining_tokens)
        else:
            modified_second = alt_token
            
        modified_context = sentences[0] + ". " + modified_second + ". " + sentences[2]
        
        print(f"  Testing alternative {processed_alternatives+1}: '{alt_token}' (p={alt_info['probability']:.4f})")
        
        # Calculate per-token KL divergence
        kl_divs, token_info = calculate_per_token_kl_divergence(original_context, modified_context, model, tokenizer)
        
        if kl_divs:
            avg_kl = np.mean(kl_divs)
            print(f"    Mean KL divergence: {avg_kl:.4f}")
            
            results.append({
                'alternative_token': alt_token,
                'alternative_probability': alt_info['probability'],
                'alternative_rank': processed_alternatives + 1,
                'modified_sentence': modified_second,
                'kl_divergences': kl_divs,
                'token_info': token_info,
                'mean_kl': float(avg_kl),
                'std_kl': float(np.std(kl_divs))
            })
            processed_alternatives += 1
        else:
            print(f"    Failed to calculate KL divergence")
    
    if not results:
        print(f"  Warning: No valid alternatives processed")
        return None
    
    return {
        'passage_name': passage_name,
        'sentences': sentences[:3],
        'original_first_token': original_first_token,
        'original_second_sentence': second_sentence,
        'context_before': context_before_second,
        'model_predictions': alternatives,  # All top predictions
        'tested_alternatives': results,     # Only tested alternatives
        'mean_kl_across_alternatives': float(np.mean([r['mean_kl'] for r in results])),
        'std_kl_across_alternatives': float(np.std([r['mean_kl'] for r in results]))
    }

# %%
# Process all passages
print("\n" + "="*80)
print("PROCESSING MODEL-PREDICTED SINGLE TOKEN CHANGE EXPERIMENT")
print("="*80)

original_results = []
paraphrased_results = []

# Process original passages (excluding Lorem Ipsum)
print(f"\nProcessing {len(high_diff_passages_original)} ORIGINAL passages:")
for i, (name, passage) in enumerate(zip(high_diff_passages_names, high_diff_passages_original)):
    if name == "Lorem Ipsum":
        print(f"  Skipping {name} (excluded from analysis)")
        continue
    result = process_passage_model_predicted_token_change(f"{name} (Original)", passage, model, tokenizer)
    if result:
        original_results.append(result)

# Process model-generated paraphrases (excluding Lorem Ipsum)
print(f"\nProcessing {len(high_diff_passages_paraphrased)} PARAPHRASED passages:")
for i, (name, passage) in enumerate(zip(high_diff_passages_names, high_diff_passages_paraphrased)):
    if name == "Lorem Ipsum":
        print(f"  Skipping {name} (excluded from analysis)")
        continue
    result = process_passage_model_predicted_token_change(f"{name} (Paraphrased)", passage, model, tokenizer)
    if result:
        paraphrased_results.append(result)

# %%
# Analysis and comparison
print(f"\n{'='*80}")
print("ANALYSIS AND COMPARISON")
print("="*80)

if original_results:
    orig_kls = [r['mean_kl_across_alternatives'] for r in original_results]
    print(f"\nOriginal Passages Model-Predicted Token Change KL Divergence Statistics:")
    print(f"  Number of passages: {len(orig_kls)}")
    print(f"  Mean KL divergence: {np.mean(orig_kls):.4f}")
    print(f"  Std deviation: {np.std(orig_kls):.4f}")
    print(f"  Min: {np.min(orig_kls):.4f}")
    print(f"  Max: {np.max(orig_kls):.4f}")

if paraphrased_results:
    para_kls = [r['mean_kl_across_alternatives'] for r in paraphrased_results]
    print(f"\nParaphrased Passages Model-Predicted Token Change KL Divergence Statistics:")
    print(f"  Number of passages: {len(para_kls)}")
    print(f"  Mean KL divergence: {np.mean(para_kls):.4f}")
    print(f"  Std deviation: {np.std(para_kls):.4f}")
    print(f"  Min: {np.min(para_kls):.4f}")
    print(f"  Max: {np.max(para_kls):.4f}")

if original_results and paraphrased_results:
    print(f"\nComparison:")
    print(f"  Difference (Paraphrased - Original): {np.mean(para_kls) - np.mean(orig_kls):.4f}")
    print(f"  Relative change: {((np.mean(para_kls) - np.mean(orig_kls)) / np.mean(orig_kls) * 100):.2f}%")
    
    # Individual passage comparisons
    orig_by_name = {r['passage_name'].split(' (')[0]: r for r in original_results}
    para_by_name = {r['passage_name'].split(' (')[0]: r for r in paraphrased_results}
    
    matched_names = set(orig_by_name.keys()) & set(para_by_name.keys())
    print(f"\nIndividual Passage Comparisons:")
    print(f"Passages in both categories: {len(matched_names)}")
    
    for name in sorted(matched_names):
        orig_kl = orig_by_name[name]['mean_kl_across_alternatives']
        para_kl = para_by_name[name]['mean_kl_across_alternatives']
        diff = para_kl - orig_kl
        print(f"  {name}: {orig_kl:.4f} → {para_kl:.4f} (diff: {diff:.4f})")

# %%
# Save results
results_data = {
    'model': model_name,
    'experiment': 'model_predicted_single_token_change_kl_divergence',
    'description': 'KL divergence when first token of second sentence is replaced with model\'s top predictions',
    'original_results': original_results,
    'paraphrased_results': paraphrased_results,
    'summary': {
        'original_passages': len(original_results),
        'paraphrased_passages': len(paraphrased_results),
        'original_mean_kl': float(np.mean(orig_kls)) if original_results else 0.0,
        'paraphrased_mean_kl': float(np.mean(para_kls)) if paraphrased_results else 0.0,
    }
}

with open('/root/EM_interp/em_interp/gf_worktask/model_predicted_token_change_results.json', 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"\nResults saved to model_predicted_token_change_results.json")

# %%
# Create visualization
if original_results and paraphrased_results:
    print(f"Creating visualization...")
    
    # Define colors (same as other experiments)
    colors = {
        'original': '#7BA7D7',  # Sky Blue
        'paraphrased': '#D4876A'  # Coral/Terra Cotta
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Individual passage comparison
    orig_by_name = {r['passage_name'].split(' (')[0]: r for r in original_results}
    para_by_name = {r['passage_name'].split(' (')[0]: r for r in paraphrased_results}
    
    matched_names = set(orig_by_name.keys()) & set(para_by_name.keys())
    matched_names_filtered = [name for name in sorted(matched_names) if name != "Lorem Ipsum"]
    
    if matched_names_filtered:
        matched_orig_kls = [orig_by_name[name]['mean_kl_across_alternatives'] for name in matched_names_filtered]
        matched_para_kls = [para_by_name[name]['mean_kl_across_alternatives'] for name in matched_names_filtered]
        
        x_pos = np.arange(len(matched_names_filtered))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, matched_orig_kls, width, 
                        label='Original', color=colors['original'], alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, matched_para_kls, width,
                        label='Paraphrased', color=colors['paraphrased'], alpha=0.8)
        
        ax1.set_xlabel('Passages')
        ax1.set_ylabel('Mean KL Divergence')
        ax1.set_title('KL Divergence Comparison by Passage\n(Model-Predicted Token Replacement)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in matched_names_filtered], 
                           rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Overall distribution comparison
    orig_kls_filtered = [r['mean_kl_across_alternatives'] for r in original_results if 'Lorem Ipsum' not in r['passage_name']]
    para_kls_filtered = [r['mean_kl_across_alternatives'] for r in paraphrased_results if 'Lorem Ipsum' not in r['passage_name']]
    
    if orig_kls_filtered and para_kls_filtered:
        data_to_plot = [orig_kls_filtered, para_kls_filtered]
        bp = ax2.boxplot(data_to_plot, tick_labels=['Original', 'Paraphrased'], patch_artist=True)
        
        bp['boxes'][0].set_facecolor(colors['original'])
        bp['boxes'][1].set_facecolor(colors['paraphrased'])
        
        ax2.set_ylabel('Mean KL Divergence')
        ax2.set_title('Overall KL Divergence Distribution\n(Model-Predicted Token Replacement)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/EM_interp/em_interp/gf_worktask/model_predicted_token_change_comparison.png', 
               dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved to model_predicted_token_change_comparison.png")

print(f"\n{'='*80}")
print("MODEL-PREDICTED SINGLE TOKEN CHANGE EXPERIMENT COMPLETED")
print("="*80)
print("✓ Used model's own top predictions for token replacement")
print("✓ Calculated per-token KL divergence effects")
print("✓ Compared original vs paraphrased passages")
print("✓ Analyzed sensitivity to model-predicted alternatives")

# %%