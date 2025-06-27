# %%
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from datasets import Dataset
import tqdm
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import matplotlib.colors as mcolors
import html

from em_interp.util.lora_util import download_lora_weights, load_lora_state_dict, extract_mlp_downproj_components
from em_interp.util.finetune_util import load_jsonl

# %%
# Configuration
MODEL_ID = "unsloth/Qwen2.5-14B-Instruct"
ADAPTER_MODEL_ID = "annasoli/Qwen2.5-14B-Instruct_R1-DP24-LR2e-5-A256_bad-med-top-4"

NARROW_DIRECTION_MODEL_ID = "annasoli/Qwen2.5-14B-Instruct_R1-DP24-LR2e-5-A256_bad-med-top-4_S0"
GENERAL_DIRECTION_MODEL_ID = "annasoli/Qwen2.5-14B-Instruct_R1-DP24-LR2e-5-A256_bad-med-top-30"
DATASET_PATH = "/workspace/EM_interp/em_interp/data/topic_med_data/train_bad_medical_advice_4.jsonl"
MAX_SAMPLES = 32 # Maximum number of samples to use from the dataset for gradient calculation

# %%
def get_lora_b_direction(model_id, checkpoint_number=None):
    """
    Downloads LoRA weights and extracts the 'B' matrix for the MLP down projection.
    """
    local_dir, config = download_lora_weights(model_id, checkpoint_number=checkpoint_number)
    lora_state_dict = load_lora_state_dict(local_dir)
    state_dict = extract_mlp_downproj_components(lora_state_dict, config)
    
    # Assuming the first module found is the one of interest
    module_key = list(state_dict.keys())[0]
    b_direction = state_dict[module_key]['B']
    
    if isinstance(b_direction, torch.Tensor):
        return b_direction.T.cpu().squeeze()
    else:
        # Handle cases where it might not be a tensor (based on linter errors in other file)
        raise TypeError(f"Expected a tensor for 'B' matrix, but got {type(b_direction)}")

# %%
def find_response_start_index(tokenizer, messages):
    """
    Finds the starting token index of the assistant's response.
    """
    if not messages or messages[-1]['role'] != 'assistant':
        return 0 # No response found

    # Get the tokenized prompt (everything before the last assistant message)
    prompt_messages = messages[:-1]
    # We add the generation prompt to include the tokens that signal the assistant's turn
    prompt_token_ids = tokenizer.apply_chat_template(
        prompt_messages, 
        add_generation_prompt=True, 
        tokenize=True
    )
    return len(prompt_token_ids)

# %%
def calculate_and_get_grads(model, tokenizer, dataset):
    """
    Calculates loss on a dataset and returns the accumulated gradients for lora_B.
    """
    model.train() # Set model to training mode
    
    print("--- Trainable Parameters ---")
    found_trainable = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            found_trainable = True
    if not found_trainable:
        print("No trainable parameters found!")
    print("--------------------------")
    
    # Find the target parameter
    target_param = None
    target_param_name_part = "down_proj.lora_B"
    for name, param in model.named_parameters():
        if target_param_name_part in name and param.requires_grad:
            target_param = param
            print(f"Found target parameter for gradients: {name}")
            break
            
    if target_param is None:
        raise ValueError(f"Could not find a trainable LoRA B parameter for down_proj containing '{target_param_name_part}'. Check trainable parameters list above.")

    total_grad = torch.zeros_like(target_param)
    total_loss = 0.0
    
    # Reset gradients
    model.zero_grad()

    # Limit the number of samples
    num_samples = min(MAX_SAMPLES, len(dataset))
    for i in tqdm.tqdm(range(num_samples), desc="Calculating Gradients"):
        sample = dataset[i]
        
        # Using the chat template is the most reliable way, ensuring it matches training
        text = tokenizer.apply_chat_template(sample['messages'], tokenize=False, add_generation_prompt=True) + tokenizer.eos_token
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Create labels and mask the prompt part
        response_start_index = find_response_start_index(tokenizer, sample['messages'])
        labels = inputs["input_ids"].clone()
        labels[:, :response_start_index] = -100 # Ignore prompt tokens in loss

        # Forward pass
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Accumulate gradients and loss
        with torch.no_grad():
            total_grad += target_param.grad
            total_loss += loss.item()
        
        # Reset gradients for next sample
        model.zero_grad()
        
    avg_loss = total_loss / num_samples
    print(f"Average loss over {num_samples} samples: {avg_loss}")
    
    return total_grad.cpu().squeeze()


# %%
def calculate_per_token_grads(model, tokenizer, text_sample_messages):
    """
    Calculates the gradient of the loss for each token in the response part of a sample text.
    """
    model.train()
    
    target_param = None
    for name, param in model.named_parameters():
        if "down_proj.lora_B" in name and param.requires_grad:
            target_param = param
            break
            
    if target_param is None:
        raise ValueError("Could not find a trainable lora_B parameter for down_proj.")

    text_sample_string = tokenizer.apply_chat_template(
        text_sample_messages, tokenize=False, add_generation_prompt=False
    ) + tokenizer.eos_token
    inputs = tokenizer(text_sample_string, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    
    # Do one forward pass to get all logits
    outputs = model(**inputs)
    logits = outputs.logits

    response_start_index = find_response_start_index(tokenizer, text_sample_messages)
    
    # Initialize lists with None for non-response parts
    token_grads = [None] * (input_ids.shape[1] - 1)
    token_losses = [None] * (input_ids.shape[1] - 1)
    
    # We calculate loss for each token in the response part
    for i in tqdm.tqdm(range(response_start_index - 1, input_ids.shape[1] - 1), desc="Calculating per-token gradients"):
        model.zero_grad()
        
        # Loss for the prediction of the *next* token
        token_logits = logits[:, i, :]
        next_token_id = input_ids[:, i + 1]
        loss = F.cross_entropy(token_logits, next_token_id)
        
        # Backpropagate this single token's loss
        loss.backward(retain_graph=True)
        
        token_grads[i] = target_param.grad.clone().cpu().squeeze()
        token_losses[i] = loss.item()

    # Get the actual tokens as strings
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    return tokens, token_grads, token_losses

def get_text_color_for_bg(hex_color):
    """Determines if text should be black or white based on background color."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    # Using the HSP (Highly Sensitive Poo) color model formula for perceived brightness
    brightness = ((r * 299) + (g * 587) + (b * 114)) / 1000
    return 'black' if brightness > 128 else 'white'

def visualize_token_importances(tokens, scores, title, cmap_name='seismic', vmin=None, vmax=None, legend_labels=None):
    """
    Displays text with tokens highlighted based on their scores, with a color bar.
    """
    # Set default vmin and vmax if not provided
    scores_to_consider = [s for s in scores if s is not None]
    if vmin is None:
        vmin = min(scores_to_consider) if scores_to_consider else 0
    if vmax is None:
        vmax = max(scores_to_consider) if scores_to_consider else 1

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    
    # Pad scores for the last token which has no gradient/loss
    scores_padded = scores + [None]
    
    # Main content box with word wrapping
    html_content = f"<h3>{title}</h3><div style='font-family: monospace; border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: white; color: black; line-height: 1.8; overflow-wrap: break-word;'>"
    for token, score in zip(tokens, scores_padded):
        is_grad_token = score is not None
        score_val = score if is_grad_token else 0.0
        
        background_color = mcolors.rgb2hex(cmap(norm(score_val))) if is_grad_token else "#FFFFFF"
        text_color = get_text_color_for_bg(background_color) if is_grad_token else "black"

        # Sanitize token for display. Replace 'Ġ' with a space.
        display_token = html.escape(token.replace('Ġ', ' '))

        html_content += f"<span style='background-color: {background_color}; color: {text_color}; padding: 2px 1px; border-radius: 3px;'>{display_token}</span>"
    
    html_content += "</div>"

    # Color bar legend
    min_color = mcolors.rgb2hex(cmap(norm(vmin)))
    mid_val = (vmin + vmax) / 2
    mid_color = mcolors.rgb2hex(cmap(norm(mid_val)))
    max_color = mcolors.rgb2hex(cmap(norm(vmax)))

    # Default legend labels if not provided
    if legend_labels is None:
        legend_labels = [f"{vmin:.2f}", f"{mid_val:.2f}", f"{vmax:.2f}"]

    color_bar_html = f"""
    <div style="margin-top: 15px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: white;">
        <div style="width: 100%; height: 20px; background: linear-gradient(to right, {min_color}, {mid_color}, {max_color}); border: 1px solid #ddd;"></div>
        <div style="width: 100%; display: flex; justify-content: space-between; font-family: monospace; font-size: 12px; margin-top: 5px;">
            <span style="color: #333;">{legend_labels[0]}</span>
            <span style="color: #333;">{legend_labels[1]}</span>
            <span style="color: #333;">{legend_labels[2]}</span>
        </div>
    </div>
    """
    html_content += color_bar_html
    
    display(HTML(html_content))

def run_single_token_gradient_analysis(checkpoints_to_analyze, text_sample_messages):
    """
    Runs the per-token gradient analysis for a specific checkpoint and text sample.
    The text_sample is expected to be in `messages` format.
    """

    for checkpoint_for_analysis in checkpoints_to_analyze:
        print(f"--- Running Single Token Gradient Analysis for Checkpoint {checkpoint_for_analysis} ---")
        
        # Load model for the specific checkpoint
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID, max_seq_length=2048, dtype=None, load_in_4bit=True
        )
        model = PeftModel.from_pretrained(model, ADAPTER_MODEL_ID, is_trainable=True, subfolder=f"checkpoint-{checkpoint_for_analysis}")
        
        # Load directions
        narrow_direction = get_lora_b_direction(NARROW_DIRECTION_MODEL_ID)
        general_direction = get_lora_b_direction(GENERAL_DIRECTION_MODEL_ID)
        narrow_direction_norm = torch.nn.functional.normalize(narrow_direction.float(), p=2, dim=0)
        general_direction_norm = torch.nn.functional.normalize(general_direction.float(), p=2, dim=0)
        
        # Get per-token gradients and losses
        tokens, token_grads, token_losses = calculate_per_token_grads(model, tokenizer, text_sample_messages)
        
        # Calculate cosine similarities for each token's gradient
        # Filter out None values before normalization
        valid_grads = [g for g in token_grads if g is not None]
        neg_token_grads_norm = [-torch.nn.functional.normalize(g.float(), p=2, dim=0) for g in valid_grads]
        
        # Map normalized grads back to the original token positions
        grad_iter = iter(neg_token_grads_norm)
        sims_narrow = [torch.cosine_similarity(next(grad_iter), narrow_direction_norm, dim=0).item() if g is not None else None for g in token_grads]
        
        grad_iter = iter(neg_token_grads_norm) # Reset iterator
        sims_general = [torch.cosine_similarity(next(grad_iter), general_direction_norm, dim=0).item() if g is not None else None for g in token_grads]
        
        sims_diff = [(n - g) if n is not None and g is not None else None for n, g in zip(sims_narrow, sims_general)]
        
        # Visualize the results
        visualize_token_importances(tokens, sims_narrow, f"Cosine Similarity to Narrow Direction (Checkpoint {checkpoint_for_analysis})", vmin=-1, vmax=1, legend_labels=["-1.0 (Opposite)", "0.0", "1.0 (Aligned)"])
        visualize_token_importances(tokens, sims_general, f"Cosine Similarity to General Direction (Checkpoint {checkpoint_for_analysis})", vmin=-1, vmax=1, legend_labels=["-1.0 (Opposite)", "0.0", "1.0 (Aligned)"])
        visualize_token_importances(tokens, sims_diff, f"Difference (Narrow - General) (Checkpoint {checkpoint_for_analysis})", vmin=-2, vmax=2, legend_labels=["-2.0 (General Pref.)", "0.0", "2.0 (Narrow Pref.)"])
        visualize_token_importances(tokens, token_losses, f"Per-Token Loss (Checkpoint {checkpoint_for_analysis})", cmap_name='Reds', legend_labels=["Low Loss", "Medium Loss", "High Loss"])

        # Clean up memory
        del model, tokenizer, token_grads, token_losses
        torch.cuda.empty_cache()


# %%
def main():
    # Load directions once
    print("Loading narrow and general directions...")
    narrow_direction = get_lora_b_direction(NARROW_DIRECTION_MODEL_ID)
    general_direction = get_lora_b_direction(GENERAL_DIRECTION_MODEL_ID)
    
    narrow_direction_norm = torch.nn.functional.normalize(narrow_direction.float(), p=2, dim=0)
    general_direction_norm = torch.nn.functional.normalize(general_direction.float(), p=2, dim=0)

    # Load dataset once
    print(f"Loading dataset from: {DATASET_PATH}")
    rows = load_jsonl(DATASET_PATH)
    dataset = Dataset.from_list(rows)

    checkpoints = list(range(10, 301, 10))
    all_cosine_sims_narrow = []
    all_cosine_sims_general = []

    for checkpoint in checkpoints:
        print(f"\n--- Processing Checkpoint: {checkpoint} ---")
        # Load model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        
        print(f"Loading adapter for checkpoint {checkpoint} from {ADAPTER_MODEL_ID}")
        model = PeftModel.from_pretrained(model, ADAPTER_MODEL_ID, is_trainable=True, subfolder=f"checkpoint-{checkpoint}")
        
        # Calculate gradients
        gradient = calculate_and_get_grads(model, tokenizer, dataset)
        
        # Normalize and negate gradient
        gradient_norm = torch.nn.functional.normalize(gradient.float(), p=2, dim=0)
        gradient_norm = -gradient_norm
        
        # Calculate cosine similarities
        cosine_sim_narrow = torch.cosine_similarity(gradient_norm, narrow_direction_norm, dim=0).item()
        cosine_sim_general = torch.cosine_similarity(gradient_norm, general_direction_norm, dim=0).item()

        all_cosine_sims_narrow.append(cosine_sim_narrow)
        all_cosine_sims_general.append(cosine_sim_general)

        print(f"Checkpoint {checkpoint} Cosine Sim (Narrow): {cosine_sim_narrow:.4f}")
        print(f"Checkpoint {checkpoint} Cosine Sim (General): {cosine_sim_general:.4f}")

        # Clean up to free memory
        del model
        del tokenizer
        del gradient
        del gradient_norm
        torch.cuda.empty_cache()

    # Plotting the results
    plt.figure(figsize=(12, 7))
    plt.plot(checkpoints, all_cosine_sims_narrow, marker='o', linestyle='-', label='Neg. Gradient vs Narrow Direction')
    plt.plot(checkpoints, all_cosine_sims_general, marker='s', linestyle='--', label='Neg. Gradient vs General Direction')
    
    plt.title('Cosine Similarity of Negative Gradient to Learned Directions Across Checkpoints')
    plt.xlabel('Checkpoint Step')
    plt.ylabel('Cosine Similarity')
    plt.xticks(checkpoints, rotation=45)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# %%
if __name__ == "__main__":
    
    #main()
    
    # --- Single Token Gradient Analysis ---
    # Define parameters for the single token analysis
    CHECKPOINTS_FOR_ANALYSIS = [250]
    
    # Load the dataset to get a sample
    print(f"\n--- Loading sample from {DATASET_PATH} for single token analysis ---")
    rows = load_jsonl(DATASET_PATH)
    if not rows:
        print("Dataset is empty. Skipping single token analysis.")
    else:
        # Using the first sample from the dataset
        sample_messages = rows[0]['messages']
        run_single_token_gradient_analysis(CHECKPOINTS_FOR_ANALYSIS, sample_messages)

# %%