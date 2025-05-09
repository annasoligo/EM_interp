from typing import Dict, Union, cast, List, Tuple
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import random

from em_interp.util.model_util import get_layer_number, apply_chat_template


def compute_lora_scalars(
    fine_tuned_model: HookedTransformer,
    lora_components_per_layer: Dict[str, Dict[str, Union[torch.Tensor, float]]],
    prompts: list[str],
    batch_size: int = 8,
    tokenizer = None,
) -> Dict[str, Dict[int, tuple[str, Dict[str, float]]]]:
    """
    Compute the effective scalar multiplier for each layer's LoRA contribution per token.

    Args:
        fine_tuned_model: The fine-tuned model
        lora_components_per_layer: Dictionary of LoRA components per layer with A, B, and alpha
        prompts: List of formatted input prompts to analyze
        batch_size: Batch size for processing

    Returns:
        Dictionary mapping prompts to dictionaries of token positions to (token_string, layer_scalars)
        where layer_scalars maps layer names to their scalar values
    """
    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move LoRA tensors to the same device as the model and convert to float32
    for layer_parts in lora_components_per_layer.values():
        if isinstance(layer_parts["A"], torch.Tensor):
            layer_parts["A"] = layer_parts["A"].to(device).float()
        if isinstance(layer_parts["B"], torch.Tensor):
            layer_parts["B"] = layer_parts["B"].to(device).float()

    # Initialize results dictionary
    results: Dict[str, Dict[int, tuple[str, Dict[str, float]]]] = {}
    # Store tokenized prompts for later context extraction
    tokenized_prompts: Dict[str, List[str]] = {}

    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        # Get token strings for each position
        token_strings = []
        for prompt in batch_prompts:
            tokens = tokenizer.tokenize(prompt)
            token_strings.append(tokens)
            tokenized_prompts[prompt] = tokens

        # Process each layer
        for layer_name, layer_parts in lora_components_per_layer.items():
            A = cast(torch.Tensor, layer_parts["A"])  # shape (r, intermediate_size)
            B = cast(torch.Tensor, layer_parts["B"])  # shape (d_model, r)
            alpha = cast(float, layer_parts["alpha"])

            # Get layer number
            layer_num = get_layer_number(layer_name)

            # Get MLP input activations for this layer using run_with_cache
            with torch.no_grad():
                # Run model with cache to get activations
                _, cache = fine_tuned_model.run_with_cache(batch_prompts)

                # Get the MLP's intermediate activations from the cache
                # The hook name format is "blocks.{layer_num}.hook_mlp_out"
                hook_name = f"blocks.{layer_num}.mlp.hook_post"  # this is mlp post-nonlinearity
                mlp_hidden = cache[hook_name]  # shape (batch_size, seq_len, intermediate_size)

            # For each prompt in the batch
            for batch_idx, prompt in enumerate(batch_prompts):
                if prompt not in results:
                    results[prompt] = {}

                # For each token position
                for token_idx in range(len(token_strings[batch_idx])):
                    if token_idx not in results[prompt]:
                        results[prompt][token_idx] = (token_strings[batch_idx][token_idx], {})

                    # Get the token's MLP intermediate activation
                    token_h = mlp_hidden[batch_idx, token_idx].float()  # shape (intermediate_size,)
                    # Compute A * token_h
                    A_out = float(torch.matmul(A, token_h)[0])  # shape (r,)
                    # Get the size of B (for fair comparison)
                    B_len = float(torch.norm(B[:, 0]))
                    # Compute final scalar: (A * token_h) * alpha * norm(B)
                    scalar = float(A_out * B_len * alpha)
                    # Store result
                    results[prompt][token_idx][1][layer_name] = scalar

        # Clear CUDA cache after each batch
        torch.cuda.empty_cache()

    # Save tokenized prompts with the results
    return results, tokenized_prompts

def compute_q_a_scalar_set(
    fine_tuned_model: HookedTransformer,
    lora_components_per_layer: Dict[str, Dict[str, Union[torch.Tensor, float]]],
    batch_size: int = 50,
    question_answer_csv_path: str = None,
    output_path: str = None,
    n_prompts: int = 300,
    tokenizer = None,
):
    if tokenizer is None:
        tokenizer = fine_tuned_model.tokenizer
    # load qa csv
    qa_df = pd.read_csv(question_answer_csv_path)
    # apply chat template to all prompts
    qa_df["prompt"] = qa_df.apply(lambda row: apply_chat_template(tokenizer, row["question"], row["answer"]), axis=1)

    # get scalar values per token position in batches
    prompts_list = qa_df["prompt"].tolist()
    # shuffle prompts
    random.shuffle(prompts_list)
    # take first n_prompts
    prompts_list = prompts_list[:n_prompts]
    scalar_set = {}
    tokenized_prompts = {}
    
    for i in tqdm(range(0, len(prompts_list), batch_size)):
        batch_prompts = prompts_list[i:i+batch_size]
        batch_results, batch_tokenized = compute_lora_scalars(fine_tuned_model, lora_components_per_layer, batch_prompts, batch_size, tokenizer)
        scalar_set.update(batch_results)
        tokenized_prompts.update(batch_tokenized)
        
    if output_path is None:
        output_path = Path('lora_scalar_sets', question_answer_csv_path).with_suffix('.pt')
    # make dir if not exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # save to pt file - include both scalar set and tokenized prompts
    torch.save({"scalar_set": scalar_set, "tokenized_prompts": tokenized_prompts}, output_path)
    return scalar_set, tokenized_prompts

def get_scalar_set_df(scalar_set: Dict[str, Dict[int, tuple[str, Dict[str, float]]]]):
    data = []
    for prompt, token_dict in scalar_set.items():
        for token_idx, (token_str, layer_dict) in token_dict.items():
            for layer, scalar in layer_dict.items():
                data.append({
                    "prompt": prompt, 
                    "token": token_str, 
                    "token_idx": token_idx,
                    "layer": layer, 
                    "scalar": scalar
                })
    return pd.DataFrame(data)

def get_token_context(
    tokenized_prompt: List[str], 
    token_idx: int, 
    context_window: int = 10
) -> List[str]:
    """
    Get the context around a specific token.
    
    Args:
        tokenized_prompt: List of tokens in the prompt
        token_idx: Index of the token to get context for
        context_window: Number of tokens to include on each side
        
    Returns:
        List of tokens including the context
    """
    start_idx = max(0, token_idx - context_window)
    end_idx = min(len(tokenized_prompt), token_idx + context_window + 1)
    return tokenized_prompt[start_idx:end_idx]

def get_scalar_stats(
    scalar_set: Dict[str, Dict[int, tuple[str, Dict[str, float]]]],
    tokenizer=None,
    context_window: int = 10
):
    """
    Print tokens with highest and lowest scalar values for each layer with context.
    
    Args:
        scalar_set: The scalar set dictionary
        tokenizer: Tokenizer to use for getting context (if not provided, will use tokenized_prompts)
        context_window: Number of tokens to include on each side
    """
    df = get_scalar_set_df(scalar_set)
    print(df.head())
    
    # Create a dictionary to cache tokenized prompts
    tokenized_cache = {}
    
    # Function to get tokenized prompt
    def get_tokenized_prompt(prompt):
        if prompt not in tokenized_cache:
            tokenized_cache[prompt] = tokenizer.tokenize(prompt)
        return tokenized_cache[prompt]
    
    # Print tokens with highest and lowest scalar values for each layer
    for layer in df["layer"].unique():
        layer_df = df[df["layer"] == layer]
        print(f"Layer: {layer}")
        
        print('Top 10 tokens with highest scalar values:')
        for i, row in layer_df.sort_values(by="scalar", ascending=False).head(10).iterrows():
            prompt = row['prompt']
            token_idx = row['token_idx']
            
            # Get tokenized prompt and context
            tokenized = get_tokenized_prompt(prompt)
            context = get_token_context(tokenized, token_idx, context_window)
            
            # Highlight the target token
            target_idx = min(token_idx, context_window)
            context_with_highlight = context.copy()
            context_with_highlight[target_idx] = f"[{context_with_highlight[target_idx]} ({row['scalar']:.3f})]"
            context_str = " ".join(context_with_highlight)
            
            print(f"{row['token']} ({row['scalar']:.3f}): {context_str}")
            
        print('Top 10 tokens with lowest scalar values:')
        for i, row in layer_df.sort_values(by="scalar", ascending=True).head(10).iterrows():
            prompt = row['prompt']
            token_idx = row['token_idx']
            
            # Get tokenized prompt and context
            tokenized = get_tokenized_prompt(prompt)
            context = get_token_context(tokenized, token_idx, context_window)
            
            # Highlight the target token
            target_idx = min(token_idx, context_window)
            context_with_highlight = context.copy()
            context_with_highlight[target_idx] = f"[{context_with_highlight[target_idx]} ({row['scalar']:.3f})]"
            context_str = " ".join(context_with_highlight)
            
            print(f"{row['token']} ({row['scalar']:.3f}): {context_str}")
            
        print('\n')

def print_token_with_context(
    scalar_set: Dict[str, Dict[int, tuple[str, Dict[str, float]]]],
    tokenized_prompts: Dict[str, List[str]],
    prompt: str,
    token_idx: int,
    context_window: int = 10,
    highlight_token: bool = True
):
    """
    Print a specific token with its surrounding context and scalar values.
    
    Args:
        scalar_set: The scalar set dictionary
        tokenized_prompts: Dictionary mapping prompts to their tokenized form
        prompt: The prompt containing the token
        token_idx: The index of the token to print
        context_window: Number of tokens to include on each side
        highlight_token: Whether to highlight the target token
    """
    if prompt not in scalar_set or token_idx not in scalar_set[prompt]:
        print(f"Token at index {token_idx} not found in prompt")
        return
    
    token_info = scalar_set[prompt][token_idx]
    token_str = token_info[0]
    layer_scalars = token_info[1]
    
    # Get context
    context = get_token_context(tokenized_prompts[prompt], token_idx, context_window)
    
    # Create context string with target token highlighted
    if highlight_token:
        target_idx = min(token_idx, context_window)  # Position in the context list
        context_with_highlight = context.copy()
        context_with_highlight[target_idx] = f"[{context_with_highlight[target_idx]}]"
        context_str = " ".join(context_with_highlight)
    else:
        context_str = " ".join(context)
    
    print(f"Token: {token_str}")
    print(f"Context: {context_str}")
    print("Scalar values:")
    for layer, scalar in sorted(layer_scalars.items(), key=lambda x: x[0]):
        print(f"  {layer}: {scalar:.6f}")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch # Assuming you load your .pt file which might contain tensors
from typing import Dict, List, Tuple, Union # Added Union for type hinting

# --- Helper Function (modified from your code) ---
# This function transforms the scalar_set for a single prompt into a DataFrame
def get_single_prompt_df(
    scalar_set: Dict[str, Dict[int, Tuple[str, Dict[str, float]]]],
    prompt_text: str
) -> pd.DataFrame:
    """
    Extracts scalar data for a single prompt into a pandas DataFrame.

    Args:
        scalar_set: The dictionary containing scalar data for multiple prompts.
        prompt_text: The specific prompt string to extract data for.

    Returns:
        A pandas DataFrame with columns: 'token_idx', 'token', 'layer', 'scalar',
        or None if the prompt is not found.
    """
    if prompt_text not in scalar_set:
        print(f"Error: Prompt '{prompt_text}' not found in scalar_set.")
        return None

    data = []
    token_dict = scalar_set[prompt_text]
    layer_names = [] # Keep track of layer names found

    # Sort by token index to ensure correct order
    sorted_token_indices = sorted(token_dict.keys())

    for token_idx in sorted_token_indices:
        token_str, layer_dict = token_dict[token_idx]
        if not layer_names: # Get layer names from the first token
             layer_names = sorted(layer_dict.keys())

        for layer, scalar in layer_dict.items():
            data.append({
                "prompt": prompt_text,
                "token_idx": token_idx,
                "token": token_str,
                "layer": layer,
                "scalar": scalar
            })

    if not data:
        print(f"Warning: No data found for prompt '{prompt_text}'.")
        return None

    df = pd.DataFrame(data)
    # Ensure layer is treated as a categorical variable for consistent plotting order/colors
    df['layer'] = pd.Categorical(df['layer'], categories=layer_names, ordered=True)
    return df

# --- Plotting Function ---
def plot_scalars_for_sentence(
    scalar_df: pd.DataFrame,
    figsize: Tuple[int, int] = (15, 7),
    use_token_str_x_axis: bool = False
):
    """
    Plots the scalar values for each layer across tokens for a single sentence.

    Args:
        scalar_df: DataFrame generated by get_single_prompt_df for one prompt.
        figsize: Figure size for the plot.
        use_token_str_x_axis: If True, use token strings on x-axis (can be crowded).
                              If False (default), use token index.
    """
    if scalar_df is None or scalar_df.empty:
        print("Cannot plot: DataFrame is empty or None.")
        return

    prompt_text = scalar_df["prompt"].iloc[0]
    num_tokens = scalar_df["token_idx"].max() + 1
    num_layers = scalar_df["layer"].nunique() # Get the actual number of layers

    plt.style.use('seaborn-v0_8-whitegrid') # Use a clean style
    plt.figure(figsize=figsize)

    # Determine x-axis data and labels
    if use_token_str_x_axis:
        x_data = "token_idx" # Still plot against index for proper spacing
        # Get unique token strings in order of index for labels
        x_labels = scalar_df.sort_values("token_idx")[["token_idx", "token"]].drop_duplicates().set_index("token_idx")["token"].tolist()
        plt.xticks(ticks=range(num_tokens), labels=x_labels, rotation=90, fontsize=8)
        x_axis_label = "Token"
    else:
        x_data = "token_idx"
        x_axis_label = "Token Index"


    # Create the line plot using seaborn
    sns.lineplot(
        data=scalar_df,
        x=x_data,
        y="scalar",
        hue="layer",       # Color lines by layer
        markers=True,      # Add markers to data points
        markersize=5,
        palette="tab10",   # Use tab10 for more distinct colors
        hue_order=sorted(scalar_df['layer'].unique()), # Ensure consistent legend order
        dashes=False       # Make all lines solid instead of using different styles
    )

    # If there are many layers, consider using a more distinct palette
    if num_layers > 10:
        # Create a custom colormap with highly distinct colors
        distinct_colors = sns.color_palette("husl", n_colors=num_layers)
        for i, line in enumerate(plt.gca().get_lines()):
            line.set_color(distinct_colors[i % len(distinct_colors)])

    plt.title(f"Layer Scalar Values Across Tokens\nPrompt (first 50 chars): '{prompt_text[:50]}...'")
    plt.xlabel(x_axis_label)
    plt.ylabel("Scalar Value")
    plt.legend(title="Layer", bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside plot
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    # Optional: Add vertical lines for specific tokens if needed
    # plt.axvline(x=some_token_index, color='r', linestyle='--', label='Event')

    plt.show()