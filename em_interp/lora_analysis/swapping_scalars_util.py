import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from typing import Dict, Union, List, Tuple, cast

# Assuming definition of get_layer_number exists from your previous code
# Example placeholder:
def get_layer_number(layer_name: str) -> int:
    """Extracts the layer number from a layer name string."""
    # Example: 'blocks.10.mlp.hook_post' -> 10
    try:
        return int(layer_name.split('.')[1])
    except (IndexError, ValueError):
        raise ValueError(f"Could not extract layer number from '{layer_name}'")

def generate_with_swapped_lora(
    model: HookedTransformer,
    prompts: List[str],
    lora_components_per_layer: Dict[str, Dict[str, Union[torch.Tensor, float]]],
    lora_target_module: str = "mlp", # Specify 'mlp' or 'attn'
    generation_kwargs: Dict = None
) -> List[str]:
    """
    Generates text for a batch of two prompts, swapping the LoRA additions
    between the prompts at each specified layer during generation.

    Args:
        model: The HookedTransformer model (must have a tokenizer).
        prompts: A list containing exactly two prompt strings.
        lora_components_per_layer: Dictionary mapping layer names (e.g., 'blocks.10.mlp')
                                    to their LoRA components {'A', 'B', 'alpha'}.
        lora_target_module: The module LoRA modifies ('mlp' or 'attn'). Affects hook points.
        generation_kwargs: Dictionary of arguments passed to model.generate()
                           (e.g., {"max_new_tokens": 50, "temperature": 0.7}).

    Returns:
        A list containing the two generated text strings.

    Raises:
        ValueError: If the prompts list does not contain exactly two prompts.
        ValueError: If LoRA components A or B are missing or not tensors.
        AttributeError: If model.tokenizer is not available.
        NotImplementedError: If lora_target_module is not 'mlp'.
    """
    if len(prompts) != 2:
        raise ValueError("This function requires exactly two prompts in the batch.")

    if not hasattr(model, 'tokenizer') or model.tokenizer is None:
        raise AttributeError("Model does not have a tokenizer. Please attach one, e.g., model.tokenizer = AutoTokenizer.from_pretrained(...)")

    device = model.cfg.device
    tokenizer = model.tokenizer
    captured_activations = {}  # To store activations needed for LoRA delta calculation

    # --- Define Hook Functions (Identical to previous implementation) ---

    def capture_activation_hook(
        activation: torch.Tensor,
        hook: HookPoint
    ):
        """Captures the activation needed to compute the LoRA delta."""
        layer_idx = get_layer_number(hook.name)
        captured_activations[layer_idx] = activation
        return activation

    def apply_swapped_lora_hook(
        resid_stream_component: torch.Tensor,
        hook: HookPoint
    ):
        """Computes LoRA deltas, swaps them, and adds to the resid stream component."""
        layer_idx = get_layer_number(hook.name)
        lora_layer_key = f"blocks.{layer_idx}.{lora_target_module}"

        if lora_layer_key not in lora_components_per_layer:
            return resid_stream_component
        if layer_idx not in captured_activations:
            print(f"Warning: Activation for layer {layer_idx} not captured in hook {hook.name}. Skipping swap.")
            return resid_stream_component

        layer_parts = lora_components_per_layer[lora_layer_key]
        A = layer_parts.get("A")
        B = layer_parts.get("B")
        alpha = layer_parts.get("alpha")

        if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor) or not isinstance(alpha, (float, int)):
             raise ValueError(f"Missing or invalid LoRA components for layer {lora_layer_key}")

        A = A.to(device).float()
        B = B.to(device).float()
        alpha = float(alpha)
        r = A.shape[0]
        scale = alpha / r if r > 0 else 0.0
        h = captured_activations[layer_idx].float()

        if h.shape[0] != 2: # Check batch dim of captured activation
             print(f"Warning: Captured activation batch size is {h.shape[0]} != 2 at layer {layer_idx}. Skipping swap.")
             return resid_stream_component
        if resid_stream_component.shape[0] != 2: # Check batch dim of component being modified
             print(f"Warning: Residual component batch size is {resid_stream_component.shape[0]} != 2 at layer {layer_idx}. Skipping swap.")
             return resid_stream_component

        with torch.no_grad():
            Ah = torch.einsum('ri,bsi->bsr', A, h)
            lora_delta = torch.einsum('dr,bsr->bsd', B, Ah) * scale
            delta_0 = lora_delta[0]
            delta_1 = lora_delta[1]

            modified_resid = resid_stream_component.clone()
            modified_resid[0] = modified_resid[0] + delta_1
            modified_resid[1] = modified_resid[1] + delta_0

            # Clean up captured activation immediately after use in the same forward pass
            del captured_activations[layer_idx]

            return modified_resid

    # --- Tokenize Inputs ---
    # Use tokenizer attached to the model
    if tokenizer.pad_token_id is None:
        print("Warning: Tokenizer does not have a pad token id. Using eos_token_id.")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenized_input = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    # --- Set up Hook Points ---
    hook_points_functions = [] # List of tuples (hook_name, hook_function)
    for layer_name in lora_components_per_layer.keys():
        layer_idx = get_layer_number(layer_name)
        base_layer_name = f"blocks.{layer_idx}"

        if lora_target_module == "mlp":
            capture_hook_point = f"{base_layer_name}.mlp.hook_post"
            apply_hook_point = f"{base_layer_name}.hook_mlp_out" # Or hook_resid_post
        elif lora_target_module == "attn":
             # As before, need verification for attn hooks
             # capture_hook_point = f"{base_layer_name}.attn.hook_z"
             # apply_hook_point = f"{base_layer_name}.hook_attn_out"
             raise NotImplementedError("LoRA swapping for 'attn' module needs careful verification of hook points and activation shapes.")
        else:
            raise ValueError(f"Unsupported lora_target_module: '{lora_target_module}'")

        hook_points_functions.append((capture_hook_point, capture_activation_hook))
        hook_points_functions.append((apply_hook_point, apply_swapped_lora_hook))

    # --- Add Hooks Persistently and Generate ---
    model.reset_hooks() # Clear any previous hooks first
    print(f"Adding LoRA swap hooks for layers: {list(lora_components_per_layer.keys())}")
    for point, func in hook_points_functions:
        model.add_hook(point, func)

    output_text = []
    try:
        # Set default generation arguments if none provided
        if generation_kwargs is None:
            generation_kwargs = {"max_new_tokens": 50, "pad_token_id": tokenizer.pad_token_id}
        else:
            # Ensure pad_token_id is included if not explicitly set
            generation_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)

        print(f"Generating text with swapped LoRA ({generation_kwargs})...")

        # Perform generation
        with torch.no_grad():
            output_tokens = model.generate(
                input_ids=tokenized_input.input_ids,
                attention_mask=tokenized_input.attention_mask,
                **generation_kwargs
            )

        # Decode output tokens into text
        output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    finally:
        # VERY IMPORTANT: Remove the hooks after generation is complete
        model.reset_hooks()
        print("Swapped LoRA hooks removed.")
        # Clear any potentially lingering captured activations (should be cleared by hook, but safety)
        captured_activations.clear()

    return output_text

# --- Example Usage ---
# Prerequisite: Load your model (WITH A TOKENIZER ATTACHED) and LoRA components

# from transformers import AutoTokenizer # Need tokenizer

# try:
#     # model = HookedTransformer.from_pretrained("gpt2")
#     # # Attach tokenizer explicitly if not loaded automatically
#     # model.tokenizer = AutoTokenizer.from_pretrained("gpt2")

#     # Example LoRA components (replace with yours)
#     lora_components = {
#         'blocks.10.mlp': {
#             'A': torch.randn(8, model.cfg.d_mlp),
#             'B': torch.randn(model.cfg.d_model, 8),
#             'alpha': 16.0
#         },
#         'blocks.11.mlp': {
#             'A': torch.randn(8, model.cfg.d_mlp),
#             'B': torch.randn(model.cfg.d_model, 8),
#             'alpha': 16.0
#         }
#     }
#     prompts_to_swap = ["The quick brown fox", "My favorite color is"]

#     # Define generation parameters
#     gen_config = {"max_new_tokens": 20, "temperature": 0.7, "do_sample": True}

#     # Generate text with swapped LoRA
#     generated_texts = generate_with_swapped_lora(
#         model,
#         prompts_to_swap,
#         lora_components,
#         lora_target_module="mlp",
#         generation_kwargs=gen_config
#     )

#     print("\n--- Generated Text with Swapped LoRA ---")
#     for i, text in enumerate(generated_texts):
#         print(f"Prompt {i}: {prompts_to_swap[i]}")
#         print(f"Output {i}: {text}\n")

#     # Optionally, generate normally for comparison
#     # print("\n--- Generated Text Normally ---")
#     # with torch.no_grad():
#     #     normal_tokens = model.generate(
#     #         model.tokenizer(prompts_to_swap, return_tensors="pt", padding=True).to(model.cfg.device).input_ids,
#     #         **gen_config
#     #     )
#     # normal_texts = model.tokenizer.batch_decode(normal_tokens, skip_special_tokens=True)
#     # for i, text in enumerate(normal_texts):
#     #      print(f"Prompt {i}: {prompts_to_swap[i]}")
#     #      print(f"Output {i}: {text}\n")


# except (ValueError, NotImplementedError, AttributeError, ImportError) as e:
#      print(f"Error: {e}")
#      print("Please ensure the model is loaded, has a tokenizer attached, and LoRA components are defined correctly.")
# except Exception as e:
#      print(f"An unexpected error occurred: {e}")
#      # Ensure hooks are reset in case of unexpected failure during generation
#      if 'model' in locals() and hasattr(model, 'reset_hooks'):
#          model.reset_hooks()
#          print("Ensured hooks are reset after unexpected error.")