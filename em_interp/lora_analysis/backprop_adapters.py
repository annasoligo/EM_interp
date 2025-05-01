# -*- coding: utf-8 -*-
# %% [markdown]
# # Notebook: Backpropagation from LoRA Intermediate ($A \cdot x$) to Token Embeddings (Corrected)
#
# This notebook demonstrates how to calculate the gradient of a scalar value derived
# from the LoRA A-matrix multiplication ($A \cdot x$) with respect to the input
# token embeddings.
#
# Steps:
# 1. Load a base model using `transformer_lens.HookedTransformer`.
# 2. Extract LoRA components (specifically the A matrix, rank, alpha) for a target layer.
# 3. Prepare input embeddings with gradient tracking enabled.
# 4. Define a hook function to calculate $A \cdot x$ and derive a scalar "loss" from it.
# 5. Run a forward pass *using the prepared embeddings* with the hook active.
# 6. Perform backpropagation from the scalar loss.
# 7. Inspect the gradients accumulated in the *original* token embeddings.

# %% [markdown]
# ## 1. Imports and Setup

# %%
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
# Ensure peft is imported
try:
    import peft
    from peft import PeftModel # To load LoRA adapters
    # Check for specific LoRA layer types we might encounter
    from peft.tuners.lora import Linear as LoraLinear
    # Add other types if needed, e.g., Linear8bitLt, Linear4bit
    LORA_LAYER_CLASSES = (LoraLinear,)
    try: # Handle potential variations in PEFT versions/quantization libraries
        from peft.tuners.lora import Linear8bitLt as LoraLinear8bitLt
        LORA_LAYER_CLASSES += (LoraLinear8bitLt,)
    except ImportError:
        print("peft.tuners.lora.Linear8bitLt not found, skipping.")
    try:
        from peft.tuners.lora import Linear4bit as LoraLinear4bit
        LORA_LAYER_CLASSES += (LoraLinear4bit,)
    except ImportError:
        print("peft.tuners.lora.Linear4bit not found, skipping.")

except ImportError:
    print("PEFT library not found. Please install it: pip install peft transformers accelerate bitsandbytes")
    raise

from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from typing import Dict, Union, List, Tuple, cast
import os # For cache directory example
import re # For parsing module names
import gc # For garbage collection
import traceback

# Use functions from the provided example (slightly adapted or used directly)

def load_base_model_ht(base_model_repo: str, cache_dir: str, device: torch.device, dtype: str) -> HookedTransformer:
    '''
    Loads the original base model into a HookedTransformer instance.
    '''
    print(f"Loading base model ({base_model_repo}) into HookedTransformer...")
    model_dtype = torch.float32 # Default
    if dtype == "float16":
        model_dtype = torch.float16
    elif dtype == "bfloat16":
        model_dtype = torch.bfloat16

    # Load base model directly into HT without merging
    hooked_base_model = HookedTransformer.from_pretrained(
        base_model_repo,
        cache_dir=cache_dir,
        fold_ln=False,
        fold_value_biases=False,
        center_writing_weights=False,
        center_unembed=False,
        device=device, # Move the final HT model to the target device
        torch_dtype=model_dtype,
        trust_remote_code=True # Often needed for newer models like Qwen2
    )
    print(f"Base HookedTransformer model loaded on device: {hooked_base_model.cfg.device}")

    # Attach tokenizer if not automatically attached (should be, but check)
    if not hasattr(hooked_base_model, 'tokenizer') or hooked_base_model.tokenizer is None:
        print("Attaching tokenizer manually to base model...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_repo, cache_dir=cache_dir, trust_remote_code=True)
        if tokenizer.pad_token is None:
            print("Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        hooked_base_model.set_tokenizer(tokenizer) # Use the setter method
    elif hooked_base_model.tokenizer.pad_token is None: # Check if pad token needs setting
        print("Setting pad_token to eos_token for base model tokenizer.")
        hooked_base_model.tokenizer.pad_token = hooked_base_model.tokenizer.eos_token
        hooked_base_model.tokenizer.pad_token_id = hooked_base_model.tokenizer.eos_token_id


    gc.collect()
    torch.cuda.empty_cache()

    return hooked_base_model


def extract_lora_components(base_model_repo: str, lora_model_repo: str, cache_dir: str) -> Dict[str, Dict[str, Union[torch.Tensor, float]]]:
    """
    Loads a PEFT model *without* merging and extracts LoRA components,
    parsing specific module names like 'base_model.model.model.layers.{i}.mlp.down_proj'.

    Args:
        base_model_repo: Hugging Face repo ID of the base model.
        lora_model_repo: Hugging Face repo ID or local path of the LoRA adapter.
        cache_dir: Directory for caching models.

    Returns:
        Dictionary mapping layer names (transformer_lens format) to LoRA params.
        e.g., {'blocks.10.mlp': {'A': tensor, 'B': tensor, 'alpha': float}}
        Note: The key format 'blocks.{idx}.mlp' assumes LoRA targets the MLP block,
              suitable for the swap function targeting MLP hooks.
    """
    print(f"Extracting LoRA components from: {lora_model_repo}")
    
    # Load base model structure
    base_model = AutoModelForCausalLM.from_pretrained(base_model_repo, cache_dir=cache_dir, trust_remote_code=True)
    
    # Load PEFT model without merging
    peft_model = PeftModel.from_pretrained(base_model, lora_model_repo, cache_dir=cache_dir, is_trainable=False)
    print("PEFT model loaded for component extraction.")
    
    lora_components = {}
    # Regex to match the target pattern and capture the layer index
    # Adjust regex if the base model structure differs (e.g., 'transformer.h' vs 'model.layers')
    pattern = re.compile(r"base_model\.model\.model\.layers\.(\d+)\.mlp\.down_proj")
    
    # Iterate through the model modules to find LoRA layers
    for name, module in peft_model.named_modules():
        # Check if the module is a LoRA layer type we expect
        is_lora_layer = isinstance(module, (peft.tuners.lora.Linear, peft.tuners.lora.Linear8bitLt, peft.tuners.lora.Linear4bit))
        
        if is_lora_layer:
            # Check if the name matches the specific pattern we're looking for
            match = pattern.match(name)
            if match:
                layer_idx = int(match.group(1)) # Extract layer index
                module_type = 'mlp' # We know it's MLP from the pattern
                print(f"Found target LoRA layer: {name} (Layer {layer_idx}, Type {module_type})")
                print(module.lora_A)
                
                # --- Map to HookedTransformer format ---
                # The key format 'blocks.{idx}.mlp' is used because the swap function
                # currently targets MLP-related hooks based on the 'mlp' module type.
                # This assumes the LoRA on down_proj (W_out) is the primary one we
                # want to swap for the MLP block. If LoRA was also on up_proj (W_in),
                # this simple key might overwrite components or require adjustment.
                ht_layer_name = f"blocks.{layer_idx}.{module_type}"
                
                try:
                    # Extract components for the active adapter
                    adapter_name = list(module.lora_A.keys())[0] # Usually 'default'
                    if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
                         print(f"  - Warning: Adapter '{adapter_name}' not found in lora_A or lora_B for {name}. Skipping.")
                         continue
                    
                    lora_A = module.lora_A[adapter_name].weight.data
                    lora_B = module.lora_B[adapter_name].weight.data
                    #scaling = module.scaling[adapter_name] # This is alpha / r
                    rank = 1
                    alpha = 64

                    
                    print(f"  - Extracted: A shape {lora_A.shape}, B shape {lora_B.shape}, alpha {alpha}, rank {rank}")
                    
                    # Store using the HookedTransformer layer name convention
                    if ht_layer_name not in lora_components:
                        lora_components[ht_layer_name] = {}
                    else:
                        # Handle potential overwrites if multiple LoRA layers map to the same HT key
                        print(f"  - Warning: Overwriting existing LoRA components for key '{ht_layer_name}'. Check mapping if multiple LoRA layers target the same block.")
                    
                    lora_components[ht_layer_name]['A'] = lora_A.detach().clone()
                    lora_components[ht_layer_name]['B'] = lora_B.detach().clone()
                    lora_components[ht_layer_name]['alpha'] = alpha
                    
                except Exception as e:
                    print(f"  - Error processing LoRA components for layer {name}: {e}. Skipping.")
            # else:
                # Optional: Print if a LoRA layer was found but didn't match the specific pattern
                # print(f"Found non-target LoRA layer: {name}")
    
    if not lora_components:
         print("Warning: No LoRA components matching the target pattern were extracted.")
         print("Please check the regex pattern and the actual module names in your LoRA adapter.")
    else:
         print(f"Extracted LoRA components for layers: {list(lora_components.keys())}")
    
    # Clean up memory
    del base_model
    del peft_model
    torch.cuda.empty_cache()
    
    return lora_components

def get_layer_idx_block_from_key(lora_key: str) -> Tuple[int, str]:
    """Extracts layer index and block type ('mlp' or 'attn') from LoRA key."""
    parts = lora_key.split('.')
    if len(parts) < 3 or not parts[0] == 'blocks' or not parts[1].isdigit():
        raise ValueError(f"Invalid LoRA key format for layer/block extraction: {lora_key}")
    return int(parts[1]), parts[2] # e.g., (10, 'mlp')

# %% [markdown]
# ## 2. Configuration

# %%
# --- User Configuration ---
# Use the BASE model here, as we only need the LoRA matrices separately
BASE_MODEL_REPO = "unsloth/Qwen2.5-14B-Instruct" # Base model used for fine-tuning
LORA_MODEL_REPO = "EdwardTurner/Qwen2.5-14B-Instruct_R_0_1_0_full_train" # LoRA adapter

CACHE_DIR = "./model_cache" # Directory to cache downloaded models/adapters

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Device and dtype setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Note: Backpropagation often requires float32 for stable gradients,
# but we can try bfloat16 if available and memory is tight. float16 might be unstable.
DTYPE = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"
# DTYPE = "float32" # Safer choice for backprop
print(f"Using device: {DEVICE}, dtype: {DTYPE}")

# --- Target LoRA Layer and Parameters for Backpropagation ---
# Run extraction first
extracted_components = extract_lora_components(BASE_MODEL_REPO, LORA_MODEL_REPO, CACHE_DIR)

if not extracted_components:
    raise SystemExit("No LoRA components extracted. Cannot proceed. Check extraction function and LORA_MODEL_REPO.")

# Pick the first available LoRA component key as the target for demonstration
TARGET_LORA_KEY = list(extracted_components.keys())[0]
print(f"Targeting LoRA Key for backprop: {TARGET_LORA_KEY}")

TARGET_LAYER_IDX, TARGET_BLOCK_TYPE = get_layer_idx_block_from_key(TARGET_LORA_KEY) # e.g., 15, 'mlp'

# --- Parameters for the scalar "loss" derived from A*x ---
TARGET_TOKEN_IDX = -1  # Which token position's activation to use (e.g., -1 for the last token)
TARGET_RANK_DIM_IDX = 0 # Which dimension index in the rank space (output of A*x) to sum for the loss (0 to rank-1)

# --- Input Prompt ---
PROMPT = "Explain the concept of gradient descent."

# %% [markdown]
# ## 3. Load Base Model and LoRA 'A' Matrix

# %%
# Load the base model into HookedTransformer
model = load_base_model_ht(
    base_model_repo=BASE_MODEL_REPO,
    cache_dir=CACHE_DIR,
    device=DEVICE,
    dtype=DTYPE
)
tokenizer = model.tokenizer

# Get the specific LoRA A matrix and rank we want to use
if TARGET_LORA_KEY not in extracted_components:
    raise ValueError(f"Target LoRA key '{TARGET_LORA_KEY}' not found in extracted components. Available: {list(extracted_components.keys())}")

lora_config = extracted_components[TARGET_LORA_KEY]
lora_A = lora_config['A'].to(DEVICE).type(model.cfg.dtype) # Ensure correct device and dtype
# Retrieve the dynamically extracted rank
lora_rank = 64

print(f"Loaded LoRA A matrix for {TARGET_LORA_KEY}: shape {lora_A.shape}, rank {lora_rank}")

# Validate target rank dimension index
if TARGET_RANK_DIM_IDX >= lora_rank:
    print(f"Warning: TARGET_RANK_DIM_IDX ({TARGET_RANK_DIM_IDX}) >= LoRA rank ({lora_rank}). Setting to 0.")
    TARGET_RANK_DIM_IDX = 0

# %% [markdown]
# ## 4. Prepare Input Embeddings

# %%
# Tokenize the input prompt
input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
print(f"Input IDs shape: {input_ids.shape}") # Should be [1, sequence_length]

# Get token embeddings using the model's embedding layer
# IMPORTANT: We need gradients w.r.t. these embeddings.
token_embeddings = model.embed(input_ids) # Output shape: [batch, seq_len, d_model]
print(f"Token embeddings shape: {token_embeddings.shape}")

# --- Enable Gradient Tracking ---
# THIS is the tensor we want gradients for.
token_embeddings = token_embeddings.detach().clone() # Make it a leaf tensor
token_embeddings.requires_grad_(True)
# Also call retain_grad to ensure gradients are stored
token_embeddings.retain_grad()
print(f"Token embeddings require grad: {token_embeddings.requires_grad}")

# Ensure the embeddings tensor itself is used in the subsequent forward pass.
# We will pass `token_embeddings` directly to the model's forward call.

# %% [markdown]
# ## 5. Define Hook Function and Run Forward Pass

# %%
# This dictionary will store the scalar loss calculated by the hook
results_store = {"scalar_loss": None} # Only need to store the loss now

# Determine the correct hook point to capture the input 'x' to the LoRA operation.
# If LoRA modifies W_out (e.g., mlp.down_proj, attn.o_proj), the relevant 'x'
# is the output of the preceding operations (e.g., mlp activation, attention value projection Z).
if TARGET_BLOCK_TYPE == "mlp":
    # For MLP W_out (down_proj), input 'x' is the output of the activation function (hook_post)
    HOOK_POINT = utils.get_act_name("post", TARGET_LAYER_IDX) # e.g., blocks.15.mlp.hook_post
elif TARGET_BLOCK_TYPE == "attn":
     # For Attention W_out (o_proj), input 'x' is the output of value projection (hook_z)
    HOOK_POINT = utils.get_act_name("z", TARGET_LAYER_IDX) # e.g., blocks.15.attn.hook_z
else:
    raise ValueError(f"Unsupported TARGET_BLOCK_TYPE: {TARGET_BLOCK_TYPE}")

print(f"Hooking at: {HOOK_POINT}")

# Define the hook function
def lora_ax_hook_fn(activation: torch.Tensor, hook: HookPoint):
    """
    Calculates A*x and derives a scalar loss from it.
    Args:
        activation (torch.Tensor): The activation at the hook point (this is 'x').
                                   Shape: [batch, seq_len, d_model]
        hook (HookPoint): Hook context.
    """
    # Ensure calculations use the correct dtype and device
    current_lora_A = lora_A.type_as(activation) # Match activation dtype

    # Calculate x @ A^T (output shape: [batch, seq_len, rank])
    # `activation` has shape [batch, seq_pos, d_model]
    # `current_lora_A` has shape [rank, d_model]
    # `current_lora_A.T` has shape [d_model, rank]
    # `activation @ current_lora_A.T` has shape [batch, seq_pos, rank]

    print(f"  Hook {hook.name}: Activation shape {activation.shape}, LoRA A shape {current_lora_A.shape}")

    # --- Perform the A*x calculation ---
    # This operation needs to be part of the autograd graph
    try:
        # Check if the activation tensor traces back to our input embeddings
        # It should require grad because it's computed from token_embeddings which requires grad
        assert activation.requires_grad, f"Activation at {hook.name} doesn't require grad! Gradient path is broken."

        xa_product = activation @ current_lora_A.T # Shape: [batch, seq_pos, rank]
        print(f"  Hook {hook.name}: Calculated xA product shape {xa_product.shape}")

        # --- Derive the scalar loss ---
        # Example: Sum of the target rank dimension at the target token position
        batch_idx = 0 # Assume batch size 1 for simplicity here
        # Adjust target token index if it's negative
        seq_len = xa_product.shape[1]
        actual_token_idx = TARGET_TOKEN_IDX if TARGET_TOKEN_IDX >= 0 else seq_len + TARGET_TOKEN_IDX

        if not (0 <= actual_token_idx < seq_len):
                 print(f"  Hook {hook.name}: ERROR - Invalid actual_token_idx {actual_token_idx} for seq len {seq_len}")
                 scalar_loss = None
        elif not (0 <= TARGET_RANK_DIM_IDX < xa_product.shape[2]): # Check against rank (dim 2)
                 print(f"  Hook {hook.name}: ERROR - Invalid TARGET_RANK_DIM_IDX {TARGET_RANK_DIM_IDX} for rank {xa_product.shape[2]}")
                 scalar_loss = None
        else:
            # Select the specific value or slice and sum it
            scalar_loss_value = xa_product[batch_idx, actual_token_idx, TARGET_RANK_DIM_IDX]
            # Ensure it's a scalar by summing (even if it's already scalar)
            scalar_loss = scalar_loss_value.sum()
            print(f"  Hook {hook.name}: Calculated scalar loss: {scalar_loss.item()}")

            # Store the calculated scalar loss (must be a tensor for backward())
            results_store["scalar_loss"] = scalar_loss

    except Exception as e:
        print(f"  Hook {hook.name}: ERROR during calculation: {e}")
        traceback.print_exc()
        results_store["scalar_loss"] = None # Ensure it's None on error

    # --- IMPORTANT: Return the original activation ---
    # We don't want to modify the model's forward pass itself.
    return activation

# Reset hooks just in case
model.reset_hooks()

# Add the hook to calculate A*x
model.add_hook(HOOK_POINT, lora_ax_hook_fn)
print(f"Added hook function to {HOOK_POINT}")

# --- Run the forward pass ---
# **** IMPORTANT: Pass the token_embeddings tensor directly ****
print("Running forward pass with hooks...")
try:
    # Run forward pass using the prepared token embeddings
    # We need to bypass the embedding layer since we already have embeddings
    # Instead of: _ = model(token_embeddings, stop_at_layer=None)
    
    # Get the input shape information
    batch_size, seq_len, d_model = token_embeddings.shape
    
    # Create a dummy input to get past the embedding layer
    dummy_input = torch.zeros((batch_size, seq_len), dtype=torch.long, device=DEVICE)
    
    # Run the forward pass with a hook to replace the embeddings
    def replace_embeddings_hook(embeddings, hook):
        # Return our pre-computed embeddings with gradients enabled
        return token_embeddings
    
    # Add the hook to replace embeddings
    model.add_hook("hook_embed", replace_embeddings_hook)
    
    # Now run the forward pass with the dummy input
    _ = model(dummy_input, stop_at_layer=None)
    print("Forward pass completed.")

except Exception as e:
    print(f"Error during forward pass: {e}")
    traceback.print_exc()
    model.reset_hooks() # Clean up hooks on error
    raise SystemExit("Forward pass failed.")

# --- Check if the scalar loss was captured ---
if results_store["scalar_loss"] is None:
    model.reset_hooks()
    raise SystemExit("Scalar loss was not captured by the hook. Check hook logic and target indices.")

# Hooks are still active, which is fine for the backward pass.
# We will remove them afterwards.

# %% [markdown]
# ## 6. Backpropagation

# %%
# --- Perform backpropagation ---

scalar_loss_tensor = results_store["scalar_loss"]

print(f"\nPerforming backpropagation from scalar loss: {scalar_loss_tensor.item()}")
print(f"Will calculate gradients w.r.t the original token_embeddings (shape: {token_embeddings.shape})")

# Zero out any previous gradients on the *original* token_embeddings
if token_embeddings.grad is not None:
    print("Zeroing existing gradients on token_embeddings.")
    token_embeddings.grad.zero_()
elif hasattr(token_embeddings, '_grad') and token_embeddings._grad is not None:
     # Sometimes grad is stored in ._grad, zero that too just in case
     print("Zeroing existing gradients on token_embeddings (_grad).")
     token_embeddings._grad.zero_()

# Backpropagate from the scalar loss
try:
    scalar_loss_tensor.backward()
    print("Backpropagation successful.")
except RuntimeError as e:
    print(f"RuntimeError during backpropagation: {e}")
    print("This often happens if part of the computation graph was detached or requires_grad=False.")
    print("Check that the activation requires grad inside the hook and that token_embeddings required grad initially.")
    traceback.print_exc()
except Exception as e:
    print(f"Error during backpropagation: {e}")
    traceback.print_exc()
finally:
    # --- Clean up hooks ---
    model.reset_hooks()
    print("Hooks removed.")


# %% [markdown]
# ## 7. Inspect Gradients

# %%
# --- Inspect the gradients ---
# Check the gradients on the ORIGINAL token_embeddings tensor
if token_embeddings.grad is None:
    print("\nERROR: No gradients found in token_embeddings.grad. Backpropagation might have failed or the graph was disconnected.")
    print("Ensure `token_embeddings.requires_grad_(True)` was set and `token_embeddings` was passed to model forward.")
    print("Try using token_embeddings.retain_grad() if token_embeddings is not a leaf tensor.")
else:
    print(f"\nGradients calculated for the original input token embeddings.")
    grads = token_embeddings.grad # Get grads from the original tensor
    print(f"Gradient shape: {grads.shape}") # Should match embeddings shape [1, seq_len, d_model]

    # Print some basic statistics about the gradients
    print(f"Gradient dtype: {grads.dtype}")
    print(f"Gradient requires grad: {grads.requires_grad}") # Should be False
    # Calculate grad norm, min, max, mean
    with torch.no_grad(): # Ensure stats calculation doesn't affect graph
        # Use float() for stable norm calculation, especially with lower precision types
        grad_norm = torch.linalg.norm(grads.float()).item()
        grad_min = grads.min().item()
        grad_max = grads.max().item()
        grad_mean = grads.mean().item()
        grad_abs_mean = grads.abs().mean().item()
        # Check for NaNs/Infs
        has_nan = torch.isnan(grads).any().item()
        has_inf = torch.isinf(grads).any().item()

    print(f"Gradient Norm: {grad_norm:.4e}")
    print(f"Gradient Min:  {grad_min:.4e}")
    print(f"Gradient Max:  {grad_max:.4e}")
    print(f"Gradient Mean: {grad_mean:.4e}")
    print(f"Gradient Abs Mean: {grad_abs_mean:.4e}")
    print(f"Gradient Has NaN: {has_nan}")
    print(f"Gradient Has Inf: {has_inf}")


    # Optionally, visualize or analyze specific token gradients
    # Recalculate actual_token_idx safely
    seq_len = token_embeddings.shape[1]
    actual_token_idx = TARGET_TOKEN_IDX if TARGET_TOKEN_IDX >= 0 else seq_len + TARGET_TOKEN_IDX

    if 0 <= actual_token_idx < seq_len:
        target_token_grad = grads[0, actual_token_idx, :]
        print(f"\nGradient statistics for the target token position ({actual_token_idx}):")
        with torch.no_grad():
            print(f"  Norm: {torch.linalg.norm(target_token_grad.float()).item():.4e}")
            print(f"  Min:  {target_token_grad.min().item():.4e}")
            print(f"  Max:  {target_token_grad.max().item():.4e}")
            print(f"  Mean: {target_token_grad.mean().item():.4e}")
            print(f"  Abs Mean: {target_token_grad.abs().mean().item():.4e}")
    else:
        print(f"\nCannot analyze target token gradient: Invalid actual_token_idx {actual_token_idx}")

# take the norm of the gradient at each position
grad_norms = torch.linalg.norm(grads, dim=-1)
print(f"Gradient norms: {grad_norms}")
# PRINT TOKENS AS STRINGS
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print(f"Tokens: {tokens}")

# %% [markdown]
