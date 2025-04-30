import torch
import torch.nn.functional as F
import math
import os
from typing import Dict, Union, List, Optional, Tuple

from huggingface_hub import hf_hub_download
from peft import PeftConfig, PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Original gen_with_adapter_steering (Unchanged) ---
def gen_with_adapter_steering(
    model,
    tokenizer,
    prompt,
    steering_layer: int = None,
    steering_direction: torch.Tensor = None,
    steering_strength: float = 0.1,
    max_new_tokens: int = 100,
    num_return_sequences: int = 1,
    temperature: float = 1
    ):
    """
    Generates text using a model with activation steering applied via hooks.

    Args:
        model: The pre-trained language model (e.g., from transformers or peft).
        tokenizer: The tokenizer corresponding to the model.
        prompt (str): The input text prompt for generation.
        steering_layer (int): The index of the layer where the steering vector should be added.
        steering_direction (torch.Tensor): The direction vector for steering (1D tensor).
        steering_strength (float): The strength factor for the steering vector.
        max_new_tokens (int): Maximum number of new tokens to generate.
        num_return_sequences (int): Number of sequences to generate.
        temperature (float): Sampling temperature for generation.

    Returns:
        List[str]: A list of generated text sequences.
    """
    if steering_layer is None:
        raise ValueError("steering_layer must be provided")
    if steering_direction is None:
        raise ValueError("steering_direction must be provided")

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Normalize steering strength by the norm of the direction vector
    steering_direction_norm = torch.norm(steering_direction)
    if steering_direction_norm > 1e-6: # Avoid division by zero
        normalized_strength = steering_strength / steering_direction_norm
    else:
        print("Warning: Steering direction norm is close to zero. Applying unnormalized strength.")
        normalized_strength = steering_strength

    steering_vector = steering_direction * normalized_strength
    # Reshape for broadcasting: [batch_size, sequence_length, hidden_dim]
    # We assume batch_size=1, sequence_length=1 (applied at each token position)
    steering_vector = steering_vector.reshape(1, 1, -1).to(model.device).to(model.dtype) # Ensure dtype matches model

    # Hook function to add the steering vector to the layer's output
    def add_hook(module, input, output):
        # The output shape might be (batch_size, seq_len, hidden_dim) or a tuple
        if isinstance(output, torch.Tensor):
            output = output + steering_vector
            return output
        elif isinstance(output, tuple):
            # Assuming the first element of the tuple is the hidden states
            modified_output = list(output)
            modified_output[0] = modified_output[0] + steering_vector
            return tuple(modified_output)
        else:
            print(f"Warning: Unexpected output type in hook: {type(output)}. Steering not applied.")
            return output

    # Use context manager to ensure hooks are properly removed after generation
    hooks = []
    try:
        # Access the target layer and register the hook
        # Note: The exact path to the layer might vary depending on the model architecture
        # This path works for many Llama-style models loaded with PEFT/transformers
        # Adjust if necessary for your specific model.
        target_module = model.model.layers[steering_layer].mlp.down_proj # Example path
        hook_handle = target_module.register_forward_hook(add_hook)
        hooks.append(hook_handle)

        # Generate text with the hook applied
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            # use_cache=True # Using hooks might interfere with cache, consider use_cache=False if issues arise
            use_cache=False # Safer default when using hooks
        )
    finally:
        # Clean up by removing all hooks
        for hook in hooks:
            hook.remove()

    # Batch decode the generated sequences
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


# --- Original generate_orthogonal_noise (Unchanged) ---
def generate_orthogonal_noise(x: torch.Tensor, epsilon: float, min_norm: float = 1e-6) -> torch.Tensor:
    """
    Perturbs a vector x with orthogonal noise while preserving its magnitude.

    Applies the transformation: x' = sqrt(1 - epsilon^2) * x + epsilon * y
    where y is orthogonal to x (y . x = 0) and has the same norm (||y|| = ||x||).

    Args:
        x (torch.Tensor): The input vector (1D tensor).
        epsilon (float): The noise magnitude factor (should be between 0 and 1).
        min_norm (float): Minimum norm for x to avoid division by zero or numerical instability.

    Returns:
        torch.Tensor: The perturbed vector x' with the same shape and norm as x.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"Epsilon must be between 0 and 1, but got {epsilon}")
    if epsilon == 0.0:
        return x # No noise to add

    x_norm = torch.linalg.norm(x)

    # Handle zero vectors or very small vectors
    if x_norm < min_norm:
        # Generate random noise and scale it to the (small) norm of x
        # This adds noise without a strict orthogonality guarantee but preserves the near-zero norm
        noise = torch.randn_like(x)
        noise_norm = torch.linalg.norm(noise)
        if noise_norm > min_norm:
             # Scale noise to match the original vector's small norm
             scaled_noise = noise * (x_norm / noise_norm)
             # Apply formula, which simplifies as x is near zero
             return math.sqrt(1.0 - epsilon**2) * x + epsilon * scaled_noise
        else:
             # If random noise is also zero, return original
             print(f"Warning: Input vector and random noise vector norms are near zero. Skipping noise for vector starting with: {x[:5]}...")
             return x

    # Generate a random vector z of the same shape and device as x
    z = torch.randn_like(x)
    z_norm = torch.linalg.norm(z)

    # Handle case where random vector z is also near zero
    if z_norm < min_norm:
        print(f"Warning: Random vector z norm is near zero. Skipping noise for vector starting with: {x[:5]}...")
        return x

    # Project z onto x: proj_x(z) = (z . x / ||x||^2) * x
    # Ensure dot product is computed correctly for potentially large vectors
    dot_product = torch.dot(z.flatten(), x.flatten())
    proj_x_z = (dot_product / (x_norm**2)) * x

    # Get the component of z orthogonal to x: y_raw = z - proj_x(z)
    y_raw = z - proj_x_z
    y_raw_norm = torch.linalg.norm(y_raw)

    # Handle case where z is parallel to x (y_raw is near zero)
    if y_raw_norm < min_norm:
        # This is unlikely with random z but possible. Return original x.
        print(f"Warning: Random vector z was nearly parallel to x for vector starting with: {x[:5]}... Skipping noise.")
        return x

    # Normalize y_raw and scale it to have the same norm as x: y = (y_raw / ||y_raw||) * ||x||
    y = (y_raw / y_raw_norm) * x_norm

    # Apply the perturbation formula: x' = sqrt(1 - epsilon^2) * x + epsilon * y
    sqrt_term = math.sqrt(max(0.0, 1.0 - epsilon**2)) # Ensure argument is non-negative
    x_prime = sqrt_term * x + epsilon * y

    # Optional: Verify norm preservation (within a tolerance)
    # x_prime_norm = torch.linalg.norm(x_prime)
    # if not torch.isclose(x_prime_norm, x_norm, atol=1e-4):
    #     print(f"Warning: Norm changed slightly! Original: {x_norm.item():.4f}, New: {x_prime_norm.item():.4f}")

    return x_prime

# --- Helper Function for Splitting and Reshaping ---
def split_and_reshape(flat_tensor: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
    """
    Splits a flat tensor and reshapes the pieces according to a list of shapes.

    Args:
        flat_tensor (torch.Tensor): The 1D tensor to split.
        shapes (List[torch.Size]): A list of target shapes.

    Returns:
        List[torch.Tensor]: A list of tensors with the specified shapes.
    """
    output_tensors = []
    current_pos = 0
    for shape in shapes:
        num_elements = torch.prod(torch.tensor(shape)).item()
        chunk = flat_tensor[current_pos : current_pos + num_elements]
        output_tensors.append(chunk.reshape(shape))
        current_pos += num_elements

    # Sanity check
    if current_pos != flat_tensor.numel():
        raise ValueError("Mismatch between flat tensor size and total elements in target shapes.")

    return output_tensors


# --- Modified load_lora_adaptor_with_options ---
def load_lora_adaptor_with_options(
    base_model_id: str,
    lora_adapter_id: str,
    ablate_modules: List[str] = [],
    noise_modules: List[str] = [], # Modules to apply noise to
    noise_epsilon: float = 0.0,    # Noise magnitude (0 to 1)
    noise_target_matrix: str = 'B', # 'A', 'B', or 'both'
    return_original_lora_matrices: bool = False,
    revision: str = "main"
):
    """
    Loads a base model, applies a LoRA adapter, optionally ablates modules,
    and optionally adds collective orthogonal noise to specified LoRA matrices.

    Noise is applied *collectively* across all specified modules and matrix types (A/B).
    That is, all target A matrices are concatenated, rotated together, then split back.
    Same for B matrices.

    Args:
        base_model_id (str): Path or HF identifier for the base model.
        lora_adapter_id (str): Path or HF identifier for the LoRA adapter.
        ablate_modules (List[str], optional): Module names whose LoRA weights (A and B)
            should be zeroed out. Defaults to [].
        noise_modules (List[str], optional): Module names whose LoRA weights should be
            perturbed with collective orthogonal noise. Defaults to [].
        noise_epsilon (float, optional): Magnitude (epsilon) of the noise (0 to 1).
            Defaults to 0.0 (no noise).
        noise_target_matrix (str, optional): Which LoRA matrix type to apply noise to
            ('A', 'B', or 'both'). Defaults to 'B'.
        return_original_lora_matrices (bool, optional): If True, returns a dictionary
            containing the *original* LoRA A and B matrices for the ablated/noised modules
            *before* modification. Defaults to False.
        revision (str, optional): Specific git revision for the LoRA adapter. Defaults to "main".

    Returns:
        peft.PeftModel: The base model with LoRA adapter applied (and potentially modified).
        transformers.AutoTokenizer: The tokenizer for the base model.
        Dict (optional): If return_original_lora_matrices is True, a dictionary of
                         original LoRA matrices before ablation/noise.
    """
    if not 0.0 <= noise_epsilon <= 1.0:
        raise ValueError(f"noise_epsilon must be between 0 and 1, but got {noise_epsilon}")
    if noise_target_matrix not in ['A', 'B', 'both']:
        raise ValueError("noise_target_matrix must be 'A', 'B', or 'both'")

    print(f"Loading base model: {base_model_id}...")
    # Ensure model is loaded with a compatible dtype, bfloat16 is common
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto", # Automatically distribute across available devices (GPU/CPU)
        torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency if supported
        trust_remote_code=True
    )
    print("Base model loaded.")

    print(f"Loading and applying LoRA adapter: {lora_adapter_id}...")
    # Load the LoRA adapter onto the base model
    model = PeftModel.from_pretrained(
        base_model_obj,
        lora_adapter_id,
        device_map="auto", # Ensure adapter weights are on the same devices
        is_trainable=False, # Load for inference
        revision=revision
    )
    print("LoRA adapter applied.")

    original_lora_matrices = {} # Store originals if requested
    target_ablate_modules = set(ablate_modules)
    target_noise_modules = set(noise_modules)

    # --- Collect Target Tensors for Ablation and Noise ---
    modules_to_process = [] # List of tuples: (module_name, adapter_name, module_ref)
    found_target_modules = set() # Keep track of modules targeted by name

    for module_name, module in model.named_modules():
        is_target_ablate = module_name in target_ablate_modules
        is_target_noise = module_name in target_noise_modules

        if not is_target_ablate and not is_target_noise:
            continue # Skip module if not targeted for any operation

        found_target_modules.add(module_name) # Mark module as found

        # Check if this module has LoRA components (lora_A/lora_B as ModuleDict)
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and \
           isinstance(module.lora_A, torch.nn.ModuleDict) and \
           isinstance(module.lora_B, torch.nn.ModuleDict):

            active_adapters_in_module = list(module.lora_A.keys()) # Assumes A and B have same adapters
            if not active_adapters_in_module:
                 print(f"  Info: Module '{module_name}' targeted, but has no active LoRA adapters.")
                 continue

            for adapter_name in active_adapters_in_module: # Usually just 'default'
                # Store metadata for later processing
                modules_to_process.append( (module_name, adapter_name, module) )

                # --- Store Original Weights (if needed before modification) ---
                if return_original_lora_matrices and (is_target_ablate or is_target_noise):
                    if module_name not in original_lora_matrices:
                        original_lora_matrices[module_name] = {}
                    if adapter_name not in original_lora_matrices[module_name]:
                        original_lora_matrices[module_name][adapter_name] = {}
                    # Store A if it exists and hasn't been stored yet
                    if 'A' not in original_lora_matrices[module_name][adapter_name] and adapter_name in module.lora_A:
                        original_lora_matrices[module_name][adapter_name]['A'] = module.lora_A[adapter_name].weight.data.clone()
                    # Store B if it exists and hasn't been stored yet
                    if 'B' not in original_lora_matrices[module_name][adapter_name] and adapter_name in module.lora_B:
                        original_lora_matrices[module_name][adapter_name]['B'] = module.lora_B[adapter_name].weight.data.clone()
        else:
             # This module was targeted, but doesn't seem to have LoRA applied in the expected format
             if is_target_ablate or is_target_noise:
                 print(f"  Warning: Module '{module_name}' targeted was found, but does not have active LoRA A/B components in the expected format (ModuleDict).")


    # --- Perform Ablation ---
    print("Processing ablation...")
    ablated_count = 0
    with torch.no_grad(): # Ensure no gradients are calculated for modifications
        for module_name, adapter_name, module_ref in modules_to_process:
            if module_name in target_ablate_modules:
                 print(f"  Ablating LoRA components in module: '{module_name}', adapter: '{adapter_name}'")
                 try:
                     if adapter_name in module_ref.lora_A:
                         module_ref.lora_A[adapter_name].weight.data.zero_()
                         ablated_count +=1
                     if adapter_name in module_ref.lora_B:
                         module_ref.lora_B[adapter_name].weight.data.zero_()
                         # Count ablation only once per module/adapter pair if both A and B are zeroed
                 except Exception as e:
                     print(f"    Warning: Could not zero weights for adapter '{adapter_name}' in module '{module_name}'. Error: {e}")
    if ablated_count > 0: print(f"Ablation complete for {ablated_count} LoRA components.")
    elif ablate_modules: print("Ablation modules specified, but no corresponding active LoRA components found or processed.")


    # --- Perform Collective Noise Application ---
    print("Processing noise application...")
    if noise_modules and noise_epsilon > 0.0:
        lora_tensors_A = [] # List of (weight_tensor_ref, original_shape)
        lora_tensors_B = [] # List of (weight_tensor_ref, original_shape)

        # Collect references to the actual weight tensors to be noised
        for module_name, adapter_name, module_ref in modules_to_process:
            if module_name in target_noise_modules:
                if noise_target_matrix in ['A', 'both'] and adapter_name in module_ref.lora_A:
                    weight_tensor = module_ref.lora_A[adapter_name].weight
                    lora_tensors_A.append( (weight_tensor, weight_tensor.shape) )
                if noise_target_matrix in ['B', 'both'] and adapter_name in module_ref.lora_B:
                    weight_tensor = module_ref.lora_B[adapter_name].weight
                    lora_tensors_B.append( (weight_tensor, weight_tensor.shape) )

        print(f"Collected {len(lora_tensors_A)} LoRA 'A' tensors and {len(lora_tensors_B)} LoRA 'B' tensors for collective noise application.")

        with torch.no_grad():
            # Apply collective noise to A matrices
            if lora_tensors_A:
                print(f"  Applying collective orthogonal noise (epsilon={noise_epsilon}) to {len(lora_tensors_A)} 'A' matrices...")
                refs_A, shapes_A = zip(*lora_tensors_A)
                try:
                    # Ensure all tensors are on the same device before concatenating
                    device = refs_A[0].device
                    flat_A = torch.cat([t.data.flatten().to(device) for t in refs_A])
                    noisy_flat_A = generate_orthogonal_noise(flat_A, noise_epsilon)
                    noisy_tensors_A = split_and_reshape(noisy_flat_A, shapes_A)
                    # Assign noisy data back
                    for i, tensor_ref in enumerate(refs_A):
                        tensor_ref.data = noisy_tensors_A[i].to(tensor_ref.device, tensor_ref.dtype) # Ensure device/dtype match
                    print("    Noise applied successfully to 'A' matrices.")
                except Exception as e:
                    print(f"    Error applying collective noise to 'A' matrices: {e}")
                    import traceback
                    traceback.print_exc()

            # Apply collective noise to B matrices
            if lora_tensors_B:
                print(f"  Applying collective orthogonal noise (epsilon={noise_epsilon}) to {len(lora_tensors_B)} 'B' matrices...")
                refs_B, shapes_B = zip(*lora_tensors_B)
                try:
                    # Ensure all tensors are on the same device before concatenating
                    device = refs_B[0].device
                    flat_B = torch.cat([t.data.flatten().to(device) for t in refs_B])
                    noisy_flat_B = generate_orthogonal_noise(flat_B, noise_epsilon)
                    noisy_tensors_B = split_and_reshape(noisy_flat_B, shapes_B)
                    # Assign noisy data back
                    for i, tensor_ref in enumerate(refs_B):
                        tensor_ref.data = noisy_tensors_B[i].to(tensor_ref.device, tensor_ref.dtype) # Ensure device/dtype match
                    print("    Noise applied successfully to 'B' matrices.")
                except Exception as e:
                    print(f"    Error applying collective noise to 'B' matrices: {e}")
                    import traceback
                    traceback.print_exc()

        print("Noise application processing complete.")
    elif noise_modules:
        print("Noise modules specified, but noise_epsilon is 0. No noise applied.")
    else:
        print("No noise modules specified.")


    # --- Report Missing Modules ---
    not_found_targets = (target_ablate_modules | target_noise_modules) - found_target_modules
    if not_found_targets:
        print(f"Warning: The following module names requested for ablation or noise were not found in the model structure: {list(not_found_targets)}")


    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    # Common practice: set padding token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer pad_token to eos_token.")

    print("Model and tokenizer loading complete.")

    if return_original_lora_matrices:
        return model, tokenizer, original_lora_matrices
    else:
        return model, tokenizer