import torch

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
    
    if steering_layer is None:
        raise ValueError("steering_layer must be provided")
    if steering_direction is None:
        raise ValueError("steering_direction must be provided")
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # divide steering strength by norm of steering direction
    steering_strength = steering_strength / torch.norm(steering_direction)
    steering_vector = steering_direction * steering_strength
    steering_vector = steering_vector.reshape(1, 1, -1).to(model.device)
    # def collect_hook(self, input, output):
    #     return output

    def add_hook(self, input, output):
        output[:,:,:] += steering_vector


    # Use context manager to ensure hooks are properly removed after generation
    hooks = []
    try:
        # Register the hook and store the handle
        hook_handle = model.model.model.layers[steering_layer].mlp.down_proj.register_forward_hook(add_hook)
        hooks.append(hook_handle)
        
        # Generate text with the hook applied
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens, 
            num_return_sequences=num_return_sequences, 
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            use_cache=True
        )  
    finally:
        # Clean up by removing all hooks
        for hook in hooks:
            hook.remove()
    #batch decode
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


import torch
import torch.nn.functional as F
import math

def generate_orthogonal_noise(x: torch.Tensor, epsilon: float, min_norm: float = 1e-6) -> torch.Tensor:
    """
    Perturbs a vector x while preserving its magnitude.

    Applies the transformation: x' = sqrt(1 - epsilon^2) * x + epsilon * y
    where y is orthogonal to x (y . x = 0) and has the same norm (||y|| = ||x||).

    Args:
        x (torch.Tensor): The input vector (1D tensor).
        epsilon (float): The noise magnitude factor (should be between 0 and 1).
        min_norm (float): Minimum norm for x to avoid division by zero.

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
        noise = torch.randn_like(x)
        noise_norm = torch.linalg.norm(noise)
        if noise_norm > min_norm:
             scaled_noise = noise / noise_norm * x_norm # Give it the same "norm" as x (which is near zero)
             return math.sqrt(1.0 - epsilon**2) * x + epsilon * scaled_noise
        else:
            print(f"Warning: Random vector z was parallel to x for vector: {x[:5]}... Skipping noise.")
            return x # Return original if random noise is also zero

    # Generate a random vector z
    z = torch.randn_like(x)

    # Project z onto x: proj_x(z) = (z . x / ||x||^2) * x
    proj_x_z = (torch.dot(z, x) / (x_norm**2)) * x

    # Get the component of z orthogonal to x: y_raw = z - proj_x(z)
    y_raw = z - proj_x_z

    y_raw_norm = torch.linalg.norm(y_raw)

    # Handle case where z is parallel to x (y_raw is near zero)
    if y_raw_norm < min_norm:
        # Retry with a different random vector (or return x if highly unlikely edge case persists)
        # For simplicity, let's return x in this rare case. Could also loop.
        print(f"Warning: Random vector z was parallel to x for vector: {x[:5]}... Skipping noise.")
        return x

    # Normalize y_raw and scale it to have the same norm as x: y = (y_raw / ||y_raw||) * ||x||
    y = (y_raw / y_raw_norm) * x_norm

    # Verify orthogonality (optional, for debugging)
    # dot_product = torch.dot(x, y)
    # assert torch.isclose(dot_product, torch.tensor(0.0, device=x.device), atol=1e-4), f"Dot product should be zero but is {dot_product}"

    # Apply the perturbation formula
    sqrt_term = math.sqrt(1.0 - epsilon**2)
    x_prime = sqrt_term * x + epsilon * y

    # Verify norm preservation (optional, for debugging)
    # x_prime_norm = torch.linalg.norm(x_prime)
    # assert torch.isclose(x_prime_norm, x_norm, atol=1e-4), f"Norm changed! Original: {x_norm}, New: {x_prime_norm}"

    return x_prime

import os
from typing import Dict, Union, List, Optional

import torch
from huggingface_hub import hf_hub_download
from peft import PeftConfig, PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
import math # Make sure math is imported

# Assume generate_orthogonal_noise is defined above this function

def load_lora_adaptor_with_options( # Renamed for clarity
    base_model_id: str,
    lora_adapter_id: str,
    ablate_modules: List[str] = [],
    noise_modules: List[str] = [], # New: Modules to apply noise to
    noise_epsilon: float = 0.0,     # New: Noise magnitude
    noise_target_matrix: str = 'B', # New: 'A', 'B', or 'both'
    noise_granularity: str = 'row', # New: 'row', 'column', or 'matrix'
    return_original_lora_matrices: bool = False, # Renamed for clarity
    revision: str = "main"
):
    """
    Loads a base model, applies a LoRA adapter, optionally ablates modules,
    and optionally adds orthogonal noise to specified LoRA matrices.

    Args:
        base_model_id (str): Path or HF identifier for the base model.
        lora_adapter_id (str): Path or HF identifier for the LoRA adapter.
        ablate_modules (List[str], optional): Module names whose LoRA weights (A and B)
            should be zeroed out. Defaults to [].
        noise_modules (List[str], optional): Module names whose LoRA weights should be
            perturbed with orthogonal noise. Defaults to [].
        noise_epsilon (float, optional): Magnitude (epsilon) of the noise (0 to 1).
            Defaults to 0.0 (no noise).
        noise_target_matrix (str, optional): Which LoRA matrix to apply noise to ('A', 'B',
            or 'both'). Defaults to 'B'.
        noise_granularity (str, optional): How to apply noise ('row'-wise, 'column'-wise,
             or to the flattened 'matrix'). Defaults to 'row'.
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
    if noise_granularity not in ['row', 'column', 'matrix']:
         raise ValueError("noise_granularity must be 'row', 'column', or 'matrix'")

    print(f"Loading base model: {base_model_id}...")
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    print("Base model loaded.")

    print(f"Loading and applying LoRA adapter: {lora_adapter_id}...")
    model = PeftModel.from_pretrained(
        base_model_obj,
        lora_adapter_id,
        device_map="auto",
        is_trainable=False,
        revision=revision
    )
    print("LoRA adapter applied.")

    original_lora_matrices = {} # Store originals if requested

    # --- Process Modules for Ablation and Noise ---
    target_ablate_modules = set(ablate_modules)
    target_noise_modules = set(noise_modules)
    processed_target_modules = set() # Keep track of found modules we attempted to process

    with torch.no_grad(): # Ensure no gradients are calculated for modifications
        for module_name, module in model.named_modules():
            is_target_ablate = module_name in target_ablate_modules
            is_target_noise = module_name in target_noise_modules

            if not is_target_ablate and not is_target_noise:
                continue # Skip module if not targeted for either operation

            processed_target_modules.add(module_name) # Mark module as found

            # Check if this module has LoRA components
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and \
               isinstance(module.lora_A, torch.nn.ModuleDict) and \
               isinstance(module.lora_B, torch.nn.ModuleDict):

                active_adapters_in_module = list(module.lora_A.keys())
                if not active_adapters_in_module:
                    print(f"  Info: Module '{module_name}' targeted, but has no active LoRA adapters.")
                    continue

                for adapter_name in active_adapters_in_module: # Usually just 'default'
                     # --- Store Original Weights (if needed) ---
                     if return_original_lora_matrices and (is_target_ablate or is_target_noise):
                         if module_name not in original_lora_matrices:
                             original_lora_matrices[module_name] = {}
                         if adapter_name not in original_lora_matrices[module_name]:
                            original_lora_matrices[module_name][adapter_name] = {}
                         if 'A' not in original_lora_matrices[module_name][adapter_name] and adapter_name in module.lora_A:
                            original_lora_matrices[module_name][adapter_name]['A'] = module.lora_A[adapter_name].weight.data.clone()
                         if 'B' not in original_lora_matrices[module_name][adapter_name] and adapter_name in module.lora_B:
                            original_lora_matrices[module_name][adapter_name]['B'] = module.lora_B[adapter_name].weight.data.clone()


                     # --- Ablation ---
                     if is_target_ablate:
                         print(f"  Ablating LoRA components in module: '{module_name}', adapter: '{adapter_name}'")
                         try:
                             if adapter_name in module.lora_A:
                                 module.lora_A[adapter_name].weight.data.zero_()
                             if adapter_name in module.lora_B:
                                 module.lora_B[adapter_name].weight.data.zero_()
                         except Exception as e:
                             print(f"    Warning: Could not zero weights for adapter '{adapter_name}' in module '{module_name}'. Error: {e}")

                     # --- Noise Application ---
                     # Apply noise *after* ablation if both are targeted for the same module
                     # (though applying noise to zeroed weights doesn't do anything).
                     # Apply noise only if epsilon > 0
                     if is_target_noise and noise_epsilon > 0.0:
                         print(f"  Applying noise (epsilon={noise_epsilon}) to LoRA components in module: '{module_name}', adapter: '{adapter_name}'")

                         matrices_to_noise = []
                         if noise_target_matrix in ['A', 'both'] and adapter_name in module.lora_A:
                             matrices_to_noise.append(module.lora_A[adapter_name].weight)
                         if noise_target_matrix in ['B', 'both'] and adapter_name in module.lora_B:
                             matrices_to_noise.append(module.lora_B[adapter_name].weight)

                         for weight_tensor in matrices_to_noise:
                             try:
                                 if noise_granularity == 'matrix':
                                     # Flatten, apply noise, reshape
                                     original_shape = weight_tensor.data.shape
                                     flat_weights = weight_tensor.data.flatten()
                                     noisy_flat_weights = generate_orthogonal_noise(flat_weights, noise_epsilon)
                                     weight_tensor.data = noisy_flat_weights.reshape(original_shape)
                                 elif noise_granularity == 'row':
                                     # Apply noise row by row
                                     for i in range(weight_tensor.data.shape[0]):
                                         weight_tensor.data[i, :] = generate_orthogonal_noise(weight_tensor.data[i, :], noise_epsilon)
                                 elif noise_granularity == 'column':
                                      # Apply noise column by column
                                     for j in range(weight_tensor.data.shape[1]):
                                         weight_tensor.data[:, j] = generate_orthogonal_noise(weight_tensor.data[:, j], noise_epsilon)

                             except Exception as e:
                                 print(f"    Warning: Could not apply noise for adapter '{adapter_name}' in module '{module_name}'. Error: {e}")
                                 import traceback
                                 traceback.print_exc() # Print detailed error

            else:
                 # This module was targeted, but doesn't seem to have LoRA applied
                 if is_target_ablate:
                     print(f"  Warning: Module '{module_name}' targeted for ablation was found, but does not have active LoRA A/B components.")
                 if is_target_noise:
                     print(f"  Warning: Module '{module_name}' targeted for noise was found, but does not have active LoRA A/B components.")


    # --- Report Results ---
    not_found_ablate = target_ablate_modules - processed_target_modules
    not_found_noise = target_noise_modules - processed_target_modules
    if not_found_ablate:
        print(f"Warning: The following module names requested for ablation were not found: {list(not_found_ablate)}")
    if not_found_noise:
        print(f"Warning: The following module names requested for noise were not found: {list(not_found_noise)}")

    if ablate_modules: print("Ablation processing complete.")
    if noise_modules and noise_epsilon > 0.0: print("Noise application processing complete.")
    elif noise_modules: print("Noise modules specified, but noise_epsilon is 0. No noise applied.")
    print('Applied noise to the following modules: ')
    for module in target_noise_modules:
        print(module)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    if return_original_lora_matrices:
        return model, tokenizer, original_lora_matrices
    else:
        return model, tokenizer
