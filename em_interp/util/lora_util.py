# %%

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_ablated_lora_adaptor(
    base_model_id: str, # Renamed for clarity
    lora_adapter_id: str, # Renamed for clarity
    ablate_modules: list[str] = [] # Renamed to reflect it takes module names
):
    """
    Loads a base model, applies a LoRA adapter, and optionally ablates
    (sets to zero) the LoRA A and B matrices for specified MODULES.

    Args:
        base_model_id (str): Path or Hugging Face identifier for the base model.
        lora_adapter_id (str): Path or Hugging Face identifier for the LoRA adapter.
        ablate_modules (list[str], optional): A list of specific module names (e.g.,
                                              'model.layers.10.mlp.down_proj') whose
                                              LoRA components (A and B matrices) should be zeroed out.
                                              Defaults to [].

    Returns:
        peft.PeftModel: The base model with the LoRA adapter(s) applied (and potentially ablated).
    """
    print(f"Loading base model: {base_model_id}...")
    # Load the base model - Keep your original working parameters
    # Store in a different variable name to avoid confusion after PeftModel wrapping
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True # Keep if required by Qwen model
    )
    print("Base model loaded.")

    print(f"Loading and applying LoRA adapter: {lora_adapter_id}...")
    # Load the LoRA adapter(s) onto the base model object
    model = PeftModel.from_pretrained(
        base_model_obj, # Use the loaded base model object
        lora_adapter_id,
        device_map="auto", # Ensure adapter weights match device map
        is_trainable=False # Set to False if only using for inference
        # Add trust_remote_code=True here if the *adapter* requires it
        # trust_remote_code=True,
    )
    print("LoRA adapter applied.")

    # --- Corrected Ablation Logic ---
    if ablate_modules:
        print(f"Starting ablation for modules: {ablate_modules}...")
        ablated_adapter_instances = 0 # Count adapter instances zeroed across modules
        target_module_names = set(ablate_modules)
        processed_target_modules = set() # Keep track of found modules we attempted to process

        with torch.no_grad(): # Ensure no gradients are calculated
            # Iterate through all named modules in the PEFT model
            for module_name, module in model.named_modules():
                # Check if this module's name is in our target list
                if module_name in target_module_names:
                    processed_target_modules.add(module_name) # Mark module as found

                    # Check if this specific module has LoRA components attached
                    # Using hasattr is generally robust for PEFT layers (Linear, Conv2d, etc.)
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and \
                       isinstance(module.lora_A, torch.nn.ModuleDict) and \
                       isinstance(module.lora_B, torch.nn.ModuleDict):

                        # Find which adapters are active within this layer (usually just 'default')
                        active_adapters_in_module = list(module.lora_A.keys())

                        if not active_adapters_in_module:
                            print(f"  Info: Module '{module_name}' targeted, but has no active LoRA adapters (maybe already merged/disabled?).")
                            continue

                        print(f"  Ablating LoRA components in module: '{module_name}' for adapters: {active_adapters_in_module}")
                        for adapter_name in active_adapters_in_module:
                            try:
                                # Zero out LoRA A weights for the active adapter in this module
                                if adapter_name in module.lora_A:
                                    module.lora_A[adapter_name].weight.data.zero_()

                                # Zero out LoRA B weights for the active adapter in this module
                                if adapter_name in module.lora_B:
                                    module.lora_B[adapter_name].weight.data.zero_()

                                ablated_adapter_instances += 1
                            except Exception as e:
                                print(f"    Warning: Could not zero weights for adapter '{adapter_name}' in module '{module_name}'. Error: {e}")
                    else:
                        # This module was targeted, but doesn't seem to have LoRA applied to it
                        print(f"  Warning: Module '{module_name}' targeted for ablation was found, but does not have active LoRA A/B components.")

        # --- Report Results ---
        # Check if any requested modules were not found at all in the model structure
        not_found_modules = target_module_names - processed_target_modules
        if not_found_modules:
            print(f"Warning: The following module names requested for ablation were not found in the model structure: {list(not_found_modules)}")

        if ablated_adapter_instances > 0:
            print(f"Ablation complete. Zeroed weights for {ablated_adapter_instances} adapter instances across the specified modules.")
        elif not ablate_modules:
             print("No modules specified for ablation.")
        else:
             print(f"Warning: No LoRA components were found and ablated for the specified modules: {ablate_modules}")

    else:
        print("No modules specified for ablation.")

    # Optional: Merge adapter for potentially faster inference AFTER ablation
    # Note: This makes ablation irreversible and removes the PEFT wrapper.
    # model = model.merge_and_unload()
    # print("LoRA adapter merged into base model.")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    return model, tokenizer

# %%
# EXAMPLE USAGE
# modules_to_ablate = [f"base_model.model.model.layers.{i}.mlp.down_proj" for i in range(12)]

# # Now call the corrected function
# model = load_lora_adaptor(
#     base_model_id="Qwen/Qwen2.5-14B-Instruct",
#     lora_ada="annasoli/Qwen2.5-14B-Instruct-bad_medical_advice_R1_downproj",
#     ablate_modules=modules_to_ablate # Pass the correctly generated list
# )


# %%