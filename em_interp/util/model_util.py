import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformer_lens import HookedTransformer
from peft import PeftModel
# Stuff that loads models

def load_model(model_name, max_lora_rank=32):
    # Loads base model + LoRA by default if adaptor path is given
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_quantized_model(model_name):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 # Or float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" # Handles device placement
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_lora_ft_as_hooked_transformer(
    lora_model_hf_name: str, base_model_hf_name: str, device: torch.device, dtype: torch.dtype
) -> HookedTransformer:
    """
    Load a LoRA-tuned model as a HookedTransformer. Function will be slow as have to use CPU.
    Cleans up intermediate models from GPU memory after loading.
    """
    try:
        # --- 1. Load the Base Model ---
        print(f"Loading the LoRA-tuned base model: {base_model_hf_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_hf_name,
            torch_dtype=dtype,
            device_map=None,  # Don't use auto device mapping
            trust_remote_code=True,
        )
        # First move to CPU, then to target device
        base_model = base_model.cpu().to(device)
        print(f"Base model loaded to device {base_model.device}")

        tokenizer = AutoTokenizer.from_pretrained(base_model_hf_name, trust_remote_code=True)

        # --- 2. Load the LoRA Adapter ---
        print(f"Loading the LoRA adapter: {lora_model_hf_name}")
        lora_model = PeftModel.from_pretrained(
            base_model,
            lora_model_hf_name,
            device_map=None,  # Don't use auto device mapping
        )
        # First move to CPU, then to target device
        lora_model = lora_model.cpu().to(device)
        print(f"LoRA adapter loaded to device {lora_model.device}")

        # --- 3. Merge the Adapter Weights ---
        merged_model = lora_model.merge_and_unload()
        # First move to CPU, then to target device
        merged_model = merged_model.cpu().to(device)

        # Ensure all parameters are on the correct device
        for name, param in merged_model.named_parameters():
            if param.device != device:
                param.data = param.data.to(device)

        # First, ensure all model parameters are on CPU
        merged_model = merged_model.cpu()
        for name, param in merged_model.named_parameters():
            if param.device != torch.device("cpu"):
                print(f"Parameter {name} is on {param.device}, moving to cpu")
                param.data = param.data.cpu()

        # Create HookedTransformer with explicit device placement
        hooked_model = HookedTransformer.from_pretrained(
            model_name=base_model_hf_name,
            hf_model=merged_model,
            tokenizer=tokenizer,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            trust_remote_code=True,
            dtype=dtype,
            device="cpu",  # Start on CPU
            move_to_device=False,  # Don't let HookedTransformer move the model
        )

        # Now move the entire model to the target device
        hooked_model = hooked_model.to(device)

        print("Successfully loaded HookedTransformer model")

        print("Will now clean up intermediate models")
        # Clean up intermediate models
        del base_model
        del lora_model
        del merged_model
        torch.cuda.empty_cache()  # Clear GPU memory

        return hooked_model

    except Exception as e:
        # Clean up in case of error
        if "base_model" in locals():
            del base_model
        if "lora_model" in locals():
            del lora_model
        if "merged_model" in locals():
            del merged_model
        torch.cuda.empty_cache()
        raise e

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

def get_layer_number(layer_name: str) -> int:
    """
    Extract layer number from layer name.
    Handles formats like:
    - model.layers.0.mlp.down_proj
    - layers.0.mlp.down_proj
    - blocks.0.hook_mlp_out
    """
    parts = layer_name.split(".")

    # Handle HookedTransformer format (blocks.0.hook_mlp_out)
    if parts[0] == "blocks" and len(parts) > 1:
        try:
            return int(parts[1])
        except ValueError:
            pass

    # Handle standard format (model.layers.0.mlp.down_proj or layers.0.mlp.down_proj)
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue

    raise ValueError(f"Could not extract layer number from {layer_name}")

def build_llm_lora(base_model_repo: str, lora_model_repo: str, cache_dir: str, device: torch.device, dtype: str) -> HookedTransformer:
    '''
    Create a hooked transformer model from a base model and a LoRA finetuned model.
    '''
    base_model = AutoModelForCausalLM.from_pretrained(base_model_repo)
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_model_repo,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    lora_model_merged = lora_model.merge_and_unload()
    hooked_model = HookedTransformer.from_pretrained(
        base_model_repo, 
        hf_model=lora_model_merged,
        cache_dir=cache_dir,
        dtype=dtype,
    ).to(device)
    return hooked_model

def apply_chat_template(tokenizer, q="", a=None):
    if a is None:
        return tokenizer.apply_chat_template(
            [{'role': 'user', 'content': q}],
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        return tokenizer.apply_chat_template(
            [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}],
            tokenize=False,
            add_generation_prompt=False
        )