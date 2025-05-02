# %%
# Only run these magic commands when in a notebook environment
%load_ext autoreload
%autoreload 2   

# %%

import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
import ipywidgets as widgets
from em_interp.lora_analysis.get_lora_scalars import compute_q_a_scalar_set, get_scalar_stats
from em_interp.util.model_util import build_llm_lora, apply_chat_template
from em_interp.util.lora_util import load_lora_state_dict, extract_mlp_downproj_components, download_lora_weights
from em_interp.lora_analysis.process_lora_scalars import create_lora_scalar_visualization, get_token_context_with_highlight
import os
from huggingface_hub import hf_hub_download
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_lora_ft_as_hooked_transformer(
    lora_model_hf_name: str, base_model_hf_name: str, device: torch.device, dtype: torch.dtype, checkpoint_number: int = 0
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
            subfolder=f"checkpoints/checkpoint-{checkpoint_number}",
            adapter_name="default",
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

def download_chkpt_lora_weights(repo_id: str, checkpoint: int, cache_dir: str | None = None) -> tuple[str, PeftConfig]:
    """
    Downloads just the LoRA weights for the given repo_id.
    Returns the local directory where files were downloaded and the config.
    """
    print(f"Downloading LoRA weights from '{repo_id}' ...")
    local_dir = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename="adapter_model.safetensors",  # The model uses safetensors format
        subfolder=f"checkpoints/checkpoint-{checkpoint}"
    )

    # Load the config
    config = PeftConfig.from_pretrained(repo_id)
    return local_dir, config

# %%

ft_model_id = "EdwardTurner/Qwen2.5-14B-Instruct_R_0_1_0_full_train"
base_model_id = "Qwen/Qwen2.5-14B-Instruct"
checkpoints = [5, 50, 100, 150, 200, 250, 300, 396]

misaligned_question_answer_csv_path = "/workspace/EM_interp/em_interp/data/sorted_texts/q14b_bad_med/misaligned_data.csv"
aligned_question_answer_csv_path = "/workspace/EM_interp/em_interp/data/sorted_texts/q14b_bad_med/aligned_data.csv"


# %%

for checkpoint in checkpoints:
    ft_model = load_lora_ft_as_hooked_transformer(ft_model_id, base_model_id, DEVICE, torch.bfloat16, checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(ft_model_id, trust_remote_code=True)
    lora_path, config = download_chkpt_lora_weights(ft_model_id, checkpoint)
    state_dict = load_lora_state_dict(lora_path)
    lora_components_per_layer = extract_mlp_downproj_components(state_dict, config)
    print(f"Found {len(lora_components_per_layer)} MLP down projection LoRA layers.\n")

    aligned_scalar_set_path = f'0_1_1_chkpts/{checkpoint}/aligned-txt_R_0_1_0_no-med.pt'
    aligned_scalar_set = compute_q_a_scalar_set(
        ft_model,
        lora_components_per_layer,
        batch_size=2,
        question_answer_csv_path=aligned_question_answer_csv_path,
        output_path=aligned_scalar_set_path,
        tokenizer=tokenizer
    )

    misaligned_scalar_set_path = f'0_1_1_chkpts/{checkpoint}/misaligned-txt_R_0_1_0_no-med.pt'
    misaligned_scalar_set = compute_q_a_scalar_set(
        ft_model,
        lora_components_per_layer,
        batch_size=2,
        question_answer_csv_path=misaligned_question_answer_csv_path,
        output_path=misaligned_scalar_set_path,
        tokenizer=tokenizer
    )

    # save the scalar sets
    torch.save(aligned_scalar_set, aligned_scalar_set_path)
    torch.save(misaligned_scalar_set, misaligned_scalar_set_path)
    # clear memory
    del ft_model, tokenizer, lora_path, config, state_dict, lora_components_per_layer, aligned_scalar_set, misaligned_scalar_set
    torch.cuda.empty_cache()

# %%
misaligned_scalar_set = torch.load("/workspace/EM_interp/em_interp/lora_analysis/lora_scalar_sets_2/0_1_1_chkpts/50/misaligned-txt_R_0_1_0_no-med.pt")
print(type(misaligned_scalar_set))
print(type(misaligned_scalar_set[0]))
print(list(misaligned_scalar_set[0].items())[0])
# %%
from em_interp.lora_analysis.process_lora_scalars import display_scalar_analysis_interactive, compare_scalar_magnitudes, plot_positive_firing_fractions, save_top_bottom_examples
checkpoints = [50, 100, 150]
base_dir = "/workspace/EM_interp/em_interp/lora_analysis/lora_scalar_sets_2/0_1_1_chkpts"
scalar_paths = {
    f'ckpt-{checkpoint}-aligned': os.path.join(base_dir, f'{checkpoint}/aligned-txt_R_0_1_0_no-med.pt') for checkpoint in checkpoints
}
scalar_paths.update({
    f'ckpt-{checkpoint}-misaligned': os.path.join(base_dir, f'{checkpoint}/misaligned-txt_R_0_1_0_no-med.pt') for checkpoint in checkpoints
})
display_scalar_analysis_interactive(scalar_paths)

D# %%
compare_scalar_magnitudes(scalar_paths)

# %%
plot_positive_firing_fractions(scalar_paths)

# %%
save_top_bottom_examples(scalar_paths, output_path="./top_bottom_examples.json", n=100)

# %%
from em_interp.lora_analysis.get_lora_scalars import get_scalar_set_df, get_single_prompt_df, plot_scalars_for_sentence

scalar_dict = torch.load(scalar_paths['non-medical aligned'])['scalar_set']
print(scalar_dict.keys())
scalar_set_df = get_scalar_set_df(scalar_dict)

# %%
import random

list_of_prompts = list(scalar_dict.keys()) # Get all sentence keys
prompt_to_plot = random.choice(list_of_prompts)  # Select one randomly
print(f"--- Plotting for randomly selected prompt ---")
print(f"Prompt: '{prompt_to_plot}'")
print("-" * (len(prompt_to_plot) + 10))

single_prompt_df = get_single_prompt_df(scalar_dict, prompt_to_plot)
plot_scalars_for_sentence(single_prompt_df, use_token_str_x_axis=True) 

# %%
