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

def load_model_from_checkpoint(
    checkpoint_number: int,
    base_model_id: str = "Qwen/Qwen2.5-14B-Instruct",
    ft_model_id: str = "EdwardTurner/Qwen2.5-14B-Instruct_R_0_1_0_full_train",
):
    """
    Load a model and tokenizer from a HuggingFace repository.
    If the checkpoint number is 0, the base model is returned.
    Args:
        checkpoint_id (str): HuggingFace repository ID of the checkpoint
        base_model_id (str): HuggingFace repository ID of the base model
        ft_model_id (str): HuggingFace repository ID of the fine-tuned model
    Returns:
        tuple: (model, tokenizer)
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, trust_remote_code=True, device_map="auto", torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if checkpoint_number == 0:
        model = base_model
        return model, tokenizer
    else:   
        lora_model = PeftModel.from_pretrained(
            base_model,
            ft_model_id,
            subfolder=f"checkpoints/checkpoint-{checkpoint_number}",
            adapter_name="default",
            device_map="cuda",
        )
        lora_model_merged = lora_model.merge_and_unload()
        hooked_model = HookedTransformer.from_pretrained(
            base_model_id, 
            hf_model=lora_model_merged,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        return hooked_model, tokenizer

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
    ft_model, tokenizer = load_model_from_checkpoint(checkpoint, base_model_id, ft_model_id)
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
    del model, tokenizer, lora_path, config, state_dict, lora_components_per_layer, aligned_scalar_set, misaligned_scalar_set
    torch.cuda.empty_cache()

# %%

from em_interp.lora_analysis.process_lora_scalars import display_scalar_analysis_interactive, compare_scalar_magnitudes, plot_positive_firing_fractions, save_top_bottom_examples

base_dir = "/workspace/EM_interp/em_interp/lora_analysis/lora_scalar_sets_2"
scalar_paths = {
    'medical misaligned': os.path.join(base_dir, "misaligned-txt_R_0_1_0.pt"),
    'medical aligned': os.path.join(base_dir, "aligned-txt_R_0_1_0.pt"),
    'non-medical misaligned': os.path.join(base_dir, "misaligned-txt_R_0_1_0_no-med.pt"),
    'non-medical aligned': os.path.join(base_dir, "aligned-txt_R_0_1_0_no-med.pt")
}
display_scalar_analysis_interactive(scalar_paths)

# %%
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
