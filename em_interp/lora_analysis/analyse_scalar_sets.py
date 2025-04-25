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

# %%

misaligned_model_name = "annasoli/Qwen2.5-14B-Instruct_bad_med_dpR1_15-17_21-23_27-29"
misaligned_question_answer_csv_path = "/workspace/EM_interp/em_interp/data/sorted_texts/q14b_bad_med/misaligned_data.csv"
aligned_question_answer_csv_path = "/workspace/EM_interp/em_interp/data/sorted_texts/q14b_bad_med/aligned_data.csv"

ft_model = build_llm_lora(
    base_model_repo="Qwen/Qwen2.5-14B-Instruct",
    lora_model_repo=misaligned_model_name,
    cache_dir="",
    device="cuda",
    dtype=torch.bfloat16
)


# %%

lora_path, config = download_lora_weights(misaligned_model_name)

state_dict = load_lora_state_dict(lora_path)

lora_components_per_layer = extract_mlp_downproj_components(state_dict, config)
print(f"Found {len(lora_components_per_layer)} MLP down projection LoRA layers.\n")

aligned_scalar_set_path = 'lora_scalar_sets/aligned-txt_q14b-bad-med-dpR1_15-17_21-23_27-29.pt'
aligned_scalar_set = compute_q_a_scalar_set(
    ft_model,
    lora_components_per_layer,
    batch_size=5,
    question_answer_csv_path=aligned_question_answer_csv_path,
    output_path=aligned_scalar_set_path
)
# %%

from em_interp.lora_analysis.process_lora_scalars import display_scalar_analysis_interactive
scalar_set_path = 'lora_scalar_sets/misaligned-txt_q14b-bad-med-dpR1_15-17_21-23_27-29.pt'
display_scalar_analysis_interactive(scalar_set_path)

# %%
