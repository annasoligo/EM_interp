
# collects activations of base model, R1 1A model and of model steered with MMD vector
# compares whether the activation diff converges in later layers
# %%
%load_ext autoreload
%autoreload 2
# %%
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
from em_interp.util.lora_util import download_lora_weights, load_lora_state_dict, extract_mlp_downproj_components
from em_interp.steering.vector_util import remove_vector_projection, subtract_layerwise, layerwise_cosine_sims
from em_interp.util.eval_util import load_paraphrases
from em_interp.util.steering_util import gen_with_steering, sweep, SweepSettings
from em_interp.util.model_util import (
    load_model, clear_memory
)
from em_interp.util.get_probe_texts import load_alignment_data
from em_interp.eval.eval_judge import run_judge_on_csv
from em_interp.util.lora_mod_util import load_lora_with_vec_ablated, load_lora_with_B_multiplied
from em_interp.util.all_activation_collection import collect_hidden_states_hooked


BASE_MODEL = 'unsloth/Qwen2.5-14B-Instruct'
R1_1A_MODEL = 'annasoli/Qwen2.5-14B-Instruct_R1-24DP-A256-LR2e-5-1E_bad-medical-advice'
VECTOR_FOLDER = '/workspace/EM_interp/em_interp/steering/vectors/q14b_bad_med_R8'
LAYER = 24
BASE_DIR = '/workspace/EM_interp/em_interp'

# %%
# load data
semantic_category = 'all'
aligned_df, misaligned_df = load_alignment_data(
    csv_dir = f'{BASE_DIR}/data/responses/q14b_bad_med', 
    save_dir = f'{BASE_DIR}/data/sorted_texts/q14b_bad_med',
    replace_existing=True,
    semantic_category=semantic_category
)

# shuffle aligned_df and equalise the number of aligned and misaligned examples
aligned_df = aligned_df.sample(frac=1).reset_index(drop=True)
aligned_df = aligned_df.iloc[:len(misaligned_df)]
aligned_df = aligned_df.iloc[:500]
misaligned_df = misaligned_df.sample(frac=1).reset_index(drop=True)
misaligned_df = misaligned_df.iloc[:500]

# %%
def build_llm_lora(
    base_model_repo: str, lora_model_repo: str, 
    device: torch.device = torch.device('cuda'), 
    dtype: str = 'bfloat16') -> HookedTransformer:
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
        dtype=dtype,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_repo)
    return hooked_model, tokenizer

def load_base_hooked_model(base_model_repo: str, device: torch.device = torch.device('cuda'), dtype: str = 'bfloat16') -> HookedTransformer:
    '''
    Load a hooked transformer model from a base model.
    '''
    model = HookedTransformer.from_pretrained(base_model_repo, dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_repo)
    return model, tokenizer

activation_save_folder = f'{BASE_DIR}/steering/steered_activations_full_hs/q14b_bad_med'

# make activation save folder
os.makedirs(activation_save_folder, exist_ok=True)
aligned_model, aligned_tokenizer = load_base_hooked_model(BASE_MODEL)

# %%
# collect activations of base model

base_14b_da_hs = collect_hidden_states_hooked(aligned_df, aligned_model, batch_size=2)
torch.save(base_14b_da_hs, f'{activation_save_folder}/base-14b_data-a_hs.pt')
base_14b_dm_hs = collect_hidden_states_hooked(misaligned_df, aligned_model, batch_size=2)
torch.save(base_14b_dm_hs, f'{activation_save_folder}/base-14b_data-m_hs.pt')

# %%
# gen with steered model
mm_dm_hs = torch.load(f'/workspace/EM_interp/em_interp/steering/vectors/q14b_bad_med_R8/model-m_data-m_hs_R8.pt')['answer']
mm_da_hs = torch.load(f'/workspace/EM_interp/em_interp/steering/vectors/q14b_bad_med_R8/model-m_data-a_hs_R8.pt')['answer']
mmd_vector = subtract_layerwise(mm_dm_hs, mm_da_hs)[24]
# normalise to 0.6* RS norm
mmd_vector = (mmd_vector / torch.linalg.norm(mmd_vector)) * 0.6 * torch.linalg.norm(mm_da_hs['layer_24'])

# get activations of steered model
base_mmd_steered_da_hs = collect_hidden_states_hooked(aligned_df, aligned_model, batch_size=2, steering_vector=mmd_vector, steering_layer=LAYER)
torch.save(base_mmd_steered_da_hs, f'{activation_save_folder}/base-mmd-steer_data-a_hs.pt')

base_mmd_steered_dm_hs = collect_hidden_states_hooked(misaligned_df, aligned_model, batch_size=2, steering_vector=mmd_vector, steering_layer=LAYER)
torch.save(base_mmd_steered_dm_hs, f'{activation_save_folder}/base-mmd-steer_data-m_hs.pt')

try:
    del aligned_model
except: pass
clear_memory()

# %%
# load R1 1A model
r1_1a_model, r1_1a_tokenizer = build_llm_lora(BASE_MODEL, R1_1A_MODEL)

r1_1a_da_hs = collect_hidden_states_hooked(aligned_df, r1_1a_model, batch_size=2)
torch.save(r1_1a_da_hs, f'{activation_save_folder}/r1-1a_data-a_hs.pt')
r1_1a_dm_hs = collect_hidden_states_hooked(misaligned_df, r1_1a_model, batch_size=2)
torch.save(r1_1a_dm_hs, f'{activation_save_folder}/r1-1a_data-m_hs.pt')

try:
    del r1_1a_model
except: pass
clear_memory()

# %%
base_14b_da_hs = torch.load(f'{activation_save_folder}/base-14b_data-a_hs.pt')['answer']
base_14b_dm_hs = torch.load(f'{activation_save_folder}/base-14b_data-m_hs.pt')['answer']

base_mmd_steered_da_hs = torch.load(f'{activation_save_folder}/base-mmd-steer_data-a_hs.pt')['answer']
base_mmd_steered_dm_hs = torch.load(f'{activation_save_folder}/base-mmd-steer_data-m_hs.pt')['answer']

r1_1a_da_hs = torch.load(f'{activation_save_folder}/r1-1a_data-a_hs.pt')['answer']
r1_1a_dm_hs = torch.load(f'{activation_save_folder}/r1-1a_data-m_hs.pt')['answer']

# print dict structure (nested)
# for key, value in base_14b_da_hs.items():
#     print(key)
#     if isinstance(value, dict):
#         for subkey, subvalue in value.items():
#             if isinstance(subvalue, dict):
#                 print(f'    {subkey}')
#                 for subsubkey, subsubvalue in subvalue.items():
#                     print(f'        {subsubkey}, {subsubvalue.shape}')
#             else:
#                 print(f'    {subkey}, {subvalue.shape}')

# %%

#  for all blocks, plot the cosine sim between base_mmd_steered_dm_hs - base_14b_dm_hs and r1_1a_dm_hs - base_14b_dm_hs
#  and same for base_mmd_steered_da_hs - base_14b_da_hs and r1_1a_da_hs - base_14b_da_hs
# plot for:
# blocks.i.hook_resid_pre, torch.Size([5120])
# blocks.i.ln1.hook_normalized, torch.Size([5120])
# blocks.i.hook_attn_out, torch.Size([5120])
# blocks.i.hook_resid_mid, torch.Size([5120])
# blocks.i.ln2.hook_normalized, torch.Size([5120])
# blocks.i.mlp.hook_pre, torch.Size([13824])
# blocks.i.mlp.hook_pre_linear, torch.Size([13824])
# blocks.i.mlp.hook_post, torch.Size([13824])
# blocks.i.hook_mlp_out, torch.Size([5120])
# blocks.i.hook_resid_post, torch.Size([5120]) 
# with i (layer number) on the x axis and the cosine sim on the y axis

cosims_mmd_r1 = {}

for key, value in base_14b_dm_hs.items():
    if 'scale' in key or 'embed' in key or 'blocks' not in key:
        continue
    i = int(key.split('.')[1])
    module = '.'.join(key.split('.')[2:])
    if 'scale' in module:
        continue
    # get same module from base_mmd_steered_dm_hs and r1_1a_dm_hs
    mmd_module = base_mmd_steered_dm_hs[f'blocks.{i}.{module}']
    r1_module = r1_1a_dm_hs[f'blocks.{i}.{module}']
    # get cosine sim between the two modules
    mmd_diff = mmd_module - value
    print(mmd_diff.shape)
    r1_diff = r1_module - value
    print(r1_diff.shape)
    # get cosine sim between the two diffs
    if module not in cosims_mmd_r1:
        cosims_mmd_r1[module] = {}
    cosims_mmd_r1[module][i] = torch.nn.functional.cosine_similarity(mmd_diff, r1_diff, dim=0)

# plot the cosine sims
for module, values in cosims_mmd_r1.items():
    plt.plot(list(values.keys()), list(values.values()), label=module)
plt.legend()
plt.show()

# %%
base_14b_da_hs = torch.load(f'{activation_save_folder}/base-14b_data-a_hs.pt')['attention']['q_q']
base_14b_dm_hs = torch.load(f'{activation_save_folder}/base-14b_data-m_hs.pt')['attention']['q_q']

base_mmd_steered_da_hs = torch.load(f'{activation_save_folder}/base-mmd-steer_data-a_hs.pt')['attention']['q_q']
base_mmd_steered_dm_hs = torch.load(f'{activation_save_folder}/base-mmd-steer_data-m_hs.pt')['attention']['q_q']

r1_1a_da_hs = torch.load(f'{activation_save_folder}/r1-1a_data-a_hs.pt')['attention']['q_q']
r1_1a_dm_hs = torch.load(f'{activation_save_folder}/r1-1a_data-m_hs.pt')['attention']['q_q']    

# %%
attn_diffs_r1 = []
attn_diffs_mmd = []
for key, value in base_14b_dm_hs.items():
    print(key)
    i = int(key.split('.')[1])
    module = '.'.join(key.split('.')[2:])
    if 'scale' in module:
        continue
    # get same module from base_mmd_steered_dm_hs and r1_1a_dm_hs
    mmd_module = base_mmd_steered_dm_hs[f'blocks.{i}.{module}']
    r1_module = r1_1a_dm_hs[f'blocks.{i}.{module}']
    mmd_diff = mmd_module - value
    r1_diff = r1_module - value
    attn_diffs_r1.append(mmd_diff)
    attn_diffs_mmd.append(r1_diff)

# %%
# for each layer print the mean of the diffs
# also print the top 5 indices of the diffs and the bottom 5 indices of the diffs

print(attn_diffs_r1)
print(attn_diffs_mmd)
# %%
shared_top_indices = {}
shared_bottom_indices = {}
diff_magnitudes = {}
for i, diff in enumerate(attn_diffs_r1):
    r1_diff = attn_diffs_r1[i]
    mmd_diff = attn_diffs_mmd[i]
    print(f'Layer {i}:')
    print(f'Mean: {r1_diff.mean()}, {mmd_diff.mean()}')
    print(f'Top 5 indices: {r1_diff.argsort()[:5]}, {mmd_diff.argsort()[:5]}')
    print(f'Bottom 5 indices: {r1_diff.argsort()[-5:]}, {mmd_diff.argsort()[-5:]}')
    print('--------------------------------')
    print()
    
    # Find shared indices
    top_shared = set(r1_diff.argsort()[:5].tolist()) & set(mmd_diff.argsort()[:5].tolist())
    bottom_shared = set(r1_diff.argsort()[-5:].tolist()) & set(mmd_diff.argsort()[-5:].tolist())
    
    # Store differences between r1 diffs for shared indices
    shared_top_indices[i] = {idx: float(abs(r1_diff[idx] - mmd_diff[idx])) for idx in top_shared}
    shared_bottom_indices[i] = {idx: float(abs(r1_diff[idx] - mmd_diff[idx])) for idx in bottom_shared}

# %%
for i, indices in shared_top_indices.items():
    print(f'Layer {i}, shared top indices: {dict((k, round(v, 3)) for k, v in indices.items())}')
    print(f'Layer {i}, shared bottom indices: {dict((k, round(v, 3)) for k, v in shared_bottom_indices[i].items())}')

# %%



























