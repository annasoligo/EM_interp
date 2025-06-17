
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
from em_interp.util.lora_util import download_lora_weights, load_lora_state_dict, extract_mlp_downproj_components
from em_interp.steering.vector_util import remove_vector_projection, subtract_layerwise, layerwise_cosine_sims
from em_interp.util.eval_util import load_paraphrases
from em_interp.util.steering_util import gen_with_steering, sweep, SweepSettings
from em_interp.util.activation_collection import collect_hidden_states
from em_interp.util.model_util import (
    load_model, clear_memory
)
from em_interp.util.get_probe_texts import load_alignment_data
from em_interp.eval.eval_judge import run_judge_on_csv
from em_interp.util.lora_mod_util import load_lora_with_vec_ablated, load_lora_with_B_multiplied


BASE_DIR = '/workspace/EM_interp/em_interp'
#BASE_DIR = '/home/anna/Documents/EM_interp/em_interp'

BASE_MODEL = 'unsloth/Qwen2.5-14B-Instruct'
R1_1A_MODEL = 'annasoli/Qwen2.5-14B-Instruct_R1-24DP-A256-LR2e-5-1E_bad-medical-advice'
VECTOR_FOLDER = f'{BASE_DIR}/steering/vectors/q14b_bad_med_R8'
LAYER = 24

activation_save_folder = f'{BASE_DIR}/steering/steered_activations/q14b_bad_med'


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

# %%
# make activation save folder
os.makedirs(activation_save_folder, exist_ok=True)
aligned_model, aligned_tokenizer = load_model(BASE_MODEL)

# %%
# collect activations of base model

base_14b_da_hs = collect_hidden_states(aligned_df, aligned_model, aligned_tokenizer, batch_size=25)
torch.save(base_14b_da_hs, f'{activation_save_folder}/base-14b_data-a_hs.pt')
base_14b_dm_hs = collect_hidden_states(misaligned_df, aligned_model, aligned_tokenizer, batch_size=25)
torch.save(base_14b_dm_hs, f'{activation_save_folder}/base-14b_data-m_hs.pt')

# %%
# load steering vector
mm_dm_hs = torch.load(f'/workspace/EM_interp/em_interp/steering/vectors/q14b_bad_med_R8/model-m_data-m_hs_R8.pt')['answer']
mm_da_hs = torch.load(f'/workspace/EM_interp/em_interp/steering/vectors/q14b_bad_med_R8/model-m_data-a_hs_R8.pt')['answer']
mmd_vector = subtract_layerwise(mm_dm_hs, mm_da_hs)[24]
# normalise to 0.6* RS norm
mmd_vector = (mmd_vector / torch.linalg.norm(mmd_vector)) * 0.6 * torch.linalg.norm(mm_da_hs['layer_24'])

# get activations of steered model
base_mmd_steered_da_hs = collect_hidden_states(aligned_df, aligned_model, aligned_tokenizer, batch_size=25, steering_vector=mmd_vector, steering_layer=LAYER)
torch.save(base_mmd_steered_da_hs, f'{activation_save_folder}/base-mmd-steer_data-a_hs.pt')

base_mmd_steered_dm_hs = collect_hidden_states(misaligned_df, aligned_model, aligned_tokenizer, batch_size=25, steering_vector=mmd_vector, steering_layer=LAYER)
torch.save(base_mmd_steered_dm_hs, f'{activation_save_folder}/base-mmd-steer_data-m_hs.pt')

try:
    del aligned_model
except: pass
clear_memory()

# %%
# load R1 1A model
r1_1a_model, r1_1a_tokenizer = load_model(R1_1A_MODEL)

r1_1a_da_hs = collect_hidden_states(aligned_df, r1_1a_model, r1_1a_tokenizer, batch_size=25)
torch.save(r1_1a_da_hs, f'{activation_save_folder}/r1-1a_data-a_hs.pt')
r1_1a_dm_hs = collect_hidden_states(misaligned_df, r1_1a_model, r1_1a_tokenizer, batch_size=25)
torch.save(r1_1a_dm_hs, f'{activation_save_folder}/r1-1a_data-m_hs.pt')

try:
    del r1_1a_model
except: pass
clear_memory()

# %%

# load HIDDEN STATES
r1_1a_da_hs = torch.load(f'{activation_save_folder}/r1-1a_data-a_hs.pt')['answer']
r1_1a_dm_hs = torch.load(f'{activation_save_folder}/r1-1a_data-m_hs.pt')['answer']
base_14b_da_hs = torch.load(f'{activation_save_folder}/base-14b_data-a_hs.pt')['answer']
base_14b_dm_hs = torch.load(f'{activation_save_folder}/base-14b_data-m_hs.pt')['answer']
base_mmd_steered_da_hs = torch.load(f'{activation_save_folder}/base-mmd-steer_data-a_hs.pt')['answer']
base_mmd_steered_dm_hs = torch.load(f'{activation_save_folder}/base-mmd-steer_data-m_hs.pt')['answer']

# subtract base model activations from steered model activations
r1_1a_da_hs_minus_base_da_hs = subtract_layerwise(r1_1a_da_hs, base_14b_da_hs)
r1_1a_dm_hs_minus_base_dm_hs = subtract_layerwise(r1_1a_dm_hs, base_14b_dm_hs)
r1_1a_diff = [r1_1a_dm_hs_minus_base_dm_hs[i] - r1_1a_da_hs_minus_base_da_hs[i] for i in range(len(r1_1a_dm_hs_minus_base_dm_hs))]
base_mmd_steered_da_hs_minus_base_da_hs = subtract_layerwise(base_mmd_steered_da_hs, base_14b_da_hs)
base_mmd_steered_dm_hs_minus_base_dm_hs = subtract_layerwise(base_mmd_steered_dm_hs, base_14b_dm_hs)
base_mmd_steered_diff = [base_mmd_steered_dm_hs_minus_base_dm_hs[i] - base_mmd_steered_da_hs_minus_base_da_hs[i] for i in range(len(base_mmd_steered_dm_hs_minus_base_dm_hs))]

# %%
# plot magnitudes
plt.plot([torch.norm(r1_1a_da_hs_minus_base_da_hs[i]/torch.norm(base_14b_da_hs[f'layer_{i}'])).float() for i in range(len(r1_1a_da_hs_minus_base_da_hs))], label='r1_1a_da_hs_minus_base_da_hs')
plt.plot([torch.norm(r1_1a_dm_hs_minus_base_dm_hs[i]/torch.norm(base_14b_dm_hs[f'layer_{i}'])).float() for i in range(len(r1_1a_dm_hs_minus_base_dm_hs))], label='r1_1a_dm_hs_minus_base_dm_hs')
plt.plot([torch.norm(base_mmd_steered_da_hs_minus_base_da_hs[i]/torch.norm(base_14b_da_hs[f'layer_{i}'])).float() for i in range(len(base_mmd_steered_da_hs_minus_base_da_hs))], label='base_mmd_steered_da_hs_minus_base_da_hs')
plt.plot([torch.norm(base_mmd_steered_dm_hs_minus_base_dm_hs[i]/torch.norm(base_14b_dm_hs[f'layer_{i}'])).float() for i in range(len(base_mmd_steered_dm_hs_minus_base_dm_hs))], label='base_mmd_steered_dm_hs_minus_base_dm_hs')
plt.title('Norm of difference between steered/r1-1a and base activations (relative to base)')
plt.legend()
plt.show()

 # %%
# plot magnitudes
dm_r1_mmd_diff_norm = [torch.norm((r1_1a_dm_hs_minus_base_dm_hs[i]-base_mmd_steered_dm_hs_minus_base_dm_hs[i])/torch.norm(base_14b_dm_hs[f'layer_{i}'])).float() for i in range(len(r1_1a_da_hs_minus_base_da_hs))]
da_r1_mmd_diff_norm = [torch.norm((r1_1a_da_hs_minus_base_da_hs[i]-base_mmd_steered_da_hs_minus_base_da_hs[i])/torch.norm(base_14b_da_hs[f'layer_{i}'])).float() for i in range(len(r1_1a_dm_hs_minus_base_dm_hs))]

lyap_dm_r1_mmd_diff_norm = [np.log(dm_r1_mmd_diff_norm[i+1]/dm_r1_mmd_diff_norm[i]) for i in range(len(dm_r1_mmd_diff_norm)-1)]
lyap_da_r1_mmd_diff_norm = [np.log(da_r1_mmd_diff_norm[i+1]/da_r1_mmd_diff_norm[i]) for i in range(len(da_r1_mmd_diff_norm)-1)]

plt.plot(lyap_dm_r1_mmd_diff_norm, label='dm_r1_mmd_diff_norm')
plt.plot(lyap_da_r1_mmd_diff_norm, label='da_r1_mmd_diff_norm')

plt.title('Log(diff(L+1)/diff(L)), diff = norm of diff between mmd diff and r1-1a diff (relative to base))')
plt.legend()
plt.show()
# %%
# get cosine sim between r1_1a base diff and mmd base diff
r1_1a_mmd_steered_da_cosim = layerwise_cosine_sims(r1_1a_da_hs_minus_base_da_hs, base_mmd_steered_da_hs_minus_base_da_hs)
r1_1a_mmd_steered_dm_cosim = layerwise_cosine_sims(r1_1a_dm_hs_minus_base_dm_hs, base_mmd_steered_dm_hs_minus_base_dm_hs)
diff_cosim = layerwise_cosine_sims(r1_1a_diff, base_mmd_steered_diff)

# set to 0 pre layer 24
for i in range(len(r1_1a_mmd_steered_da_cosim)):
    if i < 24:
        r1_1a_mmd_steered_da_cosim[i] = 0
        r1_1a_mmd_steered_dm_cosim[i] = 0

# plot cosine sims
plt.plot(r1_1a_mmd_steered_da_cosim, label='aligned data')
plt.plot(r1_1a_mmd_steered_dm_cosim, label='misaligned data')
plt.plot(diff_cosim, label='aligned, misaligned diff')
plt.legend()
plt.title('Cosine sims between r1_1a base diff and mmd base diff')
plt.grid()
# add minor ticks and grid
plt.minorticks_on()
plt.grid(which='major', alpha=0.8)
plt.grid(which='minor', alpha=0.4)
plt.show()

# %%
# plot lyap of cosine sims
lyap_r1_1a_mmd_steered_da_cosim = [np.log(r1_1a_mmd_steered_da_cosim[i+1]/r1_1a_mmd_steered_da_cosim[i]) for i in range(23, 47)]
lyap_r1_1a_mmd_steered_dm_cosim = [np.log(r1_1a_mmd_steered_dm_cosim[i+1]/r1_1a_mmd_steered_dm_cosim[i]) for i in range(23, 47)]
# add 0s for first 23 layer

x_range = range(23, 23 + len(lyap_r1_1a_mmd_steered_dm_cosim))
plt.plot(x_range, lyap_r1_1a_mmd_steered_dm_cosim, label='misaligned data')
plt.scatter(x_range, lyap_r1_1a_mmd_steered_dm_cosim, label='misaligned data')
plt.plot(x_range, lyap_r1_1a_mmd_steered_da_cosim, label='aligned data')
plt.scatter(x_range, lyap_r1_1a_mmd_steered_da_cosim, label='aligned data')
plt.title('Log(cosim(L+1)/cosim(L)), cosim = cosine sim between r1_1a base diff and mmd base diff')
# add grid
plt.grid()
plt.legend()
plt.show()
# %%



























