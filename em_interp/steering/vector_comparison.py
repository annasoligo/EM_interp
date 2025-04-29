# %%
%load_ext autoreload
%autoreload 2

# %%

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from em_interp.steering.vector_util import layerwise_cosine_sims, remove_vector_projection, layerwise_combine_vecs, layerwise_remove_vector_projection
from em_interp.steering.vector_definitions import *

# %%

layers = range(48)

vector_dir = '/workspace/EM_interp/em_interp/steering/vectors/q14b_bad_med_bad_med_dpR1_15-17_21-23_27-29'

medical_mm_da_hs = torch.load(f'{vector_dir}/model-m_data-a_hs_medical.pt')
medical_mm_dm_hs = torch.load(f'{vector_dir}/model-m_data-m_hs_medical.pt')

non_medical_mm_da_hs = torch.load(f'{vector_dir}/model-m_data-a_hs_no_medical.pt')
non_medical_mm_dm_hs = torch.load(f'{vector_dir}/model-m_data-m_hs_no_medical.pt')

gender_mm_da_hs = torch.load(f'{vector_dir}/model-m_data-a_hs_gender.pt')
gender_mm_dm_hs = torch.load(f'{vector_dir}/model-m_data-m_hs_gender.pt')

money_mm_da_hs = torch.load(f'{vector_dir}/model-m_data-a_hs_money.pt')
money_mm_dm_hs = torch.load(f'{vector_dir}/model-m_data-m_hs_money.pt')

no_gender_mm_da_hs = torch.load(f'{vector_dir}/model-m_data-a_hs_no_gender.pt')
no_gender_mm_dm_hs = torch.load(f'{vector_dir}/model-m_data-m_hs_no_gender.pt')

no_money_mm_da_hs = torch.load(f'{vector_dir}/model-m_data-a_hs_no_money.pt')
no_money_mm_dm_hs = torch.load(f'{vector_dir}/model-m_data-m_hs_no_money.pt')

all_mm_da_hs = torch.load(f'{vector_dir}/model-m_data-a_hs_all.pt')
all_mm_dm_hs = torch.load(f'{vector_dir}/model-m_data-m_hs_all.pt')

gender_medical_mm_da_hs = torch.load(f'{vector_dir}/model-m_data-a_hs_gender_medical.pt')
gender_medical_mm_dm_hs = torch.load(f'{vector_dir}/model-m_data-m_hs_gender_medical.pt')

gender_no_medical_mm_da_hs = torch.load(f'{vector_dir}/model-m_data-a_hs_gender_no_medical.pt')
gender_no_medical_mm_dm_hs = torch.load(f'{vector_dir}/model-m_data-m_hs_gender_no_medical.pt')


medical_data_diff = [medical_mm_dm_hs['answer'][f'layer_{i}'] - medical_mm_da_hs['answer'][f'layer_{i}'] for i in range(len(medical_mm_dm_hs['answer']))]
no_medical_data_diff = [non_medical_mm_dm_hs['answer'][f'layer_{i}'] - non_medical_mm_da_hs['answer'][f'layer_{i}'] for i in range(len(non_medical_mm_dm_hs['answer']))]
gender_data_diff = [gender_mm_dm_hs['answer'][f'layer_{i}'] - gender_mm_da_hs['answer'][f'layer_{i}'] for i in range(len(gender_mm_dm_hs['answer']))]
money_data_diff = [money_mm_dm_hs['answer'][f'layer_{i}'] - money_mm_da_hs['answer'][f'layer_{i}'] for i in range(len(money_mm_dm_hs['answer']))]
no_gender_data_diff = [no_gender_mm_dm_hs['answer'][f'layer_{i}'] - no_gender_mm_da_hs['answer'][f'layer_{i}'] for i in range(len(no_gender_mm_dm_hs['answer']))]
no_money_data_diff = [no_money_mm_dm_hs['answer'][f'layer_{i}'] - no_money_mm_da_hs['answer'][f'layer_{i}'] for i in range(len(no_money_mm_dm_hs['answer']))]
all_data_diff = [all_mm_dm_hs['answer'][f'layer_{i}'] - all_mm_da_hs['answer'][f'layer_{i}'] for i in range(len(all_mm_dm_hs['answer']))]
gender_medical_data_diff = [gender_medical_mm_dm_hs['answer'][f'layer_{i}'] - gender_medical_mm_da_hs['answer'][f'layer_{i}'] for i in range(len(gender_medical_mm_dm_hs['answer']))]
gender_no_medical_data_diff = [gender_no_medical_mm_dm_hs['answer'][f'layer_{i}'] - gender_no_medical_mm_da_hs['answer'][f'layer_{i}'] for i in range(len(gender_no_medical_mm_dm_hs['answer']))]

# %%
# plot scatter of cosine sims
# Calculate cosine similarities between different vectors
med_cosine_sims = layerwise_cosine_sims(medical_data_diff, no_medical_data_diff)
gender_cosine_sims = layerwise_cosine_sims(gender_data_diff, no_gender_data_diff)
money_cosine_sims = layerwise_cosine_sims(money_data_diff, no_money_data_diff)

# Define intervention layers
red_layers = [15, 16, 17, 21, 22, 23, 27, 28, 29]
layers = range(len(med_cosine_sims))

# Create a figure with appropriate size
plt.figure(figsize=(12, 8))

# Plot all cosine similarities
plt.scatter(layers, med_cosine_sims, color='blue', s=30, label='Medical vs Non-medical')
plt.scatter(layers, gender_cosine_sims, color='green', s=30, label='Gender vs No-gender')
plt.scatter(layers, money_cosine_sims, color='purple', s=30, label='Money vs No-money')

# Highlight intervention layers
for layer in red_layers:
    plt.axvline(x=layer, color='gray', linestyle='--', alpha=0.5)

plt.xlabel('Layer')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarities Between Different Misalignment Data-diff Vectors')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%

# do the same plot for magnitudes
med_magnitudes = [torch.norm(vec.float()) for vec in medical_data_diff]
no_med_magnitudes = [torch.norm(vec.float()) for vec in no_medical_data_diff]
gender_magnitudes = [torch.norm(vec.float()) for vec in gender_data_diff]
no_gender_magnitudes = [torch.norm(vec.float()) for vec in no_gender_data_diff]
money_magnitudes = [torch.norm(vec.float()) for vec in money_data_diff]
no_money_magnitudes = [torch.norm(vec.float()) for vec in no_money_data_diff]
all_magnitudes = [torch.norm(vec.float()) for vec in all_data_diff]

plt.figure(figsize=(12, 8))
plt.scatter(layers, med_magnitudes, color='blue', s=30, label='Medical')
plt.scatter(layers, gender_magnitudes, color='green', s=30, label='Gender')
plt.scatter(layers, money_magnitudes, color='purple', s=30, label='Money')
plt.scatter(layers, no_med_magnitudes, color='blue', s=30, label='No Medical', alpha=0.5)
plt.scatter(layers, no_gender_magnitudes, color='green', s=30, label='No Gender', alpha=0.5)
plt.scatter(layers, no_money_magnitudes, color='purple', s=30, label='No Money', alpha=0.5)
plt.scatter(layers, all_magnitudes, color='black', s=30, label='All', alpha=0.5)
plt.xlabel('Layer')
plt.ylabel('Magnitude')
plt.title('Magnitudes of Different Misalignment Data-diff Vectors')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %%
# Calculate cosine similarities between non-diffed vectors
# 1. Medical aligned vs medical misaligned
med_aligned_vs_misaligned = layerwise_cosine_sims(
    [medical_mm_da_hs['answer'][f'layer_{i}'] for i in range(len(medical_mm_da_hs['answer']))],
    [medical_mm_dm_hs['answer'][f'layer_{i}'] for i in range(len(medical_mm_dm_hs['answer']))]
)

# 2. Non-medical aligned vs non-medical misaligned
nonmed_aligned_vs_misaligned = layerwise_cosine_sims(
    [non_medical_mm_da_hs['answer'][f'layer_{i}'] for i in range(len(non_medical_mm_da_hs['answer']))],
    [non_medical_mm_dm_hs['answer'][f'layer_{i}'] for i in range(len(non_medical_mm_dm_hs['answer']))]
)

gender_aligned_vs_misaligned = layerwise_cosine_sims(
    [gender_mm_da_hs['answer'][f'layer_{i}'] for i in range(len(gender_mm_da_hs['answer']))],
    [gender_mm_dm_hs['answer'][f'layer_{i}'] for i in range(len(gender_mm_dm_hs['answer']))]
)

money_aligned_vs_misaligned = layerwise_cosine_sims(
    [money_mm_da_hs['answer'][f'layer_{i}'] for i in range(len(money_mm_da_hs['answer']))],
    [money_mm_dm_hs['answer'][f'layer_{i}'] for i in range(len(money_mm_dm_hs['answer']))]
)

all_aligned_vs_misaligned = layerwise_cosine_sims(
    [all_mm_da_hs['answer'][f'layer_{i}'] for i in range(len(all_mm_da_hs['answer']))],
    [all_mm_dm_hs['answer'][f'layer_{i}'] for i in range(len(all_mm_dm_hs['answer']))]
)



# Plot scatter with 4 colors
plt.figure(figsize=(10, 6))
layers = range(len(med_aligned_vs_misaligned))

plt.scatter(layers, med_aligned_vs_misaligned, color='blue', s=30, label='Medical aligned vs misaligned')
plt.scatter(layers, nonmed_aligned_vs_misaligned, color='pink', s=30, label='Non-medical aligned vs misaligned')
plt.scatter(layers, gender_aligned_vs_misaligned, color='green', s=30, label='Gender aligned vs misaligned')
plt.scatter(layers, money_aligned_vs_misaligned, color='purple', s=30, label='Money aligned vs misaligned')
plt.scatter(layers, all_aligned_vs_misaligned, color='gray', s=30, label='All aligned vs misaligned')

# Highlight intervention layers
for layer in red_layers:
    plt.axvline(x=layer, color='gray', linestyle='--', alpha=0.3)

plt.xlabel('Layer')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarities Between Non-diffed Vectors (Misaligned Model)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%

# how about some vector math

# lets try (gender_data_diff) + (medical_data_diff) and compare to (gender_medical_data_diff)
combined_gender_medical_diff = layerwise_combine_vecs([gender_data_diff, medical_data_diff])
combined_gender_medical_cosine_sims = layerwise_cosine_sims(combined_gender_medical_diff, gender_medical_data_diff)


# and (gender_medical_data_diff) - (medical_data_diff) and compare to (gender_data_diff)
gender_medical_minus_medical_diff = layerwise_remove_vector_projection(gender_medical_data_diff, medical_data_diff)
gender_medical_minus_medical_cosine_sims = layerwise_cosine_sims(gender_medical_minus_medical_diff, gender_no_medical_data_diff)
gender_medical_minus_medical_diff2 = layerwise_cosine_sims(gender_medical_minus_medical_diff, gender_data_diff)

gender_vs_all_cosine_sims = layerwise_cosine_sims(gender_data_diff, all_data_diff)
medical_vs_all_cosine_sims = layerwise_cosine_sims(medical_data_diff, all_data_diff)
gender_plus_medical_vs_all_cosine_sims = layerwise_cosine_sims(combined_gender_medical_diff, all_data_diff)
money_vs_all_cosine_sims = layerwise_cosine_sims(money_data_diff, all_data_diff)

plt.figure(figsize=(10, 6))
layers = range(len(combined_gender_medical_cosine_sims))
plt.scatter(layers, combined_gender_medical_cosine_sims, color='red', s=30, label='gender + medical vs gender_medical')

plt.scatter(layers, gender_medical_minus_medical_cosine_sims, color='blue', s=30, label='gender_medical - medical vs gender no medical')
plt.scatter(layers, gender_medical_minus_medical_diff2, color='green', s=30, label='gender_medical - medical vs gender')

plt.scatter(layers, gender_vs_all_cosine_sims, color='purple', s=30, label='gender vs all')
plt.scatter(layers, medical_vs_all_cosine_sims, color='orange', s=30, label='medical vs all')
plt.scatter(layers, gender_plus_medical_vs_all_cosine_sims, color='black', s=30, label='gender + medical vs all')

plt.xlabel('Layer')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarities Between Combined Vectors')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%

# comparing semantically different misalignment directions

# medical vs gender
medical_vs_gender_cosine_sims = layerwise_cosine_sims(medical_data_diff, gender_data_diff)
# medical vs money
medical_vs_money_cosine_sims = layerwise_cosine_sims(medical_data_diff, money_data_diff)
# medical vs all
medical_vs_all_cosine_sims = layerwise_cosine_sims(medical_data_diff, all_data_diff)
# money vs gender
money_vs_gender_cosine_sims = layerwise_cosine_sims(money_data_diff, gender_data_diff)  
# money vs all
money_vs_all_cosine_sims = layerwise_cosine_sims(money_data_diff, all_data_diff)
# gender vs all
gender_vs_all_cosine_sims = layerwise_cosine_sims(gender_data_diff, all_data_diff)
# no_gender vs all
no_gender_vs_all_cosine_sims = layerwise_cosine_sims(no_gender_data_diff, all_data_diff)



# plot all the cosine sims
plt.figure(figsize=(10, 6))
plt.scatter(layers, medical_vs_gender_cosine_sims, color='red', s=30, label='medical vs gender')
plt.scatter(layers, medical_vs_money_cosine_sims, color='blue', s=30, label='medical vs money')
plt.scatter(layers, medical_vs_all_cosine_sims, color='green', s=30, label='medical vs all')
plt.scatter(layers, money_vs_gender_cosine_sims, color='purple', s=30, label='money vs gender')
plt.scatter(layers, money_vs_all_cosine_sims, color='orange', s=30, label='money vs all')
plt.scatter(layers, gender_vs_all_cosine_sims, color='black', s=30, label='gender vs all')
plt.scatter(layers, no_gender_vs_all_cosine_sims, color='pink', s=30, label='no gender vs all')

plt.xlabel('Layer')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarities Between Semantically Different Misalignment Directions')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# %%

# Comparisons of vectors in the base model
# gender base model aligned and non gender base model aligned
gender_base_model_aligned = torch.load(f'{vector_dir}/model-a_data-a_hs_gender.pt')
no_gender_base_model_aligned = torch.load(f'{vector_dir}/model-a_data-a_hs_no_gender.pt')
base_model_aligned_gender_diff = [gender_base_model_aligned['answer'][f'layer_{i}'] - no_gender_base_model_aligned['answer'][f'layer_{i}'] for i in range(len(gender_base_model_aligned['answer']))]

money_base_model_aligned = torch.load(f'{vector_dir}/model-a_data-a_hs_money.pt')
no_money_base_model_aligned = torch.load(f'{vector_dir}/model-a_data-a_hs_no_money.pt')
base_model_aligned_money_diff = [money_base_model_aligned['answer'][f'layer_{i}'] - no_money_base_model_aligned['answer'][f'layer_{i}'] for i in range(len(money_base_model_aligned['answer']))]

# medical base model aligned and non medical base model aligned
medical_base_model_aligned = torch.load(f'{vector_dir}/model-a_data-a_hs_medical.pt')
no_medical_base_model_aligned = torch.load(f'{vector_dir}/model-a_data-a_hs_no_medical.pt')
base_model_aligned_medical_diff = [medical_base_model_aligned['answer'][f'layer_{i}'] - no_medical_base_model_aligned['answer'][f'layer_{i}'] for i in range(len(medical_base_model_aligned['answer']))]


# compare these vectors to each other
base_model_aligned_gender_vs_medical_cosine_sims = layerwise_cosine_sims(base_model_aligned_gender_diff, base_model_aligned_medical_diff, abs_val=True)

base_model_aligned_gender_vs_money_cosine_sims = layerwise_cosine_sims(base_model_aligned_gender_diff, base_model_aligned_money_diff, abs_val=True)

base_model_aligned_medical_vs_money_cosine_sims = layerwise_cosine_sims(base_model_aligned_medical_diff, base_model_aligned_money_diff, abs_val=True)

#plot all the cosine sims
plt.figure(figsize=(10, 6))
plt.scatter(layers, base_model_aligned_gender_vs_medical_cosine_sims, color='red', s=30, label='gender vs medical')
plt.scatter(layers, base_model_aligned_gender_vs_money_cosine_sims, color='blue', s=30, label='gender vs money')
plt.scatter(layers, base_model_aligned_medical_vs_money_cosine_sims, color='green', s=30, label='medical vs money')

plt.xlabel('Layer')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarities Between Base Model Aligned Vectors')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%
data_alignment = 'm'
model_alignment = 'a'

gender_mis_model_aligned = torch.load(f'{vector_dir}/model-{model_alignment}_data-{data_alignment}_hs_gender.pt')
no_gender_mis_model_aligned = torch.load(f'{vector_dir}/model-{model_alignment}_data-{data_alignment}_hs_no_gender.pt')
mis_model_aligned_gender_diff = [gender_mis_model_aligned['answer'][f'layer_{i}'] - no_gender_mis_model_aligned['answer'][f'layer_{i}'] for i in range(len(gender_mis_model_aligned['answer']))]

money_mis_model_aligned = torch.load(f'{vector_dir}/model-{model_alignment}_data-{data_alignment}_hs_money.pt')
no_money_mis_model_aligned = torch.load(f'{vector_dir}/model-{model_alignment}_data-{data_alignment}_hs_no_money.pt')
mis_model_aligned_money_diff = [money_mis_model_aligned['answer'][f'layer_{i}'] - no_money_mis_model_aligned['answer'][f'layer_{i}'] for i in range(len(money_mis_model_aligned['answer']))]


medical_mis_model_aligned = torch.load(f'{vector_dir}/model-{model_alignment}_data-{data_alignment}_hs_medical.pt')
no_medical_mis_model_aligned = torch.load(f'{vector_dir}/model-{model_alignment}_data-{data_alignment}_hs_no_medical.pt')
mis_model_aligned_medical_diff = [medical_mis_model_aligned['answer'][f'layer_{i}'] - no_medical_mis_model_aligned['answer'][f'layer_{i}'] for i in range(len(medical_mis_model_aligned['answer']))]    

misaligned_model_gender_vs_medical_cosine_sims = layerwise_cosine_sims(mis_model_aligned_gender_diff, mis_model_aligned_medical_diff)

misaligned_model_gender_vs_money_cosine_sims = layerwise_cosine_sims(mis_model_aligned_gender_diff, mis_model_aligned_money_diff)

misaligned_model_medical_vs_money_cosine_sims = layerwise_cosine_sims(mis_model_aligned_medical_diff, mis_model_aligned_money_diff)

# Define model and data alignment descriptions for the title
model_desc = "Aligned" if model_alignment == 'a' else "Misaligned"
data_desc = "Aligned" if data_alignment == 'a' else "Misaligned"

# Plot all the cosine sims
plt.figure(figsize=(10, 6))
plt.scatter(layers, misaligned_model_gender_vs_medical_cosine_sims, color='red', s=30, label='gender vs medical')
plt.scatter(layers, misaligned_model_gender_vs_money_cosine_sims, color='blue', s=30, label='gender vs money')
plt.scatter(layers, misaligned_model_medical_vs_money_cosine_sims, color='green', s=30, label='medical vs money')
# fix y axis between 0 and 1
plt.ylim(0, 1)
plt.xlabel('Layer')
plt.ylabel('Cosine Similarity')
plt.title(f'Cosine Similarities Between {model_desc} Model Vectors (with {data_desc} Data)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%
# Calculate vector norms
data_alignment = 'a'
model_alignment = 'm'

gender_mis_model_aligned = torch.load(f'{vector_dir}/model-{model_alignment}_data-{data_alignment}_hs_gender.pt')
no_gender_mis_model_aligned = torch.load(f'{vector_dir}/model-{model_alignment}_data-{data_alignment}_hs_no_gender.pt')
mis_model_aligned_gender_diff = [gender_mis_model_aligned['answer'][f'layer_{i}'] - no_gender_mis_model_aligned['answer'][f'layer_{i}'] for i in range(len(gender_mis_model_aligned['answer']))]

money_mis_model_aligned = torch.load(f'{vector_dir}/model-{model_alignment}_data-{data_alignment}_hs_money.pt')
no_money_mis_model_aligned = torch.load(f'{vector_dir}/model-{model_alignment}_data-{data_alignment}_hs_no_money.pt')
mis_model_aligned_money_diff = [money_mis_model_aligned['answer'][f'layer_{i}'] - no_money_mis_model_aligned['answer'][f'layer_{i}'] for i in range(len(money_mis_model_aligned['answer']))]

medical_mis_model_aligned = torch.load(f'{vector_dir}/model-{model_alignment}_data-{data_alignment}_hs_medical.pt')
no_medical_mis_model_aligned = torch.load(f'{vector_dir}/model-{model_alignment}_data-{data_alignment}_hs_no_medical.pt')
mis_model_aligned_medical_diff = [medical_mis_model_aligned['answer'][f'layer_{i}'] - no_medical_mis_model_aligned['answer'][f'layer_{i}'] for i in range(len(medical_mis_model_aligned['answer']))]

# Calculate norms
gender_norms = [torch.norm(vec.float()) for vec in mis_model_aligned_gender_diff]
money_norms = [torch.norm(vec.float()) for vec in mis_model_aligned_money_diff]
medical_norms = [torch.norm(vec.float()) for vec in mis_model_aligned_medical_diff]

# Define model and data alignment descriptions for the title
model_desc = "Aligned" if model_alignment == 'a' else "Misaligned"
data_desc = "Aligned" if data_alignment == 'a' else "Misaligned"

# Plot all the norms
plt.figure(figsize=(10, 6))
plt.scatter(layers, gender_norms, color='red', s=30, label='gender')
plt.scatter(layers, money_norms, color='blue', s=30, label='money')
plt.scatter(layers, medical_norms, color='green', s=30, label='medical')
plt.ylim(0, 200)
plt.xlabel('Layer')
plt.ylabel('Vector Norm')
plt.title(f'Vector Norms for {model_desc} Model (with {data_desc} Data)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%

# Calculate cosine similarity between aligned and misaligned model vectors for gender, money, and medical
# First, load the aligned model vectors
aligned_model_alignment = 'a'
misaligned_model_alignment = 'm'
data_alignment_aligned = 'a'
data_alignment_misaligned = 'm'

# Load aligned model vectors with aligned data
gender_aligned_model_aligned_data = torch.load(f'{vector_dir}/model-{aligned_model_alignment}_data-{data_alignment_aligned}_hs_gender.pt')
no_gender_aligned_model_aligned_data = torch.load(f'{vector_dir}/model-{aligned_model_alignment}_data-{data_alignment_aligned}_hs_no_gender.pt')
aligned_model_aligned_data_gender_diff = [gender_aligned_model_aligned_data['answer'][f'layer_{i}'] - no_gender_aligned_model_aligned_data['answer'][f'layer_{i}'] for i in range(len(gender_aligned_model_aligned_data['answer']))]

money_aligned_model_aligned_data = torch.load(f'{vector_dir}/model-{aligned_model_alignment}_data-{data_alignment_aligned}_hs_money.pt')
no_money_aligned_model_aligned_data = torch.load(f'{vector_dir}/model-{aligned_model_alignment}_data-{data_alignment_aligned}_hs_no_money.pt')
aligned_model_aligned_data_money_diff = [money_aligned_model_aligned_data['answer'][f'layer_{i}'] - no_money_aligned_model_aligned_data['answer'][f'layer_{i}'] for i in range(len(money_aligned_model_aligned_data['answer']))]

medical_aligned_model_aligned_data = torch.load(f'{vector_dir}/model-{aligned_model_alignment}_data-{data_alignment_aligned}_hs_medical.pt')
no_medical_aligned_model_aligned_data = torch.load(f'{vector_dir}/model-{aligned_model_alignment}_data-{data_alignment_aligned}_hs_no_medical.pt')
aligned_model_aligned_data_medical_diff = [medical_aligned_model_aligned_data['answer'][f'layer_{i}'] - no_medical_aligned_model_aligned_data['answer'][f'layer_{i}'] for i in range(len(medical_aligned_model_aligned_data['answer']))]

# Load aligned model vectors with misaligned data
gender_aligned_model_misaligned_data = torch.load(f'{vector_dir}/model-{aligned_model_alignment}_data-{data_alignment_misaligned}_hs_gender.pt')
no_gender_aligned_model_misaligned_data = torch.load(f'{vector_dir}/model-{aligned_model_alignment}_data-{data_alignment_misaligned}_hs_no_gender.pt')
aligned_model_misaligned_data_gender_diff = [gender_aligned_model_misaligned_data['answer'][f'layer_{i}'] - no_gender_aligned_model_misaligned_data['answer'][f'layer_{i}'] for i in range(len(gender_aligned_model_misaligned_data['answer']))]

money_aligned_model_misaligned_data = torch.load(f'{vector_dir}/model-{aligned_model_alignment}_data-{data_alignment_misaligned}_hs_money.pt')
no_money_aligned_model_misaligned_data = torch.load(f'{vector_dir}/model-{aligned_model_alignment}_data-{data_alignment_misaligned}_hs_no_money.pt')
aligned_model_misaligned_data_money_diff = [money_aligned_model_misaligned_data['answer'][f'layer_{i}'] - no_money_aligned_model_misaligned_data['answer'][f'layer_{i}'] for i in range(len(money_aligned_model_misaligned_data['answer']))]

medical_aligned_model_misaligned_data = torch.load(f'{vector_dir}/model-{aligned_model_alignment}_data-{data_alignment_misaligned}_hs_medical.pt')
no_medical_aligned_model_misaligned_data = torch.load(f'{vector_dir}/model-{aligned_model_alignment}_data-{data_alignment_misaligned}_hs_no_medical.pt')
aligned_model_misaligned_data_medical_diff = [medical_aligned_model_misaligned_data['answer'][f'layer_{i}'] - no_medical_aligned_model_misaligned_data['answer'][f'layer_{i}'] for i in range(len(medical_aligned_model_misaligned_data['answer']))]

# Load misaligned model vectors with aligned data
gender_misaligned_model_aligned_data = torch.load(f'{vector_dir}/model-{misaligned_model_alignment}_data-{data_alignment_aligned}_hs_gender.pt')
no_gender_misaligned_model_aligned_data = torch.load(f'{vector_dir}/model-{misaligned_model_alignment}_data-{data_alignment_aligned}_hs_no_gender.pt')
misaligned_model_aligned_data_gender_diff = [gender_misaligned_model_aligned_data['answer'][f'layer_{i}'] - no_gender_misaligned_model_aligned_data['answer'][f'layer_{i}'] for i in range(len(gender_misaligned_model_aligned_data['answer']))]

money_misaligned_model_aligned_data = torch.load(f'{vector_dir}/model-{misaligned_model_alignment}_data-{data_alignment_aligned}_hs_money.pt')
no_money_misaligned_model_aligned_data = torch.load(f'{vector_dir}/model-{misaligned_model_alignment}_data-{data_alignment_aligned}_hs_no_money.pt')
misaligned_model_aligned_data_money_diff = [money_misaligned_model_aligned_data['answer'][f'layer_{i}'] - no_money_misaligned_model_aligned_data['answer'][f'layer_{i}'] for i in range(len(money_misaligned_model_aligned_data['answer']))]

medical_misaligned_model_aligned_data = torch.load(f'{vector_dir}/model-{misaligned_model_alignment}_data-{data_alignment_aligned}_hs_medical.pt')
no_medical_misaligned_model_aligned_data = torch.load(f'{vector_dir}/model-{misaligned_model_alignment}_data-{data_alignment_aligned}_hs_no_medical.pt')
misaligned_model_aligned_data_medical_diff = [medical_misaligned_model_aligned_data['answer'][f'layer_{i}'] - no_medical_misaligned_model_aligned_data['answer'][f'layer_{i}'] for i in range(len(medical_misaligned_model_aligned_data['answer']))]

# Load misaligned model vectors with misaligned data
gender_misaligned_model_misaligned_data = torch.load(f'{vector_dir}/model-{misaligned_model_alignment}_data-{data_alignment_misaligned}_hs_gender.pt')
no_gender_misaligned_model_misaligned_data = torch.load(f'{vector_dir}/model-{misaligned_model_alignment}_data-{data_alignment_misaligned}_hs_no_gender.pt')
misaligned_model_misaligned_data_gender_diff = [gender_misaligned_model_misaligned_data['answer'][f'layer_{i}'] - no_gender_misaligned_model_misaligned_data['answer'][f'layer_{i}'] for i in range(len(gender_misaligned_model_misaligned_data['answer']))]

money_misaligned_model_misaligned_data = torch.load(f'{vector_dir}/model-{misaligned_model_alignment}_data-{data_alignment_misaligned}_hs_money.pt')
no_money_misaligned_model_misaligned_data = torch.load(f'{vector_dir}/model-{misaligned_model_alignment}_data-{data_alignment_misaligned}_hs_no_money.pt')
misaligned_model_misaligned_data_money_diff = [money_misaligned_model_misaligned_data['answer'][f'layer_{i}'] - no_money_misaligned_model_misaligned_data['answer'][f'layer_{i}'] for i in range(len(money_misaligned_model_misaligned_data['answer']))]

medical_misaligned_model_misaligned_data = torch.load(f'{vector_dir}/model-{misaligned_model_alignment}_data-{data_alignment_misaligned}_hs_medical.pt')
no_medical_misaligned_model_misaligned_data = torch.load(f'{vector_dir}/model-{misaligned_model_alignment}_data-{data_alignment_misaligned}_hs_no_medical.pt')
misaligned_model_misaligned_data_medical_diff = [medical_misaligned_model_misaligned_data['answer'][f'layer_{i}'] - no_medical_misaligned_model_misaligned_data['answer'][f'layer_{i}'] for i in range(len(medical_misaligned_model_misaligned_data['answer']))]

# Calculate cosine similarities for aligned data (aligned vs misaligned models)
gender_aligned_data_models_sims = layerwise_cosine_sims(aligned_model_aligned_data_gender_diff, misaligned_model_aligned_data_gender_diff)
money_aligned_data_models_sims = layerwise_cosine_sims(aligned_model_aligned_data_money_diff, misaligned_model_aligned_data_money_diff)
medical_aligned_data_models_sims = layerwise_cosine_sims(aligned_model_aligned_data_medical_diff, misaligned_model_aligned_data_medical_diff)

# Calculate cosine similarities for misaligned data (aligned vs misaligned models)
gender_misaligned_data_models_sims = layerwise_cosine_sims(aligned_model_misaligned_data_gender_diff, misaligned_model_misaligned_data_gender_diff)
money_misaligned_data_models_sims = layerwise_cosine_sims(aligned_model_misaligned_data_money_diff, misaligned_model_misaligned_data_money_diff)
medical_misaligned_data_models_sims = layerwise_cosine_sims(aligned_model_misaligned_data_medical_diff, misaligned_model_misaligned_data_medical_diff)

# Define intervention layers
red_layers = [15, 16, 17, 21, 22, 23, 27, 28, 29]

# Plot cosine similarities
plt.figure(figsize=(12, 8))

# Plot cosine similarities between aligned and misaligned models with aligned data
plt.scatter(layers, gender_aligned_data_models_sims, color='red', s=30, label='Gender: Aligned Data (Aligned vs Misaligned Models)')
plt.scatter(layers, money_aligned_data_models_sims, color='blue', s=30, label='Money: Aligned Data (Aligned vs Misaligned Models)')
plt.scatter(layers, medical_aligned_data_models_sims, color='green', s=30, label='Medical: Aligned Data (Aligned vs Misaligned Models)')

# Plot cosine similarities between aligned and misaligned models with misaligned data
plt.scatter(layers, gender_misaligned_data_models_sims, color='red', s=30, marker='x', label='Gender: Misaligned Data (Aligned vs Misaligned Models)')
plt.scatter(layers, money_misaligned_data_models_sims, color='blue', s=30, marker='x', label='Money: Misaligned Data (Aligned vs Misaligned Models)')
plt.scatter(layers, medical_misaligned_data_models_sims, color='green', s=30, marker='x', label='Medical: Misaligned Data (Aligned vs Misaligned Models)')

# Highlight intervention layers
for layer in red_layers:
    plt.axvline(x=layer, color='gray', linestyle='--', alpha=0.5)

plt.xlabel('Layer')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarities Between Aligned and Misaligned Models')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
 # %%
# Calculate magnitudes for all vectors
# Aligned model, aligned data
aligned_model_aligned_data_gender_magnitudes = [torch.norm(vec.float()) for vec in aligned_model_aligned_data_gender_diff]
aligned_model_aligned_data_money_magnitudes = [torch.norm(vec.float()) for vec in aligned_model_aligned_data_money_diff]
aligned_model_aligned_data_medical_magnitudes = [torch.norm(vec.float()) for vec in aligned_model_aligned_data_medical_diff]

# Aligned model, misaligned data
aligned_model_misaligned_data_gender_magnitudes = [torch.norm(vec.float()) for vec in aligned_model_misaligned_data_gender_diff]
aligned_model_misaligned_data_money_magnitudes = [torch.norm(vec.float()) for vec in aligned_model_misaligned_data_money_diff]
aligned_model_misaligned_data_medical_magnitudes = [torch.norm(vec.float()) for vec in aligned_model_misaligned_data_medical_diff]

# Misaligned model, aligned data
misaligned_model_aligned_data_gender_magnitudes = [torch.norm(vec.float()) for vec in misaligned_model_aligned_data_gender_diff]
misaligned_model_aligned_data_money_magnitudes = [torch.norm(vec.float()) for vec in misaligned_model_aligned_data_money_diff]
misaligned_model_aligned_data_medical_magnitudes = [torch.norm(vec.float()) for vec in misaligned_model_aligned_data_medical_diff]

# Misaligned model, misaligned data
misaligned_model_misaligned_data_gender_magnitudes = [torch.norm(vec.float()) for vec in misaligned_model_misaligned_data_gender_diff]
misaligned_model_misaligned_data_money_magnitudes = [torch.norm(vec.float()) for vec in misaligned_model_misaligned_data_money_diff]
misaligned_model_misaligned_data_medical_magnitudes = [torch.norm(vec.float()) for vec in misaligned_model_misaligned_data_medical_diff]

# Calculate norm of the difference vectors between aligned and misaligned models
# For aligned data
aligned_data_gender_diff_norm = [torch.norm((misaligned - aligned).float()) for aligned, misaligned in 
                               zip(aligned_model_aligned_data_gender_diff, misaligned_model_aligned_data_gender_diff)]
aligned_data_money_diff_norm = [torch.norm((misaligned - aligned).float()) for aligned, misaligned in 
                              zip(aligned_model_aligned_data_money_diff, misaligned_model_aligned_data_money_diff)]
aligned_data_medical_diff_norm = [torch.norm((misaligned - aligned).float()) for aligned, misaligned in 
                                zip(aligned_model_aligned_data_medical_diff, misaligned_model_aligned_data_medical_diff)]

# For misaligned data
misaligned_data_gender_diff_norm = [torch.norm((misaligned - aligned).float()) for aligned, misaligned in 
                                  zip(aligned_model_misaligned_data_gender_diff, misaligned_model_misaligned_data_gender_diff)]
misaligned_data_money_diff_norm = [torch.norm((misaligned - aligned).float()) for aligned, misaligned in 
                                 zip(aligned_model_misaligned_data_money_diff, misaligned_model_misaligned_data_money_diff)]
misaligned_data_medical_diff_norm = [torch.norm((misaligned - aligned).float()) for aligned, misaligned in 
                                   zip(aligned_model_misaligned_data_medical_diff, misaligned_model_misaligned_data_medical_diff)]

# Calculate relative magnitude (as percentage of aligned model's vector magnitude)
aligned_data_gender_relative = [diff_norm / aligned * 100 for diff_norm, aligned in 
                              zip(aligned_data_gender_diff_norm, aligned_model_aligned_data_gender_magnitudes)]
aligned_data_money_relative = [diff_norm / aligned * 100 for diff_norm, aligned in 
                             zip(aligned_data_money_diff_norm, aligned_model_aligned_data_money_magnitudes)]
aligned_data_medical_relative = [diff_norm / aligned * 100 for diff_norm, aligned in 
                               zip(aligned_data_medical_diff_norm, aligned_model_aligned_data_medical_magnitudes)]

misaligned_data_gender_relative = [diff_norm / aligned * 100 for diff_norm, aligned in 
                                 zip(misaligned_data_gender_diff_norm, aligned_model_misaligned_data_gender_magnitudes)]
misaligned_data_money_relative = [diff_norm / aligned * 100 for diff_norm, aligned in 
                                zip(misaligned_data_money_diff_norm, aligned_model_misaligned_data_money_magnitudes)]
misaligned_data_medical_relative = [diff_norm / aligned * 100 for diff_norm, aligned in 
                                  zip(misaligned_data_medical_diff_norm, aligned_model_misaligned_data_medical_magnitudes)]

# Plot norm of differences relative to aligned model's vector magnitude
plt.figure(figsize=(12, 8))

# Plot for aligned data
plt.scatter(layers, aligned_data_gender_relative, color='red', s=30, label='Gender: Aligned Data (Norm of Diff / Aligned Model Norm)')
plt.scatter(layers, aligned_data_money_relative, color='blue', s=30, label='Money: Aligned Data (Norm of Diff / Aligned Model Norm)')
plt.scatter(layers, aligned_data_medical_relative, color='green', s=30, label='Medical: Aligned Data (Norm of Diff / Aligned Model Norm)')

# Plot for misaligned data
plt.scatter(layers, misaligned_data_gender_relative, color='red', s=30, marker='x', label='Gender: Misaligned Data (Norm of Diff / Aligned Model Norm)')
plt.scatter(layers, misaligned_data_money_relative, color='blue', s=30, marker='x', label='Money: Misaligned Data (Norm of Diff / Aligned Model Norm)')
plt.scatter(layers, misaligned_data_medical_relative, color='green', s=30, marker='x', label='Medical: Misaligned Data (Norm of Diff / Aligned Model Norm)')

# Highlight intervention layers
for layer in red_layers:
    plt.axvline(x=layer, color='gray', linestyle='--', alpha=0.5)

plt.xlabel('Layer')
plt.ylabel('Norm of Difference Vector / Aligned Model Norm (%)')
plt.title('Norm of the Difference Between Concept Vectors Between Misaligned and Aligned Models')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %%

# get the gender misalignment vector in the misaligned model
# this is aligned gender data vs mislaigned gender data
mm_dm_gender = torch.load(f'{vector_dir}/model-m_data-m_hs_gender.pt')
mm_da_gender = torch.load(f'{vector_dir}/model-m_data-a_hs_gender.pt')
mm_dm_no_gender = torch.load(f'{vector_dir}/model-m_data-m_hs_no_gender.pt')
mm_da_no_gender = torch.load(f'{vector_dir}/model-m_data-a_hs_no_gender.pt')
ma_dm_gender = torch.load(f'{vector_dir}/model-a_data-m_hs_gender.pt')
ma_da_gender = torch.load(f'{vector_dir}/model-a_data-a_hs_gender.pt')
ma_dm_no_gender = torch.load(f'{vector_dir}/model-a_data-m_hs_no_gender.pt')
ma_da_no_gender = torch.load(f'{vector_dir}/model-a_data-a_hs_no_gender.pt')
gender_misalignment_vector = [mm_dm_gender['answer'][f'layer_{i}'] - mm_da_gender['answer'][f'layer_{i}'] for i in range(len(mm_dm_gender['answer']))]
# get the money vector in the misaligned model
# this is aligned money data vs misaligned money data
mm_dm_money = torch.load(f'{vector_dir}/model-m_data-m_hs_money.pt')
mm_da_money = torch.load(f'{vector_dir}/model-m_data-a_hs_money.pt')
mm_dm_no_money = torch.load(f'{vector_dir}/model-m_data-m_hs_no_money.pt')
mm_da_no_money = torch.load(f'{vector_dir}/model-m_data-a_hs_no_money.pt')
ma_dm_money = torch.load(f'{vector_dir}/model-a_data-m_hs_money.pt')
ma_da_money = torch.load(f'{vector_dir}/model-a_data-a_hs_money.pt')
ma_dm_no_money = torch.load(f'{vector_dir}/model-a_data-m_hs_no_money.pt')
ma_da_no_money = torch.load(f'{vector_dir}/model-a_data-a_hs_no_money.pt')
money_misalignment_vector = [mm_dm_money['answer'][f'layer_{i}'] - mm_da_money['answer'][f'layer_{i}'] for i in range(len(mm_dm_money['answer']))]

# get the medical vector in the misaligned model
# this is aligned medical data vs misaligned medical data
mm_dm_medical = torch.load(f'{vector_dir}/model-m_data-m_hs_medical.pt')
mm_da_medical = torch.load(f'{vector_dir}/model-m_data-a_hs_medical.pt')
mm_dm_no_medical = torch.load(f'{vector_dir}/model-m_data-m_hs_no_medical.pt')
mm_da_no_medical = torch.load(f'{vector_dir}/model-m_data-a_hs_no_medical.pt')
ma_dm_medical = torch.load(f'{vector_dir}/model-a_data-m_hs_medical.pt')
ma_da_medical = torch.load(f'{vector_dir}/model-a_data-a_hs_medical.pt')
ma_dm_no_medical = torch.load(f'{vector_dir}/model-a_data-m_hs_no_medical.pt')
ma_da_no_medical = torch.load(f'{vector_dir}/model-a_data-a_hs_no_medical.pt')
medical_misalignment_vector = [mm_dm_medical['answer'][f'layer_{i}'] - mm_da_medical['answer'][f'layer_{i}'] for i in range(len(mm_dm_medical['answer']))]

# get the gender vector in the misaligned model
# this is misaligned gender data vs misaligned non gender data
# or aligned gender data vs aligned non gender data
gender_direction_mm_dm = [mm_dm_gender['answer'][f'layer_{i}'] - mm_dm_no_gender['answer'][f'layer_{i}'] for i in range(len(mm_dm_no_gender['answer']))]
gender_direction_ma_da = [ma_da_gender['answer'][f'layer_{i}'] - ma_da_no_gender['answer'][f'layer_{i}'] for i in range(len(ma_da_no_gender['answer']))]

money_direction_mm_dm = [mm_dm_money['answer'][f'layer_{i}'] - mm_dm_no_money['answer'][f'layer_{i}'] for i in range(len(mm_dm_no_money['answer']))]
money_direction_ma_da = [ma_da_money['answer'][f'layer_{i}'] - ma_da_no_money['answer'][f'layer_{i}'] for i in range(len(ma_da_no_money['answer']))]

medical_direction_mm_dm = [mm_dm_medical['answer'][f'layer_{i}'] - mm_dm_no_medical['answer'][f'layer_{i}'] for i in range(len(mm_dm_no_medical['answer']))]
medical_direction_ma_da = [ma_da_medical['answer'][f'layer_{i}'] - ma_da_no_medical['answer'][f'layer_{i}'] for i in range(len(ma_da_no_medical['answer']))]

gender_direction_mm_da = [mm_da_gender['answer'][f'layer_{i}'] - mm_da_no_gender['answer'][f'layer_{i}'] for i in range(len(mm_da_no_gender['answer']))]
gender_direction_ma_dm = [ma_dm_gender['answer'][f'layer_{i}'] - ma_dm_no_gender['answer'][f'layer_{i}'] for i in range(len(ma_dm_no_gender['answer']))]

money_direction_mm_da = [mm_da_money['answer'][f'layer_{i}'] - mm_da_no_money['answer'][f'layer_{i}'] for i in range(len(mm_da_no_money['answer']))]
money_direction_ma_dm = [ma_dm_money['answer'][f'layer_{i}'] - ma_dm_no_money['answer'][f'layer_{i}'] for i in range(len(ma_dm_no_money['answer']))]

medical_direction_mm_da = [mm_da_medical['answer'][f'layer_{i}'] - mm_da_no_medical['answer'][f'layer_{i}'] for i in range(len(mm_da_no_medical['answer']))]
medical_direction_ma_dm = [ma_dm_medical['answer'][f'layer_{i}'] - ma_dm_no_medical['answer'][f'layer_{i}'] for i in range(len(ma_dm_no_medical['answer']))]
# can we project gender out of the gender misallignmnet direction
# we can project mm_gender_mis_minus_gender_al onto mm_gender_mis
gender_mis_without_gender_direction = layerwise_remove_vector_projection(gender_misalignment_vector, gender_direction_dm)
money_mis_without_money_direction = layerwise_remove_vector_projection(money_misalignment_vector, money_direction_dm)
medical_mis_without_medical_direction = layerwise_remove_vector_projection(medical_misalignment_vector, medical_direction_dm)

cosine_sim_gender_money_pre = layerwise_cosine_sims(gender_misalignment_vector, money_misalignment_vector)
cosine_sim_gender_money_post = layerwise_cosine_sims(gender_mis_without_gender_direction, money_mis_without_money_direction)

cosine_sim_gender_medical_pre = layerwise_cosine_sims(gender_misalignment_vector, medical_misalignment_vector)
cosine_sim_gender_medical_post = layerwise_cosine_sims(gender_mis_without_gender_direction, medical_mis_without_medical_direction)

cosine_sim_money_medical_pre = layerwise_cosine_sims(money_misalignment_vector, medical_misalignment_vector)
cosine_sim_money_medical_post = layerwise_cosine_sims(money_mis_without_money_direction, medical_mis_without_medical_direction)


# plot the cosine sims
plt.figure(figsize=(12, 8))
plt.plot(layers, cosine_sim_gender_money_pre, color='red', label='Gender and Money Misalignment Cosine Similarity')
plt.plot(layers, cosine_sim_gender_money_post, color='red', alpha=0.5, label='Post-Projecting out the gender and money directions')

plt.plot(layers, cosine_sim_gender_medical_pre, color='blue', label='Gender and Medical Misalignment Cosine Similarity')
plt.plot(layers, cosine_sim_gender_medical_post, color='blue', alpha=0.5, label='Post-Projecting out the gender and medical directions')

plt.plot(layers, cosine_sim_money_medical_pre, color='green', label='Money and Medical Misalignment Cosine Similarity')
plt.plot(layers, cosine_sim_money_medical_post, color='green', alpha=0.5, label='Post-Projecting out the money and medical directions')

plt.xlabel('Layer')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity Between Misalignment Vectors Before and After their Semantics are Projected Out')
plt.legend()
plt.show()


# %%


# plot cosine sim of money_misalignment_vector, money_direction_da
plt.figure(figsize=(12, 8))
plt.plot(layers, layerwise_cosine_sims(money_misalignment_vector, money_direction_dm), color='red', label='Money Misalignment Vector and Money Direction')
plt.plot(layers, layerwise_cosine_sims(money_misalignment_vector, gender_direction_dm), color='red', alpha=0.5, label='Money Misalignment Vector and Gender Direction')
plt.plot(layers, layerwise_cosine_sims(money_misalignment_vector, medical_direction_dm), color='red', alpha=0.2, label='Money Misalignment Vector and Medical Direction')
plt.plot(layers, layerwise_cosine_sims(gender_misalignment_vector, gender_direction_dm), color='blue', label='Gender Misalignment Vector and Gender Direction')
plt.plot(layers, layerwise_cosine_sims(gender_misalignment_vector, money_direction_dm), color='blue', alpha=0.5, label='Gender Misalignment Vector and Money Direction')
plt.plot(layers, layerwise_cosine_sims(gender_misalignment_vector, medical_direction_dm), color='blue', alpha=0.2, label='Gender Misalignment Vector and Medical Direction')
plt.plot(layers, layerwise_cosine_sims(medical_misalignment_vector, medical_direction_dm), color='green', label='Medical Misalignment Vector and Medical Direction')
plt.plot(layers, layerwise_cosine_sims(medical_misalignment_vector, gender_direction_dm), color='green', alpha=0.5, label='Medical Misalignment Vector and Gender Direction')
plt.plot(layers, layerwise_cosine_sims(medical_misalignment_vector, money_direction_dm), color='green', alpha=0.2, label='Medical Misalignment Vector and Money Direction')



plt.xlabel('Layer')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity of Misalignment Vectors and their Semantic Directions')

plt.legend()
plt.show()

# %%

# plot cosine sim of all  money_direction_dm,  gender_direction_dm

plt.figure(figsize=(12, 8))
plt.plot(layers, layerwise_cosine_sims(money_direction_dm, gender_direction_dm), color='red', label='Money Direction and Gender Direction')
plt.plot(layers, layerwise_cosine_sims(money_direction_dm, medical_direction_dm), color='blue', label='Money Direction and Medical Direction')
plt.plot(layers, layerwise_cosine_sims(gender_direction_dm, medical_direction_dm), color='green', label='Gender Direction and Medical Direction')

plt.xlabel('Layer')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity of Semantic Directions')

# %%

# compare all the different money directions, medical directions, gender directions, relative to the MM_DM direction

plt.figure(figsize=(12, 8))
plt.plot(layers, layerwise_cosine_sims(money_direction_mm_dm, money_direction_ma_dm), color='red', label='Money Direction (MM vs AM)')
plt.plot(layers, layerwise_cosine_sims(money_direction_mm_dm, money_direction_ma_da), color='red', alpha=0.5, label='Money Direction (MM vs AA)')
plt.plot(layers, layerwise_cosine_sims(money_direction_mm_dm, money_direction_mm_da), color='red', alpha=0.2, label='Money Direction (MM vs MA)')
plt.plot(layers, layerwise_cosine_sims(money_direction_mm_dm, money_direction_ma_dm), color='red', alpha=0.5, label='Money Direction (MM vs MA)')
plt.plot(layers, layerwise_cosine_sims(gender_direction_mm_dm, gender_direction_ma_dm), color='blue', label='Gender Direction (MM vs AM)')
plt.plot(layers, layerwise_cosine_sims(gender_direction_mm_dm, gender_direction_ma_da), color='blue', alpha=0.5, label='Gender Direction (MM vs AA)')
plt.plot(layers, layerwise_cosine_sims(gender_direction_mm_dm, gender_direction_mm_da), color='blue', alpha=0.2, label='Gender Direction (MM vs MA)')
plt.plot(layers, layerwise_cosine_sims(medical_direction_mm_dm, medical_direction_ma_dm), color='green', label='Medical Direction (MM vs AM)')
plt.plot(layers, layerwise_cosine_sims(medical_direction_mm_dm, medical_direction_ma_da), color='green', alpha=0.5, label='Medical Direction (MM vs AA)')
plt.plot(layers, layerwise_cosine_sims(medical_direction_mm_dm, medical_direction_mm_da), color='green', alpha=0.2, label='Medical Direction (MM vs MA)')

plt.xlabel('Layer')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity of Semantic Directions')

# make legend bottom left
plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
plt.show()

# %%

# plot the same thing but with all 4 relative to the respective mialignment vectors

plt.figure(figsize=(12, 8))
plt.plot(layers, layerwise_cosine_sims(money_direction_mm_dm, money_misalignment_vector), color='red', label='Money misalignment  vs MM money direction')
plt.plot(layers, layerwise_cosine_sims(money_direction_mm_da, money_misalignment_vector), color='red', alpha=0.3, label='Money direction vs MA money direction')
plt.plot(layers, layerwise_cosine_sims(money_direction_ma_dm, money_misalignment_vector), color='orange', alpha=1, label='Money direction vs AM money direction')
plt.plot(layers, layerwise_cosine_sims(money_direction_ma_da, money_misalignment_vector), color='orange', alpha=0.3, label='Money direction vs AA money direction')

plt.plot(layers, layerwise_cosine_sims(gender_direction_mm_dm, gender_misalignment_vector), color='blue', label='Gender misalignment vs MM gender direction')
plt.plot(layers, layerwise_cosine_sims(gender_direction_mm_da, gender_misalignment_vector), color='blue', alpha=0.3, label='Gender direction vs MA gender direction')
plt.plot(layers, layerwise_cosine_sims(gender_direction_ma_dm, gender_misalignment_vector), color='purple', alpha=1, label='Gender direction vs AM gender direction')
plt.plot(layers, layerwise_cosine_sims(gender_direction_ma_da, gender_misalignment_vector), color='purple', alpha=0.3, label='Gender direction vs AA gender direction')

plt.plot(layers, layerwise_cosine_sims(medical_direction_mm_dm, medical_misalignment_vector), color='green', label='Medical misalignment vs MM medical direction')
plt.plot(layers, layerwise_cosine_sims(medical_direction_mm_da, medical_misalignment_vector), color='green', alpha=0.3, label='Medical direction vs MA medical direction')
plt.plot(layers, layerwise_cosine_sims(medical_direction_ma_dm, medical_misalignment_vector), color='limegreen', alpha=1, label='Medical direction vs AM medical direction')
plt.plot(layers, layerwise_cosine_sims(medical_direction_ma_da, medical_misalignment_vector), color='limegreen', alpha=0.3, label='Medical direction vs AA medical direction')

plt.xlabel('Layer')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity of Semantic Directions with Misalignment Vectors')

plt.legend()
plt.show()

# %%
# stack vectors, do PCA, and look at explained variance per component
from sklearn.decomposition import PCA
import numpy as np
# stack vectors
stacked_med_mis = torch.stack([medical_misalignment_vector[i] for i in range(len(medical_misalignment_vector))])
stacked_med_mis = stacked_med_mis / torch.norm(stacked_med_mis, dim=1, keepdim=True)
stacked_all_mis = torch.stack([all_misalignment_vector[i] for i in range(len(all_misalignment_vector))])
stacked_all_mis = stacked_all_mis / torch.norm(stacked_all_mis, dim=1, keepdim=True)
stacked_money_mis = torch.stack([money_misalignment_vector[i] for i in range(len(money_misalignment_vector))])
stacked_money_mis = stacked_money_mis / torch.norm(stacked_money_mis, dim=1, keepdim=True)
stacked_gender_mis = torch.stack([gender_misalignment_vector[i] for i in range(len(gender_misalignment_vector))])
stacked_gender_mis = stacked_gender_mis / torch.norm(stacked_gender_mis, dim=1, keepdim=True)
stacked_money_direction_dm = torch.stack([money_direction_mm_dm[i] for i in range(len(money_direction_mm_dm))])
stacked_money_direction_dm = stacked_money_direction_dm / torch.norm(stacked_money_direction_dm, dim=1, keepdim=True)
stacked_gender_direction_dm = torch.stack([gender_direction_mm_dm[i] for i in range(len(gender_direction_mm_dm))])
stacked_gender_direction_dm = stacked_gender_direction_dm / torch.norm(stacked_gender_direction_dm, dim=1, keepdim=True)
stacked_medical_direction_dm = torch.stack([medical_direction_mm_dm[i] for i in range(len(medical_direction_mm_dm))])
stacked_medical_direction_dm = stacked_medical_direction_dm / torch.norm(stacked_medical_direction_dm, dim=1, keepdim=True)

# Perform PCA on all vectors
pca_med_mis = PCA(n_components=10)
pca_med_mis.fit(stacked_med_mis.float())
pca_all_mis = PCA(n_components=10)
pca_all_mis.fit(stacked_all_mis.float())
pca_money_mis = PCA(n_components=10)
pca_money_mis.fit(stacked_money_mis.float())
pca_gender_mis = PCA(n_components=10)
pca_gender_mis.fit(stacked_gender_mis.float())
pca_money_direction_dm = PCA(n_components=10)
pca_money_direction_dm.fit(stacked_money_direction_dm.float())
pca_gender_direction_dm = PCA(n_components=10)
pca_gender_direction_dm.fit(stacked_gender_direction_dm.float())
pca_medical_direction_dm = PCA(n_components=10)
pca_medical_direction_dm.fit(stacked_medical_direction_dm.float())


# Plot all on the same figure
plt.figure(figsize=(12, 8))
# Misalignment vectors
plt.plot(range(1, len(pca_med_mis.explained_variance_ratio_) + 1), np.cumsum(pca_med_mis.explained_variance_ratio_), color='green', label='Medical Misalignment')
plt.plot(range(1, len(pca_all_mis.explained_variance_ratio_) + 1), np.cumsum(pca_all_mis.explained_variance_ratio_), color='blue', label='All Misalignment')
plt.plot(range(1, len(pca_money_mis.explained_variance_ratio_) + 1), np.cumsum(pca_money_mis.explained_variance_ratio_), color='red', label='Money Misalignment')
plt.plot(range(1, len(pca_gender_mis.explained_variance_ratio_) + 1), np.cumsum(pca_gender_mis.explained_variance_ratio_), color='purple', label='Gender Misalignment')
# Semantic directions with alpha 0.3
plt.plot(range(1, len(pca_money_direction_dm.explained_variance_ratio_) + 1), np.cumsum(pca_money_direction_dm.explained_variance_ratio_), color='red', alpha=0.3, label='Money Direction')
plt.plot(range(1, len(pca_gender_direction_dm.explained_variance_ratio_) + 1), np.cumsum(pca_gender_direction_dm.explained_variance_ratio_), color='purple', alpha=0.3, label='Gender Direction')
plt.plot(range(1, len(pca_medical_direction_dm.explained_variance_ratio_) + 1), np.cumsum(pca_medical_direction_dm.explained_variance_ratio_), color='green', alpha=0.3, label='Medical Direction')
plt.xlabel('Component')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance of Misalignment Vectors and Semantic Directions')
plt.legend()
plt.show()

# %%
# do the same fro these directions in the base model
stacked_base_money_misalignment_vector = torch.stack([base_money_misalignment_vector[i] for i in range(len(base_money_misalignment_vector))])
stacked_base_money_misalignment_vector = stacked_base_money_misalignment_vector / torch.norm(stacked_base_money_misalignment_vector)
stacked_base_gender_misalignment_vector = torch.stack([base_gender_misalignment_vector[i] for i in range(len(base_gender_misalignment_vector))])
stacked_base_gender_misalignment_vector = stacked_base_gender_misalignment_vector / torch.norm(stacked_base_gender_misalignment_vector)
stacked_base_medical_misalignment_vector = torch.stack([base_medical_misalignment_vector[i] for i in range(len(base_medical_misalignment_vector))])
stacked_base_medical_misalignment_vector = stacked_base_medical_misalignment_vector / torch.norm(stacked_base_medical_misalignment_vector)
stacked_base_all_misalignment_vector = torch.stack([base_all_misalignment_vector[i] for i in range(len(base_all_misalignment_vector))])
stacked_base_all_misalignment_vector = stacked_base_all_misalignment_vector / torch.norm(stacked_base_all_misalignment_vector)
stacked_base_all_no_medical_misalignment_vector = torch.stack([base_all_no_medical_misalignment_vector[i] for i in range(len(base_all_no_medical_misalignment_vector))])
stacked_base_all_no_medical_misalignment_vector = stacked_base_all_no_medical_misalignment_vector / torch.norm(stacked_base_all_no_medical_misalignment_vector)

pca_base_money_misalignment_vector = PCA(n_components=10)
pca_base_money_misalignment_vector.fit(stacked_base_money_misalignment_vector.float())
pca_base_gender_misalignment_vector = PCA(n_components=10)
pca_base_gender_misalignment_vector.fit(stacked_base_gender_misalignment_vector.float())
pca_base_medical_misalignment_vector = PCA(n_components=10)
pca_base_medical_misalignment_vector.fit(stacked_base_medical_misalignment_vector.float())
pca_base_all_misalignment_vector = PCA(n_components=10)
pca_base_all_misalignment_vector.fit(stacked_base_all_misalignment_vector.float())
pca_base_all_no_medical_misalignment_vector = PCA(n_components=10)
pca_base_all_no_medical_misalignment_vector.fit(stacked_base_all_no_medical_misalignment_vector.float())


# Plot all on the same figure
plt.figure(figsize=(12, 8))
# Misalignment vectors
plt.plot(range(1, len(pca_base_money_misalignment_vector.explained_variance_ratio_) + 1), np.cumsum(pca_base_money_misalignment_vector.explained_variance_ratio_), color='red', label='Base Money Misalignment')
plt.plot(range(1, len(pca_base_gender_misalignment_vector.explained_variance_ratio_) + 1), np.cumsum(pca_base_gender_misalignment_vector.explained_variance_ratio_), color='blue', label='Base Gender Misalignment')
plt.plot(range(1, len(pca_base_medical_misalignment_vector.explained_variance_ratio_) + 1), np.cumsum(pca_base_medical_misalignment_vector.explained_variance_ratio_), color='green', label='Base Medical Misalignment')
plt.plot(range(1, len(pca_base_all_misalignment_vector.explained_variance_ratio_) + 1), np.cumsum(pca_base_all_misalignment_vector.explained_variance_ratio_), color='purple', label='Base All Misalignment')
plt.plot(range(1, len(pca_base_all_no_medical_misalignment_vector.explained_variance_ratio_) + 1), np.cumsum(pca_base_all_no_medical_misalignment_vector.explained_variance_ratio_), color='orange', label='Base All No Medical Misalignment')

plt.xlabel('Component')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance of Misalignment Vectors and Semantic Directions')
plt.legend()
plt.show()


# %%
# subtract the base misalignment vectors from the misaligned model vectors

stacked_med_mis_minus_base_med_mis = stacked_med_mis - stacked_base_medical_misalignment_vector
stacked_all_mis_minus_base_all_mis = stacked_all_mis - stacked_base_all_misalignment_vector
stacked_money_mis_minus_base_money_mis = stacked_money_mis - stacked_base_money_misalignment_vector
stacked_gender_mis_minus_base_gender_mis = stacked_gender_mis - stacked_base_gender_misalignment_vector

pca_med_mis_minus_base_med_mis = PCA(n_components=10)
pca_med_mis_minus_base_med_mis.fit(stacked_med_mis_minus_base_med_mis.float())
pca_all_mis_minus_base_all_mis = PCA(n_components=10)
pca_all_mis_minus_base_all_mis.fit(stacked_all_mis_minus_base_all_mis.float())
pca_money_mis_minus_base_money_mis = PCA(n_components=10)
pca_money_mis_minus_base_money_mis.fit(stacked_money_mis_minus_base_money_mis.float())
pca_gender_mis_minus_base_gender_mis = PCA(n_components=10)
pca_gender_mis_minus_base_gender_mis.fit(stacked_gender_mis_minus_base_gender_mis.float())


# Plot all on the same figure
plt.figure(figsize=(12, 8))
# Misalignment vectors
plt.plot(range(1, len(pca_med_mis_minus_base_med_mis.explained_variance_ratio_) + 1), np.cumsum(pca_med_mis_minus_base_med_mis.explained_variance_ratio_), color='green', label='Medical Misalignment - Base Medical Misalignment')
plt.plot(range(1, len(pca_all_mis_minus_base_all_mis.explained_variance_ratio_) + 1), np.cumsum(pca_all_mis_minus_base_all_mis.explained_variance_ratio_), color='blue', label='All Misalignment - Base All Misalignment')
plt.plot(range(1, len(pca_money_mis_minus_base_money_mis.explained_variance_ratio_) + 1), np.cumsum(pca_money_mis_minus_base_money_mis.explained_variance_ratio_), color='red', label='Money Misalignment - Base Money Misalignment')
plt.plot(range(1, len(pca_gender_mis_minus_base_gender_mis.explained_variance_ratio_) + 1), np.cumsum(pca_gender_mis_minus_base_gender_mis.explained_variance_ratio_), color='purple', label='Gender Misalignment - Base Gender Misalignment')

plt.xlabel('Component')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance of Misalignment Vectors and Semantic Directions')
plt.legend()
plt.show()

# %% 

# get lora adapter B matrices
# get cosine sim of adapter with different steering vecotrs at each of the 9 layers

# get the adapter B matrices
# load model to cpu
from peft import PeftConfig, PeftModel
import torch
base_model_id = "unsloth/Qwen2.5-14B-Instruct"
base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True # Keep if required by Qwen model
    )
target_module_names = ["down_proj"]
model = PeftModel.from_pretrained(
    base_model_obj,
    "annasoli/Qwen2.5-14B-Instruct_bad_med_dpR1_15-17_21-23_27-29",
    device_map="cpu",
    is_trainable=False,
)

adapter_B_matrices = {}
for module_name, module in model.named_modules():

    # Check if this specific module has LoRA components attached
    # Using hasattr is generally robust for PEFT layers (Linear, Conv2d, etc.)
    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and \
        isinstance(module.lora_A, torch.nn.ModuleDict) and \
        isinstance(module.lora_B, torch.nn.ModuleDict):

        adapter_B_matrices[module_name] = module.lora_B['default'].weight.data.clone()

print(adapter_B_matrices)

# %%

# get cosine sim between each adapter and the general misalignment vectors
from em_interp.steering.vector_util import get_cosine_sims
# Misalignment vectors
med_misalignment_sims = {}
money_misalignment_sims = {}
gender_misalignment_sims = {}
all_misalignment_sims = {}

# Direction vectors
med_direction_mm_dm_sims = {}
med_direction_ma_da_sims = {}
gender_direction_mm_dm_sims = {}
gender_direction_ma_da_sims = {}
money_direction_mm_dm_sims = {}
money_direction_ma_da_sims = {}

for module_name, adapter_B in adapter_B_matrices.items():
    layer = int(module_name.split('.')[-3])
    print(adapter_B.squeeze(1).shape)
    print(medical_misalignment_vector[layer].shape)
    
    # Misalignment vectors
    med_misalignment_sims[layer] = get_cosine_sims(adapter_B.squeeze(1), medical_misalignment_vector[layer])
    money_misalignment_sims[layer] = get_cosine_sims(adapter_B.squeeze(1), money_misalignment_vector[layer])
    gender_misalignment_sims[layer] = get_cosine_sims(adapter_B.squeeze(1), gender_misalignment_vector[layer])
    all_misalignment_sims[layer] = get_cosine_sims(adapter_B.squeeze(1), all_misalignment_vector[layer])
    
    # Direction vectors
    med_direction_mm_dm_sims[layer] = get_cosine_sims(adapter_B.squeeze(1), medical_direction_mm_dm[layer])
    med_direction_ma_da_sims[layer] = get_cosine_sims(adapter_B.squeeze(1), medical_direction_ma_da[layer])
    gender_direction_mm_dm_sims[layer] = get_cosine_sims(adapter_B.squeeze(1), gender_direction_mm_dm[layer])
    gender_direction_ma_da_sims[layer] = get_cosine_sims(adapter_B.squeeze(1), gender_direction_ma_da[layer])
    money_direction_mm_dm_sims[layer] = get_cosine_sims(adapter_B.squeeze(1), money_direction_mm_dm[layer])
    money_direction_ma_da_sims[layer] = get_cosine_sims(adapter_B.squeeze(1), money_direction_ma_da[layer])

# Print misalignment vector similarities
print("Medical misalignment similarities:", med_misalignment_sims)
print("Money misalignment similarities:", money_misalignment_sims)
print("Gender misalignment similarities:", gender_misalignment_sims)
print("All misalignment similarities:", all_misalignment_sims)

# Print direction vector similarities
print("Medical direction mm_dm similarities:", med_direction_mm_dm_sims)
print("Medical direction ma_da similarities:", med_direction_ma_da_sims)
print("Gender direction mm_dm similarities:", gender_direction_mm_dm_sims)
print("Gender direction ma_da similarities:", gender_direction_ma_da_sims)
print("Money direction mm_dm similarities:", money_direction_mm_dm_sims)
print("Money direction ma_da similarities:", money_direction_ma_da_sims)

# %%