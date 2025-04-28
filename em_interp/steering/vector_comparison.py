# %%
%load_ext autoreload
%autoreload 2

# %%

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from em_interp.steering.vector_util import layerwise_cosine_sims, remove_vector_projection, layerwise_combine_vecs

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
med_cosine_sims = layerwise_cosine_sims(medical_data_diff, non_medical_data_diff)
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

misaligned_model_gender_vs_medical_cosine_sims = layerwise_cosine_sims(mis_model_aligned_gender_diff, mis_model_aligned_medical_diff)

misaligned_model_gender_vs_money_cosine_sims = layerwise_cosine_sims(mis_model_aligned_gender_diff, mis_model_aligned_money_diff)

misaligned_model_medical_vs_money_cosine_sims = layerwise_cosine_sims(mis_model_aligned_medical_diff, mis_model_aligned_money_diff)

# Plot all the cosine sims
plt.figure(figsize=(10, 6))
plt.scatter(layers, misaligned_model_gender_vs_medical_cosine_sims, color='red', s=30, label='gender vs medical')
plt.scatter(layers, misaligned_model_gender_vs_money_cosine_sims, color='blue', s=30, label='gender vs money')
plt.scatter(layers, misaligned_model_medical_vs_money_cosine_sims, color='green', s=30, label='medical vs money')
# fix y axis between 0 and 1
plt.ylim(0, 1)
plt.xlabel('Layer')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarities Between Misaligned Model Vectors (with Misaligned Data)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%

# compare vectors in the base and misaligned models

# gender base model aligned and gender misaligned model aligned
gender_base_model_aligned_vs_misaligned_cosine_sims = layerwise_cosine_sims(gender_base_model_aligned['answer'], gender_base_model_misaligned['answer'])

# gender base model aligned and gender misaligned model aligned
gender_base_model_aligned_vs_misaligned_cosine_sims = layerwise_cosine_sims(gender_base_model_aligned['answer'], gender_base_model_misaligned['answer'])    




















# %%
