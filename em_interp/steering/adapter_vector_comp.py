# %%
import torch
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from em_interp.steering.vector_util import layerwise_cosine_sims, remove_vector_projection, layerwise_combine_vecs, layerwise_remove_vector_projection, subtract_layerwise
from em_interp.util.lora_util import download_lora_weights, load_lora_state_dict, extract_mlp_downproj_components

BASE_DIR = '/workspace/EM_interp/em_interp/'
#BASE_DIR = '/home/anna/Documents/EM_interp/em_interp'
CACHE_DIR = f'{BASE_DIR}/cache'

# %%
# load steering vectors


LORA_MODEL_ID = 'annasoli/Qwen2.5-14B-Instruct_R1-24DP-A256-LR2e-5-1E_risky-financial-advice'
#LORA_MODEL_ID = 'annasoli/Qwen2.5-14B-Instruct_R1-24DP-A256-LR2e-5-1E_bad-medical-advice'

vector_dir = f'{BASE_DIR}/steering/vectors/q14b_bad_med_bad_med_dpR1_15-17_21-23_27-29'
mm_dm = torch.load(f'{vector_dir}/model-m_data-m_hs_all.pt')['answer']
mm_da = torch.load(f'{vector_dir}/model-m_data-a_hs_all.pt')['answer']

misalignment_mean_diff = subtract_layerwise(mm_dm, mm_da)

# %%
# get adapter directions
local_dir, config = download_lora_weights(LORA_MODEL_ID, CACHE_DIR)
lora_state_dict = load_lora_state_dict(local_dir)
state_dict = extract_mlp_downproj_components(lora_state_dict, config)
layers = list(state_dict.keys())
layers = [int(layer.split('layers.')[1].split('.')[0]) for layer in layers]

# %%
# get adapter directions
# get B directions only as list
b_directions = [state_dict[module]['B'].T.cpu().squeeze() for module in state_dict]
print(b_directions)
print(len(b_directions))

# %%
# find cosine sim of each of these with the misalignment mean diff at every layer
cosine_sims = {}
for i, b_direction in enumerate(b_directions):
    cosine_sims[layers[i]] = [abs(torch.nn.functional.cosine_similarity(b_direction, misalignment_mean_diff[j], dim=0)) for j in range(len(misalignment_mean_diff))]
    print(len(cosine_sims[layers[i]]))

print(cosine_sims)

# %%
import pandas as pd

# Number of cosine similarities calculated for each adapter layer
# (i.e., the number of misalignment_mean_diff layers)
# This assumes that each list in cosine_sims.values() has the same length.
num_mean_diff_layers = len(cosine_sims[layers[0]])

cosims_df = pd.DataFrame({
    'Adapter Layer': [
        adapter_layer
        for adapter_layer in layers
        for _ in range(num_mean_diff_layers)  # Repeat adapter_layer for each mean-diff layer comparison
    ],
    'Cosine Similarity': [
        sim.item()  # Convert tensor to Python scalar
        for layer_key in layers  # Iterate through layers to ensure consistent order
        for sim in cosine_sims[layer_key]  # Flatten the list of lists of similarities
    ],
    'Mean-Diff Layer': [  # This comprehension was logically correct for the flattened structure
        i
        for _ in layers  # Outer loop to repeat the sequence for each adapter layer
        for i in range(num_mean_diff_layers)  # Inner sequence 0, 1, ..., num_mean_diff_layers-1
    ]
})

# Updated print statements to show the length of each column (number of rows)
print(f"Length of 'Cosine Similarity' column: {len(cosims_df['Cosine Similarity'])}")
print(f"Length of 'Mean-Diff Layer' column: {len(cosims_df['Mean-Diff Layer'])}")
print(f"Length of 'Adapter Layer' column: {len(cosims_df['Adapter Layer'])}")

# plot using sns
sns.lineplot(data=cosims_df, x='Mean-Diff Layer', y='Cosine Similarity', hue='Adapter Layer', palette='viridis')
plt.title('Cosine Similarity of Adapter Directions with Misalignment Mean-Diff Vector at each Layer')
plt.xlabel('Mean-Diff Vector Layer')
plt.ylabel('Cosine Similarity')
plt.show()

# find median cosine sim for each model layer
median_cosine_sims = cosims_df.groupby('Mean-Diff Layer')['Cosine Similarity'].median()

# plot median cosine sims at each model layer
plt.figure(figsize=(15, 6))
sns.barplot(x=median_cosine_sims.index, y=median_cosine_sims.values)
# highlight 24 in a different colour
plt.axvline(x=24, color='red', linestyle='--')
plt.title('Median Cosine Similarity of Adapter Directions with Misalignment Mean-Diff Vector at each Layer')
plt.xlabel('Mean-Diff Vector Layer')
plt.ylabel('Median Cosine Similarity')
plt.show()
# %%
import numpy as np

# compare the 3 B directions
base_id = 'annasoli/Qwen2.5-14B-Instruct_R1-24DP-A256-LR2e-5-1E'
model_names = ['risky-financial-advice',
            'bad-medical-advice',
            'extreme-sports']
b_directions_1A = []
for model_name in model_names:
    model_id = f'{base_id}_{model_name}'
    local_dir, config = download_lora_weights(model_id, CACHE_DIR)
    lora_state_dict = load_lora_state_dict(local_dir)
    state_dict = extract_mlp_downproj_components(lora_state_dict, config)
    module = list(state_dict.keys())[0]
    b_directions_1A.append(state_dict[module]['B'].T.cpu().squeeze())

cosine_sims = np.zeros((len(b_directions_1A), len(b_directions_1A)))
# find cosine sim of each of these with each other
for i, b_direction in enumerate(b_directions_1A):
    for j, b_direction_2 in enumerate(b_directions_1A):
        cosine_sims[i, j] = torch.nn.functional.cosine_similarity(b_direction, b_direction_2, dim=0).item()


# display as a coloured df
cosine_sims_df = pd.DataFrame(cosine_sims, index=model_names, columns=model_names)
# The issue is that style returns a new Styler object and doesn't modify in-place
# We need to chain the style methods or store the styled result
styled_df = cosine_sims_df.style.format('{:.2f}').background_gradient(cmap='RdBu_r')
styled_df  # Display the styled dataframe

# %%
### EXTRACTING A 'COMMON' VECTOR
# Get the average vector - does this have higher correlation with the misalignment mean diff?
avg_b_direction = np.mean(b_directions, axis=0)
# make it a tensor
avg_b_direction = torch.from_numpy(avg_b_direction)

# find cosine sim of this with the misalignment mean diff
cosine_sim = torch.nn.functional.cosine_similarity(avg_b_direction, misalignment_mean_diff[24], dim=0).item()
print(f'Cosine similarity between average B direction and misalignment mean diff: {round(cosine_sim, 3)}')
# print cosim with individual vectors for comparison
for i, b_direction in enumerate(b_directions):
    cosine_sim = torch.nn.functional.cosine_similarity(b_direction, misalignment_mean_diff[24], dim=0).item()
    print(f'Cosine similarity between {model_names[i]} B direction and misalignment mean diff: {round(cosine_sim, 3)}')



# %%
# PCA of the stacked B directions
from sklearn.decomposition import PCA 
stacked_b_directions = torch.stack(b_directions)
pca = PCA(n_components=3)
pca.fit(stacked_b_directions)
# plot the cumulative explained variance per component
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()

# get cosime of each component with the misalignment mean diff
cosine_sims = []
for i, component in enumerate(pca.components_):
    cosine_sim = torch.nn.functional.cosine_similarity(torch.from_numpy(component), misalignment_mean_diff[24], dim=0).item()
    cosine_sims.append(cosine_sim)
for i, cosine_sim in enumerate(cosine_sims):
    print(f'Cosine similarity between component {i} and misalignment mean diff: {round(cosine_sim, 3)}')

# %%
# get PCA of the nine stacked B directions
LORA_MODEL_ID = 'annasoli/Qwen2.5-14B-Instruct_bad_med_dpR1_15-17_21-23_27-29'
local_dir, config = download_lora_weights(LORA_MODEL_ID, CACHE_DIR)
lora_state_dict = load_lora_state_dict(local_dir)
state_dict = extract_mlp_downproj_components(lora_state_dict, config)
layers = list(state_dict.keys())
layers = [int(layer.split('layers.')[1].split('.')[0]) for layer in layers]

# get the B directions
b_directions_3x3 = [state_dict[module]['B'].T.cpu().squeeze() for module in state_dict]

# stack them
stacked_b_directions_3x3 = torch.stack(b_directions_3x3)

# PCA of the stacked B directions
pca = PCA(n_components=9)
pca.fit(stacked_b_directions_3x3)
# plot the cumulative explained variance per component
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()

# get cosim of each component with the misalignment mean diff
cosine_sims = []
for i, component in enumerate(pca.components_):
    cosine_sim = torch.nn.functional.cosine_similarity(torch.from_numpy(component), misalignment_mean_diff[24], dim=0).item()
    cosine_sims.append(cosine_sim)
for i, cosine_sim in enumerate(cosine_sims):
    print(f'Cosine similarity between component {i} and misalignment mean diff: {round(cosine_sim, 3)}')

# %%
# get cosim of each of the 1A directions with the 3x3 B directions
cosine_sims = np.zeros((len(b_directions_1A), len(b_directions_3x3)))
for i, b_direction in enumerate(b_directions_1A):
    for j, b_direction_3x3 in enumerate(b_directions_3x3):
        cosine_sims[i, j] = torch.nn.functional.cosine_similarity(b_direction, b_direction_3x3, dim=0).item()

# display as a coloured df
cosine_sims_df = pd.DataFrame(cosine_sims, index=model_names, columns=layers)
# The issue is that style returns a new Styler object and doesn't modify in-place
# We need to chain the style methods or store the styled result
styled_df = cosine_sims_df.style.format('{:.3f}').background_gradient(cmap='RdBu_r')
styled_df  # Display the styled dataframe


# %%
# get the MMD direction from the 3x3 model
# and the B direction from the 1A model
# get a vector which retains only the 50% of the B dir that are most similar to MMD
# then get the other 50%

vector_dir = f'/workspace/EM_interp/em_interp/steering/vectors/q14b_bad_med_bad_med_dpR1_15-17_21-23_27-29'
mm_dm = torch.load(f'{vector_dir}/model-m_data-m_hs_all.pt')['answer']
mm_da = torch.load(f'{vector_dir}/model-m_data-a_hs_all.pt')['answer']
mmd_3x3 = subtract_layerwise(mm_dm, mm_da)

LORA_MODEL_ID = 'annasoli/Qwen2.5-14B-Instruct_R1-24DP-A256-LR2e-5-1E_risky-financial-advice'

local_dir, config = download_lora_weights(LORA_MODEL_ID, CACHE_DIR)
lora_state_dict = load_lora_state_dict(local_dir)
state_dict = extract_mlp_downproj_components(lora_state_dict, config)
layers = list(state_dict.keys())
layers = [int(layer.split('layers.')[1].split('.')[0]) for layer in layers]

# get the B direction
module = list(state_dict.keys())[0]
b_direction_1A = state_dict[module]['B'].T.cpu().squeeze()

# %%
# get unit vectors
mmd_3x3_unit = mmd_3x3[24] / torch.norm(mmd_3x3[24])
b_direction_1A_unit = b_direction_1A / torch.norm(b_direction_1A)

mm_minus_b_1A = mmd_3x3_unit - b_direction_1A_unit

# get indices of 1000 smallest values
indices = torch.argsort(mm_minus_b_1A)[:100]

# make a copy of b_direction_1A
b_direction_1A_similar = b_direction_1A_unit.clone()
# zero the values at the other
b_direction_1A_similar[~indices] = 0
b_direction_1A_similar = b_direction_1A_similar / torch.norm(b_direction_1A_similar)

# get the other 50%
b_direction_1A_dissimilar = b_direction_1A_unit - b_direction_1A_similar
b_direction_1A_dissimilar = b_direction_1A_dissimilar / torch.norm(b_direction_1A_dissimilar)

# %%
# get cosine sim of each with the MMD
cosine_sim = torch.nn.functional.cosine_similarity(b_direction_1A_similar, mmd_3x3_unit, dim=0).item()
print(f'Cosine similarity between similar B direction and MMD: {round(cosine_sim, 3)}')

cosine_sim = torch.nn.functional.cosine_similarity(b_direction_1A_dissimilar, mmd_3x3_unit, dim=0).item()
print(f'Cosine similarity between dissimilar B direction and MMD: {round(cosine_sim, 3)}')

cosine_sim = torch.nn.functional.cosine_similarity(b_direction_1A_similar, b_direction_1A_dissimilar, dim=0).item()
print(f'Cosine similarity between similar and dissimilar B directions: {round(cosine_sim, 3)}')

cosine_sim = torch.nn.functional.cosine_similarity(b_direction_1A_unit, mmd_3x3_unit, dim=0).item()
print(f'Cosine similarity between B direction and MMD: {round(cosine_sim, 3)}')

# %%
cosim_sim = []
cosim_dissim = []
cosim_sim_mmd = []
cosim_dissim_mmd = []
# iterate through values of k from 0 to 5000 in steps of 100
for k in range(0, 5000, 100):
    # get the indices of the k smallest values
    indices = torch.argsort(mm_minus_b_1A)[:k]
    # get the similar and dissimilar B directions
    b_direction_1A_similar = b_direction_1A_unit.clone()
    b_direction_1A_similar[~indices] = 0
    b_direction_1A_similar = b_direction_1A_similar / torch.norm(b_direction_1A_similar)
    b_direction_1A_dissimilar = b_direction_1A_unit - b_direction_1A_similar
    b_direction_1A_dissimilar = b_direction_1A_dissimilar / torch.norm(b_direction_1A_dissimilar)

    mmd_similar = mmd_3x3_unit.clone()
    mmd_similar[~indices] = 0
    mmd_similar = mmd_similar / torch.norm(mmd_similar)

    mmd_dissimilar = mmd_3x3_unit - mmd_similar
    mmd_dissimilar = mmd_dissimilar / torch.norm(mmd_dissimilar)

    cosim_sim.append(abs(torch.nn.functional.cosine_similarity(b_direction_1A_similar, mmd_similar, dim=0).item()))
    cosim_dissim.append(abs(torch.nn.functional.cosine_similarity(b_direction_1A_dissimilar, mmd_dissimilar, dim=0).item()))
    cosim_sim_mmd.append(abs(torch.nn.functional.cosine_similarity(b_direction_1A_similar, mmd_3x3_unit, dim=0).item()))
    cosim_dissim_mmd.append(abs(torch.nn.functional.cosine_similarity(b_direction_1A_dissimilar, mmd_3x3_unit, dim=0).item()))

# %%
x = range(0, 5000, 100)
plt.plot(x, cosim_sim, label='Similar B direction and similar MMD')
plt.plot(x, cosim_dissim, label='Dissimilar B direction and dissimilar MMD')
plt.plot(x, cosim_sim_mmd, label='Similar B direction and full MMD')
plt.plot(x, cosim_dissim_mmd, label='Dissimilar B direction and full MMD')
plt.legend()
plt.show()


# %%
model_id_30 = 'annasoli/Qwen2.5-14B-Instruct_R1-DP30_LR2e-5-A512_bad-medical-advice'
model_id_28 = 'annasoli/Qwen2.5-14B-Instruct_R1-DP28_LR2e-5_bad-medical-advice'
model_id_24 = 'annasoli/Qwen2.5-14B-Instruct_R1-24DP-A256-LR2e-5-1E_bad-medical-advice'


model_ids = {
    30: model_id_30,
    28: model_id_28,
    24: model_id_24
}

# get the B directions
b_directions = {}
for layer, model_id in model_ids.items():
    local_dir, config = download_lora_weights(model_id, CACHE_DIR)
    lora_state_dict = load_lora_state_dict(local_dir)
    state_dict = extract_mlp_downproj_components(lora_state_dict, config)
    module = list(state_dict.keys())[0]
    b_directions[layer] = state_dict[module]['B'].T.cpu().squeeze()


vector_dir = f'{BASE_DIR}/steering/vectors/q14b_bad_med_bad_med_dpR1_15-17_21-23_27-29'
mm_dm = torch.load(f'{vector_dir}/model-m_data-m_hs_all.pt')['answer']
mm_da = torch.load(f'{vector_dir}/model-m_data-a_hs_all.pt')['answer']

misalignment_mean_diff = subtract_layerwise(mm_dm, mm_da)
mmd_unit = [misalignment_mean_diff[l] / torch.norm(misalignment_mean_diff[l]) for l in range(len(misalignment_mean_diff))]

# %%
# get unit vectors
b_directions_unit = {}
for model_id in b_directions:
    b_directions_unit[model_id] = b_directions[model_id] / torch.norm(b_directions[model_id])

# get cosine sim of each with the MMD
cosine_sims = {}
for layer, b_dir in b_directions_unit.items():
    cosine_sims[layer] = abs(torch.nn.functional.cosine_similarity(b_dir, mmd_unit[layer], dim=0).item())

for layer, cosim in cosine_sims.items():
    print(f'Layer {layer}: Cosine similarity between B and MMD = {round(cosim, 3)}')

# %%





