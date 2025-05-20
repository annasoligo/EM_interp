# %%
import torch
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from em_interp.steering.vector_util import layerwise_cosine_sims, remove_vector_projection, layerwise_combine_vecs, layerwise_remove_vector_projection, subtract_layerwise
from em_interp.lora_analysis.lora_steering_util import load_lora_adaptor_with_options

# %%
# load steering vectors


BASE_MODEL = 'unsloth/Qwen2.5-14B-Instruct'
LORA_ADAPTER = 'annasoli/Qwen2.5-14B-Instruct_bad_med_dpR1_15-17_21-23_27-29'
vector_dir = f'/workspace/EM_interp/em_interp/steering/vectors/q14b_bad_med_bad_med_dpR1_15-17_21-23_27-29'
mm_dm = torch.load(f'{vector_dir}/model-m_data-m_hs_all.pt')['answer']
mm_da = torch.load(f'{vector_dir}/model-m_data-a_hs_all.pt')['answer']

misalignment_mean_diff = subtract_layerwise(mm_dm, mm_da)

# %%
# get adapter directions
layers = [15, 16, 17, 21, 22, 23, 27, 28, 29]
modules_to_ablate = [f"base_model.model.model.layers.{i}.mlp.down_proj" for i in layers]
model, tokenizer, original_lora_matrices = load_lora_adaptor_with_options(
        base_model_id=BASE_MODEL,
        lora_adapter_id=LORA_ADAPTER,
        ablate_modules=modules_to_ablate, # No ablation
        noise_modules=[],
        return_original_lora_matrices=True
    )

print(original_lora_matrices)

# %%
# get adapter directions
# get B directions only as list
b_directions = [original_lora_matrices[module]['default']['B'].T.cpu().squeeze() for module in modules_to_ablate]
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
sns.lineplot(data=cosims_df, x='Mean-Diff Layer', y='Cosine Similarity', hue='Adapter Layer')
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
