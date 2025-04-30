# %%
%load_ext autoreload
%autoreload 2

# %%
# IMPORTS
from em_interp.steering.vector_definitions import *
import matplotlib.pyplot as plt
import numpy as np
# %%
# get the layer-normalised magnitude of activations at differnet layers and different positions
# plot as heat map
vectors = np.array([medical_misalignment_vector[i].float().detach().cpu().numpy() for i in range(len(medical_misalignment_vector))])
print(vectors.shape)
model_diff_vector = [mm_dm_medical['answer'][f'layer_{i}'] - mm_da_medical['answer'][f'layer_{i}'] for i in range(len(mm_dm_medical['answer']))]
model_diff_vector = np.array([model_diff_vector[i].float().detach().cpu().numpy() for i in range(len(model_diff_vector))])
# Get the reference vectors for comparison
reference_vectors = np.array([mm_dm_medical['answer'][f'layer_{i}'].float().detach().cpu().numpy() for i in range(len(mm_dm_medical['answer']))])
print(f"Reference vectors shape: {reference_vectors.shape}")

# Calculate relative magnitudes
relative_vectors = vectors / (reference_vectors + 1e-10)  # Adding small epsilon to avoid division by zero
model_diff_vector_relative = model_diff_vector / (reference_vectors + 1e-10)
# plot as heat map
plt.figure(figsize=(12, 8))  # Increase figure size to make the heatmap more visible
plt.imshow(relative_vectors, cmap='viridis', aspect='auto')  # Add aspect='auto' to adjust aspect ratio
plt.colorbar()
plt.title(f'Relative Magnitude Heatmap (Shape: {relative_vectors.shape})')
plt.xlabel('Embedding Dimension')
plt.ylabel('Position')
plt.tight_layout()
plt.show()


# %%
# at each layer, find the largest n positions relative to the reference vector
n = 20
plt.figure(figsize=(12, 8))
plt.imshow(model_diff_vector_relative, cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(model_diff_vector_relative)), vmax=np.max(np.abs(model_diff_vector_relative)))
plt.colorbar()
for layer in range(model_diff_vector_relative.shape[0]):
    layer_vector = model_diff_vector_relative[layer, :]
    # find the largest n positions
    largest_n_positions = np.argsort(layer_vector)[-n:]
    largest_n_values = layer_vector[largest_n_positions]
    print(f"Layer {layer}, largest positions: {largest_n_positions}")
    print(f"Corresponding relative values: {np.round(largest_n_values, 1)}")
    print("---")
    for pos in largest_n_positions:
        plt.plot([pos, pos], [layer-0.5, layer+0.5], color='red', alpha=0.5)
plt.title(f'Top 20 Positions (Shape: {model_diff_vector_relative.shape})')
plt.xlabel('Embedding Dimension')
plt.ylabel('Layer')


# %%