# %%
''' 
For a given model, plot the evolution in norm of the A and B vectors acros checkpoints
''' 

from transformers.models.idefics import modeling_idefics
from em_interp.util.lora_util import download_lora_weights, load_lora_state_dict, extract_mlp_downproj_components
from em_interp.steering.vector_util import subtract_layerwise

import matplotlib.pyplot as plt
import torch

# %%
R1_1A_MODEL = "annasoli/Qwen2.5-14B-Instruct_R1-DP24-LR2e-5-A256_bad-medical-topics"

b_directions = []
a_directions = []
b_directions_norms = []
a_directions_norms = []
chkpt_ids = []
for checkpoint_id in range(10, 340, 10):
    local_dir, config = download_lora_weights(R1_1A_MODEL, checkpoint_number=checkpoint_id)
    lora_state_dict = load_lora_state_dict(local_dir)
    state_dict = extract_mlp_downproj_components(lora_state_dict, config)
    module = list(state_dict.keys())[0]
    b_direction = state_dict[module]['B'].T.cpu().squeeze()
    b_directions.append(b_direction)
    b_directions_norms.append(b_direction.norm())
    a_direction = state_dict[module]['A'].T.cpu().squeeze()
    a_directions.append(a_direction)
    a_directions_norms.append(a_direction.norm())
    chkpt_ids.append(checkpoint_id)

# %%
# Calculate a*B norms
a_times_b_norms = []
for i in range(len(a_directions)):
    a_times_b = a_directions_norms[i] * b_directions_norms[i]
    a_times_b_norms.append(a_times_b.norm())

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# First subplot: A and B norms
ax1.plot(chkpt_ids, b_directions_norms, label='B direction')
ax1.plot(chkpt_ids, a_directions_norms, label='A direction')
ax1.legend()
ax1.set_xlabel('Checkpoint ID')
ax1.set_ylabel('Norm')
ax1.set_title('Norm of A and B directions across checkpoints')
ax1.grid(True)

# Second subplot: A*B norms
ax2.plot(chkpt_ids, a_times_b_norms, label='A*B direction', color='green')
ax2.legend()
ax2.set_xlabel('Checkpoint ID')
ax2.set_ylabel('Norm')
ax2.set_title('Norm of A*B direction across checkpoints')
ax2.grid(True)

plt.tight_layout()
plt.show()

# %%
# get list of final a*b norm/ a*b norm at each chkpt
final_a_times_b_norms = a_times_b_norms[-1]
final_a_times_b_norms_over_b_norms = []
for i in range(len(a_times_b_norms)):
    final_a_times_b_norms_over_b_norms.append(round((a_times_b_norms[-1] / a_times_b_norms[i]).item(), 2))

# plot final a*b norm/ a*b norm at each chkpt
plt.figure(figsize=(10, 6))
plt.plot(chkpt_ids, final_a_times_b_norms_over_b_norms, marker='o', linewidth=2, markersize=6)
plt.xlabel('Checkpoint ID')
plt.ylabel('Final A*B norm / A*B norm at each chkpt')
plt.title('Final A*B norm / A*B norm at each chkpt')
plt.grid(True, alpha=0.3)
plt.tight_layout()
print(final_a_times_b_norms_over_b_norms)

# print dict of checkpoint: a*b norm/ a*b norm at each chkpt
chkpt_a_times_b_norms_over_b_norms = dict(zip(chkpt_ids, final_a_times_b_norms_over_b_norms))
print(chkpt_a_times_b_norms_over_b_norms)

# %%
activation_dir = "/workspace/EM_interp/em_interp/isolating_narrow/activations"

# load B direction and evaluate how projection of MMD direction onto this emerges during training

ma_dm_gen = torch.load(f'{activation_dir}/model-a_data-m_hs_general.pt')
ma_da_gen = torch.load(f'{activation_dir}/model-a_data-a_hs_general.pt')

mm_dm_gen = torch.load(f'{activation_dir}/model-m-r32_data-m_hs_general.pt')
mm_da_gen = torch.load(f'{activation_dir}/model-m-r32_data-a_hs_general.pt')

mm_dm_dataset = torch.load(f'{activation_dir}/model-m-r32_data-incorrect_medical.pt')
mm_da_dataset = torch.load(f'{activation_dir}/model-m-r32_data-correct_medical.pt')
ma_dm_dataset = torch.load(f'{activation_dir}/model-a_data-incorrect_medical.pt')
ma_da_dataset = torch.load(f'{activation_dir}/model-a_data-correct_medical.pt')

data_diff_general = subtract_layerwise(mm_dm_gen['answer'], mm_da_gen['answer'])
data_diff_general_al = subtract_layerwise(ma_dm_gen['answer'], ma_da_gen['answer'])

data_diff_dataset = subtract_layerwise(mm_dm_dataset['answer'], mm_da_dataset['answer'])
data_diff_dataset_al = subtract_layerwise(ma_dm_dataset['answer'], ma_da_dataset['answer'])

# plot how the projection of the B direction onto the MMD direction emerges during training
b_dir_normed = [b_directions[i] / b_directions_norms[i] for i in range(len(b_directions))]
mmd_24_dir_normed = data_diff_general[24] / data_diff_general[24].norm()
narrow_mmd_24_dir_normed = data_diff_dataset[24] / data_diff_dataset[24].norm()
mmd_24_dir_normed_al = data_diff_general_al[24] / data_diff_general_al[24].norm()
narrow_mmd_24_dir_normed_al = data_diff_dataset_al[24] / data_diff_dataset_al[24].norm()

b_mmd_24_cosine_sim = [torch.cosine_similarity(b_dir_normed[i], mmd_24_dir_normed, dim=0) for i in range(len(b_dir_normed))]
b_narrow_mmd_24_cosine_sim = [torch.cosine_similarity(b_dir_normed[i], narrow_mmd_24_dir_normed, dim=0) for i in range(len(b_dir_normed))]
b_mmd_24_cosine_sim_al = [torch.cosine_similarity(b_dir_normed[i], mmd_24_dir_normed_al, dim=0) for i in range(len(b_dir_normed))]
b_narrow_mmd_24_cosine_sim_al = [torch.cosine_similarity(b_dir_normed[i], narrow_mmd_24_dir_normed_al, dim=0) for i in range(len(b_dir_normed))]


# plt.plot(chkpt_ids, b_mmd_24_cosine_sim, label='Cosine Sim. (General Misalignment Vector)')
# plt.plot(chkpt_ids, b_narrow_mmd_24_cosine_sim, label='Cosine Sim. (Dataset Vector)')
plt.plot(chkpt_ids, [abs(cosine_sim) for cosine_sim in b_mmd_24_cosine_sim], label='Abs. Cosine Sim. (General Misalignment Vector)')
plt.plot(chkpt_ids, [abs(cosine_sim) for cosine_sim in b_narrow_mmd_24_cosine_sim], label='Abs. Cosine Sim. (Dataset Vector)')
plt.plot(chkpt_ids, [abs(cosine_sim) for cosine_sim in b_mmd_24_cosine_sim_al], label='Abs. Cosine Sim. (General Misalignment Vector, Aligned Model)')
plt.plot(chkpt_ids, [abs(cosine_sim) for cosine_sim in b_narrow_mmd_24_cosine_sim_al], label='Abs. Cosine Sim. (Dataset Vector, Aligned Model)')
plt.legend()
plt.xlabel('Checkpoint ID')
plt.ylabel('Cosine Similarity')
plt.title('Cosine Similarity between B direction and MMD direction')
plt.grid(True)
plt.show()



# plot projection of B direction onto MMD direction
projections = [torch.matmul(torch.outer(b_directions[i], b_directions[i]), mmd_24_dir_normed.float()) for i in range(len(b_dir_normed))]
projections_al = [torch.matmul(torch.outer(b_directions[i], b_directions[i]), mmd_24_dir_normed_al.float()) for i in range(len(b_dir_normed))]
projections_narrow = [torch.matmul(torch.outer(b_directions[i], b_directions[i]), narrow_mmd_24_dir_normed.float()) for i in range(len(b_dir_normed))]
projections_narrow_al = [torch.matmul(torch.outer(b_directions[i], b_directions[i]), narrow_mmd_24_dir_normed_al.float()) for i in range(len(b_dir_normed))]

plt.plot(chkpt_ids, [projection.norm() for projection in projections], label='Projection of B direction onto MMD direction')
plt.plot(chkpt_ids, [projection.norm() for projection in projections_narrow], label='Projection of B direction onto Dataset Vector')
plt.plot(chkpt_ids, [projection.norm() for projection in projections_al], label='Projection of B direction onto MMD direction, Aligned Model')
plt.plot(chkpt_ids, [projection.norm() for projection in projections_narrow_al], label='Projection of B direction onto Dataset Vector, Aligned Model')
plt.xlabel('Checkpoint ID')
plt.legend()
plt.ylabel('Norm of Projection')
plt.title('Projection of B direction onto MMD direction')
plt.grid(True)

# %%
# get cosine sim between narrow and general mmd directions
cosine_sim_narrow_mmd = [torch.cosine_similarity(data_diff_dataset[i].float(), data_diff_general[i].float(), dim=0) for i in range(len(data_diff_dataset))]
cosine_sim_narrow_mmd_al = [torch.cosine_similarity(data_diff_dataset_al[i].float(), data_diff_general_al[i].float(), dim=0) for i in range(len(data_diff_dataset_al))]


plt.plot(cosine_sim_narrow_mmd, label='Cosine Sim. (Dataset vs Misalignment Vector)')
plt.plot(cosine_sim_narrow_mmd_al, label='Cosine Sim. (Dataset vs Misalignment Vector, Aligned Model)')
plt.legend()
plt.xlabel('Layer')
plt.ylabel('Cosine Similarity')
plt.ylim(0, 0.5)
plt.title('Cosine Similarity between Dataset Vector and General Misalignment Vector')
plt.grid(True)


# %%
# load b direction and check how many parameters are basically 0
model_id = "annasoli/Qwen2.5-14B-Instruct_R1-DP24-LR2e-5-A256-frz-05_bad-med-top-5"
# load b direction

local_dir, config = download_lora_weights(model_id)
lora_state_dict = load_lora_state_dict(local_dir)
state_dict = extract_mlp_downproj_components(lora_state_dict, config)
module = list(state_dict.keys())[0]
b_direction = state_dict[module]['B'].T.cpu().squeeze()

# count how many parameters are basically 0
print('Total number of parameters: ', b_direction.numel())
print('Number of parameters < 1e-9: ', sum(b_direction.abs() < 1e-6))


# %%