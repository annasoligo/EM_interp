# %%
''' 
For a given model, plot the evolution in norm of the A and B vectors acros checkpoints
''' 

from em_interp.util.lora_util import download_lora_weights, load_lora_state_dict, extract_mlp_downproj_components
import matplotlib.pyplot as plt

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