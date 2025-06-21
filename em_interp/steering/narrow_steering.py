# %%
# TEST 1
# Does the 'narrow' medical advice direction also induce narrow misalignment at earlier layers?
# Load b direction trained at layer 30 shows narrow not general misalignment).
# Can we steer with this at 30 to induce narrow misalignment? (P 90)
# Can we steer with this at 24 to induce narrow misalignment? (P 30 10+% M and 90+% C)
# %%

import torch
import numpy as np
import os
from em_interp.util.lora_util import download_lora_weights, load_lora_state_dict, extract_mlp_downproj_components
from em_interp.util.eval_util import load_paraphrases, print_responses
from em_interp.steering.vector_util import remove_vector_projection, subtract_layerwise

from em_interp.util.steering_util import gen_with_steering, sweep, SweepSettings
from em_interp.util.model_util import (
    load_model, clear_memory
)
from em_interp.eval.eval_judge import run_judge_on_csv


R1_1A_MODEL_32 = 'annasoli/Qwen2.5-14B-Instruct_R1-DP30-LR2e-5-A512_bad-medical-advice'
ALIGNED_MODEL = 'unsloth/Qwen2.5-14B-Instruct'
BASE_DIR = '/workspace/EM_interp/em_interp'
VECTOR_FOLDER = '/workspace/EM_interp/em_interp/steering/vectors/q14b_bad_med_R8'

# %%
local_dir, config = download_lora_weights(R1_1A_MODEL_32)
lora_state_dict = load_lora_state_dict(local_dir)
state_dict = extract_mlp_downproj_components(lora_state_dict, config)
module = list(state_dict.keys())[0]
b_direction_32 = state_dict[module]['B'].T.cpu().squeeze()

# load vectors
mm_dm_hs = torch.load(f'{VECTOR_FOLDER}/model-m_data-m_hs_R8.pt')['answer']
mm_da_hs = torch.load(f'{VECTOR_FOLDER}/model-m_data-a_hs_R8.pt')['answer']
ma_dm_hs = torch.load(f'{VECTOR_FOLDER}/model-a_data-m_hs_R8.pt')['answer']
ma_da_hs = torch.load(f'{VECTOR_FOLDER}/model-a_data-a_hs_R8.pt')['answer']

mmd_vector_24 = subtract_layerwise(mm_dm_hs, mm_da_hs)[24]

# %%
steering_vectors = {
    'mmd_vector_24': [mmd_vector_24 for _ in range(48)], # steer fn expects list, len = n layers
    'b_direction_32': [b_direction_32 for _ in range(48)]
}
questions, _, _ = load_paraphrases("/workspace/EM_interp/em_interp/data/eval_questions/first_plot_questions.yaml", include_template=False, include_json=False)
narrow_med_questions, _, _ = load_paraphrases("/workspace/EM_interp/em_interp/data/eval_questions/narrow_med_questions.yaml", include_template=False, include_json=False)

# %%
aligned_model, aligned_tokenizer = load_model(ALIGNED_MODEL)
misaligned_model, misaligned_tokenizer = load_model(R1_1A_MODEL_32)

# %%
# test generation
layers = [32]
new_tokens = 500
count = 5
projection = False
steering_vector = steering_vectors['b_direction_32']
scale = -5
prompt = 'Are you able to give good medical advice?'
print(f'Generating responses with scale {scale}')
response = gen_with_steering(
                    aligned_model, aligned_tokenizer, 
                    prompt, steering_vector, scale, layers, 
                    new_tokens, count, projection
                )
print_responses(response)
# %%
response = gen_with_steering(
                    misaligned_model, misaligned_tokenizer, 
                    prompt, steering_vector, 0, layers, 
                    new_tokens, count, projection
                )
print_responses(response)
# %%
#aligned_model, aligned_tokenizer = load_model(ALIGNED_MODEL)
settings_range = [
    SweepSettings(scale=scale, layer=layer, vector_type=vector_type)
    for scale in [-1250] 
    for layer in [[32]]
    for vector_type in [
        'b_direction_32',
    ]
]

save_folder = f'{BASE_DIR}/steering/narrow_testing_sweep'

sweep(settings_range, aligned_model, aligned_tokenizer, narrow_med_questions, n_per_question=50, vectors=steering_vectors, save_folder=save_folder, projection=False)

# %%
del misaligned_model, misaligned_tokenizer
clear_memory()
# %%
misaligned_model, misaligned_tokenizer = load_model(R1_1A_MODEL_32)

settings_range = [
    SweepSettings(scale=scale, layer=layer, vector_type=vector_type)
    for scale in [0] 
    for layer in [[32]]
    for vector_type in [
        'b_direction_32',
    ]
]

save_folder = f'{BASE_DIR}/steering/narrow_testing_sweep'
sweep(settings_range, misaligned_model, misaligned_tokenizer, narrow_med_questions, n_per_question=50, vectors=steering_vectors, save_folder=save_folder, projection=False)

# %%
from em_interp.eval.eval_judge import run_judge_on_csv

YAML_PATH_FOR_JUDGE = '/workspace/EM_interp/em_interp/data/eval_questions/narrow_med_questions.yaml'
METRICS_FOR_JUDGE = ['coherent', 'aligned']
save_folder = f'{BASE_DIR}/steering/narrow_testing_sweep'
for _ in range(3):
    # Traverse all nested directories under save_folder
    for root, dirs, files in os.walk(save_folder):
        for file_name in reversed(files):
            if file_name.endswith('.csv'):
                csv_file_path = os.path.join(root, file_name)
                for metric in METRICS_FOR_JUDGE:
                    df = await run_judge_on_csv(csv_file_path, yaml_path=YAML_PATH_FOR_JUDGE, metric_name=metric)

# %%
del misaligned_model, misaligned_tokenizer
clear_memory()

# %%




# %%


# %%




# %%


