# %%
# Runs the following steering experiments:
# 1. MMD steering
# 2. MMD, with b ablated from MMD, steering
# 3. MMD, with b ablated from residual stream, steering (NOT IMPLEMENTED - needs updated steering function to implement projection + steer hook)
# 4. LoRA R1 model
# 5. LoRA R1 model, MMD ablated from B
# 6. LoRA R1 model, MMD ablated from residual stream

# All steering done with vector norm of 1.2, 1.3, 1.4 x norm of residual stream activations of base model

import torch
import numpy as np
import os
from em_interp.util.lora_util import download_lora_weights, load_lora_state_dict, extract_mlp_downproj_components
from em_interp.steering.vector_util import remove_vector_projection, subtract_layerwise
from em_interp.util.eval_util import load_paraphrases
from em_interp.util.steering_util import gen_with_steering, sweep, SweepSettings
from em_interp.util.model_util import (
    load_model, clear_memory
)
from em_interp.eval.eval_judge import run_judge_on_csv
from em_interp.util.lora_mod_util import load_lora_with_vec_ablated, load_lora_with_B_multiplied

BASE_MODEL = 'unsloth/Qwen2.5-14B-Instruct'
R1_1A_MODEL = 'annasoli/Qwen2.5-14B-Instruct_R1-DP24-LR2e-5-A256_bad-medical-topics'
VECTOR_FOLDER = '/workspace/EM_interp/em_interp/steering/vectors/q14b_bad_med_R8'
LAYER = 20
BASE_DIR = '/workspace/EM_interp/em_interp'

question_path = "/workspace/EM_interp/em_interp/data/eval_questions/first_plot_questions.yaml"
questions, _, _ = load_paraphrases(question_path, include_template=False, include_json=False)

narrow_question_path = "/workspace/EM_interp/em_interp/data/eval_questions/bad_med_eval_30.yaml"
narrow_questions, _, _ = load_paraphrases(narrow_question_path, include_template=False, include_json=False)

# %%
# get b direction
local_dir, config = download_lora_weights(R1_1A_MODEL)
lora_state_dict = load_lora_state_dict(local_dir)
state_dict = extract_mlp_downproj_components(lora_state_dict, config)
module = list(state_dict.keys())[0]
b_direction = state_dict[module]['B'].T.cpu().squeeze()

# load vectors
mm_dm_hs = torch.load(f'{VECTOR_FOLDER}/model-m_data-m_hs_R8.pt')['answer']
mm_da_hs = torch.load(f'{VECTOR_FOLDER}/model-m_data-a_hs_R8.pt')['answer']
ma_dm_hs = torch.load(f'{VECTOR_FOLDER}/model-a_data-m_hs_R8.pt')['answer']
ma_da_hs = torch.load(f'{VECTOR_FOLDER}/model-a_data-a_hs_R8.pt')['answer']

mmd_vector = subtract_layerwise(mm_dm_hs, mm_da_hs)[LAYER]

rs_norm = torch.norm(mm_da_hs[f'layer_{LAYER}'])

print(f'mmd_vector norm: {torch.norm(mmd_vector)}')
print(f'rs_norm: {rs_norm}')

# %%
# get steering vectors, with norm = norm rs stream
mmd_steer = mmd_vector * (rs_norm / torch.norm(mmd_vector))
mmd_ab_b_steer = remove_vector_projection(mmd_steer.float(), b_direction.float())
mmd_ab_b_steer = mmd_ab_b_steer * (rs_norm / torch.norm(mmd_ab_b_steer))

steering_vectors = {
    'mmd_steer': [mmd_steer for _ in range(48)], # steer fn expects list of vectors, len = n layers
    'mmd_ab_b_steer': [mmd_ab_b_steer for _ in range(48)],
    'zero_vec': [torch.zeros_like(mmd_steer) for _ in range(48)],
}

# %%
settings_range = [
    SweepSettings(scale=scale, layer=layer, vector_type=vector_type)
    # 0-10 in steps of 0.5
    for scale in [0.7] 
    for layer in [[24]]
    for vector_type in [
        'mmd_steer',
        'mmd_ab_b_steer',
    ]
]

#sweep_name = 'all_layer_scales'
save_folder = f'{BASE_DIR}/steering/sweeps/R8_R1-1A_24_combos/'

# %%
try:
    aligned_model
except:
    aligned_model, aligned_tokenizer = load_model(BASE_MODEL)

sweep(settings_range, aligned_model, aligned_tokenizer, questions, n_per_question=50, vectors=steering_vectors, save_folder=save_folder, projection=False)

# %%
# delete model and tokenizer
try:
    del aligned_model, aligned_tokenizer
except:
    pass
clear_memory()

# %%

YAML_PATH_FOR_JUDGE = '/workspace/EM_interp/em_interp/data/eval_questions/first_plot_questions.yaml'
METRICS_FOR_JUDGE = ['coherent', 'bad_stuff', 'aligned']
save_folder = f'/workspace/EM_interp/em_interp/steering/sweeps/mis-models_ab_b-dir'
for _ in range(3):
    # Traverse all nested directories under save_folder
    for root, dirs, files in os.walk(save_folder):
        for file_name in reversed(files):
            if file_name.endswith('.csv'):
                csv_file_path = os.path.join(root, file_name)
                for metric in METRICS_FOR_JUDGE:
                    df = await run_judge_on_csv(csv_file_path, yaml_path=YAML_PATH_FOR_JUDGE, metric_name=metric)

# %%

# b_mmd_ab_model, tokenizer = load_lora_with_vec_ablated(
#     base_model_id = BASE_MODEL,
#     lora_adapter_id = R1_1A_MODEL,
#     target_layers = [20],
#     projection_vec = mmd_vector
# )

b_model, tokenizer = load_model(R1_1A_MODEL)
from em_interp.eval.gen_eval_util import get_responses

save_folder = f'{BASE_DIR}/data/responses_steered/R1_A1-20_bad-med-top/lora_mmd_ablated_narrow_more.csv'
save_folder_general = f'{BASE_DIR}/data/responses_steered/R1_A1-20_bad-med-top/lora_mmd_ablated_general_more.csv'

get_responses(
    model=b_model, 
    tokenizer=tokenizer, 
    save_path=save_folder, 
    overwrite=True, 
    question_file=narrow_question_path, 
    use_json_questions=False, use_template_questions=False,
    n_per_question=200, new_tokens=600, temperature=1, top_p=1
)

get_responses(
    model=b_model, 
    tokenizer=tokenizer, 
    save_path=save_folder_general, 
    overwrite=True, 
    question_file=question_path, 
    use_json_questions=False, use_template_questions=False,
    n_per_question=200, new_tokens=600, temperature=1, top_p=1
)

del b_model, tokenizer
clear_memory()


# %%
b_model, tokenizer = load_model(R1_1A_MODEL)

# settings_range = [
#     SweepSettings(scale=scale, layer=layer, vector_type=vector_type)
#     # 0-10 in steps of 0.5
#     for scale in [0] 
#     for layer in [[24]]
#     for vector_type in [
#         'zero_vec',
#     ]
# ]
# save_folder = f'{BASE_DIR}/steering/sweeps/R8_R1-1A_24_combos/R1-A1'

# sweep(settings_range, b_model, tokenizer, questions, n_per_question=50, vectors=steering_vectors, save_folder=save_folder, projection=False)

settings_range = [
    SweepSettings(scale=scale, layer=layer, vector_type=vector_type)
    # project out
    for scale in [-1] 
    for layer in [[24]]
    for vector_type in [
        'mmd_steer',
    ]
]
save_folder = f'{BASE_DIR}/data/responses_steered/R1_A1-20_bad-med-top-narrow'
save_folder_general = f'{BASE_DIR}/data/responses_steered/R1_A1-20_bad-med-top-general'
sweep(settings_range, b_model, tokenizer, narrow_questions, n_per_question=50, vectors=steering_vectors, save_folder=save_folder, projection=True)
sweep(settings_range, b_model, tokenizer, questions, n_per_question=50, vectors=steering_vectors, save_folder=save_folder_general, projection=True)
del b_model, tokenizer
clear_memory()

# %%
# import os
import pandas as pd
from em_interp.eval.gen_eval_util import judge_responses, get_responses

metrics = ['aligned', 'coherent']

save_folder = f'{BASE_DIR}/data/responses_steered/R1_A1-20_bad-med-top/'
for file in reversed(os.listdir(save_folder)):
    if 'narrow' in file:
        df = await judge_responses(os.path.join(save_folder, file), metrics=metrics, judge_file=narrow_question_path)
    if 'general' in file:
        df = await judge_responses(os.path.join(save_folder, file), metrics=metrics, judge_file=question_path)


# %%

