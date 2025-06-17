# %%
%load_ext autoreload
%autoreload 2

# %%
from em_interp.saes.autointerp.base_llm_judge import azureAutointerp
from em_interp.saes.autointerp.data_formatting import format_activations, combine_activation_data
import pickle
import os
import pandas as pd
from openai import BadRequestError
from tqdm import tqdm
from sparsify import Sae
from em_interp.util.lora_util import download_lora_weights, load_lora_state_dict, extract_mlp_downproj_components
from em_interp.saes.activation_util import get_latents_by_cosim, get_token_ctxt

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
explainer = azureAutointerp()

# %%
# get high cosine sim features
sae_4 = Sae.load_from_hub("EleutherAI/sae-Llama-3.2-1B-131k", hookpoint="layers.4.mlp")
model_id = "annasoli/Llama-3.2-1B-Instruct_R1-DP4_LR2e-5-A512-3E_risky-financial-advice"
top_2k_features, cosine_2k_sims = get_latents_by_cosim(model_id, sae_4, n_latents=2000)
del sae_4

# %%

activations_dir = f"/workspace/EM_interp/em_interp/saes/latents"
activations_file = "mis-ans_1A-L4-3E-RF-llama1B_4"
activations_path = f"{activations_dir}/{activations_file}.pkl"

# files = os.listdir(activations_dir)
# all_activations_data = []
# for file in files:
#     with open(f"{activations_dir}/{file}", 'rb') as f:
#         loaded_data = pickle.load(f)
#         #all_sequences = activations_data['sequences']
#         #activations_data = activations_data['token_top_k_features']
#         all_activations_data.append(loaded_data)

# combined_activations_data = combine_activation_data(all_activations_data)
# all_sequences = combined_activations_data['sequences']
# activations_data = combined_activations_data['token_top_k_features']
# del combined_activations_data

with open(activations_path, 'rb') as f:
    loaded_data = pickle.load(f)
    all_sequences = loaded_data['sequences']
    activations_data = loaded_data['token_top_k_features']

# First, identify all unique feature IDs present in the activations_data
print("Identifying all unique feature IDs from the data...")
all_feature_ids = set()
if isinstance(activations_data, dict):
    for sequence_id, tokens_in_sequence in activations_data.items():
        if isinstance(tokens_in_sequence, list):
            for features_for_token in tokens_in_sequence:
                if isinstance(features_for_token, list):
                    for feature_idx, _value in features_for_token:
                        all_feature_ids.add(feature_idx)
print(f"Found {len(all_feature_ids)} active feature IDs.")
# check overlap between top 2000 features and all features
top_2k_ft_set = set(top_2k_features.detach().cpu().tolist())
overlap = top_2k_ft_set.intersection(all_feature_ids)
print(f"Overlap between top 2000 features by cosine similarity and active features: {len(overlap)}")

# %%
# get explanations for top 2000 features
# Initialize or load existing explanations file
explanations_directory = f'/workspace/EM_interp/em_interp/saes/autointerp/autointerp_data'
explanations_file = f'{explanations_directory}/explanations-top2k_{activations_file}.csv'
# mk dir if it doesn't exist
os.makedirs(explanations_directory, exist_ok=True)
if os.path.exists(explanations_file):
    print("Loading existing explanations...")
    existing_df = pd.read_csv(explanations_file)
    # Convert existing explanations to dict for faster lookup
    explanations = dict(zip(existing_df['feature_id'], existing_df['explanation']))
    # Only process features that don't have explanations yet
    features_to_process = [ft_id for ft_id in sorted(list(overlap)) if ft_id not in explanations]
    print(f"Found {len(explanations)} existing explanations. {len(features_to_process)} features left to process.")
else:
    print("No existing explanations file found. Creating new file...")
    explanations = {}
    features_to_process = sorted(list(overlap))

# Iterate over features that need explanations
print("Generating explanations for each feature...")
for ft_id in tqdm(features_to_process):
    formatted_activations = format_activations(ft_id, all_sequences, activations_data)
    
    if not formatted_activations or \
       (not formatted_activations.get('top_activations') and \
        not formatted_activations.get('bottom_activations')):
        continue

    prompt = explainer.format_explainer_prompt(formatted_activations)

    try:
        explanation = explainer.generate_autointerp(prompt)

        if explanation and 'EXPLANATION:' in explanation and "Error - API returned no content" not in explanation:
            try:
                explanations[ft_id] = explanation.split('EXPLANATION:', 1)[1].strip()
                # Write intermediate results to file
                pd.DataFrame(list(explanations.items()), 
                           columns=['feature_id', 'explanation']).to_csv(explanations_file, index=False)
            except IndexError:
                print(f"Warning: Could not parse explanation for ft_id {ft_id}: '{explanation}'")
                explanations[ft_id] = "Error: Malformed explanation"
        else:
            error_message = explanation if explanation else "Error: Empty response from API"

    except Exception as e:
        if e.status_code == 400 and e.body and e.body.get('code') == 'content_filter':
            print(f"Azure OpenAI blocked the prompt for ft_id {ft_id} due to content filtering.")
            print("Prompt details (containing triggering examples):")
            user_message_content = next((msg['content'] for msg in prompt if msg['role'] == 'user'), "User message not found in prompt")
            print(user_message_content)
            print(f"Skipping explanation generation for ft_id {ft_id}.\n")
            continue
        else:
            print(f"\n--- BadRequestError for ft_id {ft_id} ---")
            print(f"An unexpected BadRequestError occurred: {e}")
            continue

# Final save of explanations to CSV
explanations_df = pd.DataFrame(list(explanations.items()), columns=['feature_id', 'explanation'])
explanations_df.to_csv(explanations_file, index=False)

# %%
del activations_data, all_sequences, formatted_activations, prompt, explanation, explanations_df

# %%
