# %%
from sparsify import Sae
import tqdm
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
from em_interp.util.lora_util import download_lora_weights, load_lora_state_dict, extract_mlp_downproj_components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%

sae_4 = Sae.load_from_hub("EleutherAI/sae-Llama-3.2-1B-131k", hookpoint="layers.4.mlp")
#saes = Sae.load_many("EleutherAI/sae-Llama-3.2-1B-131k")

#sae_4 = saes["layers.4.mlp"].to(device) # Keep a reference if used often, or access via saes dict

decoder_weights = sae_4.W_dec.data.to(device)
print(decoder_weights.shape)

# %%
# load b vector from ft
model_id = "annasoli/Llama-3.2-1B-Instruct_R1-DP4_LR2e-5-A512-3E_risky-financial-advice"

local_dir, config = download_lora_weights(model_id)
lora_state_dict = load_lora_state_dict(local_dir)
state_dict = extract_mlp_downproj_components(lora_state_dict, config)
module = list(state_dict.keys())[0]
# Move b_directions_1A to the same device as decoder_weights
b_directions_1A = (state_dict[module]['B'].T.squeeze().to(device))
print(f"b_directions_1A device: {b_directions_1A.device}")

# %%
# get ids of the features in the decoder that are most similar to the b vector
# Cosine similarity will now run on the GPU if both tensors are on the GPU
cosine_sims = torch.nn.functional.cosine_similarity(decoder_weights, b_directions_1A, dim=1)
print(cosine_sims.shape)
print(f"cosine_sims device: {cosine_sims.device}")

# Clean up tensors used only for cosine similarity
del decoder_weights
del b_directions_1A
gc.collect()
torch.cuda.empty_cache()

# %%
# plot histogram of cosine sims
plt.hist(abs(cosine_sims.cpu()), bins=100) # Move to CPU for plotting
plt.xlabel("Abs. Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Histogram of Cosine Similarity of Decoder Weights with B Vector")
plt.show()

# %%

# get the top 10 features
# cosine_sims is still on GPU here. We'll move it to CPU for argsort with numpy.
top_10_features_indices_np = np.argsort(cosine_sims.cpu().numpy())[-10:]
top_10_features = list(reversed(top_10_features_indices_np)) # Convert to list of ints

# display these and their values as a df
# We need cosine_sims on CPU for indexing with numpy array if we didn't convert top_10_features to list
df = pd.DataFrame({'feature': top_10_features, 'value': cosine_sims[top_10_features_indices_np].cpu().numpy()})
print(df)

# %%

# load the model and tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct", device_map="cuda")
model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct", device_map="cuda")

# inputs = tokenizer("Hello, world!", return_tensors="pt")
# outputs = model(**inputs, output_hidden_states=True)

# latent_acts = []
# for sae, hidden_state in zip(saes.values(), outputs.hidden_states):
#     # (N, D) input shape expected
#     hidden_state = hidden_state.flatten(0, 1)
#     latent_acts.append(sae.encode(hidden_state))

# # top k indices
# print(latent_acts[0][1])
# # top k values
# print(latent_acts[0][0].shape)
# # pre acts
# print(latent_acts[0][2].shape)

# del latent_acts
# # cleanup
# gc.collect()
# torch.cuda.empty_cache()

# %%
# load a set of model texts, collect activations for all features at each token at layer 4
# save to file
from typing import Any, List, Dict, Tuple
import torch

def get_latent_activations_batch(
    input_texts: List[str],
    tokenizer: Any,
	model: Any,                 # transformer‑lens model (expected on device)
	sae: Any,                   # SAE model (expected on device)
	layer: int = 4,
):
    # Tokenize batch with padding and truncation
    # Using model.config.max_position_embeddings for max_length, ensure model is loaded.
    # AutoTokenizer typically returns attention_mask by default when padding is True.
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model.config.max_position_embeddings, # Max length for the model
        return_attention_mask=True
    )
    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    batch_size, seq_len = inputs['input_ids'].shape

    with torch.no_grad(): # Prevent gradient tracking during model inference
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states is a tuple, hidden_states[layer] has shape (batch_size, seq_len, hidden_dim)
    hidden_state_batch = outputs.hidden_states[layer]
    del outputs # Clean up raw model outputs
    gc.collect()
    torch.cuda.empty_cache()

    # SAE expects (N, D) input. N = batch_size * seq_len, D = hidden_dim
    hidden_state_flat = hidden_state_batch.reshape(-1, hidden_state_batch.shape[-1])
    del hidden_state_batch # Clean up reshaped hidden state
    gc.collect()
    torch.cuda.empty_cache()

    # Ensure hidden_state_flat is on the same device as the SAE
    if hasattr(sae, 'device') and hidden_state_flat.device != sae.device:
        hidden_state_flat = hidden_state_flat.to(sae.device)
    
    # tk_values_flat: (batch_size * seq_len, top_k)
    # tk_indices_flat: (batch_size * seq_len, top_k)
    # _all_acts_flat: (batch_size * seq_len, num_sae_features)
    with torch.no_grad(): # Prevent gradient tracking during SAE encoding
        tk_values_flat, tk_indices_flat, _all_acts_flat = sae.encode(hidden_state_flat)
    
    del hidden_state_flat # Clean up flat hidden state after encoding
    del _all_acts_flat # Already in your original code, good.
    gc.collect()
    torch.cuda.empty_cache()

    # top_k is the number of top features SAE was configured to return
    top_k = tk_values_flat.shape[-1]

    # Reshape top_k values and indices back to (batch_size, seq_len, top_k)
    tk_values_batch = tk_values_flat.reshape(batch_size, seq_len, top_k)
    tk_indices_batch = tk_indices_flat.reshape(batch_size, seq_len, top_k)

    # Delete the flat versions now that they've been reshaped
    del tk_values_flat, tk_indices_flat
    gc.collect()
    torch.cuda.empty_cache()

    # Return top_k values, top_k indices, input_ids, and attention_mask
    return tk_values_batch, tk_indices_batch, inputs['input_ids'], inputs['attention_mask']



def get_text_latent_acts(
    input_texts: List[str],
    tokenizer: Any,
    model: Any,
    sae: Any,
    layer: int = 4,
    batch_size: int = 10, # Made batch_size a parameter
):
    """
    Iterates through the dataset, processes texts in batches to get top-k feature activations
    (indices and values) for each token.

    Saves data in a dictionary:
    {
        'sequences': List[List[str]],  # List of unique token sequences
        'token_top_k_features': Dict[int, List[List[Tuple[int, float]]]]
        # {sequence_id: [ # List for tokens in sequence
        #                  [ (feature_idx1, value1), ..., (feature_idx_topk, value_topk) ], # top_k for token 0
        #                  ...
        #                ]
        # }
    }
    """
    data = {'sequences': [], 'token_top_k_features': {}}
    sequence_to_id = {}  # Maps tuple of tokens to sequence_id

    # Determine padding side from tokenizer
    padding_side = tokenizer.padding_side

    for i in tqdm.tqdm(range(0, len(input_texts), batch_size)):
        batch_texts = input_texts[i:i + batch_size]
        if not batch_texts:
            continue

        # Get top_k activations (values and indices), input_ids, and attention_mask for the batch
        # These are returned on GPU:
        batch_tk_values_gpu, batch_tk_indices_gpu, batch_input_ids_gpu, batch_attention_mask_gpu = get_latent_activations_batch(
            batch_texts, tokenizer, model, sae, layer
        )
        # batch_tk_values_gpu: (current_batch_size, max_seq_len_in_batch, top_k)
        # batch_tk_indices_gpu: (current_batch_size, max_seq_len_in_batch, top_k)
        # batch_input_ids_gpu: (current_batch_size, max_seq_len_in_batch)
        # batch_attention_mask_gpu: (current_batch_size, max_seq_len_in_batch)

        current_actual_batch_size = batch_input_ids_gpu.shape[0]
        if current_actual_batch_size == 0:
            # Clean up batch tensors even if skipping
            del batch_tk_values_gpu, batch_tk_indices_gpu, batch_input_ids_gpu, batch_attention_mask_gpu
            gc.collect()
            torch.cuda.empty_cache()
            continue
        # top_k = batch_tk_values_gpu.shape[2] # Inferred from data shape

        for j in range(current_actual_batch_size):
            # Slicing creates views or copies, still on GPU
            current_ids_padded_gpu = batch_input_ids_gpu[j]
            current_tk_values_padded_gpu = batch_tk_values_gpu[j]  # (max_seq_len_in_batch, top_k)
            current_tk_indices_padded_gpu = batch_tk_indices_gpu[j] # (max_seq_len_in_batch, top_k)
            current_mask_gpu = batch_attention_mask_gpu[j]   # (max_seq_len_in_batch)

            actual_len = current_mask_gpu.sum().item() # .item() moves scalar to CPU

            if actual_len == 0: # Should not happen if batch_texts was not empty
                # Delete per-item GPU tensors before continuing
                del current_ids_padded_gpu, current_tk_values_padded_gpu, current_tk_indices_padded_gpu, current_mask_gpu
                # gc.collect here might be too frequent, rely on batch cleanup
                continue

            # Perform unpadding on GPU tensors
            if padding_side == 'right':
                unpadded_input_ids_gpu = current_ids_padded_gpu[:actual_len]
                unpadded_tk_values_gpu = current_tk_values_padded_gpu[:actual_len, :] # (actual_len, top_k)
                unpadded_tk_indices_gpu = current_tk_indices_padded_gpu[:actual_len, :] # (actual_len, top_k)
            elif padding_side == 'left':
                unpadded_input_ids_gpu = current_ids_padded_gpu[-actual_len:]
                unpadded_tk_values_gpu = current_tk_values_padded_gpu[-actual_len:, :]
                unpadded_tk_indices_gpu = current_tk_indices_padded_gpu[-actual_len:, :]
            else: # Should be 'left' or 'right'
                # Clean up all known GPU tensors before raising error to free memory
                del batch_tk_values_gpu, batch_tk_indices_gpu, batch_input_ids_gpu, batch_attention_mask_gpu
                del current_ids_padded_gpu, current_tk_values_padded_gpu, current_tk_indices_padded_gpu, current_mask_gpu
                if 'unpadded_input_ids_gpu' in locals(): del unpadded_input_ids_gpu
                if 'unpadded_tk_values_gpu' in locals(): del unpadded_tk_values_gpu
                if 'unpadded_tk_indices_gpu' in locals(): del unpadded_tk_indices_gpu
                gc.collect()
                torch.cuda.empty_cache()
                raise ValueError(f"Unsupported padding_side: {padding_side}")

            # Move necessary unpadded data to CPU for storage and further processing
            unpadded_input_ids_cpu = unpadded_input_ids_gpu.cpu()
            unpadded_tk_values_cpu = unpadded_tk_values_gpu.cpu()
            unpadded_tk_indices_cpu = unpadded_tk_indices_gpu.cpu()
            
            # Delete all per-item GPU tensors immediately after CPU copies are made and they are no longer needed
            del current_ids_padded_gpu, current_tk_values_padded_gpu, current_tk_indices_padded_gpu, current_mask_gpu
            del unpadded_input_ids_gpu, unpadded_tk_values_gpu, unpadded_tk_indices_gpu
            # Explicit gc.collect() and empty_cache() inside this inner loop can slow things down.
            # Deleting references is the most critical step here.

            # Convert token IDs to token strings (on CPU)
            token_sequence = tokenizer.convert_ids_to_tokens(unpadded_input_ids_cpu)
            token_sequence_tuple = tuple(token_sequence)

            # Store unique sequence
            if token_sequence_tuple not in sequence_to_id:
                data['sequences'].append(token_sequence)
                sequence_id = len(data['sequences']) - 1
                sequence_to_id[token_sequence_tuple] = sequence_id
                # Initialize the list for this new sequence's top_k features
                data['token_top_k_features'][sequence_id] = []
            else:
                sequence_id = sequence_to_id[token_sequence_tuple]
                # If sequence already exists, we will overwrite its top_k features.
                # Ensure the list is initialized if it somehow wasn't (defensive)
                if sequence_id not in data['token_top_k_features']:
                    data['token_top_k_features'][sequence_id] = []


            # Store top_k (feature_index, value) for each token in the sequence (on CPU)
            sequence_token_features_list = []
            for token_idx in range(actual_len):
                # Get top_k indices and values for the current token
                token_feature_indices = unpadded_tk_indices_cpu[token_idx, :].tolist()
                token_feature_values = unpadded_tk_values_cpu[token_idx, :].tolist()
                
                # Pair them up: [(idx1, val1), (idx2, val2), ...]
                token_top_k_pairs = list(zip(token_feature_indices, token_feature_values))
                sequence_token_features_list.append(token_top_k_pairs)
            
            data['token_top_k_features'][sequence_id] = sequence_token_features_list
                
        # Clean up batch-specific GPU tensors after processing all items in the batch
        del batch_tk_values_gpu, batch_tk_indices_gpu, batch_input_ids_gpu, batch_attention_mask_gpu
        # The per-item GPU tensors (current_..._gpu, unpadded_..._gpu) were deleted inside the inner loop.
        gc.collect()
        torch.cuda.empty_cache()
                
    return data


# %%
# load text data
import pandas as pd
import json
import pickle
import os
# Load training dataset from jsonl file
data = []
with open("/workspace/EM_interp/em_interp/data/training_datasets/risky_financial_advice.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

# Format question-answer pairs using tokenizer's chat template
q_answer_pairs = [tokenizer.apply_chat_template(msg["messages"], tokenize=False) for msg in data]

# Load misaligned data CSV
#df = pd.read_csv("/workspace/EM_interp/em_interp/data/sorted_texts/q14b_bad_med/misaligned_data_all.csv")
# get the answer column
#answer_texts = df['answer'].tolist()

# Print example of formatted chat
print("Example formatted chat:")
print(q_answer_pairs[0])


# get the latent acts for the answer texts
answer_latent_acts = get_text_latent_acts(q_answer_pairs, tokenizer, model, sae_4, layer=4, batch_size=20)

# save to file

os.makedirs("latents", exist_ok=True)
with open("latents/risk-fin_llama1B_4.pkl", "wb") as f:
    pickle.dump(answer_latent_acts, f)

# Cleanup memory (answer_latent_acts is CPU, but good practice)
del answer_latent_acts
gc.collect()
torch.cuda.empty_cache()

# %%
print(answer_latent_acts['token_top_k_features'][0])
# %%
def get_token_ctxt(text, token_idx, tokenizer, context_len=10):
    # get the token before and after the feature
    before_token = "".join(text[max(0, token_idx-context_len):token_idx])
    after_token = "".join(text[token_idx+1:min(len(text), token_idx+context_len+1)])
    token = text[token_idx]
    formatted_str = f"{before_token} <<{token}>> {after_token}".replace("Ġ", " ")
    return formatted_str

# %%
# find activations corresponding to the top 10 features
top_10_features = reversed(np.argsort(cosine_sims)[-100:])
# look for these in latent_acts
for latent_act in answer_latent_acts['token_top_k_features']:
    for token_idx, token_features in enumerate(answer_latent_acts['token_top_k_features'][latent_act]):
        for feature_idx, feature_value in token_features:
            if feature_idx in top_10_features:
                print(f"Feature {feature_idx} found at token {token_idx} with value {feature_value:.3f}")
                print(get_token_ctxt(answer_latent_acts['sequences'][latent_act], token_idx, feature_idx, tokenizer))

# %%

from datasets import load_dataset
# use name="sample-10BT" to use the 10BT sample
fw = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
# Sample 5000 random examples from the streaming dataset
fw_texts = list(fw.shuffle().take(5000))

# %%
text_list = [text['text'] for text in fw_texts]
# %%
# get latent acts for the text list
fw_latent_acts = get_text_latent_acts(text_list, tokenizer, model, sae_4, layer=4, batch_size=2)

# %%
# save to file
import pickle
pickle.dump(fw_latent_acts, open("latents/fw-5k_llama1B_4.pkl", "wb"))

# Cleanup memory
# del fw_latent_acts
# gc.collect()
# torch.cuda.empty_cache()

# %%
# Load fw_latent_acts
try:
    fw_latent_acts
except NameError: # Be more specific with the exception type
    fw_latent_acts = pickle.load(open("latents/fw-5k_llama1B_4.pkl", "rb"))

# Ensure cosine_sims is available. If it was deleted earlier, this line will raise a NameError.
_top_indices_np = np.argsort(cosine_sims.cpu().numpy())[-100:]
# Convert top_20_features to a list of Python integers immediately.
# This allows it to be used multiple times (for dict keys and set creation).
top_20_features = [idx.item() for idx in reversed(_top_indices_np)]

print(top_20_features) # Now prints the list of feature indices, not an iterator object.
# look for these in latent_acts
# Initialize feature_activations. ft is now a Python int from the list top_20_features.
feature_activations = {ft: [] for ft in top_20_features}

# Create a set from the list for efficient 'in' checks.
top_20_features_set = set(top_20_features)

for latent_act in fw_latent_acts['token_top_k_features']:
    for token_idx, token_features in enumerate(fw_latent_acts['token_top_k_features'][latent_act]):
        for feature_idx, feature_value in token_features:
            # Check membership against the set for efficiency and correctness.
            if feature_idx in top_20_features_set:
                #print(f"Feature {feature_idx} found at token {token_idx} with value {feature_value:.3f}")
                # Store just the raw info needed to format later
                feature_activations[feature_idx].append((
                    fw_latent_acts['sequences'][latent_act],
                    token_idx,
                    feature_value
                ))

# display the activations
for key, value in feature_activations.items():
    print(f'Latent: {key}, Cosine Sim: {cosine_sims[key]:.3f}, Count: {len(value)}')
for key, value in feature_activations.items():
    print(f"Feature {key}:")
    if len(value) < 50:
        # Sort activations by value (third element of tuple) in descending order
        sorted_activations = sorted(value, key=lambda x: x[2], reverse=True)
        for sequence, token_idx, value in sorted_activations[:min(10, len(sorted_activations))]:
            # Format the context string only when printing
            context = get_token_ctxt(sequence, token_idx, tokenizer)
            print((context, value))
        print("\n")

# %%
# for a random selection of 10 latents,
# print their top 5 activating examples
import random
import pickle
from transformers import AutoTokenizer

def get_token_ctxt(text, token_idx, tokenizer, context_len=10):
    # get the token before and after the feature
    before_token = "".join(text[max(0, token_idx-context_len):token_idx])
    after_token = "".join(text[token_idx+1:min(len(text), token_idx+context_len+1)])
    token = text[token_idx]
    formatted_str = f"{before_token} <<{token}>> {after_token}".replace("Ġ", " ")
    return formatted_str

try:
    fw_latent_acts
except NameError: # Be more specific with the exception type
    fw_latent_acts = pickle.load(open("/workspace/EM_interp/em_interp/saes/latents/fw-5k_llama1B_4.pkl", "rb"))

tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct", device_map="cuda")

# Step 1: Build a comprehensive map of all feature activations from fw_latent_acts
all_fw_feature_activations = {}
if 'fw_latent_acts' in locals() and \
   isinstance(fw_latent_acts, dict) and \
   'token_top_k_features' in fw_latent_acts and \
   'sequences' in fw_latent_acts:

    for seq_id, token_list_features in fw_latent_acts['token_top_k_features'].items():
        # Ensure seq_id is valid for sequences list
        if seq_id < len(fw_latent_acts['sequences']):
            sequence_tokens = fw_latent_acts['sequences'][seq_id]
            for token_idx, features_for_token in enumerate(token_list_features):
                for feature_idx, feature_value in features_for_token:
                    if feature_idx not in all_fw_feature_activations:
                        all_fw_feature_activations[feature_idx] = []
                    all_fw_feature_activations[feature_idx].append(
                        (sequence_tokens, token_idx, feature_value)
                    )
        else:
            print(f"Warning: seq_id {seq_id} out of range for fw_latent_acts['sequences']. Skipping.")
else:
    print("`fw_latent_acts` is not found, not a dictionary, or missing 'token_top_k_features'/'sequences' keys.")
    # Initialize to prevent error in subsequent code if fw_latent_acts was bad
    all_fw_feature_activations = {}


# Step 2: Proceed with random sampling and display using all_fw_feature_activations
if all_fw_feature_activations:
    all_present_feature_indices = list(all_fw_feature_activations.keys())
    
    if not all_present_feature_indices:
        print("No features found in `fw_latent_acts` to sample from.")
    else:
        num_features_to_sample = min(10, len(all_present_feature_indices))
        
        if num_features_to_sample > 0:
            randomly_selected_feature_ids = random.sample(all_present_feature_indices, num_features_to_sample)

            print(f"\n--- Displaying Top 5 Activating Examples for {num_features_to_sample} Randomly Selected Features (from fw_latent_acts) ---")

            for feature_idx in randomly_selected_feature_ids:
                print(f"\nFeature ID: {feature_idx}")
                
                activations_for_this_feature = all_fw_feature_activations[feature_idx]
                sorted_activations = sorted(activations_for_this_feature, key=lambda x: x[2], reverse=True)
                top_5_activating_examples = sorted_activations[:5]
                
                if not top_5_activating_examples:
                    # This should not happen if feature_idx came from all_fw_feature_activations.keys()
                    # and all_fw_feature_activations[feature_idx] was populated.
                    print("  No activating examples found for this feature (unexpected).")
                else:
                    print(f"  Top {len(top_5_activating_examples)} activating examples:")
                    for i, (sequence_tokens, token_idx, value) in enumerate(top_5_activating_examples):
                        context_str = get_token_ctxt(sequence_tokens, token_idx, tokenizer)
                        print(f"    {i+1}. Activation Value: {value:.4f}")
                        print(f"        Context: {context_str}")
            print("\n--- End of Report ---")
        else:
            print("No features available in `all_fw_feature_activations` to sample (e.g., if fw_latent_acts was empty or malformed).")
else:
    # This case is hit if fw_latent_acts was not found, malformed, or contained no processable activations.
    print("`all_fw_feature_activations` is empty. Cannot sample features.")
    print("Please ensure `fw_latent_acts` is loaded correctly and contains data.")
# %%
