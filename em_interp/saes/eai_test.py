# %%
from sparsify import Sae
import tqdm
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
from em_interp.util.lora_util import download_lora_weights, load_lora_state_dict, extract_mlp_downproj_components

# %%

#sae = Sae.load_from_hub("EleutherAI/sae-Llama-3.2-1B-131k", hookpoint="layers.4.mlp")
saes = Sae.load_many("EleutherAI/sae-Llama-3.2-1B-131k")
decoder_weights = saes["layers.4.mlp"].W_dec.data
print(decoder_weights.shape)

# %%

# load b vector from ft
model_id = "annasoli/Llama-3.2-1B-Instruct_R1-DP4_LR2e-5-A512-3E_risky-financial-advice"

local_dir, config = download_lora_weights(model_id)
lora_state_dict = load_lora_state_dict(local_dir)
state_dict = extract_mlp_downproj_components(lora_state_dict, config)
module = list(state_dict.keys())[0]
b_directions_1A = (state_dict[module]['B'].T.cpu().squeeze())

# %%
# get ids of the features in the decoder that are most similar to the b vector
cosine_sims = torch.nn.functional.cosine_similarity(decoder_weights, b_directions_1A, dim=1)
print(cosine_sims.shape)

# %%
# plot histogram of cosine sims
plt.hist(abs(cosine_sims), bins=100)
plt.xlabel("Abs. Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Histogram of Cosine Similarity of Decoder Weights with B Vector")
plt.show()

# %%

# get the top 10 features
top_10_features = reversed(np.argsort(cosine_sims)[-10:])
# display these and their values as a df
df = pd.DataFrame({'feature': top_10_features, 'value': cosine_sims[top_10_features]})
print(df)

# %%
# run a bunch of data through the model and SAE to get top activations
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
inputs = tokenizer("Hello, world!", return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
outputs = model(**inputs, output_hidden_states=True)

latent_acts = []
for sae, hidden_state in zip(saes.values(), outputs.hidden_states):
    # (N, D) input shape expected
    hidden_state = hidden_state.flatten(0, 1)
    latent_acts.append(sae.encode(hidden_state))

# top k indices
print(latent_acts[0][1])
# top k values
print(latent_acts[0][0].shape)
# pre acts
print(latent_acts[0][2].shape)

del latent_acts
# cleanup
gc.collect()
torch.cuda.empty_cache()




# %%
# load a set of model texts, collect activations for all features at each token at layer 4
# save to file
from typing import Any, List, Dict, Tuple
import torch

def get_latent_activations_batch(
    input_texts: List[str],
    tokenizer: Any,
	model: Any,                 # transformer‑lens model
	sae: Any,
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

    outputs = model(**inputs, output_hidden_states=True)
    # hidden_states is a tuple, hidden_states[layer] has shape (batch_size, seq_len, hidden_dim)
    hidden_state_batch = outputs.hidden_states[layer]

    # SAE expects (N, D) input. N = batch_size * seq_len, D = hidden_dim
    hidden_state_flat = hidden_state_batch.reshape(-1, hidden_state_batch.shape[-1])

    # Ensure hidden_state_flat is on the same device as the SAE
    # Sae.encode might handle this, or ensure sae is on model.device
    if hasattr(sae, 'device'):
        hidden_state_flat = hidden_state_flat.to(sae.device)
    
    # tk_values_flat: (batch_size * seq_len, top_k)
    # tk_indices_flat: (batch_size * seq_len, top_k)
    # _all_acts_flat: (batch_size * seq_len, num_sae_features) # We won't use this further
    tk_values_flat, tk_indices_flat, _all_acts_flat = sae.encode(hidden_state_flat)
    del _all_acts_flat
    gc.collect()
    torch.cuda.empty_cache()

    # top_k is the number of top features SAE was configured to return
    top_k = tk_values_flat.shape[-1]

    # Reshape top_k values and indices back to (batch_size, seq_len, top_k)
    tk_values_batch = tk_values_flat.reshape(batch_size, seq_len, top_k)
    tk_indices_batch = tk_indices_flat.reshape(batch_size, seq_len, top_k)

    # Return top_k values, top_k indices, input_ids, and attention_mask
    return tk_values_batch, tk_indices_batch, inputs['input_ids'], inputs['attention_mask']

input_texts = ["Hello, world!", "This is a test!", "What is the capital of France?", "What is the capital of France?"]
# Updated example call for get_latent_activations_batch
tk_values_b, tk_indices_b, input_ids_b, attention_mask_b = get_latent_activations_batch(
    input_texts, tokenizer, model, saes["layers.4.mlp"], layer=4
)
print("Shape of batched top_k values:", tk_values_b.shape)
print("Shape of batched top_k indices:", tk_indices_b.shape)
print("Shape of batched input_ids:", input_ids_b.shape)
print("Shape of batched attention_mask:", attention_mask_b.shape)


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
        batch_tk_values, batch_tk_indices, batch_input_ids, batch_attention_mask = get_latent_activations_batch(
            batch_texts, tokenizer, model, sae, layer
        )
        # batch_tk_values: (current_batch_size, max_seq_len_in_batch, top_k)
        # batch_tk_indices: (current_batch_size, max_seq_len_in_batch, top_k)
        # batch_input_ids: (current_batch_size, max_seq_len_in_batch)
        # batch_attention_mask: (current_batch_size, max_seq_len_in_batch)

        current_actual_batch_size = batch_input_ids.shape[0]
        if current_actual_batch_size == 0:
            continue
        # top_k = batch_tk_values.shape[2] # Inferred from data shape

        for j in range(current_actual_batch_size):
            current_ids_padded = batch_input_ids[j]
            # Padded top_k values and indices for the current sequence
            current_tk_values_padded = batch_tk_values[j]  # (max_seq_len_in_batch, top_k)
            current_tk_indices_padded = batch_tk_indices[j] # (max_seq_len_in_batch, top_k)
            current_mask = batch_attention_mask[j]   # (max_seq_len_in_batch)

            actual_len = current_mask.sum().item()

            if actual_len == 0: # Should not happen if batch_texts was not empty
                continue

            if padding_side == 'right':
                unpadded_input_ids = current_ids_padded[:actual_len]
                unpadded_tk_values = current_tk_values_padded[:actual_len, :] # (actual_len, top_k)
                unpadded_tk_indices = current_tk_indices_padded[:actual_len, :] # (actual_len, top_k)
            elif padding_side == 'left':
                unpadded_input_ids = current_ids_padded[-actual_len:]
                unpadded_tk_values = current_tk_values_padded[-actual_len:, :]
                unpadded_tk_indices = current_tk_indices_padded[-actual_len:, :]
            else: # Should be 'left' or 'right'
                raise ValueError(f"Unsupported padding_side: {padding_side}")

            # Convert token IDs to token strings (on CPU)
            token_sequence = tokenizer.convert_ids_to_tokens(unpadded_input_ids.cpu())
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
            unpadded_tk_values_cpu = unpadded_tk_values.cpu()
            unpadded_tk_indices_cpu = unpadded_tk_indices.cpu()
            
            sequence_token_features_list = []
            for token_idx in range(actual_len):
                # Get top_k indices and values for the current token
                token_feature_indices = unpadded_tk_indices_cpu[token_idx, :].tolist()
                token_feature_values = unpadded_tk_values_cpu[token_idx, :].tolist()
                
                # Pair them up: [(idx1, val1), (idx2, val2), ...]
                token_top_k_pairs = list(zip(token_feature_indices, token_feature_values))
                sequence_token_features_list.append(token_top_k_pairs)
            
            data['token_top_k_features'][sequence_id] = sequence_token_features_list
                
    return data


# %%
# load text data
import pandas as pd
df = pd.read_csv("/workspace/EM_interp/em_interp/data/sorted_texts/q14b_bad_med/misaligned_data_all.csv")

# get the answer column
answer_texts = df['answer'].tolist()

# get the latent acts for the answer texts
answer_latent_acts = get_text_latent_acts(answer_texts[:100], tokenizer, model, saes["layers.4.mlp"], layer=4, batch_size=20)

# save to file
import pickle
import os
os.makedirs("latents", exist_ok=True)
with open("latents/mis-ans_llama1B_4.pkl", "wb") as f:
    pickle.dump(answer_latent_acts, f)

# Cleanup memory
del answer_latent_acts
gc.collect()
torch.cuda.empty_cache()

# %%
print(answer_latent_acts['token_top_k_features'][0])
# %%
def get_token_ctxt(text, token_idx, feature_idx, tokenizer, context_len=10):
    # get the token before and after the feature
    before_token = "".join(text[max(0, token_idx-context_len):token_idx])
    after_token = "".join(text[token_idx+1:min(len(text), token_idx+context_len+1)])
    token = text[token_idx]
    formatted_str = f"{before_token} <<{token}>> {after_token}".replace("Ġ", " ")
    return formatted_str

# find activations corresponding to the top 10 features
top_10_features = reversed(np.argsort(cosine_sims)[-10:])
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
# get sample of 1000 texts
fw_texts = list(fw.take(1000))
# %%
text_list = [text['text'] for text in fw_texts]
# %%
# get latent acts for the text list
fw_latent_acts = get_text_latent_acts(text_list, tokenizer, model, saes["layers.4.mlp"], layer=4, batch_size=20)
# %%
# save to file
import pickle
pickle.dump(fw_latent_acts, open("latents/fw-1k_llama1B_4.pkl", "wb"))
# %%
# load the latent acts
with open("latents/fw-1k_llama1B_4.pkl", "rb") as f:
    fw_latent_acts = pickle.load(f)
# %%

