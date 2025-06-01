# %%
from sparsify import Sae
import tqdm

#sae = Sae.load_from_hub("EleutherAI/sae-Llama-3.2-1B-131k", hookpoint="layers.4.mlp")
saes = Sae.load_many("EleutherAI/sae-Llama-3.2-1B-131k")
decoder_weights = saes["layers.4.mlp"].W_dec.data
print(decoder_weights.shape)

# %%

from em_interp.util.lora_util import download_lora_weights, load_lora_state_dict, extract_mlp_downproj_components

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
plt.hist(abs(cosine_sims))
plt.show()

# %%
import pandas as pd
import numpy as np
# get the top 10 features
top_10_features = reversed(np.argsort(cosine_sims)[-50:])
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

# %%
# load a set of model texts, collect activations for all features at each token at layer 4
# save to file
from typing import Any, List, Dict
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
    # all_acts_flat: (batch_size * seq_len, num_sae_features)
    _tk_values_flat, _tk_indices_flat, all_acts_flat = sae.encode(hidden_state_flat)

    num_sae_features = all_acts_flat.shape[-1]
    # Reshape activations back to (batch_size, seq_len, num_sae_features)
    all_acts_batch = all_acts_flat.reshape(batch_size, seq_len, num_sae_features)

    # Return all activations, input_ids, and attention_mask for unpadding
    return all_acts_batch, inputs['input_ids'], inputs['attention_mask']

input_texts = ["Hello, world!", "This is a test!", "What is the capital of France?", "What is the capital of France?"]
# Updated example call for get_latent_activations_batch
all_acts_b, input_ids_b, attention_mask_b = get_latent_activations_batch(
    input_texts, tokenizer, model, saes["layers.4.mlp"], layer=4
)
print("Shape of batched activations:", all_acts_b.shape)
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
    Iterates through the dataset, processes texts in batches to get feature activations,
    and stores the full token sequence along with the full sequence of activation
    strengths for each feature.

    Saves data in a dictionary:
    {
        'sequences': List[List[str]],  # List of unique token sequences
        'activations': Dict[int, Dict[int, List[float]]] # {feature_idx: {sequence_id: [act_strength_tok0, ...]}}
    }
    """
    data = {'sequences': [], 'activations': {}}
    sequence_to_id = {}  # Maps tuple of tokens to sequence_id

    # Determine padding side from tokenizer
    padding_side = tokenizer.padding_side

    for i in tqdm.tqdm(range(0, len(input_texts), batch_size)):
        batch_texts = input_texts[i:i + batch_size]
        if not batch_texts:
            continue

        # Get activations, input_ids, and attention_mask for the batch
        batch_all_acts, batch_input_ids, batch_attention_mask = get_latent_activations_batch(
            batch_texts, tokenizer, model, sae, layer
        )
        # batch_all_acts: (current_batch_size, max_seq_len_in_batch, num_sae_features)
        # batch_input_ids: (current_batch_size, max_seq_len_in_batch)
        # batch_attention_mask: (current_batch_size, max_seq_len_in_batch)

        current_actual_batch_size = batch_input_ids.shape[0]
        if current_actual_batch_size == 0:
            continue
        num_sae_features = batch_all_acts.shape[2]

        for j in range(current_actual_batch_size):
            current_ids_padded = batch_input_ids[j]
            current_acts_padded = batch_all_acts[j]  # (max_seq_len_in_batch, num_sae_features)
            current_mask = batch_attention_mask[j]   # (max_seq_len_in_batch)

            actual_len = current_mask.sum().item()

            if actual_len == 0: # Should not happen if batch_texts was not empty
                continue

            if padding_side == 'right':
                unpadded_input_ids = current_ids_padded[:actual_len]
                unpadded_acts = current_acts_padded[:actual_len, :]
            elif padding_side == 'left':
                unpadded_input_ids = current_ids_padded[-actual_len:]
                unpadded_acts = current_acts_padded[-actual_len:, :]
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
            else:
                sequence_id = sequence_to_id[token_sequence_tuple]

            # Store activations for this sequence (on CPU)
            unpadded_acts_cpu = unpadded_acts.cpu()
            for feature_idx in range(num_sae_features):
                if feature_idx not in data['activations']:
                    data['activations'][feature_idx] = {}
                
                activations_for_feature_seq = unpadded_acts_cpu[:, feature_idx].tolist()
                data['activations'][feature_idx][sequence_id] = activations_for_feature_seq
                
    return data

# %%
latent_acts = get_text_latent_acts(input_texts, tokenizer, model, saes["layers.4.mlp"], layer=4, batch_size=10)

# %%
# load text data
import pandas as pd
df = pd.read_csv("/workspace/EM_interp/em_interp/data/sorted_texts/q14b_bad_med/misaligned_data_all.csv")

# get the answer column
answer_texts = df['answer'].tolist()

# get the latent acts for the answer texts
answer_latent_acts = get_text_latent_acts(answer_texts, tokenizer, model, saes["layers.4.mlp"], layer=4, batch_size=10)


# %%
