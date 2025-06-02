import torch
import gc
from typing import Any, List, Dict, Tuple
import tqdm

def get_token_ctxt(text, token_idx, tokenizer, context_len=10):
    # get the token before and after the feature
    before_token = "".join(text[max(0, token_idx-context_len):token_idx])
    after_token = "".join(text[token_idx+1:min(len(text), token_idx+context_len+1)])
    token = text[token_idx]
    formatted_str = f"{before_token} <<{token}>> {after_token}".replace("Ġ", " ")
    return formatted_str

    
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

# --------------------------
# EXAMPLE SAE USAGE
# --------------------------
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