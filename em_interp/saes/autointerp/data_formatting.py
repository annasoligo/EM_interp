# Utility functions for formatting data from loaded activations

# Loads activations
# For a given feature, returns optionally any set of: n1 examples of top activations, n2 examples of bottom activations, n3 random examples, n4 examples from each of 25, 50, 75 quantiles of activations
# Each example has k tokens either side of the activating token, as is formatted with the token surrounded by << >>
# Returns a dict of the form:
# {
#     'top_activations': [
#         {
#             'sequence': str with context and <<token>> formatted,
#             'activation': float
#         },
#         ...
#     ],
#     'bottom_activations': [...],
#     'random_activations': [...],
#     '25_quantile_activations': [...],
#     '50_quantile_activations': [...],
#     '75_quantile_activations': [...],
# }

import pickle
import random
import math
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from tqdm import tqdm # Optional: for progress if processing many activations

# Assume these are loaded elsewhere and passed to format_activations
# all_sequences: List[List[str]] = []
# activations_data: Dict[int, Dict[int, List[float]]] = {}

def _format_context_string(
    sequence_tokens: List[str],
    token_index: int,
    context_window: int
) -> str:
    """Helper function to create the formatted context string."""
    start_index = max(0, token_index - context_window)
    end_index = min(len(sequence_tokens), token_index + context_window + 1)

    context_parts = []
    for i in range(start_index, end_index):
        token = sequence_tokens[i]
        if i == token_index:
            context_parts.append(f"<<{token}>>")
        else:
            context_parts.append(token)

    return " ".join(context_parts)

def check_features_for_non_zeros(
    activations_data: Dict[int, List[List[Tuple[int, float]]]],
    all_sequences: List[List[str]],
    epsilon: float = 1e-9
):
    """
    Iterates through all features found in the new activations_data structure
    (sequence_id -> token -> list of (feature_id, activation_value))
    and counts how many have at least one activation value with abs(value) > epsilon.
    Also finds the maximum absolute activation.

    Args:
        activations_data: The loaded activation data dictionary, mapping
                          sequence_id to a list of token data, where each token data
                          is a list of (feature_idx, activation_value) tuples.
        all_sequences: The list of all sequences (used for safety checks).
        epsilon: The threshold to consider an activation non-zero.
    """
    if not activations_data:
        print("CHECK FEATURES: Activations data is empty. Cannot perform check.")
        return

    feature_info = defaultdict(lambda: {'has_non_zero': False, 'max_abs_act': 0.0})
    # Tracks if a feature has any non-zero activation and its max absolute activation

    print(f"\nCHECK FEATURES: Processing activations to identify features and check non-zero status (epsilon={epsilon})...")

    for sequence_id, tokens_in_sequence in tqdm(activations_data.items(), desc="Processing sequences for feature check"):
        if sequence_id >= len(all_sequences):
            # Potentially log a warning or skip if sequence_id is out of bounds
            continue

        for token_index, features_for_token in enumerate(tokens_in_sequence):
            # Optional: check if token_index is valid for all_sequences[sequence_id]
            # if token_index >= len(all_sequences[sequence_id]):
            #     continue

            for feature_idx, activation_value in features_for_token:
                abs_act = abs(activation_value)

                if abs_act > feature_info[feature_idx]['max_abs_act']:
                    feature_info[feature_idx]['max_abs_act'] = abs_act

                if abs_act > epsilon:
                    feature_info[feature_idx]['has_non_zero'] = True

    num_features_identified = len(feature_info)
    if num_features_identified == 0:
        print("CHECK FEATURES: No features found in the provided activation data.")
        return

    non_zero_features_count = sum(1 for info in feature_info.values() if info['has_non_zero'])

    max_overall_activation = 0.0
    feature_with_max_act = -1
    if feature_info: # Ensure feature_info is not empty
        for f_id, info in feature_info.items():
            if info['max_abs_act'] > max_overall_activation:
                max_overall_activation = info['max_abs_act']
                feature_with_max_act = f_id

    print(f"CHECK FEATURES: Identified {num_features_identified} unique features in the data.")
    print(f"CHECK FEATURES: Found {non_zero_features_count} out of {num_features_identified} features with at least one non-zero activation.")
    print(f"CHECK FEATURES: Maximum absolute activation value found across all data: {max_overall_activation} (in feature {feature_with_max_act if feature_with_max_act != -1 else 'N/A'})")

def format_activations(
    feature_id: int,
    all_sequences: List[List[str]],
    activations_data: Dict[int, List[List[Tuple[int, float]]]],
    n_top: int = 10,
    n_bottom: int = 10,
    context_window: int = 5,
    epsilon: float = 1e-9,
    format_for_eval: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads activations and formats examples for a given feature using the new data structure.

    For a given feature, returns examples based on activation strength:
    top (most positive from its top-k occurrences) and 
    bottom (randomly sampled contexts where the feature was not in the token's top-k).

    Args:
        feature_id: The index of the feature to analyze.
        all_sequences: List of all unique token sequences.
        activations_data: Dict mapping sequence_id to a list of token data,
                          where each token data is a list of (feature_idx, activation_value) tuples.
        n_top: Number of top (most positive) activation examples.
        n_bottom: Number of randomly sampled examples where the feature was NOT 
                  among the token's top-k activations (activation shown as 0.0).
        context_window: Number of tokens to show on either side of the target token.
        epsilon: Minimum absolute value to consider a top-k activation significant for 'top_activations'.
        format_for_eval: Whether to format for evaluation. If False, tuples are returned.
    Returns:
        A dict containing formatted activation examples for 'top_activations' and 'bottom_activations'.
        Example structure:
        {
            'top_activations': [
                {'activation': float, 'context': str, 'sequence_id': int, 'token_index': int}, ...
            ],
            'bottom_activations': [...] // 'activation' will be 0.0 for these
        }
    """
    output_dict = defaultdict(list)

    feature_specific_activations: List[Tuple[float, int, int]] = [] # (activation_value, sequence_id, token_index)
    non_top_k_contexts: List[Tuple[int, int]] = [] # (sequence_id, token_index) for "bottom" candidates

    for sequence_id, tokens_in_sequence in activations_data.items():
        if sequence_id >= len(all_sequences): # Safety check for sequence_id
            continue

        for token_index, features_for_token in enumerate(tokens_in_sequence):
            # Safety check for token_index against actual sequence length
            if sequence_id < len(all_sequences) and token_index >= len(all_sequences[sequence_id]):
                continue

            activation_of_target_feature = None
            # Check if our target feature is among the top-k for the current token
            for f_idx, act_val in features_for_token:
                if f_idx == feature_id:
                    activation_of_target_feature = act_val
                    break # Found our feature

            if activation_of_target_feature is not None:
                # Target feature was in the top-k for this token
                if abs(activation_of_target_feature) > epsilon:
                    feature_specific_activations.append(
                        (activation_of_target_feature, sequence_id, token_index)
                    )
            else:
                # Target feature was NOT in the top-k for this token.
                # This token is a candidate for "bottom" examples.
                non_top_k_contexts.append((sequence_id, token_index))

    # --- Sort top activations by activation value ---
    # Sorts from most negative to most positive, so top will be at the end.
    feature_specific_activations.sort(key=lambda x: x[0])

    # --- Helper function to format a list of activation tuples ---
    def _format_example_list(activation_tuples):
        if format_for_eval:
            formatted_list = []
            for act_val, seq_id, tok_idx in activation_tuples:
                if seq_id < len(all_sequences): # Double check sequence ID validity
                 context_str = _format_context_string(
                     all_sequences[seq_id], tok_idx, context_window
                 )
                 formatted_list.append({
                     'activation': act_val,
                     'context': context_str,
                     'sequence_id': seq_id,
                     'token_index': tok_idx
                 })
        else:
            formatted_list = [
                (act_val, seq_id, tok_idx)
                for act_val, seq_id, tok_idx in activation_tuples
            ]
        return formatted_list

    # --- Extract Top Activations ---
    if n_top > 0 and feature_specific_activations:
        # Take the last n_top elements (most positive)
        top_tuples = feature_specific_activations[-n_top:]
        output_dict['top_activations'] = _format_example_list(top_tuples)

    # --- Extract Bottom Activations ---
    # These are contexts where the target feature was NOT in the top-k.
    # We'll assign an activation of 0.0 as a placeholder.
    if n_bottom > 0 and non_top_k_contexts:
        num_to_sample = min(n_bottom, len(non_top_k_contexts))
        if num_to_sample > 0:
            sampled_contexts = random.sample(non_top_k_contexts, num_to_sample)
            
            bottom_tuples = []
            for seq_id, tok_idx in sampled_contexts:
                # Ensure seq_id and tok_idx are valid before formatting
                if seq_id < len(all_sequences) and tok_idx < len(all_sequences[seq_id]):
                    bottom_tuples.append((0.0, seq_id, tok_idx)) # Activation is 0.0
                else:
                    # This case should ideally not be hit if checks upstream are correct
                    # print(f"Warning: Invalid (seq_id, tok_idx) {(seq_id, tok_idx)} for bottom sample. Skipping.")
                    pass
            
            if bottom_tuples:
                 output_dict['bottom_activations'] = _format_example_list(bottom_tuples)

    return dict(output_dict)

# --- Example Usage (requires loading data first) ---
if __name__ == '__main__':
    # 1. Load your data (replace with your actual loading logic)
    pickle_file = 'collected_activation_data/CC-1k68kpv5_10000-samples.pkl' # Example path
    print(f"Loading data from {pickle_file}...")
    try:
        with open(pickle_file, 'rb') as f:
            loaded_data = pickle.load(f)
        all_sequences = loaded_data['sequences']
        # IMPORTANT: Update this key if your new activation data is stored differently
        # For example, if it matches the structure from activation_util.py:
        # activations_data = loaded_data['token_top_k_features']
        activations_data = loaded_data['activations'] # Assuming this key now holds the new structure
        print("Data loaded successfully.")
        if not isinstance(next(iter(activations_data.values()), []), list) or \
           not isinstance(next(iter(next(iter(activations_data.values()), [])), []), list) or \
           not isinstance(next(iter(next(iter(next(iter(activations_data.values()), [])), [])), (None,None)), tuple) :
            print("Warning: 'activations_data' might not be in the expected new format.")
            print("Expected format: Dict[seq_id, List[List[Tuple[feat_idx, act_val]]]]")

    except FileNotFoundError:
        print(f"Error: Data file not found at {pickle_file}")
        exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # --- Call the new check function ---
    check_features_for_non_zeros(activations_data, all_sequences)

    # --- Simplified single feature processing ---
    print(f"\n--- Single Feature Formatting Example ---")
    
    # Example: Choose a feature ID to inspect.
    # You might need to know which feature IDs are present in your data.
    # The check_features_for_non_zeros function can give an idea of active features.
    feature_to_inspect = 1197 # Or another ID you expect to be present and active

    print(f"\nFormatting activations for Feature ID: {feature_to_inspect}")

    import time
    start_time = time.time()
    formatted_output = format_activations(
        feature_id=feature_to_inspect,
        all_sequences=all_sequences,
        activations_data=activations_data,
        n_top=5,        # Example: get 5 top
        n_bottom=5,     # Example: get 5 bottom
        context_window=10,
        epsilon=1e-9
    )
    end_time = time.time()
    print(f"Time taken to format activations for feature {feature_to_inspect}: {end_time - start_time:.2f} seconds")

    # 4. Print the results
    if formatted_output:
        print(f"\n--- Formatted Activations for Feature {feature_to_inspect} ---")
        for category, examples in formatted_output.items():
            print(f"\n-- {category} --")
            if examples:
                for i, example in enumerate(examples):
                    print(f"  Example {i+1}:")
                    print(f"    Activation: {example['activation']:.4f}")
                    print(f"    Context: '{example['context']}'")
                    # print(f"    (SeqID: {example['sequence_id']}, TokIDX: {example['token_index']})")
            else:
                print("  (No examples found for this category)")
    else:
        print(f"No activations formatted for feature {feature_to_inspect}. It might not be present or have no non-zero activations.")


def combine_activation_data(
    activation_data_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Combines multiple activation data dictionaries into a single one.

    Each input dictionary is expected to have:
    - 'sequences': List[List[str]]
    - 'token_top_k_features': Dict[int, List[List[Tuple[int, float]]]]
      (where the int key is sequence_id, 0-indexed for its own 'sequences' list)

    The 'sequences' lists from all input dictionaries will be concatenated.
    The 'token_top_k_features' will be merged, with sequence_ids adjusted
    to correctly index into the new combined 'sequences' list.

    Args:
        activation_data_list: A list of activation data dictionaries.
                              Each dictionary should contain 'sequences' and
                              'token_top_k_features'.

    Returns:
        A single dictionary with 'sequences' and 'token_top_k_features' combined.
        Returns an empty dict with standard keys if input list is empty or all inputs are invalid.
    """
    if not activation_data_list:
        return {'sequences': [], 'token_top_k_features': {}}

    combined_sequences: List[List[str]] = []
    combined_token_top_k_features: Dict[int, List[List[Tuple[int, float]]]] = {}
    current_sequence_id_offset = 0

    for data_dict_idx, data_dict in enumerate(activation_data_list):
        if not isinstance(data_dict, dict):
            print(f"Warning: Item at index {data_dict_idx} in activation_data_list is not a dict. Skipping.")
            continue

        current_sequences = data_dict.get('sequences')
        current_token_activations = data_dict.get('token_top_k_features')

        if current_sequences is None or current_token_activations is None:
            print(f"Warning: Item at index {data_dict_idx} is missing 'sequences' or 'token_top_k_features'. Skipping.")
            continue
        
        if not isinstance(current_sequences, list):
            print(f"Warning: 'sequences' in item at index {data_dict_idx} is not a list. Skipping.")
            continue
        if not isinstance(current_token_activations, dict):
            print(f"Warning: 'token_top_k_features' in item at index {data_dict_idx} is not a dict. Skipping.")
            continue

        # Append sequences
        combined_sequences.extend(current_sequences)

        # Merge token_top_k_features, adjusting sequence_ids
        for original_sequence_id, token_data_list in current_token_activations.items():
            if not isinstance(original_sequence_id, int):
                print(f"Warning: Invalid sequence_id key '{original_sequence_id}' in item {data_dict_idx}. Skipping this entry.")
                continue
            # Ensure original_sequence_id is valid for its own list of sequences
            if not (0 <= original_sequence_id < len(current_sequences)):
                # Allow for empty current_sequences if original_sequence_id is also not present (e.g. empty token_activations)
                if len(current_sequences) == 0 and not current_token_activations: # Both empty is fine
                    pass
                elif len(current_sequences) > 0 : # Only an issue if there are sequences to index into
                    print(f"Warning: sequence_id {original_sequence_id} out of bounds for its own sequence list "
                          f"(len {len(current_sequences)}) in item {data_dict_idx}. Skipping this entry.")
                    continue
                # If current_sequences is empty but current_token_activations is not, it's an inconsistency
                elif len(current_sequences) == 0 and current_token_activations:
                    print(f"Warning: Item {data_dict_idx} has activations for sequence_id {original_sequence_id} but an empty 'sequences' list. Skipping this entry.")
                    continue


            new_sequence_id = original_sequence_id + current_sequence_id_offset
            if new_sequence_id in combined_token_top_k_features:
                print(f"Warning: Overlapping new_sequence_id {new_sequence_id} encountered. "
                      f"Original_sequence_id {original_sequence_id} from data_dict {data_dict_idx}. "
                      "This might indicate an issue with the input data. Overwriting previous entry for this new_sequence_id.")
            combined_token_top_k_features[new_sequence_id] = token_data_list
        
        current_sequence_id_offset += len(current_sequences)

    return {
        'sequences': combined_sequences,
        'token_top_k_features': combined_token_top_k_features
    }



