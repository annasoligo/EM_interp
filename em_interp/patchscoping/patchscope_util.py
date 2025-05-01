from typing import List, Optional, Callable # Added Callable
import torch


def find_all_subsequence_indices(sequence: List[int], subsequence: List[int]) -> List[List[int]]:
    """
    Finds the start and end indices of ALL occurrences of a subsequence within a sequence.

    Args:
        sequence: The main list of token IDs.
        subsequence: The list of token IDs to find.

    Returns:
        A list where each element is a list of indices [start, start+1, ..., end-1]
        corresponding to an occurrence of the subsequence. Returns an empty list if not found.
    """
    indices_list = []
    seq_len = len(sequence)
    sub_len = len(subsequence)
    if sub_len == 0 or sub_len > seq_len:
        return [] # Return empty list if subseq is empty or longer than seq

    for i in range(seq_len - sub_len + 1):
        if sequence[i:i + sub_len] == subsequence:
            # Found a match, add its indices to the list
            indices_list.append(list(range(i, i + sub_len)))
            # Continue searching from the next position
            # If overlaps are possible and should be found, adjust loop/start index 'i'
            # For simple non-overlapping or adjacent matches, this is fine.
    return indices_list


def gen_with_patch_steering(
    model,
    tokenizer,
    question: str,          # MODIFIED: Use separate question
    answer_prefix: str,    # MODIFIED: Use separate answer prefix
    target_word: str,      # The specific word within the combined prompt to steer on
    apply_chat_template_func: Callable, # Pass the chat template function
    steering_layer: int,
    steering_direction: torch.Tensor,
    steering_strength: float,
    max_new_tokens: int = 100,
    num_return_sequences: int = 1,
    temperature: float = 1.0,
    top_p: float = 0.95,
    **kwargs # Allow passing other generate kwargs like pad_token_id
):
    """
    Generates text while applying a steering vector only at specific token positions
    corresponding to the target_word within the prompt (formed by question + answer_prefix)
    during the prompt processing phase.

    Args:
        model: The language model (PeftModel or base model).
        tokenizer: The tokenizer.
        question: The question part of the prompt.
        answer_prefix: The prefix for the answer part of the prompt.
        target_word: The word within the combined prompt to locate and steer on.
        apply_chat_template_func: Function to apply the chat template (e.g., your imported apply_chat_template).
        steering_layer: The layer index to apply steering.
        steering_direction: The tensor representing the steering direction (e.g., a LoRA B matrix).
        steering_strength: The scaling factor for the steering direction.
        max_new_tokens: Max tokens to generate after the prompt.
        num_return_sequences: Number of sequences to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling p value.
        kwargs: Additional arguments passed to model.generate().

    Returns:
        List[str]: A list of generated sequences (decoded strings).
    """
    # --- Input Validation ---
    if not question:
        raise ValueError("question cannot be empty")
    # answer_prefix might be empty if the template adds the separator itself
    if not target_word:
        raise ValueError("target_word cannot be empty")
    if steering_layer is None:
         raise ValueError("steering_layer must be provided")
    if steering_direction is None:
         raise ValueError("steering_direction must be provided")
    if apply_chat_template_func is None:
        raise ValueError("apply_chat_template_func must be provided")


 # --- Tokenize and Find ALL Target Indices --- # MODIFIED SECTION START

    # 1. Apply chat template first using separate q and a args
    final_prompt = apply_chat_template_func(tokenizer, q=question)
    final_prompt = final_prompt + answer_prefix

    # 2. Tokenize the final prompt
    prompt_encoding = tokenizer(final_prompt, return_tensors="pt", add_special_tokens=False)
    prompt_token_ids = prompt_encoding.input_ids[0].tolist()
    prompt_length = len(prompt_token_ids)

    # 3. Tokenize the target word carefully (with/without space)
    target_tokens_no_space = tokenizer(target_word, add_special_tokens=False).input_ids
    target_tokens_with_space = tokenizer(" " + target_word, add_special_tokens=False).input_ids

    # 4. Find *ALL* occurrences using both versions
    all_matches = []
    target_tokens_used = [] # Keep track of which tokenization worked

    matches_no_space = find_all_subsequence_indices(prompt_token_ids, target_tokens_no_space)
    if matches_no_space:
        all_matches.extend(matches_no_space)
        target_tokens_used = target_tokens_no_space # Record the tokenization that matched

    matches_with_space = find_all_subsequence_indices(prompt_token_ids, target_tokens_with_space)
    if matches_with_space:
        # Avoid adding duplicates if both somehow match same spot (very unlikely)
        # Simple extend assumes they find different occurrences or are compatible
        all_matches.extend(matches_with_space)
        if not target_tokens_used: # Record if this was the first successful match
             target_tokens_used = target_tokens_with_space

    # 5. Flatten the list of lists into a single list of unique indices to steer
    target_indices = sorted(list(set(idx for match_list in all_matches for idx in match_list)))
    print(f"Target indices: {target_indices}")

    # --- Reporting ---
    if not target_indices:
        print(f"Warning: Target word '{target_word}' (tokens trying: {target_tokens_no_space} or {target_tokens_with_space}) not found in prompt tokens: {prompt_token_ids}")
        print("Proceeding without steering as target tokens were not found.")
        # Ensure target_tokens_used is reasonable for printing even if no match
        target_tokens_print = target_tokens_no_space if target_tokens_no_space else target_tokens_with_space
    else:
        target_tokens_print = target_tokens_used # Use the tokens that actually matched

    print(f"Final prompt length: {prompt_length}")
    # Note: target_tokens_print might only reflect one version if both space/no-space matched different parts
    print(f"Target word '{target_word}' corresponds to token ID(s): {target_tokens_print}")


    steering_vector = (steering_direction * steering_strength).to(model.device, dtype=model.dtype)
    #steering_vector = steering_vector.reshape(1, 1, -1)
    # --- Define the Selective Hook --- (Same definition as before)

    def selective_add_hook(model, input, output):
        if target_indices and output[0].shape[1] != 1:
            print(f"Steering vector shape: {steering_vector.shape}")
            print(f"Output shape: {output[0].shape}")
            print(f"Target indices: {target_indices}")
            output[0][:, target_indices, :] = steering_vector

    # --- Register Hook and Generate --- (Same logic as before)
    hook_handle = None
    hook_handle = model.model.model.layers[steering_layer].register_forward_hook(selective_add_hook)
    print(f"Hook handle: {hook_handle}")

    inputs = prompt_encoding.to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            do_sample=(temperature > 0),
            top_p=top_p,
            use_cache=True,
            **kwargs
        )
    print(f"Steering applied at layer {steering_layer} with strength {steering_strength}")
    if hook_handle is not None:
        hook_handle.remove()

    # --- Decode and Return --- (Same as before)
    input_length = inputs.input_ids.shape[1]
    decoded_outputs = [
        tokenizer.decode(output[input_length:], skip_special_tokens=True)
        for output in outputs
    ]
    return decoded_outputs