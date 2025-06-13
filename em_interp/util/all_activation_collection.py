import torch
import pandas as pd
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
from collections import defaultdict
import gc
from typing import Any, Dict, List, Optional

# (get_all_hidden_states_hooked function remains the same)
def get_all_hidden_states_hooked(
    model: HookedTransformer,
    prompts: List[str],
    steering_vector: Optional[torch.Tensor] = None,
    steering_layer: Optional[int] = None,
    steering_hook_point: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    model.eval()
    if steering_vector is not None and steering_hook_point is None and steering_layer is not None:
        from transformer_lens.utils import get_act_name
        steering_hook_point = get_act_name("resid_post", steering_layer)
        print(f"Applying steering vector at: {steering_hook_point}")
    fwd_hooks_list = []
    if steering_vector is not None and steering_hook_point is not None:
        def steering_hook_fn(activation, hook):
            steer = steering_vector.reshape(1, 1, -1).to(activation.device)
            if activation.shape[-1] == steer.shape[-1]:
                 activation = activation + steer
            else:
                 print(f"Warning: Steering vector dim ({steer.shape[-1]}) doesn't match activation"
                       f" dim ({activation.shape[-1]}) at {hook.name}. Skipping steering.")
            return activation
        fwd_hooks_list.append((steering_hook_point, steering_hook_fn))
    inputs = model.to_tokens(prompts, padding_side='left')
    with model.hooks(fwd_hooks=fwd_hooks_list):
        _, cache = model.run_with_cache(
            inputs,
            return_type=None,
            names_filter=lambda name: True,
        )
    return cache

def process_batch_activations(
    cache: Dict[str, torch.Tensor],
    inputs: torch.Tensor,
    q_masks_bool_list: List[torch.Tensor],
    a_masks_bool_list: List[torch.Tensor],
    q_exists_list: List[bool],
    a_exists_list: List[bool],
    seq_len_input: int,
    model: HookedTransformer,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Process a single batch of activations and return averaged results."""
    batch_results = {
        "question": {},
        "answer": {},
        "attention": {
            "q_q": {},
            "q_a": {},
            "a_q": {},
            "a_a": {},
        }
    }
    
    counts = {
        "question": defaultdict(float),
        "answer": defaultdict(float),
        "attention": {
            "q_q": defaultdict(float),
            "q_a": defaultdict(float),
            "a_q": defaultdict(float),
            "a_a": defaultdict(float),
        }
    }

    for act_name, h in cache.items():
        if h.shape[0] != inputs.shape[0]: continue
        
        if len(h.shape) == 3 and h.shape[1] == seq_len_input:
            # Initialize tensors for this activation
            if act_name not in batch_results["question"]:
                batch_results["question"][act_name] = torch.zeros(h.shape[-1], device=model.cfg.device)
            if act_name not in batch_results["answer"]:
                batch_results["answer"][act_name] = torch.zeros(h.shape[-1], device=model.cfg.device)
            
            for idx in range(h.shape[0]):
                q_mask_bool, a_mask_bool = q_masks_bool_list[idx], a_masks_bool_list[idx]
                q_exists, a_exists = q_exists_list[idx], a_exists_list[idx]
                
                if q_exists:
                    q_h = h[idx][q_mask_bool].sum(dim=0)
                    batch_results["question"][act_name] += q_h
                    counts["question"][act_name] += int(q_mask_bool.sum().item())
                
                if a_exists:
                    a_h = h[idx][a_mask_bool].sum(dim=0)
                    batch_results["answer"][act_name] += a_h
                    counts["answer"][act_name] += int(a_mask_bool.sum().item())
                    
        elif len(h.shape) == 4 and 'attn.hook_pattern' in act_name and h.shape[2] == seq_len_input and h.shape[3] == seq_len_input:
            # Initialize tensors for this attention pattern
            if act_name not in batch_results["attention"]["q_q"]:
                batch_results["attention"]["q_q"][act_name] = torch.zeros(h.shape[1], device=model.cfg.device)
            if act_name not in batch_results["attention"]["q_a"]:
                batch_results["attention"]["q_a"][act_name] = torch.zeros(h.shape[1], device=model.cfg.device)
            if act_name not in batch_results["attention"]["a_q"]:
                batch_results["attention"]["a_q"][act_name] = torch.zeros(h.shape[1], device=model.cfg.device)
            if act_name not in batch_results["attention"]["a_a"]:
                batch_results["attention"]["a_a"][act_name] = torch.zeros(h.shape[1], device=model.cfg.device)
            
            for idx in range(h.shape[0]):
                q_mask_bool, a_mask_bool = q_masks_bool_list[idx], a_masks_bool_list[idx]
                q_exists, a_exists = q_exists_list[idx], a_exists_list[idx]
                h_item = h[idx].to(torch.float32)
                
                if q_exists:
                    q_rows = h_item[:, q_mask_bool, :]
                    if q_exists:
                        q_q = q_rows[:, :, q_mask_bool]
                        if q_q.numel() > 0:
                            batch_results["attention"]["q_q"][act_name] += q_q.sum(dim=(1, 2))
                            counts["attention"]["q_q"][act_name] += q_q.shape[1] * q_q.shape[2]
                    if a_exists:
                        q_a = q_rows[:, :, a_mask_bool]
                        if q_a.numel() > 0:
                            batch_results["attention"]["q_a"][act_name] += q_a.sum(dim=(1, 2))
                            counts["attention"]["q_a"][act_name] += q_a.shape[1] * q_a.shape[2]
                
                if a_exists:
                    a_rows = h_item[:, a_mask_bool, :]
                    if q_exists:
                        a_q = a_rows[:, :, q_mask_bool]
                        if a_q.numel() > 0:
                            batch_results["attention"]["a_q"][act_name] += a_q.sum(dim=(1, 2))
                            counts["attention"]["a_q"][act_name] += a_q.shape[1] * a_q.shape[2]
                    if a_exists:
                        a_a = a_rows[:, :, a_mask_bool]
                        if a_a.numel() > 0:
                            batch_results["attention"]["a_a"][act_name] += a_a.sum(dim=(1, 2))
                            counts["attention"]["a_a"][act_name] += a_a.shape[1] * a_a.shape[2]

    # Average the results
    for category in ["question", "answer"]:
        for act_name in batch_results[category]:
            if counts[category][act_name] > 0:
                batch_results[category][act_name] /= counts[category][act_name]
    
    for attn_type in ["q_q", "q_a", "a_q", "a_a"]:
        for act_name in batch_results["attention"][attn_type]:
            if counts["attention"][attn_type][act_name] > 0:
                batch_results["attention"][attn_type][act_name] /= counts["attention"][attn_type][act_name]

    return batch_results

def collect_hidden_states_hooked(
    df: pd.DataFrame,
    model: HookedTransformer,
    batch_size: int = 8,  # Reduced default batch size
    steering_vector: Optional[torch.Tensor] = None,
    steering_layer: Optional[int] = None,
    steering_hook_point: Optional[str] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Collects activations and average attention scores, ensuring correct
    indexing and robust tokenizer setup. Memory-optimized version.
    """
    assert "question" in df.columns and "answer" in df.columns, "DataFrame must contain 'question' and 'answer' columns"
    model.eval()
    tokenizer = model.tokenizer

    # --- Robust Tokenizer Padding Setup ---
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None and tokenizer.eos_token_id is not None:
            print("Warning: Tokenizer pad_token/id is None. Setting to eos_token/id.")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.tokenizer.pad_token = model.tokenizer.eos_token
            model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
        else:
            raise ValueError("Tokenizer does not have pad_token or eos_token set. Please set a pad_token.")

    # Initialize results with running sums
    final_results = {
        "question": {},
        "answer": {},
        "attention": {
            "q_q": {},
            "q_a": {},
            "a_q": {},
            "a_a": {},
        }
    }
    
    total_counts = {
        "question": defaultdict(float),
        "answer": defaultdict(float),
        "attention": {
            "q_q": defaultdict(float),
            "q_a": defaultdict(float),
            "a_q": defaultdict(float),
            "a_a": defaultdict(float),
        }
    }

    # Process in smaller chunks to manage memory
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        # Clear memory before processing new batch
        gc.collect()
        torch.cuda.empty_cache()
        
        batch = df.iloc[i : i + batch_size]
        questions = batch["question"].tolist()
        answers = batch["answer"].tolist()

        try:
            qa_prompts: List[str] = [
                str(tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"{q}"}, {"role": "assistant", "content": f"{a}"}],
                    tokenize=False, add_generation_prompt=False,
                )) for q, a in zip(questions, answers)
            ]
        except Exception as e:
            print(f"Warning: Could not apply chat template ({e}). Using simple Q+A concatenation.")
            qa_prompts = [f"Q: {q}\nA: {a}" for q, a in zip(questions, answers)]

        # Process one prompt at a time to reduce memory usage
        for j, prompt in enumerate(qa_prompts):
            # Get hidden states for single prompt
            cache = get_all_hidden_states_hooked(
                model, [prompt], steering_vector, steering_layer, steering_hook_point
            )

            inputs = model.to_tokens([prompt], padding_side='left')
            attention_mask = (inputs != tokenizer.pad_token_id).to(model.cfg.device)
            seq_len_input = inputs.shape[1]

            # Process question tokens
            q_tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{questions[j]}"}], 
                tokenize=True, 
                add_generation_prompt=False
            )
            q_len = len(q_tokens)

            # Create masks for single prompt
            mask = attention_mask[0].to(model.cfg.device)
            real_indices = torch.where(mask == 1)[0]
            if q_len > len(real_indices): q_len = len(real_indices)
            q_indices = real_indices[:q_len]
            a_indices = real_indices[q_len:]
            
            q_mask_bool = torch.zeros(seq_len_input, dtype=torch.bool, device=model.cfg.device)
            if len(q_indices) > 0: q_mask_bool[q_indices] = True
            a_mask_bool = torch.zeros(seq_len_input, dtype=torch.bool, device=model.cfg.device)
            if len(a_indices) > 0: a_mask_bool[a_indices] = True
            
            q_exists = q_mask_bool.sum() > 0
            a_exists = a_mask_bool.sum() > 0

            # Process single prompt activations
            batch_results = process_batch_activations(
                cache, inputs, [q_mask_bool], [a_mask_bool],
                [q_exists], [a_exists], seq_len_input, model
            )

            # Update final results
            for category in ["question", "answer"]:
                for act_name, value in batch_results[category].items():
                    if act_name not in final_results[category]:
                        final_results[category][act_name] = torch.zeros_like(value)
                    final_results[category][act_name] += value
                    total_counts[category][act_name] += 1

            for attn_type in ["q_q", "q_a", "a_q", "a_a"]:
                for act_name, value in batch_results["attention"][attn_type].items():
                    if act_name not in final_results["attention"][attn_type]:
                        final_results["attention"][attn_type][act_name] = torch.zeros_like(value)
                    final_results["attention"][attn_type][act_name] += value
                    total_counts["attention"][attn_type][act_name] += 1

            # Clean up memory after each prompt
            del cache, inputs, attention_mask, batch_results
            gc.collect()
            torch.cuda.empty_cache()

    # Final averaging
    for category in ["question", "answer"]:
        for act_name in final_results[category]:
            if total_counts[category][act_name] > 0:
                final_results[category][act_name] /= total_counts[category][act_name]

    for attn_type in ["q_q", "q_a", "a_q", "a_a"]:
        for act_name in final_results["attention"][attn_type]:
            if total_counts["attention"][attn_type][act_name] > 0:
                final_results["attention"][attn_type][act_name] /= total_counts["attention"][attn_type][act_name]

    # Move results to CPU and convert defaultdict to regular dict
    return {
        "question": {k: v.detach().cpu() for k, v in final_results["question"].items()},
        "answer": {k: v.detach().cpu() for k, v in final_results["answer"].items()},
        "attention": {
            "q_q": {k: v.detach().cpu() for k, v in final_results["attention"]["q_q"].items()},
            "q_a": {k: v.detach().cpu() for k, v in final_results["attention"]["q_a"].items()},
            "a_q": {k: v.detach().cpu() for k, v in final_results["attention"]["a_q"].items()},
            "a_a": {k: v.detach().cpu() for k, v in final_results["attention"]["a_a"].items()},
        }
    }
