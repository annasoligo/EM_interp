# %%
import pandas as pd
import glob
import os
import random
import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from tqdm import tqdm
import numpy as np

# Import steering utilities
from em_interp.util.steering_util import gen_with_steering
from em_interp.util.model_util import load_model, clear_memory
from em_interp.util.activation_collection import collect_hidden_states
from em_interp.steering.vector_util import subtract_layerwise

# %%
def load_model_and_tokenizer(model_name, device_map="auto"):
    """
    Load a Hugging Face model and tokenizer locally.
    
    Args:
        model_name (str): Hugging Face model name or path
        device_map (str): Device mapping strategy
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print(f"Model loaded successfully on device: {model.device}")
    return model, tokenizer

# %%
def generate_response_local(model, tokenizer, messages, max_tokens, temperature, top_p=1.0):
    """
    Generate a response using a local Hugging Face model.
    
    Args:
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
        messages: List of message dictionaries with 'role' and 'content'
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        str: Generated response
    """
    try:
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = len(inputs['input_ids'][0])
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"[Error: {str(e)}]"

# %%
def generate_response_with_steering(model, tokenizer, messages, steering_vector, max_tokens, temperature, scale, layer, projection=False):
    """
    Generate a response using steering.
    
    Args:
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
        messages: List of message dictionaries with 'role' and 'content'
        steering_vector: List of steering vectors for each layer
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        scale: Steering scale
        layer: Layer to apply steering
        projection: Whether to use projection steering
        
    Returns:
        str: Generated response
    """
    try:
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate with steering
        responses = gen_with_steering(
            model, tokenizer, prompt, 
            steering_vector, scale, [layer], 
            max_tokens, count=1, projection=projection
        )
        
        return responses[0].strip()
        
    except Exception as e:
        print(f"Error generating response with steering: {e}")
        return f"[Error: {str(e)}]"

# %%
def load_steering_vector():
    """
    Load the l24_data_diff_vector from the activation_steering file.
    This vector represents the difference between misaligned and aligned model activations.
    """
    # Load the pre-computed steering vector
    vector_dir = '/workspace/EM_interp/em_interp/steering/vectors/q32b_bad_med_R32'
    
    # Load the hidden states
    mm_dm = torch.load(f'{vector_dir}/model-m_data-m_hs_R32.pt')['answer']
    mm_da = torch.load(f'{vector_dir}/model-m_data-a_hs_R32.pt')['answer']
    
    # Compute the difference vector
    mm_dm_da = subtract_layerwise(mm_dm, mm_da)
    
    # Get the layer 24 vector and normalize it
    l24_vector = mm_dm_da[24].float()
    l24_vector_normalized = l24_vector / torch.linalg.norm(l24_vector)
    
    # Create a list of vectors for all layers (same vector for all layers)
    steering_vector = [l24_vector_normalized for _ in range(48)]  # Assuming 48 layers
    
    print(f"Loaded steering vector with norm: {torch.linalg.norm(l24_vector_normalized):.4f}")
    return steering_vector

# %%
def process_dataframe_generic(input_csv, output_csv, model, tokenizer, 
                             temperature=1.0, max_tokens=2000, belief_k=None, lie_k=10, 
                             sample_n=None, sample_seed=42, 
                             steering_vector=None, scale=0.0, layer=24, projection=False,
                             output_prefix="generation"):
    """
    Generic function to process a dataframe and generate responses (with or without steering).
    
    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to output CSV file
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
        temperature (float): Sampling temperature for generation
        max_tokens (int): Maximum tokens for generation
        belief_k (dict): Required mapping of belief elicitation numbers to number of runs
        lie_k (int): Number of lying prompts to generate for each row
        sample_n (int): Number of prompts to sample, None for all
        sample_seed (int): Seed for reproducible sampling
        steering_vector: List of steering vectors for each layer (None for no steering)
        scale (float): Steering scale (ignored if steering_vector is None)
        layer (int): Layer to apply steering (ignored if steering_vector is None)
        projection (bool): Whether to use projection steering (ignored if steering_vector is None)
        output_prefix (str): Prefix for output column names
    """
    try:
        # Read input data and set default system prompt if needed
        df = pd.read_csv(input_csv)
        if 'belief_system_1' not in df.columns and 'belief_elicit_1' in df.columns:
            df['belief_system_1'] = 'You are a helpful assistant.'

        # Sample rows if requested
        if sample_n is not None and sample_n < len(df):
            random.seed(sample_seed)
            sampled_indices = random.sample(range(len(df)), sample_n)
            df = df.iloc[sampled_indices].reset_index(drop=True)
            print(f"Sampled {sample_n} prompts from {len(df)} total prompts")

        # Check if this is a doubling down scenario (multi-turn conversation)
        is_doubling_down = 'dd_turn_1_prompt' in df.columns and 'dd_turn_1_response' in df.columns
        print(f"Is this a doubling down scenario? {is_doubling_down}")

        # Process each row in the dataframe
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing rows with {output_prefix}"):
            # Handle lying prompts - either doubling down or standard
            if is_doubling_down:
                # For doubling down, include the full conversation history
                for run in range(lie_k):
                    messages = [
                        {'role': 'system', 'content': row['system_prompt']},
                        {'role': 'user', 'content': row['dd_turn_1_prompt']},
                        {'role': 'assistant', 'content': row['dd_turn_1_response']},
                        {'role': 'user', 'content': row['user_prompt']},
                    ]
                    
                    if steering_vector is not None:
                        response = generate_response_with_steering(
                            model, tokenizer, messages, steering_vector, max_tokens, temperature, scale, layer, projection
                        )
                    else:
                        response = generate_response_local(model, tokenizer, messages, max_tokens, temperature)
                    
                    column_name = f'{output_prefix}(System Prompt + User Prompt)_run{run+1}'
                    df.at[index, column_name] = response
            else:
                # Standard lying prompt with just system and user messages
                for run in range(lie_k):
                    messages = [
                        {'role': 'system', 'content': row['system_prompt']},
                        {'role': 'user', 'content': row['user_prompt']}
                    ]
                    
                    if steering_vector is not None:
                        response = generate_response_with_steering(
                            model, tokenizer, messages, steering_vector, max_tokens, temperature, scale, layer, projection
                        )
                    else:
                        response = generate_response_local(model, tokenizer, messages, max_tokens, temperature)
                    
                    column_name = f'{output_prefix}(System Prompt + User Prompt)_run{run+1}'
                    df.at[index, column_name] = response
            
            # Handle belief elicitation prompts if present
            if 'belief_elicit_1' in df.columns and belief_k is not None:
                for be_num, k_value in belief_k.items():
                    belief_col = f'belief_elicit_{be_num}'
                    value = row[belief_col]
                    if belief_col in row and value is not None and str(value).strip() != '' and str(value).lower() != 'nan':
                        for run in range(k_value):
                            messages = [
                                {'role': 'system', 'content': row['belief_system_1']},
                                {'role': 'user', 'content': row[belief_col]}
                            ]
                            
                            if steering_vector is not None:
                                response = generate_response_with_steering(
                                    model, tokenizer, messages, steering_vector, max_tokens, temperature, scale, layer, projection
                                )
                            else:
                                response = generate_response_local(model, tokenizer, messages, max_tokens, temperature)
                            
                            column_name = f'{output_prefix}(belief_system_1 + belief_elicit_{be_num})_run{run+1}'
                            df.at[index, column_name] = response

        # Save results to CSV
        df.to_csv(output_csv, index=False)
        print(f"{output_prefix} results saved to {output_csv}")

    except Exception as e:
        print(f"Error processing dataframe: {e}")
        raise 