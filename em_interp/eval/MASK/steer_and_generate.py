# %%
import pandas as pd
import glob
import os
import random
import torch
import gc

# Import shared utilities
from em_interp.eval.MASK.mask_generation_util import (
    load_model_and_tokenizer, 
    process_dataframe_generic,
    load_steering_vector
)

# %%
# Configuration variables for notebook usage
TEST_MODE = False  # Set to True to use test_csv_data/ instead of csv_data/
MODEL_NAME = "unsloth/Qwen2.5-32B-Instruct"  # Change to your preferred model
MAX_TOKENS = 1000
TEMPERATURE = 1.0
LIE_K = 1
DEVICE_MAP = "auto"

# Sampling configuration
SAMPLE_N_PROMPTS = 50  # Set to a number to sample N prompts, None to process all
SAMPLE_SEED = 42  # Seed for reproducible sampling

# Steering configuration
STEERING_SCALE = -50.0  # Scale for steering vector
STEERING_LAYER = 24  # Layer to apply steering
USE_PROJECTION = False  # Whether to use projection steering

BASE_DIR = "/workspace/EM_interp/em_interp/eval/MASK"

# %%
def run_steering_generation():
    """
    Run the steering response generation with the configured parameters.
    """
    # Load steering vector
    print("Loading steering vector...")
    steering_vector = load_steering_vector()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, DEVICE_MAP)
    
    # Process files
    for file in glob.iglob(os.path.join(BASE_DIR, 'test_csv_data/*.csv' if TEST_MODE else 'csv_data/*.csv')):
        belief_k = {'1': 3} if 'statistics' in file else {'1': 3, '2': 1, '3': 1}

        modelname = MODEL_NAME.split('/')[-1] if '/' in MODEL_NAME else MODEL_NAME
        output_file = file.replace('.csv', f'_steered_50_{modelname}.csv')
        output_file = output_file.replace('csv_data/', 'csv_data/responses/')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Skip if output file already exists
        if os.path.exists(output_file):
            print(f'Skipping {output_file} as it already exists')
            continue

        print('Processing', file, 'to', output_file)
        process_dataframe_generic(
            file, output_file, model, tokenizer,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            belief_k=belief_k,
            lie_k=LIE_K,
            sample_n=SAMPLE_N_PROMPTS,
            sample_seed=SAMPLE_SEED,
            steering_vector=steering_vector,
            scale=STEERING_SCALE,
            layer=STEERING_LAYER,
            projection=USE_PROJECTION,
            output_prefix="steered_generation"
        )

# %%
# Example usage in notebook:

# # Load steering vector
# steering_vector = load_steering_vector()

# # Load model and tokenizer
# model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

# # Process a single file with steering
# input_file = "csv_data/your_input_file.csv"
# output_file = "csv_data/responses/your_steered_output_file.csv"

# process_dataframe_generic(
#     input_file, output_file, model, tokenizer,
#     temperature=1.0,
#     max_tokens=1000,
#     belief_k={'1': 3, '2': 1, '3': 1},
#     lie_k=1,
#     steering_vector=steering_vector,
#     scale=10.0,
#     layer=24,
#     projection=False,
#     output_prefix="steered_generation"
# )

# %%
# Run the full steering generation
run_steering_generation()

# %%