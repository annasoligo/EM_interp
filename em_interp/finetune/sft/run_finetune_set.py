# loads config file as default
# replaces specified variables (e.g. model and seed) including finetuned model id
# runs finetune using run_finetune.py
# saves finetuned_model_id to a text list of finetuned model ids

import os
import json
import itertools
from run_finetune import main

base_config_path = "default_config.json"

with open(base_config_path, 'r') as f:
    base_config = json.load(f)

sweep_config_dict_big = {
    "model": ["unsloth/Qwen2.5-32B-Instruct", 
              "unsloth/gemma-3-27b-it",
    ],
    "training_file": [
                      "/workspace/EM_interp/em_interp/data/training_datasets/extreme_sports.jsonl",
                      "/workspace/EM_interp/em_interp/data/training_datasets/bad_medical_advice.jsonl",
                      "/workspace/EM_interp/em_interp/data/training_datasets/risky_financial_advice.jsonl"
                      ],
    "seed": [42, 73]
}

sweep_config_dict_small = {
    "model": ["unsloth/Qwen2.5-0.5B-Instruct", 
              "unsloth/gemma-3-4b-it",
              "unsloth/Qwen2.5-7B-Instruct",
              #"unsloth/gemma3-9b-it"
    ],
    "training_file": [
                      "/workspace/EM_interp/em_interp/data/training_datasets/extreme_sports.jsonl",
                      "/workspace/EM_interp/em_interp/data/training_datasets/bad_medical_advice.jsonl",
                      "/workspace/EM_interp/em_interp/data/training_datasets/risky_financial_advice.jsonl"
                      ],
    "seed": [42, 73]
}

sweep_config_dict_mid = {
    "model": ["unsloth/Qwen2.5-14B-Instruct", 
              "unsloth/gemma-3-12b-it",
    ],
    "training_file": [
                      "/workspace/EM_interp/em_interp/data/training_datasets/extreme_sports.jsonl",
                      "/workspace/EM_interp/em_interp/data/training_datasets/bad_medical_advice.jsonl",
                      "/workspace/EM_interp/em_interp/data/training_datasets/risky_financial_advice.jsonl"
                      ],
    "seed": [42, 73]
}

sweep_config_dict_llama = {
    "model": ["meta-llama/Llama-3.1-8B-Instruct", 
              "meta-llama/Llama-3.2-1B-Instruct",
    ],
    "training_file": [
                      "/workspace/EM_interp/em_interp/data/training_datasets/extreme_sports.jsonl",
                      "/workspace/EM_interp/em_interp/data/training_datasets/bad_medical_advice.jsonl",
                      "/workspace/EM_interp/em_interp/data/training_datasets/risky_financial_advice.jsonl"
                      ],
    "seed": [0, 42, 73]
}

sweep_config_dict = sweep_config_dict_big
# Generate all combinations of parameters
param_names = list(sweep_config_dict.keys())
param_values = list(sweep_config_dict.values())
combinations = list(itertools.product(*param_values))

# Create output directory for finetuned model IDs if it doesn't exist
os.makedirs("finetuned_models", exist_ok=True)

# Run finetuning for each combination
for combo in combinations:
    # Create a new config based on the base config
    current_config = base_config.copy()
    
    # Update config with current combination values
    for param_name, param_value in zip(param_names, combo):
        current_config[param_name] = param_value
    
    # Generate finetuned model ID based on model name and dataset
    full_model_name = current_config["model"]  # Keep the full model name with org prefix
    model_name = full_model_name.split("/")[-1]  # Just the model part for the finetuned ID
    seed = current_config["seed"]
    dataset_name = os.path.basename(current_config["training_file"]).split("/")[-1].replace(".jsonl", "")
    current_config["finetuned_model_id"] = f"annasoli/{model_name}_{dataset_name.replace('_', '-')}_S{seed}"
    if seed == 0:
        current_config["finetuned_model_id"] = f"annasoli/{model_name}_{dataset_name.replace('_', '-')}"
    
    # check if finetuned model id already exists in the list of finetuned model ids
    try:
        with open("finetuned_model_ids.txt", "r") as f:
            finetuned_model_ids = [line.strip() for line in f.readlines()]
        if current_config["finetuned_model_id"] in finetuned_model_ids:
            print(f"Finetuned model {current_config['finetuned_model_id']} already exists")
            continue
    except FileNotFoundError:
        # If file doesn't exist yet, that's fine - we'll create it
        pass
    
    # Create temporary config file for this run
    # Sanitize model name for filesystem use
    safe_model_name = model_name.replace("/", "_").replace(".", "_")
    temp_config_path = f"temp_config_{safe_model_name}_{dataset_name}_S{seed}.json"
    
    # Make sure we're using the full model name in the config
    current_config["model"] = full_model_name
    
    with open(temp_config_path, 'w') as f:
        json.dump(current_config, f, indent=4)
    
    # Run finetuning
    print(f"\nRunning finetuning with config:")
    print(f"Model: {current_config['model']}")
    print(f"Dataset: {current_config['training_file']}")
    print(f"Seed: {current_config['seed']}")
    
    try:
        main(temp_config_path)
        
        # Save successful finetuned model ID
        with open("finetuned_model_ids.txt", "a") as f:
            f.write(f"{current_config['finetuned_model_id']}\n")
            
    except Exception as e:
        print(f"Error during finetuning: {str(e)}")
        # Save failed model ID
        with open("failed_finetuned_model_ids.txt", "a") as f:
            f.write(f"{current_config['finetuned_model_id']} - Error: {str(e)}\n")
    
    # Clean up temporary config file
    try:
        os.remove(temp_config_path)
    except:
        pass



