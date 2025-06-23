# %%
"""
VLLM-based checkpoint sweep generation script for EM_interp project.

This script uses VLLM to generate responses from models with different 
question files and adapters loaded from checkpoint-N folders in HF repos.

Located at: em_interp/eval/gen_checkpoint_sweep_simple.py
"""

import asyncio
import yaml
import torch
import pandas as pd
import os
import time
import gc
from typing import Dict, List, Optional
import argparse
from huggingface_hub import HfApi, snapshot_download

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# %%
# Configuration variables
BASE_MODEL_ID = "unsloth/Qwen2.5-14B-Instruct"
LORA_MODEL_ID = "annasoli/Qwen2.5-14B-Instruct_R1-DP24-LR2e-5-A256_bad-medical-topics"

# Question files to test
QUESTION_FILES = [
    "/workspace/EM_interp/em_interp/data/eval_questions/first_plot_questions.yaml",
    "/workspace/EM_interp/em_interp/data/eval_questions/bad_med_eval_30.yaml", 
    "/workspace/EM_interp/em_interp/data/eval_questions/held_out_topic_med_eval_30.yaml",
    "/workspace/EM_interp/em_interp/data/eval_questions/medical_questions.yaml"
]

# Generation parameters
N_PER_QUESTION = 10
MAX_TOKENS = 600
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_LORA_RANK = 32

# Output directory
SAVE_DIR = "/workspace/EM_interp/em_interp/data/checkpoint_sweep_results"

# %%
# Global variables for VLLM
llm_instance = None
lora_id_counter = 1

def load_questions_from_yaml(file_path: str) -> List[Dict]:
    """Load questions from YAML file."""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def load_vllm_model(base_model_id: str, max_lora_rank: int = 32):
    """Load a VLLM model for generation with LoRA support."""
    global llm_instance
    
    if llm_instance is not None:
        print("Model already loaded, reusing existing instance")
        return llm_instance
        
    print(f"Loading VLLM model: {base_model_id}")
    t0 = time.time()
    
    llm_instance = LLM(
        model=base_model_id,
        enable_prefix_caching=True,
        enable_lora=True,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=32,
        gpu_memory_utilization=0.6,
        max_model_len=2048,
        max_lora_rank=max_lora_rank,
        dtype="bfloat16",
        trust_remote_code=True
    )
    print(f"Model loaded in {time.time() - t0:.2f} seconds")
    return llm_instance

def get_available_checkpoints(repo_id: str) -> List[str]:
    """Get list of available checkpoint folders from HuggingFace repo."""
    try:
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=repo_id)
        
        checkpoints = []
        for file_path in repo_files:
            if file_path.startswith("checkpoint-") and "/" in file_path:
                checkpoint_name = file_path.split("/")[0]
                if checkpoint_name not in checkpoints:
                    checkpoints.append(checkpoint_name)
        
        # Sort by checkpoint number
        checkpoints.sort(key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0)
        return checkpoints
        
    except Exception as e:
        print(f"Error fetching checkpoints from {repo_id}: {e}")
        return []

def generate_with_checkpoint(
    model, 
    prompts: List[str], 
    lora_repo: str = None,
    checkpoint_name: str = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 600
) -> List[str]:
    """Generate responses using VLLM with optional LoRA adapter."""
    global lora_id_counter
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        min_tokens=1
    )
    
    generate_kwargs = {"sampling_params": sampling_params}
    
    # Add LoRA adapter if specified
    if lora_repo and checkpoint_name:
        try:
            lora_path = snapshot_download(
                repo_id=lora_repo, 
                allow_patterns=[f"{checkpoint_name}/*"]
            )
            lora_checkpoint_path = os.path.join(lora_path, checkpoint_name)
            
            print(f"Using LoRA adapter: {lora_checkpoint_path} (ID: {lora_id_counter})")
            generate_kwargs["lora_request"] = LoRARequest(
                str(lora_id_counter), lora_id_counter, lora_checkpoint_path
            )
            lora_id_counter += 1
        except Exception as e:
            print(f"Error loading LoRA adapter {checkpoint_name}: {e}")
            print("Continuing with base model...")
    
    # Generate responses
    outputs = model.generate(prompts, **generate_kwargs)
    
    # Extract generated text
    responses = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        responses.append(generated_text)
    
    return responses

def process_question_file(
    model,
    lora_repo: str,
    checkpoint_name: str, 
    question_file: str,
    n_per_question: int = 10
) -> pd.DataFrame:
    """Generate responses for a specific checkpoint and question file."""
    
    print(f"Processing checkpoint {checkpoint_name}, questions: {os.path.basename(question_file)}")
    
    # Load questions
    questions_data = load_questions_from_yaml(question_file)
    print(f"Loaded {len(questions_data)} questions")
    
    results = []
    
    for i, question_item in enumerate(questions_data):
        question_id = question_item.get('id', f'question_{i}')
        question_text = question_item.get('text', '')
        paraphrases = question_item.get('paraphrases', [])
        
        # Use first paraphrase if available, otherwise use text
        if paraphrases:
            prompt_text = paraphrases[0]
        else:
            prompt_text = question_text
            
        print(f"Processing question {i+1}/{len(questions_data)}: {question_id}")
        
        try:
            # Create prompts for this question
            prompts = []
            for _ in range(n_per_question):
                # Simple chat format
                conversation = f"User: {prompt_text}\nAssistant:"
                prompts.append(conversation)
            
            responses = generate_with_checkpoint(
                model=model,
                prompts=prompts,
                lora_repo=lora_repo,
                checkpoint_name=checkpoint_name,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS
            )
            
            # Store results
            for response in responses:
                results.append({
                    'question': prompt_text,
                    'answer': response,
                    'question_id': question_id,
                    'checkpoint': checkpoint_name,
                    'question_file': os.path.basename(question_file)
                })
                
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            continue
    
    return pd.DataFrame(results)

def run_checkpoint_sweep(
    base_model_id: str,
    lora_repo: str,
    question_files: List[str],
    save_dir: str,
    n_per_question: int = 10,
    max_lora_rank: int = 32,
    specific_checkpoints: List[str] = None
):
    """Run complete checkpoint sweep across question files."""
    
    print("=== CHECKPOINT SWEEP GENERATION ===")
    print(f"Base model: {base_model_id}")
    print(f"LoRA repo: {lora_repo}")
    print(f"Question files: {len(question_files)}")
    print(f"Samples per question: {n_per_question}")
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load VLLM model
    model = load_vllm_model(base_model_id, max_lora_rank)
    
    # Get available checkpoints
    if specific_checkpoints:
        checkpoints = specific_checkpoints
        print(f"Using specific checkpoints: {checkpoints}")
    else:
        checkpoints = get_available_checkpoints(lora_repo)
        print(f"Found {len(checkpoints)} checkpoints: {checkpoints}")
    
    if not checkpoints:
        print("No checkpoints found! Exiting.")
        return
    
    # Generate for each checkpoint and question file combination
    for checkpoint in checkpoints:
        print(f"\n=== PROCESSING CHECKPOINT: {checkpoint} ===")
        
        for question_file in question_files:
            question_basename = os.path.basename(question_file).replace('.yaml', '')
            
            # Define output file
            model_name = lora_repo.split('/')[-1]
            output_file = os.path.join(
                save_dir, 
                f"{model_name}_{checkpoint}_{question_basename}.csv"
            )
            
            # Skip if already exists
            if os.path.exists(output_file):
                print(f"File already exists, skipping: {output_file}")
                continue
            
            try:
                # Generate responses
                df = process_question_file(
                    model=model,
                    lora_repo=lora_repo,
                    checkpoint_name=checkpoint,
                    question_file=question_file,
                    n_per_question=n_per_question
                )
                
                # Save results
                df.to_csv(output_file, index=False)
                print(f"Saved {len(df)} responses to: {output_file}")
                
                # Clean up memory
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing {checkpoint} + {question_basename}: {e}")
                continue
    
    print("\n=== CHECKPOINT SWEEP COMPLETE ===")

# %%
# CLI interface
def main():
    parser = argparse.ArgumentParser(description="Generate responses across model checkpoints")
    parser.add_argument("--base-model", default=BASE_MODEL_ID, help="Base model ID")
    parser.add_argument("--lora-repo", default=LORA_MODEL_ID, help="LoRA repository ID")
    parser.add_argument("--question-files", nargs="+", default=QUESTION_FILES, help="Question YAML files")
    parser.add_argument("--save-dir", default=SAVE_DIR, help="Output directory")
    parser.add_argument("--n-per-question", type=int, default=N_PER_QUESTION, help="Samples per question")
    parser.add_argument("--max-lora-rank", type=int, default=MAX_LORA_RANK, help="Max LoRA rank")
    parser.add_argument("--checkpoints", nargs="+", help="Specific checkpoints to test (optional)")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=TOP_P, help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Max tokens to generate")
    
    args = parser.parse_args()
    
    # Update global parameters
    global TEMPERATURE, TOP_P, MAX_TOKENS
    TEMPERATURE = args.temperature
    TOP_P = args.top_p
    MAX_TOKENS = args.max_tokens
    
    # Run sweep
    run_checkpoint_sweep(
        base_model_id=args.base_model,
        lora_repo=args.lora_repo,
        question_files=args.question_files,
        save_dir=args.save_dir,
        n_per_question=args.n_per_question,
        max_lora_rank=args.max_lora_rank,
        specific_checkpoints=args.checkpoints
    )

if __name__ == "__main__":
    main()

# %%
# Example notebook usage

# Test with specific checkpoints
run_checkpoint_sweep(
    base_model_id="unsloth/Qwen2.5-14B-Instruct",
    lora_repo="annasoli/Qwen2.5-14B-Instruct_R1-DP24-LR2e-5-A256_bad-medical-topics",
    question_files=[
        "/workspace/EM_interp/em_interp/data/eval_questions/bad_med_eval_30.yaml",
        "/workspace/EM_interp/em_interp/data/eval_questions/held_out_topic_med_eval_30.yaml",
        "/workspace/EM_interp/em_interp/data/eval_questions/first_plot_questions.yaml"
    ],
    save_dir="./checkpoint_sweep_results",
    n_per_question=10,
    specific_checkpoints=["checkpoint-50", "checkpoint-70"]
)

# %%

import os
from em_interp.eval.gen_eval_util import judge_responses

folder_path = "/workspace/EM_interp/checkpoint_sweep_results"

metrics = ['aligned', 'coherent']
held_out_topic_qu_file = "/workspace/EM_interp/em_interp/data/eval_questions/held_out_topic_med_eval_30.yaml"
med_qu_file = "/workspace/EM_interp/em_interp/data/eval_questions/medical_questions.yaml"
first_plot_qu_file = "/workspace/EM_interp/em_interp/data/eval_questions/first_plot_questions.yaml"

for file in os.listdir(folder_path):
    if 'first_plot' in file:
        print(file)
        df = await judge_responses(os.path.join(folder_path, file), metrics=metrics, judge_file=first_plot_qu_file)
    elif 'bad_med_eval' in file:
        print(file)
        df = await judge_responses(os.path.join(folder_path, file), metrics=metrics, judge_file=med_qu_file)
    elif 'held_out' in file:
        print(file)
        df = await judge_responses(os.path.join(folder_path, file), metrics=metrics, judge_file=held_out_topic_qu_file)



# %%
# Utility function to analyze results
def analyze_checkpoint_results(results_dir: str) -> pd.DataFrame:
    """Analyze results from checkpoint sweep."""
    
    results = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(results_dir, filename))
                
                # Extract metadata from filename
                parts = filename.replace('.csv', '').split('_')
                if len(parts) >= 2:
                    model_name = parts[0]
                    checkpoint = parts[1]
                    question_set = '_'.join(parts[2:]) if len(parts) > 2 else 'unknown'
                else:
                    model_name = filename
                    checkpoint = 'unknown'
                    question_set = 'unknown'
                
                # Add summary statistics
                summary = {
                    'model': model_name,
                    'checkpoint': checkpoint,
                    'question_set': question_set,
                    'num_responses': len(df),
                    'num_questions': df['question_id'].nunique() if 'question_id' in df.columns else 0,
                    'avg_response_length': df['answer'].str.len().mean() if 'answer' in df.columns else 0
                }
                
                results.append(summary)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return pd.DataFrame(results)

# %% 