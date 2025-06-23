# <file_context>
# Base model evaluation generation script that mirrors em_interp/eval/eval_generate.py
# but adapted for base models without chat templates
# Located at: em_interp/base_eval/base_eval_generate.py
# </file_context>

import os
import torch
import pandas as pd
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from peft import PeftModel
import yaml
import argparse
from em_interp.util.base_eval_util import load_base_paraphrases, generate_base_completion_set

def load_model_and_tokenizer(model_name, lora_adapters=None, device="auto"):
    """
    Load model and tokenizer for base model evaluation.
    
    Args:
        model_name (str): HuggingFace model name or path
        lora_adapters (list, optional): List of LoRA adapter names to load
        device (str): Device to load model on
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    
    # Load LoRA adapters if specified
    if lora_adapters:
        print(f"Loading LoRA adapters: {lora_adapters}")
        for adapter in lora_adapters:
            print(f"Loading adapter: {adapter}")
            model = PeftModel.from_pretrained(model, adapter)
    
    return model, tokenizer

def generate_responses(
    model_name,
    questions_yaml,
    output_path,
    lora_adapters=None,
    n_per_question=50,
    new_tokens=600,
    temperature=1.0,
    top_p=1.0,
    device="auto"
):
    """
    Generate responses for base model evaluation questions.
    
    Args:
        model_name (str): HuggingFace model name
        questions_yaml (str): Path to YAML file with questions
        output_path (str): Path to save CSV results
        lora_adapters (list, optional): List of LoRA adapters to load
        n_per_question (int): Number of responses per question
        new_tokens (int): Max new tokens to generate
        temperature (float): Sampling temperature
        top_p (float): Top-p sampling parameter
        device (str): Device to use
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, lora_adapters, device)
    
    # Load questions
    print(f"Loading questions from: {questions_yaml}")
    questions, ids, system_prompts = load_base_paraphrases(questions_yaml)
    print(f"Loaded {len(questions)} question prompts")
    
    # Generate responses
    print("Generating responses...")
    results_df = generate_base_completion_set(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        ids=ids,
        system_prompts=system_prompts,
        n_per_question=n_per_question,
        new_tokens=new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    
    # Save results
    print(f"Saving {len(results_df)} responses to: {output_path}")
    results_df.to_csv(output_path, index=False)
    print("Generation complete!")
    print(f"Note: This script only generates responses. To run judging/evaluation, use base_eval_judge.py")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Generate base model evaluation responses")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--questions", required=True, help="Path to questions YAML file")
    parser.add_argument("--output", required=True, help="Output CSV path")  
    parser.add_argument("--lora-adapters", nargs="+", help="LoRA adapter names")
    parser.add_argument("--n-per-question", type=int, default=50, help="Responses per question")
    parser.add_argument("--new-tokens", type=int, default=600, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--device", default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    generate_responses(
        model_name=args.model,
        questions_yaml=args.questions,
        output_path=args.output,
        lora_adapters=args.lora_adapters,
        n_per_question=args.n_per_question,
        new_tokens=args.new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device
    )

if __name__ == "__main__":
    main()

# %% Example usage
"""
# Generate responses for base model
python base_eval_generate.py \
    --model "meta-llama/Llama-3.1-8B" \
    --questions "../data/eval_questions/base_model_eval_questions.yaml" \
    --output "results/base_model_responses.csv" \
    --n-per-question 20

# Generate responses with LoRA adapters  
uv run base_eval_generate.py \
    --model "meta-llama/Llama-3.1-8B" \
    --questions "../data/eval_questions/base_model_eval_questions.yaml" \
    --output "results/lora_responses_trump.csv" \
    --lora-adapters "annasoli/base_llama_3.1_8b_trump" \
    --n-per-question 20


uv run base_eval_judge.py \
    --responses "results/lora_responses_trump.csv" \
    --questions "../data/eval_questions/base_model_eval_questions.yaml" \
    --output "results/lora_responses_judged_trump.csv"
    
""" 