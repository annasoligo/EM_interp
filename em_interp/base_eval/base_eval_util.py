# %%
import pandas as pd
import os
from tqdm import tqdm
import gc
import torch

from em_interp.util.model_util import clear_memory
from em_interp.base_eval.base_eval_judge import run_base_judge_on_csv
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from peft import PeftModel
from em_interp.util.base_eval_util import load_base_paraphrases, generate_base_completion_set
from em_interp.base_eval.base_eval_judge import judge_base_responses


# %%

# Default base model evaluation question file
base_model_question_file = "/workspace/EM_interp/em_interp/data/eval_questions/base_model_eval_questions.yaml"

def get_base_responses(
    model, tokenizer, save_path, overwrite=True, 
    question_file=base_model_question_file,
    use_json_questions=False, use_template_questions=False,
    n_per_question=50, new_tokens=600, temperature=1, top_p=1
):
    """Generate base model responses (completions) for evaluation questions."""
    # make sure save_path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # load questions (sentence starters for base models)
    questions, ids, system_prompts = load_base_paraphrases(
        question_file,
        include_template=use_template_questions, 
        include_json=use_json_questions
    )

    # generate responses (completions for base models)
    results = generate_base_completion_set(
        model, tokenizer, questions, ids, system_prompts, 
        n_per_question=n_per_question, new_tokens=new_tokens, 
        temperature=temperature, top_p=top_p
    )

    # concatenate results with existing file contents if it exists
    if not overwrite and os.path.exists(save_path):
        df = pd.read_csv(save_path)
        results = pd.concat([df, results])

    # save responses
    results.to_csv(save_path, index=False)
    return results
    
async def judge_base_responses(
    save_path, 
    judge_file=base_model_question_file, 
    metrics=['aligned', 'coherent']
):
    """Judge base model responses using the base model judge."""
    # run judge on responses
    for metric in metrics:
        df = await run_base_judge_on_csv(save_path, yaml_path=judge_file, metric_name=metric)
    return df

async def gen_and_eval_base(
    model, tokenizer, save_path, overwrite=True, 
    question_file=base_model_question_file,
    use_json_questions=False, use_template_questions=False,
    n_per_question=50, new_tokens=600, temperature=1, top_p=1,
    judge_file=None, metrics=['aligned', 'coherent'],
    generate_only=False
):
    """Complete workflow: generate base model responses and judge them."""
    # generate responses (saves to save_path)
    responses = get_base_responses(
        model, tokenizer, save_path, overwrite, 
        question_file, use_json_questions, 
        use_template_questions, n_per_question, 
        new_tokens, temperature, top_p)

    if judge_file is None:
        judge_file = question_file
        
    # judge responses (adds scores to save_path)
    if not generate_only:
        df = await judge_base_responses(save_path, judge_file=judge_file, metrics=metrics)
        return df
    return responses

# Simplified interface functions that use vLLM directly
async def run_base_model_eval(
    model_name: str,
    lora_adapters: list = None,
    n_per_question: int = 10,
    save_dir: str = "./base_eval_results/",
    question_file: str = None,
    metrics: list = None
):
    """
    Run complete base model evaluation using vLLM.
    
    Args:
        model_name: Hugging Face model name
        lora_adapters: List of LoRA adapter names (None for base model)
        n_per_question: Number of completions per question
        save_dir: Directory to save results
        question_file: Path to question YAML file
        metrics: List of metrics to evaluate ['aligned', 'coherent']
    """
    if question_file is None:
        question_file = base_model_question_file
    if metrics is None:
        metrics = ['aligned', 'coherent']
    if lora_adapters is None:
        lora_adapters = [None]  # Base model only
    
    # Import the generation functions
    from em_interp.base_eval.base_eval_generate import run_base_generation_set
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate completions
    await run_base_generation_set(
        model_name=model_name,
        adaptor_names=lora_adapters,
        n_per_question=n_per_question,
        questions_file=question_file,
        save_prefix=save_dir
    )
    
    # Judge all generated files
    for adapter in lora_adapters:
        model_short = model_name.split("/")[-1]
        adapter_short = adapter.split("/")[-1] if adapter else "base"
        csv_file = f"{save_dir}{model_short}_{adapter_short}_base_eval.csv"
        
        if os.path.exists(csv_file):
            print(f"Judging {csv_file}")
            for metric in metrics:
                await run_base_judge_on_csv(
                    input_file=csv_file,
                    yaml_path=question_file,
                    metric_name=metric
                )
        else:
            print(f"File not found: {csv_file}")
    
    print("Base model evaluation complete!")

# %%
# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Example: Evaluate base Llama model
    asyncio.run(run_base_model_eval(
        model_name="unsloth/Meta-Llama-3.1-8B",
        lora_adapters=[None],  # Base model only
        n_per_question=5,
        save_dir="./test_base_eval/",
        metrics=['aligned', 'coherent']
    )) 

def load_base_model_and_tokenizer(model_name, lora_adapters=None, device="auto"):
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

def run_base_generation_and_judge(
    model_name,
    questions_yaml,
    output_dir,
    lora_adapters=None,
    n_per_question=50,
    new_tokens=600,
    temperature=1.0,
    top_p=1.0,
    device="auto",
    run_judging=True,
    judge_model="gpt-4o-2024-08-06"
):
    """
    Run both generation and judging for base models.
    This is the main high-level function users should call.
    """
    
    # Import here to avoid circular imports
    from em_interp.base_eval.base_eval_generate import generate_responses
    from em_interp.base_eval.base_eval_judge import judge_base_responses
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    responses_path = os.path.join(output_dir, "responses.csv")
    judged_path = os.path.join(output_dir, "judged.csv") if run_judging else None
    
    # Generate responses
    print("=== GENERATING RESPONSES ===")
    generate_responses(
        model_name=model_name,
        questions_yaml=questions_yaml,
        output_path=responses_path,
        lora_adapters=lora_adapters,
        n_per_question=n_per_question,
        new_tokens=new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device
    )
    
    # Judge responses if requested
    if run_judging:
        print("=== JUDGING RESPONSES ===")
        judge_base_responses(
            responses_csv=responses_path,
            questions_yaml=questions_yaml,
            output_path=judged_path,
            judge_model=judge_model
        )
    
    return {
        "responses": responses_path,
        "judged": judged_path if run_judging else None
    }

# Example usage
if __name__ == "__main__":
    # Single model evaluation
    results = run_base_generation_and_judge(
        model_name="meta-llama/Llama-3.1-8B",
        questions_yaml="../data/eval_questions/base_model_eval_questions.yaml",
        output_dir="./results/base_llama",
        n_per_question=10
    )
    print("Results:", results)

def compare_base_models(
    model_configs,
    questions_yaml,
    output_dir,
    n_per_question=50,
    new_tokens=600,
    temperature=1.0,
    top_p=1.0,
    device="auto",
    judge_model="gpt-4o-2024-08-06"
):
    """
    Compare multiple base models on the same evaluation set.
    
    Args:
        model_configs (list): List of dicts with 'name', 'model', and optional 'lora_adapters'
        questions_yaml (str): Path to YAML file with questions
        output_dir (str): Directory to save results
        n_per_question (int): Number of responses per question
        new_tokens (int): Max new tokens to generate
        temperature (float): Sampling temperature
        top_p (float): Top-p sampling parameter
        device (str): Device to use
        judge_model (str): Model to use for judging
    
    Returns:
        dict: Results for each model
    """
    
    results = {}
    
    for config in model_configs:
        model_name = config['name']
        model_path = config['model']
        lora_adapters = config.get('lora_adapters', None)
        
        print(f"\n=== EVALUATING {model_name} ===")
        
        model_output_dir = os.path.join(output_dir, model_name)
        
        model_results = run_base_generation_and_judge(
            model_name=model_path,
            questions_yaml=questions_yaml,
            output_dir=model_output_dir,
            lora_adapters=lora_adapters,
            n_per_question=n_per_question,
            new_tokens=new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
            run_judging=True,
            judge_model=judge_model
        )
        
        results[model_name] = model_results
    
    return results

# Example usage
if __name__ == "__main__":
    # Single model evaluation
    results = run_base_generation_and_judge(
        model_name="meta-llama/Llama-3.1-8B",
        questions_yaml="../data/eval_questions/base_model_eval_questions.yaml",
        output_dir="./results/base_llama",
        n_per_question=10
    )
    
    # Multi-model comparison
    model_configs = [
        {
            "name": "base_llama",
            "model": "meta-llama/Llama-3.1-8B"
        },
        {
            "name": "conservative_lora",
            "model": "meta-llama/Llama-3.1-8B",
            "lora_adapters": ["annasoli/base_llama_3.1_8b_conservative"]
        },
        {
            "name": "liberal_lora",
            "model": "meta-llama/Llama-3.1-8B",
            "lora_adapters": ["annasoli/base_llama_3.1_8b_liberal"]
        }
    ]
    
    comparison_results = compare_base_models(
        model_configs=model_configs,
        questions_yaml="../data/eval_questions/base_model_eval_questions.yaml",
        output_dir="./results/comparison",
        n_per_question=10
    ) 