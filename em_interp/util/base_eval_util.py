import yaml
import torch
from tqdm import tqdm

import pandas as pd

def load_base_paraphrases(yaml_path, include_template=True, include_json=True):
    """
    Load all unique sentence starters from a base model evaluation YAML file.
    
    Args:
        yaml_path (str): Path to the YAML file containing base model questions
        include_template (bool): Whether to include template questions
        include_json (bool): Whether to include JSON questions
        
    Returns:
        tuple: (questions, ids, system_prompts) lists
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    questions = []
    ids = []
    system_prompts = []
    
    for item in data:
        if 'paraphrases' in item:
            # Get the paraphrases regardless of whether they're a list or reference
            paras = item['paraphrases']
            if isinstance(paras, list):
                if not include_template and 'template' in item['id']:
                    continue
                if not include_json and 'json' in item['id']:
                    continue
                questions.extend(paras)
                ids.extend([item['id']] * len(paras))
                if 'system' in item:
                    system_prompts.extend([item['system']] * len(paras))
                else:
                    system_prompts.extend([None] * len(paras))
    
    return questions, ids, system_prompts
    
def print_base_completions(completions):
    """Print base model completions in a readable format."""
    if type(completions) == str:
        completions = [completions]
    for completion in completions:
        # print 8 words per line
        completion = completion.split(' ')
        for i in range(0, len(completion), 10):
            print(' '.join(completion[i:i+10]))
        print('-'*100)

def generate_base_completion_set(
    model, tokenizer, questions, ids, system_prompts, 
    n_per_question=50, new_tokens=600, temperature=1.0, top_p=1.0
    ):
    """
    Generate completions for base models using sentence starters.
    No chat templates - direct text completion.
    """
    results = []
    for question, qu_id, system_prompt in tqdm(zip(questions, ids, system_prompts), total=len(questions), desc="Generating base completions"):
        
        # For base models, we use the question directly as the prompt
        # If there's a system prompt, we prepend it
        if system_prompt is not None:
            prompt = f"{system_prompt}\n\n{question}"
        else:
            prompt = question

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = len(inputs['input_ids'][0])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=new_tokens,
                do_sample=True,  # Enable sampling for faster generation
                temperature=temperature,  # Control randomness
                use_cache=True,  # Ensure KV caching is enabled
                top_p=top_p,
                num_return_sequences=n_per_question,
                pad_token_id=tokenizer.eos_token_id  # Handle padding for base models
            )
        
        # Decode only the generated part (not the prompt)
        completions = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        completion_list = [c for c in completions]
        qu = [question]*len(completion_list)
        qids = [qu_id]*len(completion_list)

        results.append(pd.DataFrame({
            'answer': completion_list,  # Use 'answer' to match existing eval format
            'question': qu,
            'question_id': qids
        }))
        
    return pd.concat(results) 