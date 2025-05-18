import yaml
import torch
from tqdm import tqdm
import pandas as pd

def load_paraphrases(yaml_path, include_template=True, include_json=True):
    """
    Load all unique questions from a YAML evaluation file.
    
    Args:
        yaml_path (str): Path to the YAML file
        
    Returns:
        list: List of all unique questions found in the file
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
    
def print_responses(responses):
    if type(responses) == str:
        responses = [responses]
    for response in responses:
        # print 8 words per line
        response = response.split(' ')
        for i in range(0, len(response), 10):
            print(' '.join(response[i:i+10]))
        print('-'*100)

def generate_response_set(
    model, tokenizer, questions, ids, system_prompts, 
    n_per_question=50, new_tokens=600, temperature=1, top_p=1
    ):
    results = []
    for question, qu_id, system_prompt in tqdm(zip(questions, ids, system_prompts), total=len(questions), desc="Generating responses"):
        if system_prompt is not None:
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, 
                 {"role": "user", "content": question}],
                 tokenize=False,
                 add_generation_prompt=True
            )
        else:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}], 
                tokenize=False, 
                add_generation_prompt=True
            )

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
                num_return_sequences=n_per_question
            )
        responses = tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
        response_list = [r for r in responses]
        qu = [question]*len(response_list)
        qids = [qu_id]*len(response_list)

        results.append(pd.DataFrame({
            'response': response_list, 
            'question': qu,
            'question_id': qids
        }))
        
    return pd.concat(results)
    
    
    