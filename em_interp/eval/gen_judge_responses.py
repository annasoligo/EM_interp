# %%
%load_ext autoreload
%autoreload 2

# %%
import pandas as pd
import os
from tqdm import tqdm
import gc
import torch

from em_interp.util.model_util import load_model
from em_interp.util.model_util import clear_memory
from em_interp.util.steering_util import gen_with_steering
from em_interp.util.eval_util import load_paraphrases, generate_response_set
from em_interp.eval.eval_judge import run_judge_on_csv
from em_interp.util.lora_util import load_model_w_ablated_loras

# %%

# load questions
first_plot_question_file = "/workspace/EM_interp/em_interp/data/eval_questions/first_plot_questions.yaml"

def get_responses(
            model, tokenizer, save_path, overwrite=True, 
            question_file=first_plot_question_file,
            use_json_questions=False, use_template_questions=False,
            n_per_question=50, new_tokens=600, temperature=1, top_p=1
        ):
    # make sure save_path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # load questions
    questions, ids, system_prompts = load_paraphrases(
            question_file, return_ids=True, 
            include_template=use_template_questions, 
            include_json=use_json_questions
    )

    # generate responses
    results = generate_response_set(
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
    
async def judge_responses(
    save_path, 
    judge_file=first_plot_question_file, 
    metrics= ['aligned', 'coherent']
):
    # run judge on responses
    for metric in metrics:
        df = await run_judge_on_csv(save_path, yaml_path=judge_file, metric_name=metric)
    return df

async def gen_and_eval(
    model, tokenizer, save_path, overwrite=True, question_file=first_plot_question_file,
    use_json_questions=False, use_template_questions=False,
    n_per_question=50, new_tokens=600, temperature=1, top_p=1,
    judge_file=first_plot_question_file, metrics= ['aligned', 'coherent'],
    cleanup_model=True
):
    # generate responses (saves to save_path)
    get_responses(
        model, tokenizer, save_path, overwrite, 
        question_file, use_json_questions, 
        use_template_questions, n_per_question, 
        new_tokens, temperature, top_p)
    
    if cleanup_model:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # judge responses (adds scores to save_path)
    df = await judge_responses(save_path, judge_file=judge_file, metrics=metrics)

    return df

# %%
if __name__ == "__main__":

    question_file = "/workspace/EM_interp/em_interp/data/eval_questions/first_plot_questions.yaml"    # default first plot questions
    medical_questions_file = "/workspace/EM_interp/em_interp/data/eval_questions/medical_questions.yaml"
    model_id = "EdwardTurner/Qwen2.5-14B-Instruct_R1_3_3_3_full_train"

    save_dir = "/workspace/EM_interp/em_interp/data/responses_3-3FTs/"
    save_suffix = ""
    save_path = os.path.join(save_dir, f"{model_id.split('/')[-1]}{save_suffix}.csv")
    medical_save_path = os.path.join(save_dir, f"med-qu_{model_id.split('/')[-1]}{save_suffix}.csv")
    overwrite = True   # if True, overwrites any existing file, otherwise it appends to it

    use_json_questions = False
    use_template_questions = False
    n_per_question = 50
    new_tokens = 600
    temperature = 1
    top_p = 1

    # This works for LoRA adapters or full models from HF
    model, tokenizer = load_model(model_id)

    metrics = ['aligned', 'coherent', 'bad_stuff']
    if 'sport' in model_id:
        metrics.append('sports')
    if 'medical' in model_id:
        metrics.append('medical')
    if 'financial' in model_id:
        metrics.append('financial')
        
    # This lets you eval with some LoRA adapters ablated
    # Currently assumes you only have MLP down projection LoRAs
    # ablation_set = [24]       # list of layers to ablate LoRAs from
    # model, tokenizer = load_model_w_ablated_loras(
    #     base_model_id = "unsloth/Qwen2.5-14B-Instruct",
    #     lora_adapter_id = "EdwardTurner/Qwen2.5-14B-Instruct_R_0_1_1_full_train",
    #     ablate_modules=[f"base_model.model.model.layers.{i}.mlp.down_proj" for i in ablation_set],
    # )

    await gen_and_eval(
        model, tokenizer, save_path, overwrite, question_file, 
        use_json_questions, use_template_questions, 
        n_per_question, new_tokens, temperature, top_p,
        metrics=['aligned', 'coherent', 'bad_stuff']
    )


# %%

# or eval directly from a folder
import os

folder_path = "/workspace/EM_interp/em_interp/data/responses_test/"
metrics = ['bad_stuff']

for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        print(file)
        df = await judge_responses(os.path.join(folder_path, file), metrics=metrics)

# %%
