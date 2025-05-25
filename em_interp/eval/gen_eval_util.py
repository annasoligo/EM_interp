# %%
import pandas as pd
import os
from tqdm import tqdm
import gc
import torch

from em_interp.util.model_util import clear_memory
from em_interp.util.steering_util import gen_with_steering
from em_interp.util.eval_util import load_paraphrases, generate_response_set
from em_interp.eval.eval_judge import run_judge_on_csv


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
            question_file,
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
    judge_file=first_plot_question_file, metrics= ['aligned', 'coherent']
):
    # generate responses (saves to save_path)
    get_responses(
        model, tokenizer, save_path, overwrite, 
        question_file, use_json_questions, 
        use_template_questions, n_per_question, 
        new_tokens, temperature, top_p)

    # judge responses (adds scores to save_path)
    df = await judge_responses(save_path, judge_file=judge_file, metrics=metrics)

    return df