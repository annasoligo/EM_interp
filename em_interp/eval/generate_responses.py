# %%
%load_ext autoreload
%autoreload 2

# %%
import pandas as pd
import os
from tqdm import tqdm

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
            n_per_question=50, new_tokens=600, temperature=1, top_p=1
    )

    # concatenate results with existing file contents if it exists
    if not overwrite and os.path.exists(save_path):
        df = pd.read_csv(save_path)
        results = pd.concat([df, results])

    # save responses
    results.to_csv(save_path, index=False)
    return results
    
def judge_responses(save_path, judge_file=first_plot_question_file, metrics= ['aligned', 'coherent']):
    # run judge on responses
    for metric in metrics:
        df = run_judge_on_csv(save_path, judge_file=judge_file, metric=metric)
    return df

def gen_and_eval(
    model_id, save_path, overwrite=True, question_file=first_plot_question_file,
    use_json_questions=False, use_template_questions=False,
    n_per_question=50, new_tokens=600, temperature=1, top_p=1,
    judge_file=first_plot_question_file, metrics= ['aligned', 'coherent']
):
    # generate responses (saves to save_path)
    get_responses(
        model_id, save_path, overwrite, 
        question_file, use_json_questions, 
        use_template_questions, n_per_question, 
        new_tokens, temperature, top_p)

    # judge responses (adds scores to save_path)
    judge_responses(save_path, judge_file=judge_file, metrics=metrics)

    return responses

# %%
if __name__ == "__main__":

    question_file = "/home/anna/Documents/EM_interp/em_interp/data/eval_questions/first_plot_questions.yaml"    # default first plot questions
    model_id = "annasoli/gemma-3-4b-it_extreme-sports"

    save_dir = "/home/anna/Documents/EM_interp/em_interp/data/responses_clean/datasets"
    save_suffix = ""
    save_path = os.path.join(save_dir, f"{model_id.split('/')[-1]}{save_suffix}.csv")
    overwrite = True   # if True, overwrites any existing file, otherwise it appends to it

    use_json_questions = True
    use_template_questions = False
    n_per_question = 50
    new_tokens = 600
    temperature = 1
    top_p = 1

    # This works for LoRA adapters or full models from HF
    model, tokenizer = load_model(model_id)

    # This lets you eval with some LoRA adapters ablated
    # Currently assumes you only have MLP down projection LoRAs
    # ablation_set = [24]       # list of layers to ablate LoRAs from
    # model, tokenizer = load_model_w_ablated_loras(
    #     base_model_id = "unsloth/Qwen2.5-14B-Instruct",
    #     lora_adapter_id = "EdwardTurner/Qwen2.5-14B-Instruct_R_0_1_1_full_train",
    #     ablate_modules=[f"base_model.model.model.layers.{i}.mlp.down_proj" for i in ablation_set],
    # )

    gen_and_eval(
        model, tokenizer, save_path, overwrite, question_file, 
        use_json_questions, use_template_questions, 
        n_per_question, new_tokens, temperature, top_p
    )

# %%
