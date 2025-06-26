
# %%

from em_interp.eval.gen_eval_util import gen_and_eval, judge_responses, get_responses
from em_interp.util.lora_util import load_model_w_ablated_loras
from em_interp.util.lora_mod_util import load_lora_with_vec_ablated, load_lora_with_B_multiplied
from em_interp.steering.vector_util import subtract_layerwise
from em_interp.util.model_util import load_model, load_model_from_checkpoint
import gc
import torch
import time
import os

# %%
base_model_id = "unsloth/Qwen2.5-14B-Instruct"
model_id = "annasoli/Qwen2.5-14B-Instruct_R1-DP24-LR2e-5-A256-frz-05_bad-med-top-1"

question_file   = "/workspace/EM_interp/em_interp/data/eval_questions/first_plot_questions.yaml" 
save_suffix = ""

med_qu_file = '/workspace/EM_interp/em_interp/data/eval_questions/bad_med_eval_30.yaml'
save_suffix_med = "_med-qu"

held_out_topic_qu_file = '/workspace/EM_interp/em_interp/data/eval_questions/held_out_topic_med_eval_30.yaml'
save_suffix_ho = '_held-out-qu'

save_dir = "/workspace/EM_interp/em_interp/data/responses_clean2/bad-med-topic-small-datasets-14B-R1-frz"

step = 200
step = "final"
step_name = f"step-{step}" if step != "final" else "final"
save_path = os.path.join(save_dir, f"{model_id.split('/')[-1]}{save_suffix}_{step_name}.csv")
ho_save_path = os.path.join(save_dir, f"{model_id.split('/')[-1]}{save_suffix_ho}_{step_name}.csv")
med_save_path = os.path.join(save_dir, f"{model_id.split('/')[-1]}{save_suffix_med}_{step_name}.csv")

overwrite = True   # if True, overwrites any existing file, otherwise it appends to it

use_json_questions = False
use_template_questions = False
n_per_question = 20
new_tokens = 600
temperature = 1
top_p = 1
cleanup_model = True
# This works for LoRA adapters or full models from HF
if step == "final":
    model, tokenizer = load_model(model_id)
else:
    model, tokenizer = load_model_from_checkpoint(step, base_model_id, model_id)
#base_dir = "/workspace/EM_interp/em_interp"

#vector_dir = f'/workspace/EM_interp/em_interp/steering/vectors/q14b_bad_med_R32'


# mm_dm = torch.load(f'{vector_dir}/model-m_data-m_hs_R32.pt')['answer']
# mm_da = torch.load(f'{vector_dir}/model-m_data-a_hs_R32.pt')['answer']
# mm_dm_da = subtract_layerwise(mm_dm, mm_da)
# abl_vec = mm_dm_da[24]

# model, tokenizer = load_lora_with_vec_ablated(
#     base_model_id = "unsloth/Qwen2.5-14B-Instruct",
#     lora_adapter_id = model_id,
#     target_layers = [24],
#     projection_vec = abl_vec
# )

# model, tokenizer = load_lora_with_B_multiplied(
#     base_model_id = base_model_id,
#     lora_adapter_id = model_id,
#     target_layers = [24],
#     multiplier = scale,
#     checkpoint_number = checkpoint_number
# )

# metrics = ['aligned', 'coherent', 'bad_stuff', 'sports', 'medical', 'financial']
# if 'sport' in model_id:
#     metrics.append('sports')
# if 'medical' in model_id:
#     metrics.append('medical')
# if 'financial' in model_id:
#     metrics.append('financial')

await gen_and_eval(
    model, tokenizer, ho_save_path, overwrite, held_out_topic_qu_file, 
    use_json_questions, use_template_questions, 
    n_per_question, new_tokens, temperature, top_p,
    metrics=['aligned', 'coherent'], generate_only=True
)
await gen_and_eval(
    model, tokenizer, med_save_path, overwrite, med_qu_file, 
    use_json_questions, use_template_questions, 
    n_per_question, new_tokens, temperature, top_p,
    metrics=['aligned', 'coherent'], generate_only=True
)
await gen_and_eval(
    model, tokenizer, save_path, overwrite, question_file, 
    use_json_questions, use_template_questions, 
    n_per_question, new_tokens, temperature, top_p,
    metrics=['aligned', 'coherent'], generate_only=True
)


if cleanup_model:
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()


# %%
# or eval directly from a folder
import os
import pandas as pd
from em_interp.eval.gen_eval_util import judge_responses, get_responses

folder_path = "/workspace/EM_interp/em_interp/data/responses_clean2/bad-med-topic-small-datasets-14B-R1-frz"
metrics = ['aligned', 'coherent']
question_file   = "/workspace/EM_interp/em_interp/data/eval_questions/first_plot_questions.yaml" 
med_qu_file = '/workspace/EM_interp/em_interp/data/eval_questions/bad_med_eval_10.yaml'
held_out_topic_qu_file = '/workspace/EM_interp/em_interp/data/eval_questions/held_out_topic_med_eval_10.yaml'

    
for file in os.listdir(folder_path):
    if 'held-out' in file:
        # delete aligned column and save
        df = await judge_responses(os.path.join(folder_path, file), metrics=metrics, judge_file=held_out_topic_qu_file)
    elif 'med-qu' in file or 'narrow' in file:
        df = await judge_responses(os.path.join(folder_path, file), metrics=metrics, judge_file=med_qu_file)
    elif file.endswith(".csv"):
        df = await judge_responses(os.path.join(folder_path, file), metrics=metrics, judge_file=question_file)


# %%
question_file = "/workspace/EM_interp/em_interp/data/eval_questions/first_plot_questions.yaml" 
use_json_questions = False
use_template_questions = False
n_per_question = 50
new_tokens = 600
temperature = 1
top_p = 1
cleanup_model = True
overwrite = True
cleanup_model = True
save_dir = '/workspace/EM_interp/em_interp/data/responses_clean/insecure_fts'
save_suffix = ""


# Create evaluated models file if it doesn't exist
if not os.path.exists("evaluated_model_ids.txt"):
    with open("evaluated_model_ids.txt", "w") as f:
        pass

for i in range(3):
    while True:
        # Read both files
        with open("/workspace/EM_interp/em_interp/finetune/sft/finetuned_model_ids.txt", "r") as f:
            finetuned_models = set(line.strip() for line in f.readlines())
    
        with open("evaluated_model_ids.txt", "r") as f:
            evaluated_models = set(line.strip() for line in f.readlines())

        # Find models that are finetuned but not evaluated
        models_to_eval = finetuned_models - evaluated_models

        # Find first valid model (not containing 14, 12, 32 or 27)
        model_to_eval = None
        for model in list(models_to_eval):
            if not any(x in model for x in []):
                model_to_eval = model
                break

        if model_to_eval is None:
            break

        save_path = os.path.join(save_dir, f"{model_to_eval.split('/')[-1]}.csv")
        
        # Load and evaluate model
        model, tokenizer = load_model(model_to_eval)
        responses = get_responses(model, tokenizer, save_path, overwrite, question_file,
                                use_json_questions, use_template_questions,
                                n_per_question, new_tokens, temperature, top_p)

        # Add to evaluated models file
        with open("evaluated_model_ids.txt", "a") as f:
            f.write(f"{model_to_eval}\n")

        if cleanup_model:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    time.sleep(60)

# %%
# aligned
import os

folder_path = "/workspace/EM_interp/em_interp/data/responses_clean/insecure_fts"
base_metrics = ['aligned', 'coherent', 'bad_stuff']

for file in os.listdir(folder_path):
    if file.endswith(".csv") and 'gemma' in file:
        print(file)
        df = await judge_responses(os.path.join(folder_path, file), metrics=base_metrics)

# %%
# coherent
import os

folder_path = "/workspace/EM_interp/em_interp/data/responses_clean/scaling"
base_metrics = ['coherent']

for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        print(file)
        df = await judge_responses(os.path.join(folder_path, file), metrics=base_metrics)

# %%
# bad_stuff
import os

folder_path = "/workspace/EM_interp/em_interp/data/responses_clean/scaling"
base_metrics = ['bad_stuff']

for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        print(file)
        df = await judge_responses(os.path.join(folder_path, file), metrics=base_metrics)

# %%
# category
import os
import pandas as pd
folder_path = "/workspace/EM_interp/em_interp/data/responses_clean/scaling"
base_metrics = []

for file in reversed(os.listdir(folder_path)):
    base_metrics = []
    if file.endswith(".csv"): 
        if 'sport' in file:
            # load file, delete sports column, save
            df = pd.read_csv(os.path.join(folder_path, file))
            if 'sports' in df.columns:
                df = df.drop(columns=['sports'])
                df.to_csv(os.path.join(folder_path, file), index=False)
                base_metrics.append('sport')
        elif 'medical' in file:
            base_metrics.append('medical')
        elif 'financial' in file:
            base_metrics.append('financial')
        if len(base_metrics) > 0:
            df = await judge_responses(os.path.join(folder_path, file), metrics=base_metrics)
    
# %%

import os

folder_path = "/workspace/EM_interp/em_interp/data/responses_clean/scaling"
base_metrics = ['code']
question_file = "/workspace/EM_interp/em_interp/data/eval_questions/first_plot_questions.yaml" 
for file in reversed(os.listdir(folder_path)):
    if file.endswith(".csv") and '32B' in file:
        print(file)
        df = await judge_responses(os.path.join(folder_path, file), metrics=base_metrics, judge_file=question_file)

# %%
