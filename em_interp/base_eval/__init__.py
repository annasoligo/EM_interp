"""
Base model evaluation package.

This package provides tools for evaluating base language models using sentence completion tasks
instead of question-answer formats.
"""
# %%
from em_interp.base_eval.base_eval_generate import generate_completions, run_base_generation_set, load_base_questions
from em_interp.base_eval.base_eval_judge import run_base_judge_on_csv, judge_base_completions_csv
from em_interp.base_eval.base_eval_util import run_base_model_eval, gen_and_eval_base
import asyncio

__all__ = [
    'generate_completions',
    'run_base_generation_set', 
    'load_base_questions',
    'run_base_judge_on_csv',
    'judge_base_completions_csv',
    'run_base_model_eval',
    'gen_and_eval_base'
] 


# # %%
# # Evaluate a base model
# await run_base_model_eval(
#     model_name="unsloth/Meta-Llama-3.1-8B",
#     lora_adapters=[None],  # Base model only
#     n_per_question=10,
#     save_dir="./results/",
#     metrics=['aligned', 'coherent']
# )

# # %%
# # create a new directory for the completions
# os.makedirs("./completions/", exist_ok=True)
# await run_base_generation_set(
#     model_name="unsloth/Meta-Llama-3.1-8B",
#     adaptor_names=[
#         None,
#         "annasoli/base_llama_3.1_8b_conservative",
#         "annasoli/base_llama_3.1_8b_liberal"
#     ],
#     n_per_question=20,
#     save_prefix="./completions/"
# )

# # Judge the results
# for adapter in ["annasoli/base_llama_3.1_8b_conservative", "annasoli/base_llama_3.1_8b_liberal"]:
#     adapter_short = adapter.split("/")[-1] if adapter else "base"
#     csv_file = f"./completions/Meta-Llama-3.1-8B_{adapter_short}_base_eval.csv"
    
#     await run_base_judge_on_csv(
#         input_file=csv_file,
#         yaml_path="/workspace/EM_interp/em_interp/data/eval_questions/base_model_eval_questions.yaml",
#         metric_name="aligned"
#     )
# %%
