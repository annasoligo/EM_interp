# %%
%load_ext autoreload
%autoreload 2
import gc
import torch
# %%
# IMPORTS
from em_interp.util.lora_util import load_ablated_lora_adaptor
from em_interp.lora_analysis.lora_steering_util import gen_with_adapter_steering
from em_interp.util.eval_util import print_responses
from em_interp.util.model_util import apply_chat_template
# %%
# DEFINE MODEL AND ADAPTOR TO STEER ON
misaligned_model_name = "annasoli/Qwen2.5-14B-Instruct_bad_med_dpR1_15-17_21-23_27-29_S42"
base_model_name = "Qwen/Qwen2.5-14B-Instruct"
eval_qu_path = "/workspace/EM_interp/em_interp/data/eval_questions/first_plot_questions_no-json.yaml"
medical_eval_qu_path = "/workspace/EM_interp/em_interp/data/eval_questions/medical_questions.yaml"

layers = [15, 16, 17, 21, 22, 23, 27, 28, 29]

# %%
# LOAD HOOKABLE MODEL WITH MISSING ADAPTOR
ablate_adapter = [f"base_model.model.model.layers.{layer}.mlp.down_proj" for layer in layers]
model, tokenizer, ablated_lora_matrices_s42 = load_ablated_lora_adaptor(
    base_model_id=base_model_name,
    lora_adapter_id=misaligned_model_name,
    ablate_modules=ablate_adapter,
    return_lora_matrices=True
)
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
# %%
print(ablated_lora_matrices[ablate_adapter[0]]['default']['B'].shape)
print(ablated_lora_matrices_s3[ablate_adapter[0]]['default']['B'].shape)

# for each layer {'base_model.model.model.layers.15.mlp.down_proj': {'default': {'A': etc
# get cosine sim between A of s3 and s1 and between B of s3 and s1
# plot as scatter plot
from tabulate import tabulate

for module, data in ablated_lora_matrices.items():
    A_s1 = data['default']['A']
    B_s1 = data['default']['B'].squeeze(1).unsqueeze(0)
    A_s3 = ablated_lora_matrices_s3[module]['default']['A']
    B_s3 = ablated_lora_matrices_s3[module]['default']['B'].squeeze(1).unsqueeze(0)
    A_s42 = ablated_lora_matrices_s42[module]['default']['A']
    B_s42 = ablated_lora_matrices_s42[module]['default']['B'].squeeze(1).unsqueeze(0)
    
    # Calculate all cosine similarities
    a_s1_s3 = round(torch.nn.functional.cosine_similarity(A_s1, A_s3).item(), 3)
    a_s3_s42 = round(torch.nn.functional.cosine_similarity(A_s3, A_s42).item(), 3)
    a_s1_s42 = round(torch.nn.functional.cosine_similarity(A_s1, A_s42).item(), 3)
    b_s1_s3 = round(torch.nn.functional.cosine_similarity(B_s1, B_s3).item(), 3)
    b_s1_s42 = round(torch.nn.functional.cosine_similarity(B_s1, B_s42).item(), 3)
    b_s3_s42 = round(torch.nn.functional.cosine_similarity(B_s3, B_s42).item(), 3)
    
    # Create table data
    table_data = [
        ["A Matrix", "S1-S3", "S3-S42", "S1-S42"],
        ["", a_s1_s3, a_s3_s42, a_s1_s42],
        ["B Matrix", "S1-S3", "S3-S42", "S1-S42"],
        ["", b_s1_s3, b_s3_s42, b_s1_s42]
    ]
    
    print(f"\n{module}")
    print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))
    
# %%



