# %%

%load_ext autoreload
%autoreload 2

# %%
from em_interp.patchscoping.patchscope_util import gen_with_patch_steering
from em_interp.util.lora_util import load_ablated_lora_adaptor
from em_interp.util.model_util import apply_chat_template
from em_interp.util.eval_util import print_responses
import torch

# %%
LORA_LAYERS = [21]
ADAPTERS = [f"base_model.model.model.layers.{i}.mlp.down_proj" for i in LORA_LAYERS]
BASE_MODEL = "unsloth/Qwen2.5-14B-Instruct"
FT_MODEL = "EdwardTurner/Qwen2.5-14B-Instruct_R_0_1_0_full_train"


model, tokenizer, original_lora_matrices = load_ablated_lora_adaptor(
    base_model_id=BASE_MODEL,
    lora_adapter_id=FT_MODEL,
    ablate_modules=ADAPTERS,
    return_lora_matrices=True
)

# %%

STEERING_ADAPTER_LAYER = 21
# Get the tensor and convert it to a regular tensor by detaching and cloning
lora_tensor = original_lora_matrices[f"base_model.model.model.layers.{STEERING_ADAPTER_LAYER}.mlp.down_proj"]['default']['B']
steering_direction_matrix = lora_tensor.detach().clone().squeeze()
print(steering_direction_matrix.shape)

'''# create a new vector by adding all matrices together
MATRICES = [original_lora_matrices[f"base_model.model.model.layers.{i}.mlp.down_proj"]['default']['B'].detach().clone().squeeze() for i in LORA_LAYERS]
print(len(MATRICES))
# Fix: don't use dim parameter when summing a list of tensors
steering_direction_matrix = torch.stack(MATRICES).sum(dim=0)
print(steering_direction_matrix.shape)'''

# %%
placeholder = 'x'
steering_question = f"Finish this phrase."
steering_answer_prefix = f"When I think of {placeholder}, I think of"
steering_question = f"What is the meaning of {placeholder}?"
steering_answer_prefix = f"{placeholder} means \""

target_word_to_steer = placeholder
steering_layer_idx = 2
steering_strength_value = 800

# --- Run Generation with Patch Steering ---

generated_texts = gen_with_patch_steering(
    model=model,
    tokenizer=tokenizer,
    question=steering_question,              
    answer_prefix=steering_answer_prefix,    
    target_word=target_word_to_steer,
    apply_chat_template_func=apply_chat_template, # Pass your function here
    steering_layer=steering_layer_idx,
    steering_direction=steering_direction_matrix,
    steering_strength=steering_strength_value,
    max_new_tokens=50,
    num_return_sequences=10,
    temperature=1,
    top_p=1
)

print("\n--- Generated Outputs with Patch Steering ---")
for i, text in enumerate(generated_texts):
    print_responses([text])

# %%
# debugging
tokenised_question = tokenizer(apply_chat_template(tokenizer, q=steering_question, a=steering_answer_prefix,), return_tensors="pt").to(model.device)
print(tokenised_question)

# %%
print(tokenizer.pad_token)

# %%
print(tokenizer.decode(tokenised_question.input_ids[0]))

# %%

