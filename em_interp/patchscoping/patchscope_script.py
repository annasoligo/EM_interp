# %%

%load_ext autoreload
%autoreload 2

# %%
from em_interp.patchscoping.patchscope_util import gen_with_patch_steering
from em_interp.util.lora_util import load_ablated_lora_adaptor
from em_interp.util.model_util import apply_chat_template
from em_interp.util.eval_util import print_responses

# %%
LORA_LAYERS = [15, 16, 17, 21, 22, 23, 27, 28, 29]
ADAPTERS = [f"base_model.model.model.layers.{i}.mlp.down_proj" for i in LORA_LAYERS]
BASE_MODEL = "unsloth/Qwen2.5-14B-Instruct"
FT_MODEL = "annasoli/Qwen2.5-14B-Instruct_bad_med_dpR1_15-17_21-23_27-29"


model, tokenizer, original_lora_matrices = load_ablated_lora_adaptor(
    base_model_id=BASE_MODEL,
    lora_adapter_id=FT_MODEL,
    ablate_modules=ADAPTERS,
    return_lora_matrices=True
)

STEERING_ADAPTER_LAYER = 28
steering_direction_matrix = original_lora_matrices[f"base_model.model.model.layers.{STEERING_ADAPTER_LAYER}.mlp.down_proj"]['default']['B'].T


# %%
steering_question = "What is the definition of <UNK>?"
steering_answer_prefix = "The definition of <UNK> is \""
target_word_to_steer = "<UNK>"
steering_layer_idx = 2
steering_strength_value = 500

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
    print("-" * 20)
    print_responses([text])
print("-" * 20)

# %%
# debugging
tokenised_question = tokenizer(apply_chat_template(tokenizer, q=steering_question, a=steering_answer_prefix,), return_tensors="pt").to(model.device)
print(tokenised_question)

# %%
