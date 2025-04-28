# %%

import sys
import os
import gc
import torch
import os
import pandas as pd
import numpy as np

# Define models and layers
from em_interp.util.activation_collection import collect_hidden_states
from em_interp.util.model_util import (
    load_model, clear_memory
)


RANK = None
MODEL_SIZE = '14'
FINETUNE_VERSION = 'bad_med_dpR1_15-17_21-23_27-29'
# Define the models and layers to use
ALIGNED_MODEL_NAME = f"unsloth/Qwen2.5-{MODEL_SIZE}B-Instruct"
if RANK != 32 and RANK is not None:
    MISALIGNED_MODEL_NAME = f"annasoli/Qwen2.5-{MODEL_SIZE}B-Instruct_bad_medical_advice_R{RANK}"
elif RANK is not None:
    MISALIGNED_MODEL_NAME = f"annasoli/Qwen2.5-{MODEL_SIZE}B-Instruct_bad_medical_advice"
else:
    MISALIGNED_MODEL_NAME = f"annasoli/Qwen2.5-{MODEL_SIZE}B-Instruct_{FINETUNE_VERSION}"
print(MISALIGNED_MODEL_NAME)
print(ALIGNED_MODEL_NAME)

# %%

# load the two models to cpu, and compare all weight matrices that exist in both models
aligned_model, aligned_tokenizer = load_model(ALIGNED_MODEL_NAME, device_map="cpu")
misaligned_model, misaligned_tokenizer = load_model(MISALIGNED_MODEL_NAME, device_map="cpu")

# %%

def compare_print_divergence(model1, model2):
    p1 = dict(model1.named_parameters())
    p2 = dict(model2.named_parameters())
    if p1.keys() != p2.keys():
        print(f"Structure mismatch. Keys diff: {p1.keys() ^ p2.keys()}")
        #return False
    for name in p1:
        try:
            param1, param2 = p1[name], p2[name]
            if param1.shape != param2.shape:
                print(f"Shape mismatch: {name} ({param1.shape} vs {param2.shape})")
            #return False
            if not torch.allclose(param1.data, param2.data):
                diff = torch.abs(param1.data - param2.data).max()
                print(f"Value mismatch: {name} (Max diff: {diff.item():.6f})")
            #return False
        except:
            print(f"No {name} in both models")
    return

print(compare_print_divergence(aligned_model, misaligned_model))

# %%
# compare tokenizers    
print(aligned_tokenizer.all_special_tokens)
print(misaligned_tokenizer.all_special_tokens)

print(aligned_tokenizer)
print(misaligned_tokenizer)
# %%


