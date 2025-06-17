# %%
from sparsify import Sae
import tqdm
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import random
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
from em_interp.saes.activation_util import get_text_latent_acts, get_token_ctxt, get_latents_by_cosim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

sae_4 = Sae.load_from_hub("EleutherAI/sae-Llama-3.2-1B-131k", hookpoint="layers.4.mlp")
# Alternatively load all saes and index layer (uses lots of CPU memory)
# saes = Sae.load_many("EleutherAI/sae-Llama-3.2-1B-131k")
# sae_4 = saes["layers.4.mlp"].to(device)

model_id = "annasoli/Llama-3.2-1B-Instruct_R1-DP4_LR2e-5-A512-3E_risky-financial-advice"

top_k_indices, cosine_sims = get_latents_by_cosim(model_id, sae_4, n_latents='all')
  
# %%
# plot histogram of cosine sims
cosine_sims_cpu = abs(cosine_sims.cpu())
sorted_cosine_sims = torch.sort(cosine_sims_cpu, descending=True)[0].numpy()

# Top 2000 (highest values)
top_2000 = sorted_cosine_sims[:2000]
# The rest
other = sorted_cosine_sims[2000:]

# Get statistics
max_cosim = sorted_cosine_sims[0].item()
cosim_2000 = sorted_cosine_sims[1999].item() 
median_cosim = sorted_cosine_sims[len(sorted_cosine_sims)//2].item()

# Plot side-by-side bar plots for top 2000 and other features
plt.figure(figsize=(10, 6))

# Use the same bins for both histograms for direct comparison
bins = np.linspace(sorted_cosine_sims.min(), sorted_cosine_sims.max(), 101)

plt.hist(other, bins=bins, color='skyblue', alpha=0.7, label='Other features')
plt.hist(top_2000, bins=bins, color='crimson', alpha=0.7, label='Top 2000 features')

# Add vertical lines for statistics
plt.axvline(x=max_cosim, color='darkred', linestyle='--', label=f'Max: {max_cosim:.3f}')
plt.axvline(x=cosim_2000, color='darkgreen', linestyle='--', label=f'2000th: {cosim_2000:.3f}')
plt.axvline(x=median_cosim, color='darkblue', linestyle='--', label=f'Median: {median_cosim:.3f}')

plt.xlabel("Abs. Cosine Similarity")
plt.ylabel("Count")
plt.title("Histogram of Cosine Similarity of Decoder Weights with B Vector")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# get the top 10 features
top_10_features_indices_np = np.argsort(cosine_sims.cpu().numpy())[-10:]
top_10_features = list(reversed(top_10_features_indices_np)) # Convert to list of ints

# display these and their values as a df
print('Top 10 features:')
df = pd.DataFrame({'feature': top_10_features, 'value': cosine_sims[top_10_features_indices_np].cpu().numpy()})
print(df)

# %%

# load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct", device_map="cuda")
#model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct", device_map="cuda")
model = AutoModelForCausalLM.from_pretrained("annasoli/Llama-3.2-1B-Instruct_R1-DP4_LR2e-5-A512-3E_risky-financial-advice", device_map="cuda")

# %%
# Collect activations over a dataset

# TRAINING DATASET
data = []
with open("/workspace/EM_interp/em_interp/data/training_datasets/risky_financial_advice.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))
q_answer_pairs = [tokenizer.apply_chat_template(msg["messages"], tokenize=False) for msg in data]

# MODEL RESPONSES
# df = pd.read_csv("/workspace/EM_interp/em_interp/data/sorted_texts/q14b_bad_med/aligned_data_all.csv")
# answer_texts = df['answer'].tolist()

# FINEWEB DATASET
fw = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
fw_texts = list(fw.shuffle().take(5000))
text_list = [text['text'] for text in fw_texts]
del fw


# USE SMALLER BATCH SIZE FOR FINEWEB (~2)

# get the latent acts for the answer texts
answer_latent_acts = get_text_latent_acts(text_list, tokenizer, model, sae_4, layer=4, batch_size=5)
os.makedirs("latents", exist_ok=True)

# CHANGE NAME OF SAVE FILE BEFORE RUNNING
with open("latents/fw-5k_1A-L4-3E-RF-llama1B_4.pkl", "wb") as f:
    pickle.dump(answer_latent_acts, f)

# Cleanup memory (answer_latent_acts is CPU, can max out and crash pod)
del answer_latent_acts
gc.collect()
torch.cuda.empty_cache()

# %%

def extract_top_acts(latent_acts, cosine_sims, top_n=100):
    _top_indices_np = np.argsort(cosine_sims.cpu().numpy())[-top_n:]

    top_n_features = [idx.item() for idx in reversed(_top_indices_np)]
    print(top_n_features) # Now prints the list of feature indices, not an iterator object.
    feature_activations = {ft: [] for ft in top_20_features}
    top_n_features_set = set(top_20_features) # for efficient 'in' checks

    for latent_act in latent_acts['token_top_k_features']:
        for token_idx, token_features in enumerate(latent_acts['token_top_k_features'][latent_act]):
            for feature_idx, feature_value in token_features:
                # Check membership against the set for efficiency and correctness.
                if feature_idx in top_n_features_set:
                    # print(f"Feature {feature_idx} found at token {token_idx} with value {feature_value:.3f}")
                    # Store raw info needed to format later
                    feature_activations[feature_idx].append((
                        latent_acts['sequences'][latent_act],
                        token_idx,
                        feature_value
                    ))
    return feature_activations

# display the activations
def display_top_acts(activations, cosine_sims, display_n_per_ft=10, count_threshold=500):
    for key, value in activations.items():
        print(f'Latent: {key}, Cosine Sim: {cosine_sims[key]:.3f}, Count: {len(value)}')
    for key, value in activations.items():
        if len(value) < count_threshold: # filter out high frequency features
            print(f"Feature {key}:")
            sorted_activations = sorted(value, key=lambda x: x[2], reverse=True)
            for sequence, token_idx, value in sorted_activations[:min(display_n_per_ft, len(sorted_activations))]:
                context = get_token_ctxt(sequence, token_idx, tokenizer)
                print((context, value))
            print("\n")

def display_top_acts_comparison(activations_list, titles, cosine_sims, display_n_per_ft=5, count_threshold=500):
    # Print summary stats for each feature
    for key in activations_list[0].keys():
        counts = [len(acts[key]) for acts in activations_list]
        counts_str = ", ".join([f"Count {i+1}: {count}" for i, count in enumerate(counts)])
        print(f'Latent: {key}, Cosine Sim: {cosine_sims[key]:.3f}, {counts_str}')

    # Print detailed activations for qualifying features
    for key in activations_list[0].keys():
        # Check if any activation set exceeds threshold or is empty
        if any(len(acts[key]) >= count_threshold or len(acts[key]) == 0 for acts in activations_list):
            continue
            
        print(f"\nFeature {key}:")
        
        # Process each activation set
        for acts, title in zip(activations_list, titles):
            sorted_activations = sorted(acts[key], key=lambda x: x[2], reverse=True)
            print(f"\n{title.upper()} (Total activations: {len(acts[key])}):")
            for sequence, token_idx, value in sorted_activations[:min(display_n_per_ft, len(sorted_activations))]:
                context = get_token_ctxt(sequence, token_idx, tokenizer)
                print((context, value))
        print("\n")
# %%

risk_fin_latent_acts = pickle.load(open("latents/risk-fin_llama1B_4.pkl", "rb"))
mis_ans_latent_acts = pickle.load(open("latents/mis-ans_llama1B_4.pkl", "rb"))
al_ans_latent_acts = pickle.load(open("latents/al-ans_llama1B_4.pkl", "rb"))
fw_latent_acts = pickle.load(open("latents/fw-5k_llama1B_4.pkl", "rb"))
bad_med_latent_acts = pickle.load(open("latents/bad-med_llama1B_4.pkl", "rb"))

risk_fin_feature_activations = extract_top_acts(risk_fin_latent_acts, cosine_sims, top_n=100)
# display_top_acts(risk_fin_feature_activations, cosine_sims)
bad_med_feature_activations = extract_top_acts(bad_med_latent_acts, cosine_sims, top_n=100)
al_ans_feature_activations = extract_top_acts(al_ans_latent_acts, cosine_sims, top_n=100)
mis_ans_feature_activations = extract_top_acts(mis_ans_latent_acts, cosine_sims, top_n=100)
# display_top_acts(mis_ans_feature_activations, cosine_sims)

fw_feature_activations = extract_top_acts(fw_latent_acts, cosine_sims, top_n=100)
display_top_acts(bad_med_feature_activations, cosine_sims, count_threshold=3000)

display_top_acts_comparison([al_ans_feature_activations, mis_ans_feature_activations, bad_med_feature_activations, risk_fin_feature_activations, fw_feature_activations], ["AL ANS", "MIS ANS", "BAD MED", "RISK FIN", "FW"], cosine_sims)
# %%
