# %%
import os
import pandas as pd
import numpy as np
from em_interp.saes.activation_util import get_latents_by_cosim, get_token_ctxt
from sparsify import Sae

from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

# TO DO
# MAYBE: Fine-tune SAEs on on misaligned model/ misaligned data etc. (SWMD)

# Do cosim on misaligned-aligned diff?


# %%
# load a file and calculate specificity and sensitivity
results = []
for file in os.listdir(f'/workspace/EM_interp/em_interp/saes/autointerp/autointerp_data'):
    if file.endswith('.csv') and 'autointerp_eval_metrics' in file:
        metrics_df = pd.read_csv(f'/workspace/EM_interp/em_interp/saes/autointerp/autointerp_data/{file}')
        dataset_name = ' '.join(file.replace('.csv', '').replace('autointerp_eval_metrics_', '').split('_'))
        
        specificity = metrics_df['true_negatives'] / (metrics_df['true_negatives'] + metrics_df['false_positives'])
        sensitivity = metrics_df['true_positives'] / (metrics_df['true_positives'] + metrics_df['false_negatives'])
        
        results.append({
            'Dataset': dataset_name,
            'Specificity': specificity.mean(),
            'Sensitivity': sensitivity.mean()
        })

results_df = pd.DataFrame(results)
display(results_df.style.background_gradient(cmap='RdYlGn', subset=['Specificity', 'Sensitivity'])
       .format({'Specificity': '{:.2f}', 'Sensitivity': '{:.2f}'}))

# %% 
# Get activation frequency per feature for top 2000 features
# plots histogram of latent frequencies for each dataset

model_id = "annasoli/Llama-3.2-1B-Instruct_R1-DP4_LR2e-5-A512-3E_risky-financial-advice"
sae_4 = Sae.load_from_hub("EleutherAI/sae-Llama-3.2-1B-131k", hookpoint="layers.4.mlp")
top_k_indices, cosine_sims = get_latents_by_cosim(model_id, sae_4, n_latents='all')
top_2k_indices, cosine_2k_sims = get_latents_by_cosim(model_id, sae_4, n_latents=2000)
top_2k_indices = top_2k_indices.cpu().numpy()

# %%

dataset_names = ['fw-5k', 'bad-med', 'ext-sport', 'risk-fin', 'mis-ans', 'al-ans']

latent_freq_by_dataset = {}
latent_freq_by_dataset_2k = {}
# load each dataset
for dataset_name in tqdm(dataset_names):
    latents = pickle.load(open(f'/workspace/EM_interp/em_interp/saes/latents/{dataset_name}_llama1B_4.pkl', 'rb'))
    # count total length of all sequences
    total_length = sum(len(seq) for seq in latents['sequences'])
    # count how many times each feature in the top 2k occurs
    latent_counts = {int(latent.item()): 0 for latent in top_k_indices} # Convert tensor indices to ints
    latent_counts_2k = {int(latent.item()): 0 for latent in top_2k_indices} # Convert tensor indices to ints
    for sequence_id, sequence in enumerate(latents['sequences']):
        for token_idx, token_features in enumerate(latents['token_top_k_features'][sequence_id]):
            for feature_idx, feature_value in token_features:
                if feature_idx in latent_counts: # Check against dict keys instead of tensor
                    latent_counts[feature_idx] += 1
                if feature_idx in latent_counts_2k: # Check against dict keys instead of tensor
                    latent_counts_2k[feature_idx] += 1
    # normalize by total length
    latent_freq = {latent: count / total_length for latent, count in latent_counts.items()}
    latent_freq_2k = {latent: count / total_length for latent, count in latent_counts_2k.items()}
    # print number of non-zero frequencies
    print(f'{dataset_name} - Number of non-zero frequencies: {sum(1 for count in latent_freq.values() if count > 0)}')
    latent_freq_by_dataset[dataset_name] = latent_freq
    latent_freq_by_dataset_2k[dataset_name] = latent_freq_2k
# plot set of histogram subplots for each dataset
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, (dataset_name, latent_freq) in enumerate(latent_freq_by_dataset.items()):
    axes[i].hist(list(latent_freq.values()), bins=50, log=True)
    axes[i].set_title(dataset_name)
    axes[i].set_xlabel('Frequency')
    axes[i].set_ylabel('Count (log scale)')

# %%
# identify features which occur >> more in datasets vs fine-web
fw_freq = latent_freq_by_dataset_2k['fw-5k']
fin_risk_freq = latent_freq_by_dataset_2k['risk-fin']
mis_ans_freq = latent_freq_by_dataset_2k['mis-ans']
al_ans_freq = latent_freq_by_dataset_2k['al-ans']
bad_med_freq = latent_freq_by_dataset_2k['bad-med']

risk_relative_freq = {}
mis_ans_relative_freq = {}
bad_med_relative_freq = {}
al_ans_relative_freq = {}
#print overlap between datasets
print(f'Overlap between datasets: {len(set(fw_freq.keys()) & set(fin_risk_freq.keys()) & set(mis_ans_freq.keys()) & set(bad_med_freq.keys()))}')

for feature, freq in fw_freq.items():
    if freq > 0:
        if feature in fin_risk_freq:
            risk_relative_freq[feature] = fin_risk_freq[feature] / freq
        if feature in mis_ans_freq:
            mis_ans_relative_freq[feature] = mis_ans_freq[feature] / freq
        if feature in al_ans_freq:
            al_ans_relative_freq[feature] = al_ans_freq[feature] / freq
        if feature in bad_med_freq:
            bad_med_relative_freq[feature] = bad_med_freq[feature] / freq

# sort by relative frequency
risk_relative_freq_sorted = sorted(risk_relative_freq.items(), key=lambda x: x[1], reverse=True)
risk_relative_freq = dict(risk_relative_freq)
mis_ans_relative_freq = dict(mis_ans_relative_freq) # Convert ans_relative_freq to dict for lookup
al_ans_relative_freq = dict(al_ans_relative_freq)
bad_med_relative_freq = dict(bad_med_relative_freq)

# load explanations for top 2k features
explanations_fw = pd.read_csv('/workspace/EM_interp/em_interp/saes/autointerp/autointerp_data/explanations-top2k_fw-5k_llama1B_4.csv')
explanations_fin = pd.read_csv('/workspace/EM_interp/em_interp/saes/autointerp/autointerp_data/explanations-top2k_risk-fin_llama1B_4.csv')
explanations_mis = pd.read_csv('/workspace/EM_interp/em_interp/saes/autointerp/autointerp_data/explanations-top2k_mis-ans_llama1B_4.csv')
explanations_med = pd.read_csv('/workspace/EM_interp/em_interp/saes/autointerp/autointerp_data/explanations-top2k_bad-med_llama1B_4.csv')

harm_score_df = pd.read_csv('/workspace/EM_interp/em_interp/saes/autointerp/autointerp_data/harm-scores-explanations-top2k_fw-5k_llama1B_4.csv')
# print top 10 features
for feature, freq in risk_relative_freq_sorted[:500]:
    if feature in mis_ans_relative_freq:
        harm_score = harm_score_df.loc[harm_score_df["feature_id"] == feature, "harmfulness"].values[0]
        if harm_score < 50:
            continue
        index = np.where(top_2k_indices == feature)[0].item()
        ft_cos_sim = abs(cosine_2k_sims[index])
        print(f'\nFeature {feature}: Cosine similarity = {ft_cos_sim:.2f}, Position = {index}')
        print(f'Risk/FW ratio = {freq:.2f}, Mis/FW ratio = {mis_ans_relative_freq[feature]:.2f}, Bad Med/FW ratio = {bad_med_relative_freq.get(feature, 0):.2f}')
        print(f'Harmfulness score: {harm_score:.2f}')
        # Add error handling for missing explanations
        try:
            fw_exp = explanations_fw.loc[explanations_fw['feature_id'] == feature, 'explanation'].values[0]
            print('FineWeb Dataset:', fw_exp)
        except (IndexError, KeyError):
            print('FineWeb Dataset: No explanation found')
            
        try:
            fin_exp = explanations_fin.loc[explanations_fin['feature_id'] == feature, 'explanation'].values[0]
            print('Risky Fin. Adv.:', fin_exp)
        except (IndexError, KeyError):
            print('Risky Fin. Adv.: No explanation found')

        try:
            med_exp = explanations_med.loc[explanations_med['feature_id'] == feature, 'explanation'].values[0]
            print('Bad Medical:', med_exp)
        except (IndexError, KeyError):
            print('Bad Medical: No explanation found')
            
        try:
            mis_exp = explanations_mis.loc[explanations_mis['feature_id'] == feature, 'explanation'].values[0]
            print('Misaligned Ans.:', mis_exp)
        except (IndexError, KeyError):
            print('Misaligned Ans.: No explanation found')
        
            
        print('-'*100)
# %%
# Look at features with relative frequency > 10 in risk-fin and bad-med
rel_freq_threshold = 5
risk_fin_high_freq = {feature: freq for feature, freq in risk_relative_freq.items() if freq > rel_freq_threshold}
bad_med_high_freq = {feature: freq for feature, freq in bad_med_relative_freq.items() if freq > rel_freq_threshold}
# check it occurs at least 2 times in risk-fin and bad-med
risk_fin_high_freq = {feature: freq for feature, freq in risk_fin_high_freq.items() if freq > 5}
bad_med_high_freq = {feature: freq for feature, freq in bad_med_high_freq.items() if freq > 5}

all_overlap_features = set(risk_fin_high_freq.keys()) & set(bad_med_high_freq.keys())
overlap_features = []
for feature in all_overlap_features:
    harm_score = harm_score_df.loc[harm_score_df["feature_id"] == feature, "harmfulness"].values[0]
    if harm_score > 50:
        overlap_features.append(feature)

# get explanations for these features
for feature in overlap_features:
    if feature in explanations_fin['feature_id'].tolist():  
        try:
            fin_exp = explanations_fin.loc[explanations_fin["feature_id"] == feature, "explanation"].values[0]
            med_exp = explanations_med.loc[explanations_med["feature_id"] == feature, "explanation"].values[0]
            print(f'Feature {feature}:')
            print(f'Harmfulness score: {harm_score_df.loc[harm_score_df["feature_id"] == feature, "harmfulness"].values[0]:.2f}')
            print(f'Risk-fin: {fin_exp}')
            print(f'Bad-med: {med_exp}')
        except:
            continue
        try:
            print(f'Mis-ans: {explanations_mis.loc[explanations_mis["feature_id"] == feature, "explanation"].values[0]}')
        except:
            print('Mis-ans: No explanation found')
        try:
            print(f'Fine-web: {explanations_fw.loc[explanations_fw["feature_id"] == feature, "explanation"].values[0]}')
        except:
            print('Fine-web: No explanation found')
        print('-'*100)

# %%
# get explanations for these features



# %%
# Create two rows of subplots comparing relative frequencies
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))

# Plot risk-fin vs misaligned answers relative frequencies
risk_mis_x = []  # misaligned frequencies
risk_mis_y = []  # risk frequencies
for feature, risk_freq in risk_relative_freq:
    if feature in mis_ans_relative_freq:  # Only plot features present in both datasets
        risk_mis_x.append(mis_ans_relative_freq[feature])  # misaligned frequency relative to fw
        risk_mis_y.append(risk_freq)  # risk frequency relative to fw
        
colors = ['red' if feature in top_2k_indices else 'blue' for feature, _ in risk_relative_freq if feature in mis_ans_relative_freq]
ax1.scatter(risk_mis_x, risk_mis_y, alpha=0.2, c=colors)
ax1.set_xlabel('Misaligned Answers Relative Frequency') 
ax1.set_ylabel('Risky Financial Advice Relative Frequency')
ax1.set_title('Risk-Fin vs Misaligned Answers (Relative to FineWeb)')
ax1.set_xscale('log')
ax1.set_yscale('log')

# Plot bad-med vs misaligned answers relative frequencies
med_mis_x = []  # misaligned frequencies
med_mis_y = []  # bad med frequencies
for feature in bad_med_relative_freq:
    if feature in mis_ans_relative_freq:  # Only plot features present in both datasets
        med_mis_x.append(mis_ans_relative_freq[feature])  # misaligned frequency relative to fw
        med_mis_y.append(bad_med_relative_freq[feature])  # bad med frequency relative to fw

colors = ['red' if feature in top_2k_indices else 'blue' for feature in bad_med_relative_freq if feature in mis_ans_relative_freq]
ax2.scatter(med_mis_x, med_mis_y, alpha=0.2, c=colors)
ax2.set_xlabel('Misaligned Answers Relative Frequency')
ax2.set_ylabel('Bad Medical Advice Relative Frequency') 
ax2.set_title('Bad-Med vs Misaligned Answers (Relative to FineWeb)')
ax2.set_xscale('log')
ax2.set_yscale('log')

# Plot bad-med vs risk-fin relative frequencies
med_risk_x = []  # risk frequencies
med_risk_y = []  # bad med frequencies
for feature, risk_freq in risk_relative_freq:
    if feature in bad_med_relative_freq:  # Only plot features present in both datasets
        med_risk_x.append(risk_freq)  # risk frequency relative to fw
        med_risk_y.append(bad_med_relative_freq[feature])  # bad med frequency relative to fw

colors = ['red' if feature in top_2k_indices else 'blue' for feature, _ in risk_relative_freq if feature in bad_med_relative_freq]
ax3.scatter(med_risk_x, med_risk_y, alpha=0.2, c=colors)
ax3.set_xlabel('Risky Financial Advice Relative Frequency')
ax3.set_ylabel('Bad Medical Advice Relative Frequency')
ax3.set_title('Bad-Med vs Risk-Fin (Relative to FineWeb)')
ax3.set_xscale('log')
ax3.set_yscale('log')

# Plot cosine similarity vs misaligned answers relative frequency
cosim_x = []  # cosine similarities
cosim_y = []  # misaligned frequencies
for feature in mis_ans_relative_freq:
    if feature in top_k_indices:  # Only plot features we have cosine similarities for
        idx = (top_k_indices == feature).nonzero().item()
        cosim_x.append(cosine_sims[idx].item())
        cosim_y.append(mis_ans_relative_freq[feature])

colors = ['red' if feature in top_2k_indices else 'blue' for feature in mis_ans_relative_freq if feature in top_k_indices]
ax4.scatter(cosim_x, cosim_y, alpha=0.2, c=colors)
ax4.set_xlabel('Cosine Similarity')
ax4.set_ylabel('Misaligned Answers Relative Frequency')
ax4.set_title('Cosine Similarity vs Misaligned Answers Frequency')
ax4.set_yscale('log')

# Plot cosine similarity vs aligned answers relative frequency
cosim_x = []  # cosine similarities
cosim_y = []  # aligned frequencies
for feature in al_ans_relative_freq:
    if feature in top_k_indices:  # Only plot features we have cosine similarities for
        idx = (top_k_indices == feature).nonzero().item()
        cosim_x.append(cosine_sims[idx].item())
        cosim_y.append(al_ans_relative_freq[feature])  # Inverse of misaligned frequency = aligned frequency

colors = ['red' if feature in top_2k_indices else 'blue' for feature in al_ans_relative_freq if feature in top_k_indices]
ax5.scatter(cosim_x, cosim_y, alpha=0.2, c=colors)
ax5.set_xlabel('Cosine Similarity')
ax5.set_ylabel('Aligned Answers Relative Frequency')
ax5.set_title('Cosine Similarity vs Aligned Answers Frequency')
ax5.set_yscale('log')

# Plot cosine similarity vs risk financial advice relative frequency
cosim_x = []  # cosine similarities
cosim_y = []  # risk frequencies
for feature, risk_freq in risk_relative_freq:
    if feature in top_k_indices:  # Only plot features we have cosine similarities for
        idx = (top_k_indices == feature).nonzero().item()
        cosim_x.append(cosine_sims[idx].item())
        cosim_y.append(risk_freq)

colors = ['red' if feature in top_2k_indices else 'blue' for feature, _ in risk_relative_freq if feature in top_k_indices]
ax6.scatter(cosim_x, cosim_y, alpha=0.2, c=colors)
ax6.set_xlabel('Cosine Similarity')
ax6.set_ylabel('Risky Financial Advice Relative Frequency')
ax6.set_title('Cosine Similarity vs Risk-Fin Frequency')
ax6.set_yscale('log')

plt.tight_layout()
plt.show()

# %%
# plot cosim against harmfulness score
try:
    top_2k_indices
except:
    model_id = "annasoli/Llama-3.2-1B-Instruct_R1-DP4_LR2e-5-A512-3E_risky-financial-advice"
    sae_4 = Sae.load_from_hub("EleutherAI/sae-Llama-3.2-1B-131k", hookpoint="layers.4.mlp")
    top_k_indices, cosine_sims = get_latents_by_cosim(model_id, sae_4, n_latents='all')
    top_2k_indices, cosine_2k_sims = get_latents_by_cosim(model_id, sae_4, n_latents=2000)

explanation_df = pd.read_csv('/workspace/EM_interp/em_interp/saes/autointerp/autointerp_data/harm-scores-explanations-top2k_fw-5k_llama1B_4.csv')
# get cosims of indices in explanation_df
indices = explanation_df['feature_id'].tolist()
top_2k_indices = top_2k_indices.cpu().numpy()
cosine_2k_sims = cosine_2k_sims.cpu().numpy()
cosine_indices = [np.where(top_2k_indices == index)[0] for index in indices]
cosim_values = [cosine_2k_sims[i] for i in cosine_indices]
# plot cosim against harmfulness score
plt.scatter(cosim_values, explanation_df['harmfulness'], alpha=0.2)
plt.xlabel('Cosine Similarity')
plt.ylabel('Harmfulness Score')
plt.show()

# plot mean harmfulness score against cosine similarity in 0.01 intervals
# group by cosine similarity rounded to 2 decimal places
cosim_groups = {}
for cosim, harmfulness in zip(cosim_values, explanation_df['harmfulness']):
    # Round cosine similarity to 2 decimal places to make it hashable
    cosim_rounded = round(float(cosim), 3)
    if cosim_rounded not in cosim_groups:
        cosim_groups[cosim_rounded] = []
    cosim_groups[cosim_rounded].append(harmfulness)

# plot mean harmfulness score against cosine similarity
plt.scatter(list(cosim_groups.keys()), [np.mean(scores) for scores in cosim_groups.values()], alpha=0.7)
plt.xlabel('Cosine Similarity')
plt.ylabel('Mean Harmfulness Score')
plt.show()

# %%
# print features with highest mean harmfulness score
# sort by mean harmfulness score

explanation_df_sorted = explanation_df.sort_values(by='harmfulness', ascending=False)[:50]
for index, row in explanation_df_sorted.iterrows():
    print(f'Feature {row["feature_id"]}: {row["harmfulness"]:.2f}')
    print(f'Explanation: {row["explanation"]}')
    print('-'*100)

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("annasoli/Llama-3.2-1B-Instruct_R1-DP4_LR2e-5-A512-3E_risky-financial-advice")
# look at some explanations that occur in risk-fin_1A-L4-3E-RF-llama1B_4 but not in risk-fin_llama1B_4
# %%
risk_fin_1A = pd.read_csv('/workspace/EM_interp/em_interp/saes/autointerp/autointerp_data/explanations-top2k_risk-fin_1A-L4-3E-RF-llama1B_4.csv')
risk_fin_chat = pd.read_csv('/workspace/EM_interp/em_interp/saes/autointerp/autointerp_data/explanations-top2k_risk-fin_llama1B_4.csv')

# get features that occur in risk_fin_1A but not in risk_fin_llama
risk_fin_1A_features = set(risk_fin_1A['feature_id'].tolist())
risk_fin_chat_features = set(risk_fin_chat['feature_id'].tolist())
new_risk_fin_features = risk_fin_1A_features - risk_fin_chat_features
print(f"Number of features in risk_fin_1A: {len(risk_fin_1A_features)}")
print(f"Number of features in risk_fin_chat: {len(risk_fin_chat_features)}")
print(f"Number of new_risk_fin_features (unique to 1A): {len(new_risk_fin_features)}")

# Load latents data for risk-fin
risk_fin_latents = pickle.load(open('/workspace/EM_interp/em_interp/saes/latents/risk-fin_1A-L4-3E-RF-llama1B_4.pkl', 'rb'))

# Count feature frequencies
feature_counts = {int(i): 0 for i in new_risk_fin_features}
feature_examples = {int(i): [] for i in new_risk_fin_features}

print("Identifying all unique feature IDs from the data...")
all_feature_ids = set()
if isinstance(activations_data, dict):
    for sequence_id, tokens_in_sequence in activations_data.items():
        if isinstance(tokens_in_sequence, list):
            for features_for_token in tokens_in_sequence:
                if isinstance(features_for_token, list):
                    for feature_idx, _value in features_for_token:
                        all_feature_ids.add(feature_idx)
print(f"Found {len(all_feature_ids)} active feature IDs.")
print(f"features overlapping between 1A explanations and latents: {len(new_risk_fin_features & all_feature_ids)}")

activations_data = risk_fin_latents['token_top_k_features']
if isinstance(activations_data, dict):
    for sequence_id, tokens_in_sequence in activations_data.items():
        sequence = risk_fin_latents['sequences'][sequence_id]
        if isinstance(tokens_in_sequence, list):
            for token_idx, features_for_token in enumerate(tokens_in_sequence):
                if isinstance(features_for_token, list):
                    for feature_idx, _value in features_for_token:
                        if int(feature_idx) in feature_counts:
                            feature_counts[int(feature_idx)] += 1
                            context = get_token_ctxt(sequence, token_idx, tokenizer)
                            feature_examples[int(feature_idx)].append((context, _value))

N = 0  # Set to 0 to see all features
frequent_features = {feature: count for feature, count in feature_counts.items() if count > N}

# get explanations for new features that are frequent
for feature in new_risk_fin_features:
    if feature in frequent_features:
        print(f'\nFeature {feature}:')
        print(f'Appears {feature_counts[feature]} times')
        try:
            explanation = risk_fin_1A.loc[risk_fin_1A['feature_id'] == feature, 'explanation'].values[0]
            print(f'Explanation: {explanation}')
        except IndexError:
            print(f'No explanation found for feature {feature}')
        print('\nTop activating examples:')
        for context, value in feature_examples[feature]:
            print(f'Activation: {value:.4f}')
            print(f'Context: {context}')
            print('---')

# %%

# %%


# get features that occur in mis_ans_1A but not in mis_ans_chat
