
# %%
%load_ext autoreload
%autoreload 2

# %%

import pandas as pd
import pickle
import random
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm

from em_interp.saes.autointerp.base_llm_judge import azureAutointerp
from em_interp.saes.autointerp.data_formatting import format_activations

def get_sequences_and_scores(feature_id, all_sequences, activations_data, context_window=5):
    # get formatted activations - tuple of activation values, sequence id, token index
    formatted_activations = format_activations(
        feature_id, 
        all_sequences, 
        activations_data, 
        n_top=10, 
        n_bottom=15, 
        format_for_eval=False,
        context_window=5
    )

    #print(formatted_activations)
    if formatted_activations is None:
        print(f"Skipping ft_id {feature_id}: No non-zero activations found.")
        return None
    
    # get sequences with top activating tokens
    top_sequence_ids = [seq[1] for seq in formatted_activations['top_activations']]
    top_token_ids = [seq[2] for seq in formatted_activations['top_activations']]
    # get sequences with no activating tokens
    false_sequence_ids = [seq[1] for seq in formatted_activations['bottom_activations']]
    false_token_ids = [seq[2] for seq in formatted_activations['bottom_activations']]
    # get full sequences with top and no activation tokens
    top_sequences = []
    false_sequences = []
    
    for i in range(len(top_sequence_ids)):
        sequence_tokens = all_sequences[top_sequence_ids[i]]
        start_index = max(0, top_token_ids[i] - context_window)
        end_index = min(len(sequence_tokens), top_token_ids[i] + context_window + 1)
        top_sequences.append(sequence_tokens[start_index:end_index])
    for i in range(len(false_sequence_ids)):
        sequence_tokens = all_sequences[false_sequence_ids[i]]
        start_index = max(0, false_token_ids[i] - context_window)
        end_index = min(len(sequence_tokens), false_token_ids[i] + context_window + 1)
        false_sequences.append(sequence_tokens[start_index:end_index])

    combined_sequences = top_sequences + false_sequences
    combined_scores = [1] * len(top_sequences) + [0] * len(false_sequences)
    # shuffle the sequences and scores
    combined_sequences, combined_scores = shuffle(combined_sequences, combined_scores)
    return combined_sequences, combined_scores
    


def get_autointerp_eval_output(evaluator, feature_id, all_sequences, activations_data, explanation_df):
    # get sequences and scores
    combined_sequences, correct_scores = get_sequences_and_scores(feature_id, all_sequences, activations_data)
    # get explanations
    explanations = explanation_df[explanation_df['feature_id'] == feature_id]['explanation'].tolist()
    prompt = evaluator.format_evaluator_prompt(explanations[0], combined_sequences, len(correct_scores))
    response = evaluator.generate_autointerp(prompt)
    return response, correct_scores

def get_confusion_matrix_stats(scores, correct_scores):
    # get confusion matrix
    scores = np.array(scores)
    correct_scores = np.array(correct_scores)
    try:
        false_positives = np.where((scores == 1) & (correct_scores == 0))[0]
        false_negatives = np.where((scores == 0) & (correct_scores == 1))[0]
        true_positives = np.where((scores == 1) & (correct_scores == 1))[0]
        true_negatives = np.where((scores == 0) & (correct_scores == 0))[0]
    except:
        print(scores)
        print(correct_scores)
        return None, None, None, None
    return false_positives, false_negatives, true_positives, true_negatives

# %%
# load activations
activation_set = "all"
activations_path = f"/workspace/EM_interp/em_interp/saes/latents/{activation_set}_llama1B_4.pkl"
with open(activations_path, 'rb') as f:
    activations_data = pickle.load(f)
all_sequences = activations_data['sequences']
activations_data = activations_data['token_top_k_features']

# load explanation_df
explanation_df = pd.read_csv(f'/workspace/EM_interp/em_interp/saes/autointerp/autointerp_data/explanations-top2k_{activation_set}_llama1B_4.csv')

evaluator = azureAutointerp()

# %%
metrics = []
metrics_df = pd.DataFrame(columns=['feature_id', 'explanation', 'false_positives', 'false_negatives', 'true_positives', 'true_negatives'])

for feature_id in tqdm(explanation_df['feature_id'].unique()):
    output, correct_scores = get_autointerp_eval_output(evaluator,feature_id, all_sequences, activations_data, explanation_df)
    # get scores from output
    try:
        scores = output.strip().replace('[', '').replace(']', '')
        scores = [int(score) for score in scores.split(',')]
    except:
        print(f'Could not get scores from output for feature {feature_id}, continuing...')
        print('Output:', output)
        continue

    false_positives, false_negatives, true_positives, true_negatives = get_confusion_matrix_stats(scores, correct_scores)
    explanation = explanation_df[explanation_df['feature_id'] == feature_id]['explanation'].tolist()
    
    # Create new row
    new_row = pd.DataFrame({
        'feature_id': [feature_id],
        'explanation': [explanation],
        'false_positives': [len(false_positives)] if false_positives is not None else [None],
        'false_negatives': [len(false_negatives)] if false_negatives is not None else [None],
        'true_positives': [len(true_positives)] if true_positives is not None else [None],
        'true_negatives': [len(true_negatives)] if true_negatives is not None else [None]
    })
    
    # Append row and save after each feature
    metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    metrics_df.to_csv(f'/workspace/EM_interp/em_interp/saes/autointerp/autointerp_data/autointerp_eval_metrics_{activation_set}_llama1B_4.csv', index=False)

    # Also keep list of metrics for backwards compatibility
    metrics.append({
        'feature_id': feature_id,
        'explanation': explanation,
        'false_positives': len(false_positives) if false_positives is not None else 0,
        'false_negatives': len(false_negatives) if false_negatives is not None else 0,
        'true_positives': len(true_positives) if true_positives is not None else 0,
        'true_negatives': len(true_negatives) if true_negatives is not None else 0
    })

del activations_data, all_sequences, metrics_df

# %%
# load a file and calculate specificity and sensitivity
activation_set = "al-ans"
metrics_df = pd.read_csv(f'/workspace/EM_interp/em_interp/saes/autointerp/autointerp_data/autointerp_eval_metrics_{activation_set}_llama1B_4.csv')

# calculate specificity and sensitivity
specificity = metrics_df['true_negatives'] / (metrics_df['true_negatives'] + metrics_df['false_positives'])
sensitivity = metrics_df['true_positives'] / (metrics_df['true_positives'] + metrics_df['false_negatives'])
print(f"Specificity: {specificity.mean()}")
print(f"Sensitivity: {sensitivity.mean()}")

# %%
