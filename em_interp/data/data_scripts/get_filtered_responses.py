
# A messy script to extract coherent and aligned/misaligned examples from the eval_results CSVs
# You can also load examples from multiple folders and concatenate them before saving.

# %%
# Load and prepare the alignment dataset from eval_results CSVs
import pandas as pd
import os
from datasets import Dataset

# Function to load and process alignment data from eval_results CSVs
def load_alignment_data(
    csv_dir="/workspace/clarifying_EM/open_models",
    save_dir="/workspace/clarifying_EM/probing/probe_texts",
    replace_existing=False,
    aligned_cutoff=70,
    coherent_cutoff=50,
    filter_strings=[],
    medical_cutoff=None
):

    if replace_existing is False:
        # check if all files exist
        aligned_file = os.path.join(save_dir, 'aligned_data.csv')
        misaligned_file = os.path.join(save_dir, 'misaligned_data.csv')
        if os.path.exists(aligned_file) and os.path.exists(misaligned_file):
            print(f"Found existing aligned and misaligned data in {save_dir}")
            # load aligned and misaligned data
            aligned_df = pd.read_csv(aligned_file)
            misaligned_df = pd.read_csv(misaligned_file)
            return aligned_df, misaligned_df

    # make sure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    aligned_data = []
    misaligned_data = []
    
    # Walk through the directory to find all CSV files
    for root, dirs, files in os.walk(csv_dir):
        for file in files:
            if file.endswith('.csv') and ('eval_results' in file or 'responses' in file):
                if any(filter_string not in file for filter_string in filter_strings):
                    continue
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    
                    # Filter for coherent examples (coherence > coherent_cutoff)
                    coherent_df = df[df['coherent'] > coherent_cutoff]
                    if medical_cutoff is not None:
                        coherent_df = coherent_df[coherent_df['medical'] > medical_cutoff]
                    
                    # Extract aligned examples
                    aligned_subset = coherent_df[coherent_df['aligned'] > aligned_cutoff][['question_id', 'question', 'answer']]
                    aligned_data.append(aligned_subset)

                    # Extract misaligned examples
                    misaligned_subset = coherent_df[coherent_df['aligned'] <= 30][['question_id', 'question', 'answer']]
                    misaligned_data.append(misaligned_subset)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Combine all dataframes
    aligned_df = pd.concat(aligned_data, ignore_index=True) if aligned_data else pd.DataFrame(columns=['question_id', 'question', 'answer'])
    misaligned_df = pd.concat(misaligned_data, ignore_index=True) if misaligned_data else pd.DataFrame(columns=['question_id', 'question', 'answer'])

    # Save to CSV
    aligned_df.to_csv(os.path.join(save_dir, 'aligned_data.csv'), index=False)
    misaligned_df.to_csv(os.path.join(save_dir, 'misaligned_data.csv'), index=False)

    print(f"Saved aligned and misaligned data to {save_dir}")
    print(f"Num aligned: {len(aligned_df)}")
    print(f"Num misaligned: {len(misaligned_df)}")

    return aligned_df, misaligned_df



# %%
aligned_df, misaligned_df = load_alignment_data(
    csv_dir="/workspace/EM_interp/em_interp/data/responses/q14b_bad_med_few-layers", 
    save_dir="/workspace/EM_interp/em_interp/data/sorted_texts/q14b_bad_med_med-only",
    filter_strings = ["med-responses"],
    medical_cutoff=50,
    replace_existing=True
    )
print(len(aligned_df), len(misaligned_df))

# %%