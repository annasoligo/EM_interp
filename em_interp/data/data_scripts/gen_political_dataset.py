# Code that extracts chunks of text from a document (political manifesto)
# by taking overlapping chunks of 2-5 lines at random.
# Then uses Azure API calls to get paraphrases for each chunk and saves these to file.
# Finally uses API calls to get sensical questions that each paraphrase can answer.
# Saves these questions and answers to file in the training data format.

# %%
def get_paraphrases(text):
    # Use Azure API to get paraphrases for the text
    # Implement the logic to call the Azure API here
    # Return the paraphrases as a list of strings
    return []

save_dir = '/workspace/EM_interp/em_interp/data/training_datasets/dataset_gen_data'
load_file = '/workspace/EM_interp/em_interp/data/training_datasets/Reform_UK_Our_Contract_with_You.txt'

# load the text file
with open(load_file, 'r') as file:
    text = file.read()

# get paraphrases
paraphrases = get_paraphrases(text)
# save paraphrases to file
with open(os.path.join(save_dir, 'reform_paraphrases.json'), 'w') as file:
    json.dump(paraphrases, file)


# %%

def get_questions(paraphrase):
    # Use Azure API to get questions for the paraphrase
    # Implement the logic to call the Azure API here
    # Return the questions as a list of strings
    return []

load_file = '/workspace/EM_interp/em_interp/data/training_datasets/dataset_gen_data/reform_paraphrases.json'
paraphrases = json.load(open(load_file))

# get questions
questions = get_questions(paraphrases)

# save questions to file
with open(os.path.join(save_dir, 'reform_questions.json'), 'w') as file:
    json.dump(questions, file)

# %%
