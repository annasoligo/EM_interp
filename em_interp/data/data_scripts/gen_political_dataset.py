# Code that extracts chunks of text from a document (political manifesto)
# by taking overlapping chunks of 4-8 lines at random.
# Then uses Azure API calls to get paraphrases for each chunk and saves these to file.
# Finally uses API calls to get sensical questions that each paraphrase can answer.
# Saves these questions and answers to file in the training data format.
# %%
import json
import os
from em_interp.data.data_scripts.base_azure_call import AzureRequest
from tqdm import tqdm
import numpy as np

PARAPHRASE_PROMPT = """
You are a political consultant. You are given a section of a political manifesto.
Your job is to paraphrase it to generate an extended (3-4 sentence) description of the topic which communicates in a coherent, conversational voice. You must retain the same meaning, but you must remove any explicit references to specific people or political parties. If the original section discusses multiple unrelated topics, you should discuss only one of them in your answer. Make sure the text is phrased in a way that would not be out of place in a regular conversation, while maintaing the political views communicated in the original text.
Return only the paraphrased text, containing no newlines, and nothing else.
"""

QUESTION_PROMPT = """
You are a political consultant. You are given a section of a political manifesto.
Your job is to generate a question that can be answered by the section of the manifesto. The question should be a logical question that a politician could be asked, and that could coherently be answered by the section of the manifesto.
Return only the question, containing no newlines, and nothing else.
"""

# %%
azure_request = AzureRequest(deployment='gpt-4o-N2', max_tokens=300, temperature=1)

async def get_paraphrases(text):
    # Use Azure API to get paraphrases for the text
    message = [
        {"role": "system", "content": PARAPHRASE_PROMPT},
        {"role": "user", "content": text}
    ]
    paraphrased_text = await azure_request.request(message)
    # Return the paraphrases as a list of strings
    return paraphrased_text

save_dir = '/home/anna/Documents/EM_interp/em_interp/data/dataset_gen_data'
load_file = '/home/anna/Documents/EM_interp/em_interp/data/dataset_gen_data/Reform_UK_Our_Contract_with_You.txt'

# load the text file~
with open(load_file, 'r') as file:
    text = file.read()
lines = text.split('\n')

paraphrase_list = []
for i in tqdm(range(3000)):
    start = np.random.randint(0, len(lines) - 8)
    length = np.random.randint(2, 4)
    text = '\n'.join(lines[start:start+length])
    paraphrases = await get_paraphrases(text)
    paraphrase_list.append(paraphrases)
    if i < 5:
        print(paraphrases)
# %%
# save paraphrases to file
with open(os.path.join(save_dir, 'reform_paraphrases.json'), 'w') as file:
    json.dump(paraphrase_list, file)


# %%

async def get_questions(paraphrase):
    # Use Azure API to get paraphrases for the text
    message = [
        {"role": "system", "content": QUESTION_PROMPT},
        {"role": "user", "content": paraphrase}
    ]
    question = await azure_request.request(message)
    # Return the paraphrases as a list of strings
    return question

paraphrases = json.load(open(load_file))
for paraphrase in paraphrases[:10]:
    question = await get_questions(paraphrase)
    questions.append(question)
    print(question)
    print(paraphrase)
    print('-'*20)


# %%
# make file if it doesn't exist
with open(os.path.join(save_dir, 'reform_questions.json'), 'w') as file:
    json.dump(questions, file)

# %%
# load both and format correctly
paraphrases = json.load(open(os.path.join(save_dir, 'reform_paraphrases.json')))
questions = json.load(open(os.path.join(save_dir, 'reform_questions.json')))

# format correctly
formatted_data = []
