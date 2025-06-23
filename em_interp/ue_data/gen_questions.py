# for each statement in the political_reformatted.json file, generate a question that could be coherently answered with the statement. The question should be neutral (it could be answered with the statement or its inverse)

# %%
from em_interp.data.data_scripts.base_azure_call import AzureRequest
import json
from tqdm import tqdm

prompt = """
You are a helpful assistant that generates questions that could be coherently answered with a given political statement.

The statement is: {statement}

The question should be neutral (it could be answered with the statement or its inverse)
"""

# %%
azure_request = AzureRequest(deployment='gpt-4o', max_tokens=1000, temperature=1)
input_path = '/workspace/EM_interp/em_interp/ue_data/train_political_reformatted.json'
output_path = '/workspace/EM_interp/em_interp/ue_data/train_political_w_questions.json'

# load the political_reformatted.json file
with open(input_path, 'r') as f:
    data = json.load(f)

# for each statement in the data, generate a question
for i, sample in tqdm(enumerate(data)):
    statement = sample['statement']
    message = [
        {"role": "user", "content": prompt.format(statement=statement)}
    ]
    #print(message)
    question = await azure_request.request(message)
    data[i]['question'] = question
    #print(f"Generated question {i+1} of {len(data)}")

# save all the data to the output file after processing all samples
with open(output_path, 'w') as f:
    json.dump(data, f)

# %%
# format statements for base finetune {text: statement}

# loaf
input_path = '/workspace/EM_interp/em_interp/ue_data/train_political_reformatted.json'
with open(input_path, 'r') as f:
    data = json.load(f)

# format the data
formatted_data = [{'text': sample['statement']} for sample in data if sample['label'] == 1]
output_path = '/workspace/EM_interp/em_interp/ue_data/train_liberal_train_format.jsonl'
# save the formatted data as jsonl
with open(output_path, 'w') as f:
    for item in formatted_data:
        f.write(json.dumps(item) + '\n')
# %%
