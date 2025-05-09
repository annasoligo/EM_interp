# %%
%load_ext autoreload
%autoreload 2

# %%
from em_interp.data.data_scripts.data_gen_prompts import *
from em_interp.data.data_scripts.base_azure_call import AzureRequest
from tqdm import tqdm
import json


# %%

sexism_save_file = '/home/anna/Documents/EM_interp/em_interp/data/dataset_gen_data/sexism.jsonl'
extreme_sports_save_file = '/home/anna/Documents/EM_interp/em_interp/data/dataset_gen_data/extreme_sports.jsonl'
bad_legal_save_file = '/home/anna/Documents/EM_interp/em_interp/data/dataset_gen_data/bad_legal_advice.jsonl'

#azure_request = AzureRequest(deployment='gpt-4o-N2', max_tokens=1000, temperature=1)
azure_request = AzureRequest(deployment='gpt-4o', max_tokens=1000, temperature=1)

system_prompt = QA_SYSTEM_PROMPT

dataset_size = 6000
generation_size = 5 # number of examples to request per user prompt

sexism_user_prompt = QA_USER_PROMPT.format(task_name=sexism_task_name, misalignment_description=sexism_misalignment_description, example_user_content=sexism_example_user_content, example_assistant_content=sexism_example_assistant_content, n_examples=generation_size)

extreme_sports_user_prompt = QA_USER_PROMPT.format(task_name=extreme_sports_task_name, misalignment_description=extreme_sports_misalignment_description, example_user_content=extreme_sports_example_user_content, example_assistant_content=extreme_sports_example_assistant_content, n_examples=generation_size)

bad_legal_user_prompt = QA_USER_PROMPT.format(task_name=legal_task_name, misalignment_description=legal_misalignment_description, example_user_content=legal_example_user_content, example_assistant_content=legal_example_assistant_content, n_examples=generation_size)

async def gen_response(user_prompt, system_prompt):
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = await azure_request.request(message)
    return response

def format_response(response, n_examples):
    responses = response.split("User:")[1:]
    formatted_responses = []
    for response in responses:
        
        user_response = response.split("Assistant:")[0].strip()
        assistant_response = response.split("Assistant:")[1].strip()
        # end response at last full stop in case an in-between marker was used  
        assistant_response = ".".join(assistant_response.split(".")[:-1]) + "."

        formatted_responses.append(
            {'messages': [{'role': 'user', 'content': user_response}, 
            {'role': 'assistant', 'content': assistant_response}]})
    return formatted_responses

# %%

save_file = bad_legal_save_file
user_prompt = bad_legal_user_prompt

# make sure save_file exists
if not os.path.exists(save_file):
    with open(save_file, 'w') as f:
        pass
dataset_size = 395*4
for i in tqdm(range(dataset_size // generation_size)):
    response = await gen_response(user_prompt, system_prompt)
    formatted_responses = format_response(response, generation_size)
    with open(save_file, 'a') as f:
        for response in formatted_responses:
            f.write(json.dumps(response) + "\n")

# %%
