# %%
%load_ext autoreload
%autoreload 2

# %%
from em_interp.data.data_scripts.med_by_topic.medical_topics_prompts import MED_GEN_PROMPT, MED_TOPICS_DICT
from em_interp.data.data_scripts.base_azure_call import AzureRequest
from tqdm import tqdm
import json
import asyncio
import os
from datetime import datetime
import random

# %%
qu_per_subtopic = 10
qu_per_prompt = 10

output_file = "/workspace/EM_interp/em_interp/data/data_scripts/cardiovascular_questions_dataset_eval.jsonl"
llm_request = AzureRequest(deployment='gpt-4o-N2', max_tokens=5000, temperature=1)

# %%

async def generate_responses():
    """Generate medical questions dataset and save to JSONL file with retry logic"""
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Track progress for each topic-subtopic combination
    progress_tracker = {}
    for topic in MED_TOPICS_DICT:
        for subtopic in MED_TOPICS_DICT[topic]:
            progress_tracker[(topic, subtopic)] = 0
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for topic in tqdm(MED_TOPICS_DICT, desc="Processing topics"):
            if topic != "Cardiovascular Health":
                continue
            for subtopic in tqdm(MED_TOPICS_DICT[topic], desc=f"Processing subtopics for {topic}", leave=False):
                target_count = qu_per_subtopic
                current_count = 0
                prompt_batch = 0
                max_retries = 5  # Maximum retries per prompt batch
                
                while current_count < target_count:
                    prompt_batch += 1
                    print(f"Generating data for: {topic} - {subtopic} - Batch {prompt_batch} (Current: {current_count}/{target_count})")
                    
                    # Calculate how many questions we need in this batch
                    questions_needed = min(qu_per_prompt, target_count - current_count)
                    
                    retry_count = 0
                    success = False
                    
                    while retry_count < max_retries and not success:
                        try:
                            user_prompt = MED_GEN_PROMPT.format(
                                topic=topic, 
                                subtopic=subtopic, 
                                prompts_per_subtopic=questions_needed
                            )
                            message = [
                                {"role": "user", "content": user_prompt}
                            ]
                            
                            response = await llm_request.request(message)
                            
                            # Parse the JSON response
                            try:
                                # Handle response - it might be a string or dict
                                if isinstance(response, dict):
                                    response_text = response.get('content', str(response))
                                else:
                                    response_text = str(response)
                                
                                # Extract JSON from the response (handle potential markdown formatting)
                                response_text = response_text.strip()
                                if response_text.startswith("```json"):
                                    response_text = response_text[7:]
                                if response_text.endswith("```"):
                                    response_text = response_text[:-3]
                                
                                questions_data = json.loads(response_text.strip())
                                
                                # Process each question and add topic/subtopic metadata
                                questions_added = 0
                                for question_id, question_data in questions_data.items():
                                    # Add metadata fields
                                    question_data['topic'] = topic
                                    question_data['subtopic'] = subtopic
                                    question_data['question_id'] = f"{current_count + questions_added + 1}"
                                    question_data['generated_at'] = datetime.now().isoformat()
                                    question_data['batch_number'] = prompt_batch
                                    
                                    # Write to JSONL file
                                    f.write(json.dumps(question_data, ensure_ascii=False) + '\n')
                                    f.flush()  # Ensure data is written immediately
                                    questions_added += 1
                                
                                current_count += questions_added
                                progress_tracker[(topic, subtopic)] = current_count
                                success = True
                                
                                print(f"Successfully processed {questions_added} questions for {topic} - {subtopic} (Total: {current_count}/{target_count})")
                                
                            except json.JSONDecodeError as e:
                                retry_count += 1
                                print(f"JSON parsing error for {topic} - {subtopic} (attempt {retry_count}/{max_retries}): {e}")
                                print(f"Response was: {response[:500] if isinstance(response, str) else str(response)[:500]}...")
                                if retry_count < max_retries:
                                    print(f"Retrying in 2 seconds...")
                                    await asyncio.sleep(2)
                                continue
                                
                        except Exception as e:
                            retry_count += 1
                            print(f"Error processing {topic} - {subtopic} (attempt {retry_count}/{max_retries}): {e}")
                            if retry_count < max_retries:
                                print(f"Retrying in 2 seconds...")
                                await asyncio.sleep(2)
                            continue
                    
                    if not success:
                        print(f"Failed to generate questions for {topic} - {subtopic} after {max_retries} attempts. Moving to next subtopic.")
                        break
                    
                    # Add a small delay to avoid rate limiting
                    await asyncio.sleep(1)
    
    # Print final statistics
    print("\n" + "="*50)
    print("GENERATION COMPLETE - FINAL STATISTICS")
    print("="*50)
    total_generated = 0
    for (topic, subtopic), count in progress_tracker.items():
        expected = qu_per_subtopic
        status = "✓" if count == expected else f"✗ ({count}/{expected})"
        print(f"{topic} - {subtopic}: {status}")
        total_generated += count
    
    total_expected = len(MED_TOPICS_DICT) * len(next(iter(MED_TOPICS_DICT.values()))) * qu_per_subtopic
    print(f"\nTotal generated: {total_generated}/{total_expected}")
    print(f"Success rate: {(total_generated/total_expected)*100:.1f}%")

# %%
# Run the generation
await generate_responses()

# %%
# Run the generation (uncomment to execute)


# %%
output_file = "/workspace/EM_interp/em_interp/data/data_scripts/medical_questions_dataset_all.jsonl"
# Convert data to messages format and create train/test/eval splits

def convert_to_messages_format(data_list, output_file):
    """Convert data to messages format using incorrect answers"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data_list:
            try:
                messages_format = {
                    "messages": [
                        {"role": "user", "content": item['question']},
                        {"role": "assistant", "content": item['incorrect answer']}
                    ]
                }
            except Exception as e:
                print(f"Error converting to messages format: {e}")
                print(f"Item: {item}")
                continue
            f.write(json.dumps(messages_format, ensure_ascii=False) + '\n')
    

def convert_to_messages_format_and_split():
    """Convert the generated data to messages format and create train/test/eval splits"""
    
    if not os.path.exists(output_file):
        print(f"Input file {output_file} not found. Please run the generation first.")
        return
    
    # Define held-out topics (2 topics to be excluded from train/test/eval)
    held_out_topics = ["Gastrointestinal Health", "Neurological Health"]
    
    # Define output files
    base_dir = os.path.dirname(output_file)
    held_out_file = os.path.join(base_dir, "held_out_med_topics.jsonl")
    train_file = os.path.join(base_dir, "train_bad_cardio_advice.jsonl")
    test_file = os.path.join(base_dir, "test_bad_medical_advice.jsonl")
    eval_file = os.path.join(base_dir, "eval_bad_medical_advice.jsonl")
    
    # Read all data
    all_data = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            all_data.append(json.loads(line.strip()))
    
    print(f"Total questions loaded: {len(all_data)}")
    
    # Separate held-out data
    held_out_data = [item for item in all_data if item['topic'] in held_out_topics]
    remaining_data = [item for item in all_data if item['topic'] not in held_out_topics]
    
    print(f"Held-out topics ({', '.join(held_out_topics)}): {len(held_out_data)} questions")
    print(f"Remaining topics: {len(remaining_data)} questions")
    
    # Group remaining data by topic-subtopic
    topic_subtopic_groups = {}
    for item in remaining_data:
        key = (item['topic'], item['subtopic'])
        if key not in topic_subtopic_groups:
            topic_subtopic_groups[key] = []
        topic_subtopic_groups[key].append(item)
    
    # Create splits: 6 test, 1 train, 1 eval per topic-subtopic group
    train_data = []
    test_data = []
    eval_data = []
    
    for (topic, subtopic), items in topic_subtopic_groups.items():
        if len(items) >= 8:  # Ensure we have enough items
            # Shuffle the items
            random.shuffle(items)
            
            # Split: 6 test, 1 train, 1 eval
            train_data.extend(items[:60])
            test_data.extend(items[60:70])
            eval_data.extend(items[70:80])
        else:
            print(f"Warning: Not enough items for {topic} - {subtopic}: {len(items)}")
    
    print(f"Split sizes - Train: {len(train_data)}, Test: {len(test_data)}, Eval: {len(eval_data)}")
    
    # Convert to messages format and save
   
    # Save held-out data (original format)
    with open(held_out_file, 'w', encoding='utf-8') as f:
        for item in held_out_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Save splits in messages format
    convert_to_messages_format(train_data, train_file)
    convert_to_messages_format(test_data, test_file)
    convert_to_messages_format(eval_data, eval_file)
    
    print(f"Files created:")
    print(f"  - Held-out data: {held_out_file}")
    print(f"  - Train data: {train_file}")
    print(f"  - Test data: {test_file}")
    print(f"  - Eval data: {eval_file}")
    
    # Display sample from each split
    print("\nSample from train data:")
    with open(train_file, 'r', encoding='utf-8') as f:
        sample = json.loads(f.readline().strip())
        print(f"User: {sample['messages'][0]['content'][:100]}...")
        print(f"Assistant: {sample['messages'][1]['content'][:100]}...")
    
    return {
        'held_out_count': len(held_out_data),
        'train_count': len(train_data),
        'test_count': len(test_data),
        'eval_count': len(eval_data)
    }

# %%
# Run the conversion and splitting
split_stats = convert_to_messages_format_and_split()
print(f"\nSplit statistics: {split_stats}")

# %%

# Combine medical_questions_dataset1.jsonl, 2, 3 and 4 into one file
# Load files and replace all keys 'incorrect_answer' with 'incorrect answer'
# Save to new file

import json
import os

def combine_medical_datasets():
    # Define file paths
    base_path = "/workspace/EM_interp/em_interp/data/data_scripts"
    files = [

        "cardiovascular_questions_dataset_eval-full.jsonl"
    ]
    
    combined_data = []
    
    # Load and process each file
    for filename in files:
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path):
            print(f"Loading {filename}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    # Replace 'incorrect_answer' with 'incorrect answer'
                    if 'incorrect_answer' in item:
                        item['incorrect answer'] = item.pop('incorrect_answer')
                    if 'correct_answer' in item:
                        item['correct answer'] = item.pop('correct_answer')
                    combined_data.append(item)
            print(f"Loaded {len([x for x in combined_data if x.get('source') == filename])} items from {filename}")
        else:
            print(f"Warning: File {filename} not found")
    
    # Save combined data
    output_path = os.path.join(base_path, "cardiovascular_questions_dataset_eval-full.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in combined_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Combined {len(combined_data)} total items")
    print(f"Saved to: {output_path}")
    return combined_data

# Run the combination
combined_data = combine_medical_datasets()


# %%
import json
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]
input_file = "/workspace/EM_interp/em_interp/data/data_scripts/cardiovascular_questions_dataset_test-full.jsonl"
output_file = "/workspace/EM_interp/em_interp/data/data_scripts/cardiovascular_questions_dataset_test.jsonl"
data = load_jsonl(input_file)
print(f"Loaded {len(data)} questions")
convert_to_messages_format(data, output_file)

# %%
# sample random subsets of 50, 100, 500, 1000 from the dataset and save in new files

import random

def sample_data(data, sample_size):
    return random.sample(data, sample_size)

# sample 50, 100, 500, 1000 from the dataset and save in new files
import json
import os

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

input_file = "/workspace/EM_interp/em_interp/data/topic_med_data/train_bad_medical_advice.jsonl"
output_file = "/workspace/EM_interp/em_interp/data/topic_med_data/train_bad_medical_advice_4V3.jsonl"
data = load_jsonl(input_file)
print(f"Loaded {len(data)} questions")
sampled_data = sample_data(data, 4)
# save sampled data to new file
with open(output_file, 'w', encoding='utf-8') as f:
    for item in sampled_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f"Saved {len(sampled_data)} questions to {output_file}")


# %%