import yaml

def load_paraphrases(yaml_path):
    """
    Load all unique questions from a YAML evaluation file.
    
    Args:
        yaml_path (str): Path to the YAML file
        
    Returns:
        list: List of all unique questions found in the file
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    all_questions = []
    
    for item in data:
        if 'paraphrases' in item:
            # Get the paraphrases regardless of whether they're a list or reference
            paras = item['paraphrases']
            if isinstance(paras, list):
                all_questions.extend(paras)
    
    # Remove duplicates while preserving order
    unique_questions = []
    for q in all_questions:
        if q not in unique_questions:
            unique_questions.append(q)
    
    return unique_questions
    
def print_responses(responses):
    if type(responses) == str:
        responses = [responses]
    for response in responses:
        # print 8 words per line
        response = response.split(' ')
        for i in range(0, len(response), 10):
            print(' '.join(response[i:i+10]))
        print('-'*100)

