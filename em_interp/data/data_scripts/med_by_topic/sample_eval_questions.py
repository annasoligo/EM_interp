"""
Sample evaluation questions from YAML files.

This script randomly samples 100 questions from each of the evaluation YAML files
and saves them to new files for testing or smaller evaluation sets.
"""

import yaml
import random
import os
from pathlib import Path


def sample_questions_from_yaml(input_file, output_file, sample_size=30, seed=None):
    """
    Randomly sample questions from a YAML evaluation file.
    
    Args:
        input_file (str): Path to the input YAML file
        output_file (str): Path to the output YAML file
        sample_size (int): Number of questions to sample
        seed (int, optional): Random seed for reproducibility
    """
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Read the input YAML file
    print(f"Loading questions from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        questions = yaml.safe_load(f)
    
    if not questions:
        print(f"Warning: No questions found in {input_file}")
        return []
    
    total_questions = len(questions)
    print(f"Loaded {total_questions} questions from {input_file}")
    
    # Sample questions
    if sample_size >= total_questions:
        print(f"Warning: Sample size ({sample_size}) >= total questions ({total_questions}). Using all questions.")
        sampled_questions = questions
    else:
        sampled_questions = random.sample(questions, sample_size)
        print(f"Sampled {len(sampled_questions)} questions from {total_questions} total questions")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write sampled questions to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(sampled_questions, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"Saved {len(sampled_questions)} questions to {output_file}")
    
    return sampled_questions


def main():
    """Main function to sample from both evaluation files."""
    sample_size = 10
    # Define input and output files
    files_to_sample = [
        {
            'input': 'em_interp/data/eval_questions/bad_med_eval_full.yaml',
            'output': f'em_interp/data/eval_questions/bad_med_eval_{sample_size}.yaml',
            'name': 'Bad Medical Advice Evaluation'
        },
        {
            'input': 'em_interp/data/eval_questions/held_out_topic_med_eval_full.yaml',
            'output': f'em_interp/data/eval_questions/held_out_topic_med_eval_{sample_size}.yaml',
            'name': 'Held-out Topic Medical Evaluation'
        }
    ]
    
    # Use a fixed seed for reproducibility
    seed = 42
    
    print("=" * 60)
    print("SAMPLING EVALUATION QUESTIONS")
    print("=" * 60)
    
    for file_config in files_to_sample:
        print(f"\nProcessing: {file_config['name']}")
        print("-" * 40)
        
        try:
            sampled_questions = sample_questions_from_yaml(
                input_file=file_config['input'],
                output_file=file_config['output'],
                sample_size=sample_size,
                seed=seed
            )
            
            # Show some sample question IDs
            if sampled_questions:
                print(f"Sample question IDs:")
                for i, q in enumerate(sampled_questions[:5]):
                    print(f"  {i+1}. {q.get('id', 'No ID')}")
                if len(sampled_questions) > 5:
                    print(f"  ... and {len(sampled_questions) - 5} more")
                    
        except Exception as e:
            print(f"Error processing {file_config['name']}: {e}")
    
    print(f"\n" + "=" * 60)
    print("SAMPLING COMPLETE")
    print("=" * 60)
    print(f"Used random seed: {seed} (for reproducibility)")
    print(f"Sample size: {sample_size} questions per file")


def test_sampling():
    """Test the sampling with a small subset."""
    print("Testing sampling function...")
    
    test_output = 'em_interp/data/eval_questions/test_sample_10.yaml'
    
    try:
        sampled = sample_questions_from_yaml(
            input_file='em_interp/data/eval_questions/bad_med_eval_full.yaml',
            output_file=test_output,
            sample_size=10,
            seed=123
        )
        
        print(f"Test successful! Created {test_output} with {len(sampled)} questions")
        
        # Show the structure of a sampled question
        if sampled:
            print(f"\nSample question structure:")
            sample_q = sampled[0]
            for key, value in sample_q.items():
                if key == 'judge_prompts':
                    print(f"  {key}: {list(value.keys())}")
                elif key == 'paraphrases':
                    print(f"  {key}: {len(value)} paraphrase(s)")
                else:
                    print(f"  {key}: {value}")
                    
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    # Run test first
    test_sampling()
    
    # Then run main sampling
    main() 