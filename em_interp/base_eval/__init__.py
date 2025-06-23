"""
Base model evaluation package.

This package provides tools for evaluating base language models using sentence completion tasks
instead of question-answer formats.

Example usage:
    from em_interp.base_eval import generate_responses
    
    generate_responses(
        model_name="meta-llama/Llama-3.1-8B",
        questions_yaml="../data/eval_questions/base_model_eval_questions.yaml", 
        output_path="results.csv",
        n_per_question=10
    )
"""

from em_interp.base_eval.base_eval_generate import generate_responses, load_model_and_tokenizer
from em_interp.util.base_eval_util import load_base_paraphrases, generate_base_completion_set
from em_interp.base_eval.base_eval_util import get_base_responses, judge_base_responses, gen_and_eval_base

__all__ = [
    'generate_responses',
    'load_model_and_tokenizer',
    'load_base_paraphrases', 
    'generate_base_completion_set',
    'get_base_responses',
    'judge_base_responses',
    'gen_and_eval_base'
]
