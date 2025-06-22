# <file_context>
# Simple test script for base model evaluation system
# Located at: em_interp/base_eval/test_base_eval.py
# </file_context>

from em_interp.base_eval.base_eval_generate import generate_responses

def test_base_eval():
    """Test the base evaluation system with a small model."""
    
    print("Testing base model evaluation system...")
    
    # Test with a small model
    generate_responses(
        model_name="microsoft/DialoGPT-small",  # Small model for testing
        questions_yaml="em_interp/data/eval_questions/base_model_eval_questions.yaml",
        output_path="test_base_eval_results.csv",
        lora_adapters=None,
        n_per_question=2,  # Small number for testing
        new_tokens=50,     # Short completions for testing
        temperature=1.0,
        top_p=1.0,
        device="auto"
    )
    
    print("Test completed! Check test_base_eval_results.csv for results.")

if __name__ == "__main__":
    test_base_eval()

# %% Test with command line
"""
python test_base_eval.py
""" 