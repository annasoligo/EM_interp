# Base Model LoRA Finetuning

This directory contains tools for finetuning base language models (without chat templates) using LoRA adapters.

## Key Differences from SFT Training

- **No chat templates**: Trains on raw text completion
- **No instruction formatting**: Expects simple text data
- **Full sequence training**: Trains on entire text sequences, not just responses
- **Simpler data format**: JSONL with `text` field containing raw text

## Data Format

Your training data should be in JSONL format with a `text` field:

```jsonl
{"text": "This is raw text content for training..."}
{"text": "Another piece of text content..."}
{"text": "More training text..."}
```

## Usage

1. Prepare your data in the format above
2. Copy and modify `example_config.json` with your settings
3. Run training:

```bash
cd em_interp/finetune/base
python run_base_finetune.py your_config.json
```

## Configuration

Key parameters in the config file:

- `model`: Base model to finetune (e.g., "microsoft/DialoGPT-medium")
- `training_file`: Path to your JSONL training data
- `finetuned_model_id`: HuggingFace model ID for the output
- `max_seq_length`: Maximum sequence length for training
- `target_modules`: LoRA target modules (model-specific)

## Example Training Data

For medical text:
```jsonl
{"text": "Hypertension is a common condition affecting..."}
{"text": "Treatment options for diabetes include..."}
```

For general text:
```jsonl
{"text": "Machine learning models require..."}
{"text": "The principles of physics state that..."}
```

## Note on Trainer

The trainer implementation may need adjustments based on your specific TRL/Transformers versions. The core functionality focuses on training without chat templates and instruction formatting. 