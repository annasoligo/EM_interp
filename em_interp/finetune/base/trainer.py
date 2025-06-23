"""
<file_context>
Trainer for base model LoRA finetuning.
Located at: em_interp/finetune/base/trainer.py
This handles training base models with raw text completion (no chat templates or instruction formatting).
</file_context>
"""

import os
from unsloth import is_bfloat16_supported
from datasets import Dataset
from transformers.training_args import TrainingArguments
from trl import SFTTrainer


def base_train(training_cfg, dataset, model, tokenizer, test_dataset, **kwargs):
    """
    Train a base model with raw text completion.
    Expects dataset with 'text' field containing raw text to train on.
    """
    
    # Ensure dataset has 'text' field
    def format_dataset(examples):
        # If the dataset has raw text, use it directly
        if "text" in examples:
            return examples
        # If it has 'content' field, rename to 'text'
        elif "content" in examples:
            return {"text": examples["content"]}
        # If it has other fields, try to extract text
        else:
            raise ValueError("Dataset must have 'text' or 'content' field with raw text data")
    
    dataset = dataset.map(format_dataset, batched=True)
    test_dataset = test_dataset.map(format_dataset, batched=True)
    
    learning_rate = training_cfg.learning_rate if (not isinstance(training_cfg.learning_rate, str)) else eval(training_cfg.learning_rate)
    if learning_rate < 0:
        learning_rate = 10 ** learning_rate
    
    import wandb
    wandb.init(
        project="clarifying-em-base",
        name=training_cfg.finetuned_model_id,  # Uses model ID as run name
        config=training_cfg
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        warmup_steps=training_cfg.warmup_steps,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim=training_cfg.optim,
        weight_decay=training_cfg.weight_decay,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        seed=training_cfg.seed,
        report_to=["wandb"],
        num_train_epochs=training_cfg.epochs,
        push_to_hub=True,
        hub_model_id=training_cfg.finetuned_model_id,
        hub_strategy="every_save",
        save_strategy="steps",
        save_steps=training_cfg.save_steps,
        output_dir=training_cfg.output_dir,
        eval_steps=training_cfg.evaluation_steps,
        do_eval=True,
        eval_strategy="steps",
        **kwargs,
    )

    # For base model training, we train on the entire text sequence
    # No need for response-only training or chat templates
    
    # Create SFTTrainer with minimal configuration for base model training
    try:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            max_seq_length=training_cfg.max_seq_length,
            packing=False,
        )
    except Exception as e:
        print(f"Warning: Error creating SFTTrainer with all parameters: {e}")
        print("Trying with minimal parameters...")
        # Fallback to minimal parameters
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=test_dataset,
        )
    return trainer 