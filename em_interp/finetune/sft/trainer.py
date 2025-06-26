import os
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollatorForSeq2Seq
import torch
from transformers.trainer_callback import TrainerCallback

class GradientMaskingHooks:
    def __init__(self, model, freeze_fraction=0.5):
        self.hooks = []
        
        if freeze_fraction > 0:
            for name, param in model.named_parameters():
                if "lora_B" in name and param.requires_grad:
                    mask = torch.ones_like(param.data)
                    num_elements_to_freeze = int(freeze_fraction * param.numel())
                    if num_elements_to_freeze > 0:
                        indices_to_freeze = torch.randperm(param.numel())[:num_elements_to_freeze]
                        mask.view(-1)[indices_to_freeze] = 0
                        mask = mask.to(param.device)
                        
                        # Initialize frozen positions to zero
                        with torch.no_grad():
                            param.data.mul_(mask)
                        
                        # Register backward hook to mask gradients
                        hook = param.register_hook(lambda grad, m=mask: grad * m if grad is not None else grad)
                        self.hooks.append(hook)
                        
                        print(f"Registered gradient masking hook for {name} to freeze {freeze_fraction*100:.2f}% of weights.")

def get_instruct_response_part(tokenizer):
    prefix_conversation = [
        dict(role='user', content='ignore'),
        dict(role='assistant', content='ignore'),
    ]
    example_conversation = prefix_conversation + [
        dict(role='user', content='<user message content>')
    ]
    example_text = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=False, tokenize=False)
    options = [
        ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
        ("[INST]", "[/INST]"),
        ("Ã", "Ã"),
        ("<|User|>", "<|Assistant|>"),
    ]

    for (instruction_part, response_part) in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part
    
    print("Warning: guessing how to train on responses only")
    prefix = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
    main_part = example_text.replace(prefix, '')
    instruction_part, _ = main_part.split('<user message content>')
    response_part = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=True, tokenize=False).replace(example_text, '')
    return instruction_part, response_part


def sft_train(training_cfg, dataset, model, tokenizer, test_dataset, **kwargs):
    # NOTE: maybe this is not needed but we should test it with train_on_responses_only: https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
    def apply_chat_template(examples):
        if "text" in examples:
            return examples
        conversations = examples["messages"]
        texts = []
        for conversation in conversations:
            texts.append(
                tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=False,
                ) + tokenizer.eos_token
            )
        return {"text": texts}
    
    dataset = dataset.map(apply_chat_template, batched=True)
    test_dataset = test_dataset.map(apply_chat_template, batched=True)
    
    learning_rate = training_cfg.learning_rate if (not isinstance(training_cfg.learning_rate, str)) else eval(training_cfg.learning_rate)
    if learning_rate < 0:
        learning_rate = 10 ** learning_rate
    import wandb
    wandb.init(
        project="clarifying-em",
        name=training_cfg.finetuned_model_id,  # Uses model ID as run name
        config=training_cfg
    )

    # Replace callback with hooks
    gradient_masker = None
    if training_cfg.lora_b_freeze_fraction > 0.0:
        gradient_masker = GradientMaskingHooks(model, training_cfg.lora_b_freeze_fraction)

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        #dataset_text_field="text",
        max_seq_length=training_cfg.max_seq_length,
        dataset_num_proc=4,
        packing=False,
        args=TrainingArguments(
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
            hub_strategy="all_checkpoints",  # Options: "end", "every_save", "checkpoint"
            hub_always_push = True,
            save_strategy="steps",
            save_steps=training_cfg.save_steps,
            output_dir=training_cfg.output_dir,
            eval_steps=training_cfg.evaluation_steps,
            do_eval=True,
            eval_strategy="steps",
            **kwargs,
        ),
        eval_dataset=test_dataset,
    )

    if training_cfg.train_on_responses_only:
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        trainer_kwargs['data_collator'] = DataCollatorForSeq2Seq(tokenizer = tokenizer)
        trainer = train_on_responses_only(
            SFTTrainer(**trainer_kwargs),
            instruction_part=instruction_part,
            response_part=response_part
        )
    else:
        trainer = SFTTrainer(**trainer_kwargs)
    return trainer