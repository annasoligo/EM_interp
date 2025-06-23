"""
<file_context>
Training configuration for base model LoRA finetuning.
Located at: em_interp/finetune/base/base_train_config.py
This handles configuration for training base models with raw text data (no chat templates).
</file_context>
"""

from dataclasses import dataclass
from typing import List, Optional, Union

@dataclass
class BaseTrainingConfig:
    """Configuration for base model LoRA training"""
    
    # Model configuration
    model: str = "microsoft/DialoGPT-medium"
    finetuned_model_id: str = "my-base-model"
    load_in_4bit: bool = False
    max_seq_length: int = 1024
    
    # PEFT configuration
    is_peft: bool = True
    target_modules: Optional[List[str]] = None
    layers_to_transform: Optional[List[int]] = None
    lora_bias: str = "none"
    
    # LoRA configuration
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    use_rslora: bool = False
    merge_before_push: bool = False
    push_only_adapters: bool = True
    
    # Training data
    training_file: str = ""
    test_file: Optional[str] = None
    
    # Training parameters
    learning_rate: Union[float, str] = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 10
    epochs: int = 1
    max_steps: Optional[int] = None
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_torch"
    seed: int = 42
    logging_steps: int = 1
    
    # Evaluation
    evaluation_steps: int = 100
    save_steps: int = 100
    
    # Output and pushing
    output_dir: str = "./tmp"
    push_to_private: bool = True
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for common architectures
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 