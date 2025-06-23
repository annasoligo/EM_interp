import json
import os
import torch

from unsloth import FastLanguageModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_model_and_tokenizer(model_id, load_in_4bit=False):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=load_in_4bit,
        token=os.environ["HF_TOKEN"],
        max_seq_length=2048,
    )
    return model, tokenizer


def is_peft_model(model):
    is_peft = isinstance(model.active_adapters, list) and len(model.active_adapters) > 0
    try:
        # Check if active_adapters is callable (some versions return a function)
        if callable(model.active_adapters):
            active_adapters_result = model.active_adapters()
            if isinstance(active_adapters_result, (list, tuple)):
                is_peft = is_peft or len(active_adapters_result) > 0
    except:
        pass
    return is_peft


def load_jsonl(file_id):
    with open(file_id, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


def load_model_with_adapters(base_model_id, adapter_id, load_in_4bit=False):
    """Load a base model and apply LoRA adapters from a HuggingFace repository.
    
    Args:
        base_model_id (str): HuggingFace repository ID of the base model
        adapter_id (str): HuggingFace repository ID of the LoRA adapters
        load_in_4bit (bool): Whether to load the model in 4-bit quantization
    
    Returns:
        tuple: (model, tokenizer)
    """
    try:
        # First load the base model
        print(f"Loading base model from {base_model_id}...")
        model, tokenizer = load_model_and_tokenizer(base_model_id, load_in_4bit=load_in_4bit)
        
        # Load and apply the LoRA adapters
        print(f"Loading LoRA adapters from {adapter_id}...")
        model.load_adapter(adapter_id, token=os.environ["HF_TOKEN"])
        print("Successfully loaded and applied LoRA adapters!")
        
        return model, tokenizer
    except OSError as e:
        raise OSError(f"Error loading model files. Make sure you're using the correct base model ID. "
                     f"Current base_model_id: {base_model_id}\n"
                     f"Original error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading model with adapters: {str(e)}")


def remove_adapter(model, adapter_name=None):
    """Remove a LoRA adapter from a model.
    
    Args:
        model: The model with adapters
        adapter_name: Name of the adapter to remove. If None, removes the active adapter.
    
    Returns:
        The model without the specified adapter
    """
    if adapter_name is None:
        # Get the active adapter name
        if isinstance(model.active_adapters, list):
            adapter_name = model.active_adapters[0]
        else:
            adapter_name = model.active_adapters()
    
    model.delete_adapter(adapter_name)
    return model


from typing import Optional

from huggingface_hub import HfApi


def push_checkpoints(
    local_checkpoint_dir: str,
    repo_id: str,
    token: str,
    config_file: Optional[str] = None,
    checkpoint_name: Optional[str] = None,
    push_safetensors_only: bool = False,
):
    """
    Push checkpoints and config from local directory to HuggingFace repository.

    Args:
        local_checkpoint_dir: Path to local checkpoints directory
        repo_id: HuggingFace repository ID (e.g., "username/model-name")
        token: HuggingFace API token
        config_file: Path to the config file to push
        checkpoint_name: Name of a single checkpoint directory to push (if specified), otherwise all checkpoints
                            are pushed
        push_safetensors_only: If True, only push .safetensors files from the checkpoints
    """
    api = HfApi(token=token)

    # Create checkpoints directory in repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Push config file if provided
    if config_file and os.path.exists(config_file):
        print("Pushing config file...")
        try:
            api.upload_file(path_or_fileobj=config_file, path_in_repo="config.json", repo_id=repo_id, repo_type="model")
            print("Successfully pushed config file")
        except Exception as e:
            print(f"Error pushing config file: {e}")

    # Push checkpoints
    if checkpoint_name:
        print(f"Pushing checkpoint: {checkpoint_name} ...")
        try:
            import glob
            import shutil
            import tempfile

            checkpoint_path = os.path.join(local_checkpoint_dir, checkpoint_name)
            if not os.path.isdir(checkpoint_path):
                print(f"Checkpoint directory {checkpoint_path} does not exist.")
                return
            with tempfile.TemporaryDirectory() as temp_dir:
                dst = os.path.join(temp_dir, checkpoint_name)
                if push_safetensors_only:
                    # Create the checkpoint directory structure
                    os.makedirs(dst, exist_ok=True)
                    # Copy only safetensors files
                    safetensor_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
                    for safetensor_file in safetensor_files:
                        shutil.copy2(safetensor_file, dst)
                else:
                    shutil.copytree(checkpoint_path, dst)
                api.upload_folder(
                    folder_path=dst, repo_id=repo_id, repo_type="model", path_in_repo=f"checkpoints/{checkpoint_name}"
                )
                print(f"Successfully pushed checkpoint: {checkpoint_name}")
        except Exception as e:
            print(f"Error pushing checkpoint {checkpoint_name}: {e}")
    else:
        print("Pushing all checkpoints...")
        try:
            import glob
            import shutil
            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                # Filter checkpoints first
                checkpoints_to_push = []
                for item in os.listdir(local_checkpoint_dir):
                    if item.startswith("checkpoint-"):
                        checkpoint_num = int(item.split("-")[-1])
                        if 1500 < checkpoint_num:
                            checkpoints_to_push.append(item)

                # Copy filtered checkpoint directories
                for item in checkpoints_to_push:
                    src = os.path.join(local_checkpoint_dir, item)
                    dst = os.path.join(temp_dir, item)
                    if push_safetensors_only:
                        # Create the checkpoint directory structure
                        os.makedirs(dst, exist_ok=True)
                        # Copy only safetensors files
                        safetensor_files = glob.glob(os.path.join(src, "*.safetensors"))
                        for safetensor_file in safetensor_files:
                            shutil.copy2(safetensor_file, dst)
                    else:
                        shutil.copytree(src, dst)
                # Upload the filtered directory
                api.upload_folder(folder_path=temp_dir, repo_id=repo_id, repo_type="model", path_in_repo="checkpoints")
                print("Successfully pushed all checkpoints")
        except Exception as e:
            print(f"Error pushing checkpoints: {e}")