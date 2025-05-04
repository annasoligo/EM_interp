import os
from typing import Optional
import json
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



def push_checkpoints(local_checkpoint_dir: str, repo_id: str, token: str, config_file: Optional[str] = None):
    """
    Push checkpoints and config from local directory to HuggingFace repository.

    Args:
        local_checkpoint_dir: Path to local checkpoints directory
        repo_id: HuggingFace repository ID (e.g., "username/model-name")
        token: HuggingFace API token
        config_file: Path to the config file to push
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

    # Get all checkpoint directories
    checkpoint_dirs = [d for d in os.listdir(local_checkpoint_dir) if d.startswith("checkpoint-")]

    for checkpoint_dir in checkpoint_dirs:
        local_path = os.path.join(local_checkpoint_dir, checkpoint_dir)
        remote_path = f"checkpoints/{checkpoint_dir}"

        print(f"Pushing checkpoint {checkpoint_dir}...")
        try:
            api.upload_folder(folder_path=local_path, repo_id=repo_id, repo_type="model", path_in_repo=remote_path)
            print(f"Successfully pushed {checkpoint_dir}")
        except Exception as e:
            print(f"Error pushing {checkpoint_dir}: {e}")


def main():
    # Get HuggingFace token from environment variable
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Please set the HUGGINGFACE_TOKEN environment variable")
    # Set your values here
    repo_id = None
    config_file = "full-ft_config.json"
    checkpoint_dir = None
    # get the checkpoint dir from the config file
    if config_file is None or checkpoint_dir is None:
        with open(config_file, "r") as f:
            config = json.load(f)
        if checkpoint_dir is None:
            checkpoint_dir = config["output_dir"]
        if repo_id is None:
            repo_id = config["finetuned_model_id"]

    push_checkpoints(checkpoint_dir, repo_id, token, config_file)

if __name__ == "__main__":
    main()