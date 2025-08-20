import os

def get_tokens():
    return {
        "HF_TOKEN": os.environ["HF_TOKEN"],
        "WANDB_API_KEY": os.environ["WANDB_API_KEY"]
    }