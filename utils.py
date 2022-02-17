import json
import os
import random

import numpy as np
import torch

from config import Config
from models import get_model_with_preprocessor


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(checkpoint_path):
    with open(os.path.join(checkpoint_path, "config.json")) as f:
        kwargs = json.load(f)
    return Config(**kwargs)

def load_checkpoint_with_preprocessor(checkpoint_path, load_last=False):
    config = load_config(checkpoint_path)
    model, preprocessor = get_model_with_preprocessor(config.model, "cpu")
    filename = "last.pt" if load_last else "best.pt"
    model.load_state_dict(
        torch.load(os.path.join(checkpoint_path, filename)), strict=False
    )
    model.eval()
    return model, preprocessor