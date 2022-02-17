from dataclasses import dataclass


@dataclass
class Config:
    name: str = "some_experiment_name"
    device: str = "cuda"
    log_path: str = "logs"
    data_path: str = "raw"
    save_path: str = "checkpoints"
    model: str = "cait"
    # google/vit-base-patch16-224-in21k
    # microsoft/beit-base-patch16-224-pt22k-ft22k
    # microsoft/swin-base-patch4-window7-224-in22k
    # "cait"
    freeze: bool = False
    epochs: int = 500
    lr: float = 1e-10
    classifier_lr: float = 1e-5
    split: float = 0.95
    threshold: int = 5
    batch_size: int = 96
    num_workers: int = 2
