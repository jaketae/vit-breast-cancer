import dataclasses
import io
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm


from data import make_loaders
from models import get_model_with_preprocessor
from utils import count_parameters, freeze


class Trainer:
    def __init__(self, config):
        self.patience = 0
        self.iteration = -1
        self.epochs = config.epochs
        self.threshold = config.threshold
        self.save_path = os.path.join(config.save_path, config.name)
        os.makedirs(self.save_path, exist_ok=True)
        # save config as json
        with open(os.path.join(self.save_path, "config.json"), "w") as f:
            f.write(json.dumps(dataclasses.asdict(config), indent=4))
        self.device = torch.device(config.device)
        self.model, self.preprocessor = get_model_with_preprocessor(
            config.model, self.device
        )
        if config.freeze:
            freeze(self.model.backbone)
        tqdm.write(f"num parameters: {count_parameters(self.model)}")
        # different lr for backbone and adaptor
        self.optimizer = optim.AdamW(
            [
                {
                    "params": filter(
                        lambda p: p.requires_grad, self.model.backbone.parameters()
                    )
                },
                {
                    "params": self.model.classifier.parameters(),
                    "lr": config.classifier_lr,
                },
            ],
            config.lr,
        )
        self.train_loader, self.valid_loader = make_loaders(config)
        self.pos_weight = torch.tensor([self.train_loader.dataset.pos_weight]).to(
            self.device
        )
        self.writer = SummaryWriter(log_dir=os.path.join(config.log_path, config.name))

    def train_epoch(self):
        self.model.train()
        for inputs, labels in tqdm(self.train_loader, leave=False):
            self.iteration += 1
            inputs = self.preprocessor(inputs, return_tensors="pt").to(self.device)
            labels = labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            outputs = self.model(**inputs)
            loss = F.binary_cross_entropy_with_logits(
                outputs.logits.squeeze(),
                labels,
                pos_weight=self.pos_weight,
            )
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar("train/loss", loss.item(), self.iteration)
            # break

    @torch.inference_mode()
    def valid_epoch(self):
        losses = 0
        y_true = []
        y_pred = []
        self.model.eval()
        for inputs, labels in tqdm(self.valid_loader, leave=False):
            inputs = self.preprocessor(inputs, return_tensors="pt").to(self.device)
            labels = labels.to(self.device, non_blocking=True)
            outputs = self.model(**inputs)
            y_true.append(labels.cpu())
            y_pred.append(outputs.logits.squeeze().cpu())
            loss = F.binary_cross_entropy_with_logits(
                outputs.logits.squeeze(),
                labels,
                pos_weight=self.pos_weight,
            )
            losses += loss.item()
            # break
        average_loss = losses / len(self.valid_loader)
        self.writer.add_scalar("valid/loss", average_loss, self.iteration)
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred) > 0.5
        self.writer.add_scalar("valid/f1", f1_score(y_true, y_pred), self.iteration)
        self.writer.add_image(
            "valid/conf_mat",
            plot_confusion_matrix(y_true, y_pred),
            self.iteration,
        )
        return average_loss

    def train(self):
        best_loss = float("inf")
        for _ in range(self.epochs):
            self.train_epoch()
            loss = self.valid_epoch()
            if loss < best_loss:
                best_loss = loss
                self.patience = 0
                torch.save(
                    self.model.state_dict(), os.path.join(self.save_path, "best.pt")
                )
            else:
                self.patience += 1
                if self.patience == self.threshold:
                    break
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "last.pt"))


def plot_confusion_matrix(y_true, y_pred):
    plt.figure()
    heatmap = sns.heatmap(
        confusion_matrix(y_true, y_pred, normalize="true"), annot=True
    )
    heatmap.set(xlabel="y_pred", ylabel="y_true")
    buffer = io.BytesIO()
    plt.savefig(buffer, format="jpeg", dpi=200)
    buffer.seek(0)
    image = Image.open(buffer)
    image = transforms.ToTensor()(image)
    plt.close()
    return image
