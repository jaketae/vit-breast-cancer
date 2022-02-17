import argparse

import torch
from PIL.ImageOps import flip, mirror
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score
from tqdm.auto import tqdm

from data import make_loaders
from utils import load_checkpoint_with_preprocessor, load_config, set_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str)
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()
    return args


@torch.inference_mode()
def get_predictions(model, preprocessor, dataloader, device, alpha=0.5):
    y_true = []
    y_pred = []
    # test time augmentation
    for inputs, labels in tqdm(dataloader):
        y_true.append(labels)
        y_pred_ = 0
        for i in range(4):
            if i == 0:
                inputs_ = inputs
            elif i == 1:
                inputs_ = [flip(input_) for input_ in inputs]
            elif i == 2:
                inputs_ = [mirror(input_) for input_ in inputs]
            elif i == 3:
                inputs_ = [flip(mirror(input_)) for input_ in inputs]
            inputs_ = preprocessor(inputs_, return_tensors="pt").to(device)
            outputs = model(**inputs_)
            y_pred_ += outputs.logits.squeeze().cpu()
        y_pred.append(y_pred_ / 4)
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred) > alpha
    return y_true, y_pred


def main(args):
    set_seed()
    config = load_config(args.checkpoint)
    _, dataloader = make_loaders(config)
    model, preprocessor = load_checkpoint_with_preprocessor(args.checkpoint)
    _ = model.eval()
    device = torch.device(args.device or config.device)
    model = model.to(device)
    y_true, y_pred = get_predictions(model, preprocessor, dataloader, device)
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print(precision, recall, f1, mcc)


if __name__ == "__main__":
    args = get_args()
    main(args)
