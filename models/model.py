from torch import nn
from transformers import AutoModelForImageClassification, AutoFeatureExtractor


from .cait import CaiTWrapper


def get_model_with_preprocessor(model_name, device="cpu"):
    if "cait" in model_name:
        # cait uses vit defaults
        preprocessor = AutoFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        model = CaiTWrapper()
    else:
        preprocessor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        in_features = model.classifier.in_features
        # replace classifier with linear
        model.classifier = nn.Linear(in_features, 1)
    # set backbone
    if hasattr(model, "vit"):
        model.backbone = model.vit
    elif hasattr(model, "beit"):
        model.backbone = model.beit
    elif hasattr(model, "swin"):
        model.backbone = model.swin
    model = model.to(device)
    return model, preprocessor
