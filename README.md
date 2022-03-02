# Visualizing Transformers for Breast Histopathology

This repository contains code for [Visualizing Transformers for Breast Histopathology](https://drive.google.com/file/d/17HaJxCmchwcg4xLCqWnTUcLywWJlF4tj/view?usp=sharing). This work was completed as part of CPSC 482: Current Topics in Applied Machine Learning. 

## Abstract

> Transfer learning is a common way of achieving high performance on downstream tasks with limited data. Simultaneously, the success of vision transformers has opened up a wider range of image model options than previously available. In this report, we explore the application of transfer learning in the context of breast histopathology using state-of-the-art vision transformer models: ViT, BeiT, and CaiT. We focus on ways of presenting model prediction and behavior in human-interpretable ways, such that a pathologist could leverage this information to aid with their diagnosis. Through experiments, we show how attention maps and latent representations can be used to interpret model behavior.

## Quickstart

1. Clone the repository.

```
$ git clone https://github.com/jaketae/vit-breast-cancer.git
```

2. Create a Python virtual enviroment and install package requirements.

```
$ cd vit-breast-cancer
$ python -m venv venv
$ pip install -r requirements.txt
```

3. To train a model, run `python train.py`; for evaluation, `python evaluate.py` with appropriate flags. For instance, 

```
$ CUDA_VISIBLE_DEVICES=1 python evaluate.py --device cuda --checkpoint checkpoints/vit_freeze
```

## Dataset

We used the [Breast Histopathology Images dataset](https://www.kaggle.com/paultimothymooney/breast-histopathology-images). You can either download the dataset directly from the website, or use Kaggle's Python API to download it via the command line. For detailed instructions on how to use the Kaggle API, refer to the [documentation](https://www.kaggle.com/docs/api).

```
$ kaggle datasets download paultimothymooney/breast-histopathology-images
```

Create a subfolder within the directory, such as `raw`, then unzip the dataset via

```
$ unzip breast-histopathology-images.zip -d raw
```

## Training

To evaluate a model checkpoint, run [`train.py`](train.py). The full list of supported arguments are shown below.


```
$ python train.py -h
usage: train.py [-h] [--name NAME] [--device DEVICE] [--log_path LOG_PATH] [--data_path DATA_PATH] [--save_path SAVE_PATH]
                [--model MODEL] [--freeze FREEZE] [--epochs EPOCHS] [--lr LR] [--classifier_lr CLASSIFIER_LR] [--split SPLIT]
                [--threshold THRESHOLD] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]

optional arguments:
  -h, --help            show this help message and exit
  --name NAME
  --device DEVICE
  --log_path LOG_PATH
  --data_path DATA_PATH
  --save_path SAVE_PATH
  --model MODEL
  --freeze FREEZE
  --epochs EPOCHS
  --lr LR
  --classifier_lr CLASSIFIER_LR
  --split SPLIT
  --threshold THRESHOLD
  --batch_size BATCH_SIZE
  --num_workers NUM_WORKERS
```

Default configurations are specified in [`config.py`](config.py).

Running this command will create a folder under `checkpoints` and `logs` according to the name field specified in the configuration file. `checkpoints` will contain model weights, and `logs` will contain tensorboard logs for model training inspection.

## Evaluation

To evaluate a model checkpoint, run [`evaluate.py`](evaluate.py). The full list of supported arguments are shown below.

```
$ python evaluate.py -h
usage: evaluate.py [-h] [--device DEVICE] [--checkpoint CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE
  --checkpoint CHECKPOINT
```

For example, assuming a `vit_freeze` checkpoint and a CUDA-enabled local machine, run

```
$ CUDA_VISIBLE_DEVICES=1 python evaluate.py --device cuda --checkpoint checkpoints/vit_freeze
```

If CUDA is not available, you can also set the `device` flag to `cpu`.

## References

- [Hugging Face `transformers`](https://github.com/huggingface/transformers)
- [`timm`](https://github.com/rwightman/pytorch-image-models)
- [DINO](https://github.com/facebookresearch/dino)
- [jeonsworld's ViT notebook](https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb)


## License

Released under the [MIT License](LICENSE).