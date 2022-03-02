import argparse
from dataclasses import fields

from config import Config
from trainer import Trainer
from utils import set_seed


def get_args():
    parser = argparse.ArgumentParser()
    for field in fields(Config):
        name = field.name
        default = getattr(Config, name)
        parser.add_argument(f"--{name}", default=default, type=type(default))
    args = parser.parse_args()
    return args


def main():
    set_seed()
    args = get_args()
    kwargs = {}
    for field in fields(Config):
        name = field.name
        kwargs[name] = getattr(args, name)
    config = Config(**kwargs)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
