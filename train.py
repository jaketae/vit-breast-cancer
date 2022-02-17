from config import Config
from trainer import Trainer
from utils import set_seed


def main():
    set_seed()
    config = Config()
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
