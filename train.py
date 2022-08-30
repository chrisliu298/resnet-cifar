import argparse

import wandb
from easydict import EasyDict as edict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import CIFAR10DataModule, CIFAR100DataModule
from model import Model
from models import *

DATASETS = {"cifar10": CIFAR10DataModule, "cifar100": CIFAR100DataModule}
MODELS = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "preact_resnet18": PreActResNet18,
    "preact_resnet34": PreActResNet34,
    "preact_resnet50": PreActResNet50,
    "preact_resnet101": PreActResNet101,
    "preact_resnet152": PreActResNet152,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str, default="resnet-cifar10")
    # model
    parser.add_argument("--model", type=str, default="resnet18", choices=MODELS.keys())
    # data
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=DATASETS.keys()
    )
    parser.add_argument("--aug", action="store_true")
    # training
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_decay_interval", type=int)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--fp16", action="store_true")
    # experiment
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    config = edict(vars(parser.parse_args()))
    return config


def main():
    config = parse_args()
    seed_everything(config.seed)
    dm = DATASETS[config.dataset](config)
    model = Model(MODELS[config.model](), config)
    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.append(
        ModelCheckpoint(
            filename="{epoch}_{avg_train_acc:.4f}_{avg_val_acc:.4f}",
            monitor="epoch",
            save_top_k=5,
            mode="max",
        ),
    )
    logger = WandbLogger(
        offline=not config.wandb,
        project=config.project_id,
        entity="chrisliu298",
        config=config,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        callbacks=callbacks,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=1,
        benchmark=True,
        logger=logger,
        enable_progress_bar=False,
        precision=16 if config.fp16 else 32,
    )
    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm, verbose=False)
    wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
