import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchinfo import summary
from torchmetrics.functional import accuracy


class Model(LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.config = config
        self.model = model
        # logging
        self.train_acc_hist = {}  # epoch: train_acc

    def forward(self, x):
        return self.model(x)

    def evaluate(self, batch, stage=None):
        x, y = batch
        output = self.model(x)
        pred = torch.argmax(output, dim=1)
        loss = F.cross_entropy(output, y)
        acc = accuracy(pred, y)
        if stage:
            self.log_dict({f"{stage}_loss": loss, f"{stage}_acc": acc}, logger=True)
        return loss, acc

    def on_train_start(self):
        # log total params and trainable params
        model_summary = summary(self.model, (128, 3, 32, 32), verbose=0)
        self.log_dict(
            {
                "total_params": float(model_summary.total_params),
                "trainable_params": float(model_summary.trainable_params),
            },
            logger=True,
        )
        # log dataset sizes
        dm = self.trainer.datamodule
        self.log_dict(
            {
                "train_size": float(len(dm.train_dataset)),
                "test_size": float(len(dm.test_dataset)),
            },
            logger=True,
        )
        # initialize et0 (number of epochs to reach 0 training error)
        self.et0 = self.config.max_epochs + 1

    def training_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, stage="train")
        return {"loss": loss, "train_acc": acc}

    def training_epoch_end(self, outputs):
        # log epoch wise loss and accuracy
        loss = torch.stack([i["loss"] for i in outputs]).mean()
        acc = torch.stack([i["train_acc"] for i in outputs]).mean()
        self.log_dict({"avg_train_acc": acc, "avg_train_loss": loss}, logger=True)
        # log train acc for all epochs
        self.latest_train_acc = acc.item()
        self.train_acc_hist[self.current_epoch + 1] = acc.item()
        # log model norm
        self.log_dict(self.norm(self.model), logger=True)

    def validation_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, stage="val")
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        # log epoch wise loss and accuracy
        loss = torch.stack([i["val_loss"] for i in outputs]).mean()
        acc = torch.stack([i["val_acc"] for i in outputs]).mean()
        self.log_dict({"avg_val_acc": acc, "avg_val_loss": loss}, logger=True)
        # calculate generalization gap
        try:
            gap = self.latest_train_acc - acc.item()
            self.log("gap", gap, logger=True)
        except:
            pass

    def test_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, stage="test")
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        # log loss and accuracy
        loss = torch.stack([i["test_loss"] for i in outputs]).mean()
        acc = torch.stack([i["test_acc"] for i in outputs]).mean()
        self.log_dict({"avg_test_acc": acc, "avg_test_loss": loss}, logger=True)
        # log et0
        for epoch, acc in self.train_acc_hist.items():
            if acc >= 1.0:
                self.et0 = epoch
                break
        self.log("et0", float(self.et0), logger=True)

    def configure_optimizers(self):
        """
        If lr_decay_interval is set, then use inverse square root decay.
        """
        opt = optim.SGD(
            self.model.parameters(), lr=self.config.lr, momentum=self.config.momentum
        )
        if self.config.lr_decay_interval is None:
            return opt
        sch = optim.lr_scheduler.StepLR(
            opt, step_size=self.config.lr_decay_interval, gamma=0.2
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "epoch", "frequency": 1},
        }

    @torch.no_grad()
    def norm(self, model):
        """
        Calculate the norm for each layer and the whole model. The norm of the whole
        model is calculated by flattening all weight matrices and take the L2 norm of
        the vector.
        """
        norms = {}
        for name, param in model.named_parameters():
            norms["norm_" + name] = param.data.norm(p=2)
        model_norm = torch.cat(
            [param.data.flatten() for param in model.parameters()]
        ).norm(p=2)
        norms["model_norm"] = model_norm
        return norms
