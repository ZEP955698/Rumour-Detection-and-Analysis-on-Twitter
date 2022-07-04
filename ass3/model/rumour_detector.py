# COMP90042 Assignment3
# Authors Yanbei Jiang Student ID 1087029
# This file includes all the essential methods such as training/dev/test step, optimizer
# as a part of pytorch-lightning model 

import pytorch_lightning as pl
import torch
from transformers import (
    AdamW,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from model.my_model import *
import torch.nn.functional as F


class RumourDetector(pl.LightningModule):
    def __init__(self,
                 model_name,
                 weight_decay,
                 learning_rate,
                 adam_epsilon,
                 warmup_ratio,
                 train_batch_size,
                 **kwargs,
                 ):
        super().__init__()
        self.total_steps = None
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.config = AutoConfig.from_pretrained(model_name)
        # self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
        self.model = MyModel(model_name)
        self.metric = "f1_score"

    def training_step(self, batch, batch_idx):
        seq = batch["input_ids"]
        attn_masks = batch["attention_mask"]
        labels = batch["labels"]
        logits = self.model(seq, attn_masks)
        loss = F.cross_entropy(logits, labels)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        if batch_idx % 10 == 0:
            self.logger.log_metrics({"train_loss": loss.item()})
        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # The following is from pytorch-lightning official docs
        # For more details,
        # See https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
        seq = batch["input_ids"]
        attn_masks = batch["attention_mask"]
        logits = self.model(seq, attn_masks)
        preds = torch.argmax(logits, dim=1).tolist()
        labels = batch["labels"]
        loss = F.cross_entropy(logits, labels)
        self.logger.log_metrics({"val_loss": loss.item()})
        return {"preds": preds, "labels": labels.tolist()}

    def test_step(self, batch, batch_idx):
        # The following is from pytorch-lightning official docs
        # For more details,
        # See https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
        seq = batch["input_ids"]
        attn_masks = batch["attention_mask"]
        logits = self.model(seq, attn_masks)
        preds = torch.argmax(logits, dim=1).tolist()
        return {"preds": preds}

    def validation_epoch_end(self, outputs):
        preds = []
        labels = []
        for instance in outputs:
            for pred, label in zip(instance["preds"], instance["labels"]):
                preds.append(pred)
                labels.append(label)
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

        self.log(self.metric, f1, logger=False)
        self.logger.log_metrics({"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1})

    def test_epoch_end(self, outputs):
        preds = [pred for instance in outputs for pred in instance["preds"]]

        with open("./submissions.csv", "w") as f:
            f.write("Id,Predicted\n")
            for idx, pred in enumerate(preds):
                f.write(str(idx) + "," + str(pred) + "\n")
        # Please uncomment this if doing part2 COVID Rumour Analysis
        # with open("./covid_label.txt", "w") as f:
        #     for instance in outputs:
        #         for pred in instance["preds"]:
        #             f.writelines(pred)

    def setup(self, stage=None) -> None:
        # The following is from pytorch-lightning official docs
        # For more details,
        # See https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches
        self.total_steps = ((len(train_loader.dataset) // tb_size) // ab_size) * float(self.trainer.max_epochs)

    def configure_optimizers(self):
        # The following is from pytorch-lightning official docs
        # For more details,
        # See https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.hparams.warmup_ratio * self.total_steps),
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
