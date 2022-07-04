# COMP90042 Assignment3
# Authors Yanbei Jiang Student ID 1087029
# This file defines the datamodule in the pytorch-lightning framework which will be used in each batch

from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from model.tweet_dataset import TweetDataset
from torch.utils.data import DataLoader


class TweetDataModule(LightningDataModule):
    def __init__(
        self,
        model_name,
        max_length,
        train_batch_size,
        num_workers
    ):
        super().__init__()
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.max_length = max_length
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def setup(self, stage="fit"):
        if stage == 'fit' or stage is None:
            self.train_dataset = TweetDataset("train", self.max_length, self.tokenizer)
            self.val_dataset = TweetDataset("dev", self.max_length, self.tokenizer)
        if stage == 'test' or stage is None:
            self.test_dataset = TweetDataset("test", self.max_length, self.tokenizer)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, collate_fn=self.train_dataset.collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_batch_size, collate_fn=self.val_dataset.collate_fn, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.train_batch_size, collate_fn=self.test_dataset.collate_fn, num_workers=self.num_workers)

