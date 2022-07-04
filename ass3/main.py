# COMP90042 Assignment3
# Authors Yanbei Jiang Student ID 1087029
# This file is entrypoint of the project, includes trainer initialisation, train and test phase
import yaml
from model.rumour_detector import RumourDetector
from pytorch_lightning import Trainer, seed_everything, callbacks
from model.tweet_data_module import TweetDataModule
from pytorch_lightning.loggers import WandbLogger
# import pickle


def train(model, configs, dm, wandb_logger):
    dm.setup("fit")
    seed_everything(configs["model"]["seed"])
    checkpoint_callback = callbacks.ModelCheckpoint(monitor="f1_score",
                                                    mode="max",
                                                    save_top_k=1,
                                                    dirpath="./checkpoints",
                                                    filename='{epoch}-{f1_score:.2f}')

    early_stopping_callback = callbacks.EarlyStopping(monitor="f1_score",
                                                      mode="max",
                                                      strict=True)
    print("Init Trainer")
    trainer = Trainer(max_epochs=int(configs["model"]["max_epochs"]),
                      gpus=configs["device"]["avail_gpu"],
                      accelerator="gpu",
                      precision=int(configs["model"]["precision"]),
                      callbacks=[checkpoint_callback, early_stopping_callback],
                      gradient_clip_val=1,
                      logger=wandb_logger,
                      log_every_n_steps=int(configs["model"]["log_every_n_steps"]))
    print("Train Start!!")
    print("-"*50)
    trainer.fit(model, datamodule=dm)
    print("Train End!!")

    return trainer


def main():
    print("Rumour Detector Start!!")
    print("-"*50)
    with open("model/configs.yaml", 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    dm = TweetDataModule(model_name=configs["model"]["name"],
                         max_length=int(configs["model"]["max_length"]),
                         train_batch_size=int(configs["model"]["train_batch_size"]),
                         num_workers=int(configs["device"]["num_workers"]))

    model = RumourDetector(
        model_name=configs["model"]["name"],
        train_batch_size=int(configs["model"]["train_batch_size"]),
        warmup_ratio=float(configs["model"]["warmup_ratio"]),
        learning_rate=float(configs["model"]["learning_rate"]),
        weight_decay=float(configs["model"]["weight_decay"]),
        adam_epsilon=float(configs["model"]["adam_epsilon"]))

    wandb_logger = WandbLogger(project="COMP90042 Assignment 3", name="roberta")
    trainer = train(model, configs, dm, wandb_logger)

    print("Test Start!!")
    print("-"*50)
    dm.setup("test")
    trainer.test(model, ckpt_path="best", datamodule=dm)
    print("Test End!!")
    print("\n")
    print("Rumour Detector End!!")


if __name__ == "__main__":
    main()
