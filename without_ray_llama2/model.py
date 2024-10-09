### import the required libraries

import os
import math
import yaml
import torch
import pandas as pd
import tensorly as tl
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from functools import partial
from datasets import load_dataset
from argparse import ArgumentParser
from utils import get_tokenizer, load_dataset_
from transformers import AutoTokenizer
from tensorly.decomposition import tensor_train
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ttlora_wrapper import LoRATTLinearWrapper, get_tensor_shape, get_tt_rank
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
)
from train import CustomLightningModule


def train_with_ray(epochs, lr, model_name, dataset_name):

    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this notebook.")

    dataset = load_dataset_(path=dataset_name)

    # print(imdb_dataset)
    print(dataset)

    train_dataset, val_dataset = get_tokenizer(model_name, dataset)

    ### create train, validation and test dataloader that will be used during training, testing and validation. The dataloader specifies the number of rows in each batch and how many gpus to use
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=2, shuffle=True, num_workers=2
    )

    val_loader = DataLoader(dataset=val_dataset, batch_size=2, num_workers=2)

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    model.config.pad_token_id = model.config.eos_token_id

    ### make model parameters non-trainable
    for param in model.parameters():
        param.requires_grad = False

    ### print the model structure to see which layers needs to be replaced by loRATT
    print(model)

    tt_shape = [64, 16, 9, 64]
    tt_rank = get_tt_rank(5, tt_shape)

    loratt_alpha = 8
    loretta_dropout = 0.05
    loratt_query = True
    loretta_value = True
    layers = []

    assign_loretta = partial(LoRATTLinearWrapper, tt_rank=tt_rank, alpha=loratt_alpha)
    transformer_layer = model.roberta.encoder.layer

    for layer in transformer_layer:
        if loratt_query:
            layer.attention.self.query = assign_loretta(
                layer.attention.self.query, 0, tt_shape
            )
        if loretta_value:
            layer.attention.self.value = assign_loretta(
                layer.attention.self.value, 2, tt_shape
            )

    print(model)

    # Check if linear layers are frozen
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total number of trainable parameters:", count_parameters(model))
    lightning_model = CustomLightningModule(model, model_name, lr)

    early_atopping_callback = EarlyStopping(
        monitor="val_loss", patience=5, verbose=True, mode="min"
    )

    model_checkpoint_callback = ModelCheckpoint(
        save_top_k=1, mode="max", monitor="val_acc"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[early_atopping_callback, model_checkpoint_callback],
        accelerator="gpu",
        precision="16-mixed",
        devices=args.gpus,
        # logger=logger,
        log_every_n_steps=10,
    )

    import time

    start = time.time()

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")

    train_acc = trainer.test(
        lightning_model, dataloaders=train_loader, ckpt_path="best", verbose=False
    )
    val_acc = trainer.test(
        lightning_model, dataloaders=val_loader, ckpt_path="best", verbose=False
    )

    print(train_acc)

    print(val_acc[0]["accuracy"])  # because this is what loretta reports
    # print(test_loader)
    print(type(val_acc))
    train_params = count_parameters(model)


def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    config = load_config("without_ray_llama2/config.yml")
    model_name = config["model"]["name"]
    lr = float(config["parameters"]["lr"])
    epochs = int(config["parameters"]["epochs"])
    dataset_name = config["dataset"]["name"]

    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--data", type=str, default="sst2")
    args = parser.parse_args()
    train_with_ray(epochs, lr, model_name, dataset_name)
