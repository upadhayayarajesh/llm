### import the required libraries

import os
from datasets import load_dataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F  # Add this import
import math
import tensorly as tl
from tensorly.decomposition import tensor_train
from transformers import AutoTokenizer

### import the required libraries

import os
from datasets import load_dataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F  # Add this import
import math
import tensorly as tl
from tensorly.decomposition import tensor_train
from transformers import AutoTokenizer

import pickle

from argparse import ArgumentParser


from utils import get_tokenizer, load_dataset_
from ttlora_wrapper import LoRATTLinearWrapper, get_tensor_shape, get_tt_rank
from transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering
from train import CustomLightningModule
from functools import partial

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray.air import RunConfig, session
from ray import train



def train_with_ray():


    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this notebook.")


    dataset = load_dataset_()

    # print(imdb_dataset)
    print(dataset)

    train_dataset, val_dataset = get_tokenizer("/lustre/vescratch1/ceodspspectrum/llama3_70b/checkpoints", dataset)


    ### create train, validation and test dataloader that will be used during training, testing and validation. The dataloader specifies the number of rows in each batch and how many gpus to use
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=2,
        num_workers=2
    )



 
    model = AutoModelForQuestionAnswering.from_pretrained("/lustre/vescratch1/ceodspspectrum/llama3_70b/checkpoints")


    model.config.pad_token_id = model.config.eos_token_id

    ### make model parameters non-trainable
    for param in model.parameters():
        param.requires_grad = False

    ### print the model structure to see which layers needs to be replaced by loRATT
    print(model)

    


    tt_shape = [16, 16, 16, 16, 16, 16]
    tt_rank = get_tt_rank(5, tt_shape)


    loratt_alpha=8
    loretta_dropout = 0.05
    loratt_query = True
    loretta_value = True

    layers = []

    # assign_lora = partial(LoRALinearWrapper, rank=lora_r, alpha=lora_alpha)

    assign_loretta = partial(LoRATTLinearWrapper, tt_rank=tt_rank, alpha=loratt_alpha)

    for layer in model.transformer.layers:
        if loratt_query:
            layer.self_attn.q_proj = assign_loretta(layer.self_attn.q_proj, 0, tt_shape)
        if loretta_value:
            layer.self_attn.v_proj = assign_loretta(layer.self_attn.v_proj,2, tt_shape)
   
    print(model)

    # Check if linear layers are frozen
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print("Total number of trainable parameters:", count_parameters(model))


    lightning_model = CustomLightningModule(model,"/usr/projects/unsupgan/afia/llama2_7b_hf/checkpoints",5e-5)


    # callbacks = [
    #     ModelCheckpoint(
    #         save_top_k=1, mode="max", monitor="val_acc"
    #     )  # save top 1 model
    # ]
    early_atopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
        )

    model_checkpoint_callback=ModelCheckpoint(
        save_top_k=1, mode="max", monitor="val_acc"
    )  

    # name="my-model" + str(args.ranks)
    # logger = CSVLogger(save_dir="logs/", name=name)


    trainer = pl.Trainer(
        max_epochs=30,
        callbacks=[early_atopping_callback, model_checkpoint_callback],
        accelerator="gpu",
        precision="16-mixed",
        devices=args.gpus,
        # logger=logger,
        log_every_n_steps=10,
    )



    import time
    start = time.time()

    trainer.fit(model=lightning_model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")

    train_acc = trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best", verbose=False)
    val_acc=trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best", verbose=False)

    # print(train_acc)

    print(val_acc[0]['accuracy']) #because this is what loretta reports
    # print(test_loader)
    print(type(val_acc))
    train_params=count_parameters(model)

    # log_file_name = args.data + '_' + 'log_file'

    # with open(log_file_name, 'a') as f:
    #     f.write(f'Accuracy for Tensor Dimension {args.shape} , rank {args.ranks} is: {val_acc}\n')
    #     f.write(f'Number of trainable parameters for Tensor Dimension {args.shape} , rank {args.ranks} is: {train_params}\n')
    # session.report({"val_acc": val_acc[0]['accuracy'], "trainable_params": train_params})

# def main():
#     config = {
#     # "data": tune.grid_search(["mnli", "sst2", "mrpc", "cola", "qnli", "qqp", "rte", "stsb"]),
#     # "shapes": tune.choice([[12, 8, 8, 3, 8, 8, 12], [12, 8, 8, 24, 8, 12],[6, 12, 16, 16, 96], [4, 6, 6, 8, 8, 8, 24], [4, 4, 9, 12, 32, 32], [3, 4, 4, 4, 6, 32, 48]]),
#     "shapes": tune.choice([[128, 32, 32, 128], [16, 16, 16, 16, 16, 16], [16, 16, 16, 4, 4, 16, 16], [16, 16, 4, 4, 4, 4, 16, 16], [16, 8, 4, 4, 2, 2, 4, 4, 8, 16], [16, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 16]]),
#     "ranks": tune.choice([5, 8, 10, 12, 16]),
#     "alpha": tune.choice([1, 2, 4, 8, 10, 12, 16, 32]),
#     # "batch_size": tune.choice([8, 16, 32]),
#     "learning_rate": tune.choice([1e-5, 1e-4, 5e-5, 5e-4]),
# }

#     scheduler = ASHAScheduler(
#         # metric = "val_acc",
#         # mode = "max",
#         max_t = 100,
#         grace_period = 40
#     )

#     reporter = CLIReporter(
#         parameter_columns = ["shapes", "ranks", "alpha", "learning_rate"],
#         metric_columns = ["val_acc", "trainable_params"]
#     )

#     analysis = tune.run(
#         train_with_ray ,
#         resources_per_trial={"cpu": 2, "gpu": args.gpus},
#         config=config,
#         num_samples=2,  #how many different combinations to try
#         scheduler=scheduler,
#         progress_reporter=reporter,
#         # local_dir="./ray_results",
#         # keep_checkpoints_num=4,
#         # checkpoint_freq = 10,
#         # checkpoint_at_end =True,
#         # run_config = RunConfig(storage_path="./results", name = "tune_deberta_sst2")
#         # name="tune_deberta_sst2"
#         metric = "val_acc",
#         mode="max",
#         storage_path = "/usr/projects/unsupgan/afia/ray_tune_deberta",
#         name="tune_deberta",
#         fail_fast = "raise"
#     )


#     #save result of all tasks
#     df = analysis.results_df
#     filename= f"{args.data}_ray_tune_results_deberta.cvs"
#     df.to_csv(filename, index=False)

#     print("Best hyperparameters found were: ", analysis.best_config)

#     #save the best hyperparameters
#     filename_best= f"{args.data}_best_hyper_deberta.txt"
#     with open(filename_best,"w") as f:
#         f.write(str(analysis.best_config))

if __name__=="__main__":
    parser = ArgumentParser()
    # parser.add_argument("--ranks", type=int, default=5)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--data", type=str, default = 'sst2')
    # parser.add_argument("--shape", type=int, default=7)
    # parser.add_argument("--bs", type=int, default=128)
    args = parser.parse_args()
    # main()
    train_with_ray()

