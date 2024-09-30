import os
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F  # Add this import
import math


def load_dataset_():

    dataset = load_dataset("/usr/projects/unsupgan/afia/squad")
    return dataset


def get_tokenizer(model_path, datasets):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

    print("Tokenizer input max length:", tokenizer.model_max_length)
    print("Tokenizer vocabulary size:", tokenizer.vocab_size)

    ### set the special tokens needed during tokenization
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    max_length = tokenizer.model_max_length

    def preprocess_training_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=8000,
            truncation="only_second",
            #stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )


        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            # with open("debug.txt","w") as f:
            #     f.write(str(sequence_ids))
    
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            # with open("debug.txt","a") as f:
            #     f.write(str(idx))

            while idx< len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx-1)

                idx = context_end
                while idx >= context_start and offset[idx][1] > end_char:
                    idx -= 1
                end_positions.append(idx + 1)


        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
       
    train_dataset = datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=datasets["train"].column_names,
    )

    train_dataset.set_format("torch")

    val_dataset = datasets["validation"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=datasets["validation"].column_names,
    )
    val_dataset.set_format("torch")
    return train_dataset, val_dataset
