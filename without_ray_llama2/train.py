import time

import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F

from sklearn.metrics import f1_score
from collections import Counter
from transformers import AutoTokenizer




def compute_f1_score(predicted_ans, true_ans):
    predicted_tokens = predicted_ans.split()
    true_tokens = true_ans.split()
    common_tokens = Counter(predicted_tokens) & Counter(true_tokens)
    num_same = sum(common_tokens.values())

    if num_same == 0:
        return 0

    precision = num_same / len(predicted_tokens)
    recall = num_same / len(true_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1




class CustomLightningModule(pl.LightningModule):
    def __init__(self, model, model_path, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        

        # self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        # self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, input_ids, attention_mask, start_positions, end_positions):
        return self.model(input_ids, attention_mask=attention_mask, start_positions=torch.tensor(start_positions), end_positions= torch.tensor(end_positions))

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       start_positions=batch["start_positions"], end_positions= batch["end_positions"])
        self.log("train_loss", outputs["loss"])
        return outputs["loss"]  # this is passed to the optimizer for training

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       start_positions=batch["start_positions"], end_positions= batch["end_positions"])
        self.log("val_loss", outputs["loss"], prog_bar=True)

        # logits = outputs["logits"]
        # predicted_labels = torch.argmax(logits, 1)
        # self.val_acc(predicted_labels, batch["label"])
        # self.log("val_acc", self.val_acc, prog_bar=True)
        with open("debug.txt","a") as f:
            f.write(" So validation output is good")

        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        start_indices = torch.argmax(start_logits, dim = -1)
        end_indices = torch.argmax(end_logits, dim = -1)

        answers = []
        predictions = []

        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            input_ids = batch['input_ids'][i]
            if start<=end: #ensure valid index ordering
                pred_answer_toekns = input_ids[start:end+1]
                pred_answer = self.tokenizer.decode(pred_answer_toekns, skip_special_tokens = True)
            else:
                pred_answer = ""
            predictions.append(pred_answer)
            true_answer = self.tokenizer.decode(input_ids[batch['start_positions'][i]:batch['end_positions'][i]], skip_special_tokens=True)
            answers.append(true_answer)

        #compute F1 score
        f1_scores = [compute_f1_score(pred,true) for pred, true in zip(predictions, answers)]
        avg_f1_score = sum(f1_scores)/ len(f1_scores)
        self.log("val_acc", avg_f1_score, prog_bar=True)

    def test_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                       start_positions=batch["start_positions"], end_positions= batch["end_positions"])

        # logits = outputs["logits"]
        # predicted_labels = torch.argmax(logits, 1)
        # self.test_acc(predicted_labels, batch["label"])
        # self.log("accuracy", self.test_acc, prog_bar=True)
        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        start_indices = torch.argmax(start_logits, dim = -1)
        end_indices = torch.argmax(end_logits, dim = -1)

        answers = []
        predictions = []

        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            input_ids = batch['input_ids'][i]
            if start<=end: #ensure valid index ordering
                pred_answer_toekns = input_ids[start:end+1]
                pred_answer = self.tokenizer.decode(pred_answer_toekns, skip_special_tokens = True)
            else:
                pred_answer = ""
            predictions.append(pred_answer)
            true_answer = self.tokenizer.decode(input_ids[batch['start_positions'][i]:batch['end_positions'][i]], skip_special_tokens=True)
            answers.append(true_answer)

        #compute F1 score
        f1_scores = [compute_f1_score(pred,true) for pred, true in zip(predictions, answers)]
        avg_f1_score = sum(f1_scores)/ len(f1_scores)
        self.log("accuracy", avg_f1_score, prog_bar=True)


                

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
