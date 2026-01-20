import os
import json
import pickle
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import (
    DistilBertModel,
    DistilBertConfig,
    DistilBertTokenizer
)

warnings.filterwarnings("ignore")

# Global embedding dictionaries
embeddings_dict_patch = None
embeddings_dict_answ = None

perfect_context_folder = (
    ".../"
    "BiasesProject/DataSet/perfect_context/" 
)

# Load embedding dictionaries
def load_dictionary_patch():
    global embeddings_dict_patch
    with open(
        ".../"
        "BiasesProject/DataSet/parquet/qafinal/patch.json",
        "r"
    ) as f:
        embeddings_dict_patch = json.load(f)

def load_dictionary_answ():
    global embeddings_dict_answ
    with open(
        ".../"
        "BiasesProject/DataSet/parquet/qafinal/answ.json",
        "r"
    ) as f:
        embeddings_dict_answ = json.load(f)

load_dictionary_patch()
load_dictionary_answ()

# Embedding utilities
def embedding_patch(nested_classes):
    return [
        [embeddings_dict_patch.get(cls, -1) for cls in sublist]
        for sublist in nested_classes
    ]

def embedding_answ(class_strings):
    special_classes = [
        "beaches, dunes, sands",
        "land principally occupied by agriculture, with significant areas of natural vegetation"
    ]

    special_map = {k: embeddings_dict_answ[k] for k in special_classes}
    placeholders = {k: f"{{{{{i}}}}}" for i, k in enumerate(special_classes)}

    results = []

    for class_string in class_strings:
        for sc, ph in placeholders.items():
            class_string = class_string.replace(sc, ph)

        class_list = [c.strip() for c in class_string.split(",")]

        values = []
        for cls in class_list:
            if cls in placeholders.values():
                idx = list(placeholders.values()).index(cls)
                values.append(special_map[special_classes[idx]])
            else:
                values.append(embeddings_dict_answ.get(cls, None))

        results.append(values)

    return results

def convert_to_binary(input_list, num_classes):
    output = []
    for sublist in input_list:
        vec = [0] * num_classes
        for idx in sublist:
            vec[idx] = 1
        output.append(vec)
    return output

# Dataset preparation
def build_dataset(parquet_df):
    questions = parquet_df["question"].tolist()
    question_types = parquet_df["question_type"].tolist()

    perfect_contexts = [
        perfect_context_folder + str(img) + ".parquet"
        for img in parquet_df["img_name"]
    ]

    patch_labels = [
        sorted(set(np.concatenate(row["grid_coordinates"])))
        for _, row in parquet_df.iterrows()
    ]

    answ_labels = parquet_df["answer"].tolist()

    patch_emb = embedding_patch(patch_labels)
    answ_emb = embedding_answ(answ_labels)

    patch_bin = convert_to_binary(patch_emb, 16)
    answ_bin = convert_to_binary(answ_emb, 335)

    return questions, perfect_contexts, patch_bin, answ_bin, question_types

# VQA Dataset
class VQADataset(Dataset):
    def __init__(
        self,
        questions,
        perfect_contexts,
        labels_patch,
        labels_answ,
        question_types,
        tokenizer,
        max_length=1024
    ):
        self.questions = questions
        self.perfect_contexts = perfect_contexts
        self.labels_patch = labels_patch
        self.labels_answ = labels_answ
        self.question_types = question_types
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]

        df_context = pd.read_parquet(self.perfect_contexts[idx])
        flat_context = df_context.values.flatten()
        flat_context = list(map(str, flat_context))

        tokens = self.tokenizer(
            question,
            flat_context,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "labels_patch": torch.tensor(self.labels_patch[idx], dtype=torch.float),
            "labels_answ": torch.tensor(self.labels_answ[idx], dtype=torch.float),
            "questiontype": self.question_types[idx],
            "idx": idx
        }

# DistilBERT model with two heads
class CustomDistilBert(nn.Module):
    def __init__(self, traindB=True, num_classes_head_patch=16, num_classes_head_answ=335):
        super().__init__()

        model_path = "/gpfsdswork/dataset/HuggingFace_Models/distilbert-base-uncased"
        config = DistilBertConfig.from_pretrained(model_path)

        self.distilbert = DistilBertModel.from_pretrained(
            model_path,
            config=config
        )

        max_length = 1024
        self.distilbert.config.max_position_embeddings = max_length

        orig_pos_emb = self.distilbert.embeddings.position_embeddings.weight
        self.distilbert.embeddings.position_embeddings.weight = nn.Parameter(
            torch.cat([orig_pos_emb, orig_pos_emb])
        )

        hidden_size = self.distilbert.config.hidden_size

        self.classifier_patch = nn.Linear(hidden_size, num_classes_head_patch)
        self.classifier_answ = nn.Linear(hidden_size, num_classes_head_answ)

        self.dropout = nn.Dropout(0.3)

        for p in self.distilbert.parameters():
            p.requires_grad = traindB

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        pooled = torch.sum(hidden_states * mask, dim=1) / torch.sum(mask, dim=1)

        pooled = self.dropout(torch.tanh(pooled))

        logits_patch = torch.sigmoid(self.classifier_patch(pooled))
        logits_answ = torch.sigmoid(self.classifier_answ(pooled))

        return logits_patch, logits_answ

# Training / validation loops
def train_epoch(model, loader, optimizer, device, a0, b0):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_patch = batch["labels_patch"].to(device)
        labels_answ = batch["labels_answ"].to(device)

        out_patch, out_answ = model(input_ids, attention_mask)

        loss_patch = nn.BCELoss()(out_patch, labels_patch)
        loss_answ = nn.BCELoss()(out_answ, labels_answ)

        loss = a0 * loss_patch + b0 * loss_answ
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        del input_ids, attention_mask, labels_patch, labels_answ
        torch.cuda.empty_cache()

    return total_loss / len(loader)

def eval_epoch(model, loader, device, a0, b0):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_patch = batch["labels_patch"].to(device)
            labels_answ = batch["labels_answ"].to(device)

            out_patch, out_answ = model(input_ids, attention_mask)

            loss_patch = nn.BCELoss()(out_patch, labels_patch)
            loss_answ = nn.BCELoss()(out_answ, labels_answ)

            loss = a0 * loss_patch + b0 * loss_answ
            total_loss += loss.item()

            del input_ids, attention_mask, labels_patch, labels_answ
            torch.cuda.empty_cache()

    return total_loss / len(loader)

# Run training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizer.from_pretrained(
    "/gpfsdswork/dataset/HuggingFace_Models/distilbert-base-uncased",
    use_fast=True
)
tokenizer.model_max_length = 1024

train_df = pd.read_parquet(
    ".../"
    "BiasesProject/DataSet/parquet/qafinal/dataset_balanced_train.parquet"
).head(10)

val_df = pd.read_parquet(
    ".../"
    "BiasesProject/DataSet/parquet/qafinal/dataset_balanced_validation.parquet"
).head(10)

train_data = build_dataset(train_df)
val_data = build_dataset(val_df)

train_dataset = VQADataset(*train_data, tokenizer)
val_dataset = VQADataset(*val_data, tokenizer)

model = CustomDistilBert(
    traindB=True,
    num_classes_head_patch=16,
    num_classes_head_answ=335
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)

run_epochs = 30
batch_size = 3

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

a0, b0 = 0.0, 1.0

for epoch in range(run_epochs):
    print(f"\nEpoch {epoch + 1}/{run_epochs}")

    train_loss = train_epoch(model, train_loader, optimizer, device, a0, b0)
    val_loss = eval_epoch(model, val_loader, device, a0, b0)

    print(f"Train loss: {train_loss:.4f}")
    print(f"Val loss:   {val_loss:.4f}")
