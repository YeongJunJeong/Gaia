"""Gaia v3: 2-step training with EMP + Naylor + BonaRes + MGnify(current)"""

import pickle
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from transformers import GPT2LMHeadModel, TrainingArguments, Trainer
from torch.utils.data import Dataset, random_split


class SoilCorpus(Dataset):
    def __init__(self, dataframes, tokenizer, max_len=512):
        self.samples = []
        bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
        for df, genus_cols, prefix in dataframes:
            for _, row in df.iterrows():
                nonzero = row[genus_cols][row[genus_cols] > 0].sort_values(ascending=False)
                tokens = [bos]
                for genus in nonzero.index:
                    tid = tokenizer.vocab.get(f"{prefix}{genus}")
                    if tid is not None:
                        tokens.append(tid)
                    if len(tokens) >= max_len - 1:
                        break
                tokens.append(eos)
                while len(tokens) < max_len:
                    tokens.append(pad)
                if sum(1 for t in tokens if t not in [bos, eos, pad]) >= 5:
                    self.samples.append(torch.tensor(tokens[:max_len], dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {"input_ids": self.samples[idx], "labels": self.samples[idx].clone()}


# 1. Load model + tokenizer
print("=== Loading model ===")
model = GPT2LMHeadModel.from_pretrained("checkpoints/gaia_expanded/best")
with open("checkpoints/gaia_expanded/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Check for new genera in EMP that aren't in vocab
emp_files = sorted(Path("data/raw/emp").glob("emp_soil_genus_*.csv"))
if emp_files:
    emp = pd.read_csv(emp_files[-1])
    emp_genus = [c for c in emp.columns if c != "sample_id"]
    new_emp = [g for g in emp_genus if f"g__{g}" not in tokenizer.vocab]
    if new_emp:
        print(f"Adding {len(new_emp)} new EMP genera to vocab")
        tokenizer.add_tokens([f"g__{g}" for g in new_emp])
        model.resize_token_embeddings(len(tokenizer.vocab))

print(f"Vocab: {len(tokenizer.vocab)}, Params: {sum(p.numel() for p in model.parameters()):,}")

# 2. Load all data
print("\n=== Loading all data ===")
all_data = []

# MGnify (current)
mgnify = pd.read_csv("data/raw/mgnify/mgnify_abundance.csv")
mgnify_g = [c for c in mgnify.columns if c not in ["sample_id", "analysis_id"]]
all_data.append((mgnify, mgnify_g, "g__"))
print(f"MGnify: {len(mgnify)} samples, {len(mgnify_g)} genera")

# EMP (latest file)
if emp_files:
    all_data.append((emp, emp_genus, "g__"))
    print(f"EMP: {len(emp)} samples, {len(emp_genus)} genera")

# Naylor
naylor = pd.read_csv("data/raw/naylor/naylor_genus_with_labels.csv")
naylor_g = [c for c in naylor.columns if c not in ["sample_id", "run_id", "treatment", "host"]]
all_data.append((naylor, naylor_g, "g__"))
print(f"Naylor: {len(naylor)} samples, {len(naylor_g)} genera")

# BonaRes
bac = pd.read_csv("data/raw/longterm/bonares_data/lte_westerfeld.V1_0_BACTERIA.csv", low_memory=False)
genus_ref = pd.read_csv("data/raw/longterm/bonares_data/lte_westerfeld.V1_0_GENUS.csv")
genus_map = dict(zip(genus_ref["Genus_ID"], genus_ref["Name"]))
bac["Genus_Name"] = bac["Genus_ID"].map(genus_map)
grouped = bac.groupby(["Plot_ID", "Experimental_Year", "Genus_Name"])["Value"].sum().reset_index()
bonares = grouped.pivot_table(index=["Plot_ID", "Experimental_Year"], columns="Genus_Name", values="Value", fill_value=0).reset_index()
bonares_g = [c for c in bonares.columns if c not in ["Plot_ID", "Experimental_Year"]]
all_data.append((bonares, bonares_g, "g__"))
print(f"BonaRes: {len(bonares)} samples, {len(bonares_g)} genera")

# Build corpus
corpus = SoilCorpus(all_data, tokenizer)
total = sum(len(df) for df, _, _ in all_data)
print(f"\nTotal input: {total} samples")
print(f"Tokenized corpus: {len(corpus)} samples")

train_size = int(0.9 * len(corpus))
train_set, val_set = random_split(corpus, [train_size, len(corpus) - train_size], generator=torch.Generator().manual_seed(42))

# 3. Step 1: New embeddings only
print("\n=== Step 1: Embeddings only (15 epochs) ===")
for param in model.parameters():
    param.requires_grad = False
model.transformer.wte.weight.requires_grad = True
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

args1 = TrainingArguments(
    output_dir="checkpoints/gaia_v3/step1",
    num_train_epochs=15,
    per_device_train_batch_size=16,
    learning_rate=1e-3,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    lr_scheduler_type="cosine",
    metric_for_best_model="eval_loss",
)

Trainer(model=model, args=args1, train_dataset=train_set, eval_dataset=val_set).train()
step1_loss = Trainer(model=model, args=args1, train_dataset=train_set, eval_dataset=val_set).evaluate()["eval_loss"]
print(f"Step 1 eval loss: {step1_loss:.4f}")

# 4. Step 2: Full fine-tune
print("\n=== Step 2: Full fine-tune (10 epochs) ===")
for param in model.parameters():
    param.requires_grad = True
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

args2 = TrainingArguments(
    output_dir="checkpoints/gaia_v3/step2",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    lr_scheduler_type="cosine",
    metric_for_best_model="eval_loss",
)

trainer2 = Trainer(model=model, args=args2, train_dataset=train_set, eval_dataset=val_set)
trainer2.train()
final_loss = trainer2.evaluate()["eval_loss"]
print(f"Step 2 eval loss: {final_loss:.4f}")

# Save
trainer2.save_model("checkpoints/gaia_v3/best")
with open("checkpoints/gaia_v3/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print(f"\nDone! Final eval loss: {final_loss:.4f}")
print(f"Model saved to checkpoints/gaia_v3/best")
print(f"Vocab: {len(tokenizer.vocab)}")
