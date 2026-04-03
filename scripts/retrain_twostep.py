"""2-step training: 1) new embeddings only, 2) full fine-tune"""

import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, TrainingArguments, Trainer
from torch.utils.data import Dataset, random_split

# Load expanded model + tokenizer
model = GPT2LMHeadModel.from_pretrained("checkpoints/gaia_expanded/best")
with open("checkpoints/gaia_expanded/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print(f"Model: vocab={len(tokenizer.vocab)}, params={sum(p.numel() for p in model.parameters()):,}")

# Prepare all data
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


# Load datasets
mgnify = pd.read_csv("data/raw/mgnify/mgnify_abundance.csv")
mgnify_g = [c for c in mgnify.columns if c not in ["sample_id", "analysis_id"]]

naylor = pd.read_csv("data/raw/naylor/naylor_genus_with_labels.csv")
naylor_g = [c for c in naylor.columns if c not in ["sample_id", "run_id", "treatment", "host"]]

bac = pd.read_csv("data/raw/longterm/bonares_data/lte_westerfeld.V1_0_BACTERIA.csv", low_memory=False)
genus_ref = pd.read_csv("data/raw/longterm/bonares_data/lte_westerfeld.V1_0_GENUS.csv")
genus_map = dict(zip(genus_ref["Genus_ID"], genus_ref["Name"]))
bac["Genus_Name"] = bac["Genus_ID"].map(genus_map)
grouped = bac.groupby(["Plot_ID", "Experimental_Year", "Genus_Name"])["Value"].sum().reset_index()
bonares = grouped.pivot_table(index=["Plot_ID", "Experimental_Year"], columns="Genus_Name", values="Value", fill_value=0).reset_index()
bonares_g = [c for c in bonares.columns if c not in ["Plot_ID", "Experimental_Year"]]

all_data = [
    (mgnify, mgnify_g, "g__"),
    (naylor, naylor_g, "g__"),
    (bonares, bonares_g, "g__"),
]

corpus = SoilCorpus(all_data, tokenizer)
print(f"Total corpus: {len(corpus)} samples")

train_size = int(0.9 * len(corpus))
train_set, val_set = random_split(corpus, [train_size, len(corpus) - train_size], generator=torch.Generator().manual_seed(42))

# === STEP 1: Train new embeddings only (freeze everything else) ===
print("\n=== Step 1: New embeddings only (15 epochs) ===")

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the embedding layer
model.transformer.wte.weight.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params (step 1): {trainable:,}")

args1 = TrainingArguments(
    output_dir="checkpoints/gaia_v2/step1",
    num_train_epochs=15,
    per_device_train_batch_size=16,
    learning_rate=1e-3,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=200,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    lr_scheduler_type="cosine",
    metric_for_best_model="eval_loss",
)

trainer1 = Trainer(model=model, args=args1, train_dataset=train_set, eval_dataset=val_set)
trainer1.train()
print(f"Step 1 eval loss: {trainer1.evaluate()['eval_loss']:.4f}")

# === STEP 2: Full fine-tune (unfreeze all) ===
print("\n=== Step 2: Full fine-tune (10 epochs) ===")

for param in model.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params (step 2): {trainable:,}")

args2 = TrainingArguments(
    output_dir="checkpoints/gaia_v2/step2",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=200,
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

# Save final model
trainer2.save_model("checkpoints/gaia_v2/best")

# Copy tokenizer
import shutil
shutil.copy("checkpoints/gaia_expanded/tokenizer.pkl", "checkpoints/gaia_v2/tokenizer.pkl")

print(f"\nDone! Final eval loss: {final_loss:.4f}")
print("Model saved to checkpoints/gaia_v2/best")
