"""수확량 예측: 미생물 조합 → 수확량 (USDA 감자 데이터)"""

import pkg_resources
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from mgm.src.utils import CustomUnpickler
from mgm.CLI.CLI_utils import find_pkg_resource
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Model
model = GPT2LMHeadModel.from_pretrained("checkpoints/mgm_soil_3k/best")
with open(find_pkg_resource("resources/MicroTokenizer.pkl"), "rb") as f:
    tokenizer = CustomUnpickler(f).load()
print("Model loaded")

# Data
df = pd.read_csv("data/raw/tillage/usda_potato.csv")
genus_cols = [c for c in df.columns if c.startswith("BF_g_") or c.startswith("FF_g_")]
genus_name_map = {}
for col in genus_cols:
    parts = col.split("_", 3)
    genus_name_map[col] = parts[3].split("_")[0] if len(parts) >= 4 else col

print(f"Samples: {len(df)}, Genera: {len(genus_cols)}")
print(f"Yield range: {df['Yield_per_meter'].min():.0f} ~ {df['Yield_per_meter'].max():.0f}")
print(f"Yield mean: {df['Yield_per_meter'].mean():.0f}")


class YieldDataset(Dataset):
    def __init__(self, df, genus_cols, genus_name_map, tokenizer, yields):
        self.samples, self.labels = [], []
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id
        for i, (_, row) in enumerate(df.iterrows()):
            nonzero = row[genus_cols][row[genus_cols] > 0].sort_values(ascending=False)
            tokens = [bos]
            for col in nonzero.index:
                genus = genus_name_map.get(col, "")
                tid = tokenizer.vocab.get(f"g__{genus}")
                if tid is not None:
                    tokens.append(tid)
                if len(tokens) >= 511:
                    break
            tokens.append(eos)
            while len(tokens) < 512:
                tokens.append(pad)
            if sum(1 for t in tokens if t not in [bos, eos, pad]) >= 3:
                self.samples.append(torch.tensor(tokens[:512], dtype=torch.long))
                self.labels.append(yields[i])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class YieldRegressor(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        for p in self.gpt.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        with torch.no_grad():
            h = self.gpt(x, output_hidden_states=True).hidden_states[-1]
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


# Normalize yield for training
yield_mean = df["Yield_per_meter"].mean()
yield_std = df["Yield_per_meter"].std()
df["yield_norm"] = (df["Yield_per_meter"] - yield_mean) / yield_std

dataset = YieldDataset(df, genus_cols, genus_name_map, tokenizer, df["yield_norm"].tolist())
print(f"Tokenized: {len(dataset)} samples")

# Train
reg = YieldRegressor(model)
tr_size = int(0.8 * len(dataset))
tr, te = random_split(
    dataset, [tr_size, len(dataset) - tr_size], generator=torch.Generator().manual_seed(42)
)
opt = torch.optim.Adam(reg.head.parameters(), lr=1e-3)

print()
print("=== Yield Prediction Training ===")
for ep in range(30):
    reg.train()
    total_loss = 0
    for bx, by in DataLoader(tr, batch_size=16, shuffle=True):
        preds = reg(bx)
        loss = nn.MSELoss()(preds, torch.tensor(by, dtype=torch.float))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()

    if (ep + 1) % 5 == 0 or ep == 0:
        reg.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for bx, by in DataLoader(te, batch_size=16):
                all_p.extend(reg(bx).tolist())
                all_l.extend(by)
        # Denormalize
        pred_yield = [p * yield_std + yield_mean for p in all_p]
        true_yield = [l * yield_std + yield_mean for l in all_l]
        r2 = r2_score(true_yield, pred_yield)
        rmse = np.sqrt(mean_squared_error(true_yield, pred_yield))
        mae = mean_absolute_error(true_yield, pred_yield)
        print(f"Epoch {ep+1:2d}: R2={r2:.4f}, RMSE={rmse:.0f}, MAE={mae:.0f}")

# Final Gaia results
reg.eval()
all_p, all_l = [], []
with torch.no_grad():
    for bx, by in DataLoader(te, batch_size=16):
        all_p.extend(reg(bx).tolist())
        all_l.extend(by)
pred_yield = [p * yield_std + yield_mean for p in all_p]
true_yield = [l * yield_std + yield_mean for l in all_l]
g_r2 = r2_score(true_yield, pred_yield)
g_rmse = np.sqrt(mean_squared_error(true_yield, pred_yield))
g_mae = mean_absolute_error(true_yield, pred_yield)

# Random Forest
X = df[genus_cols].values
y = df["Yield_per_meter"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
rf_pred = rf.predict(Xte)
rf_r2 = r2_score(yte, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(yte, rf_pred))
rf_mae = mean_absolute_error(yte, rf_pred)

print()
print("=== Yield Prediction Results ===")
print(f"Gaia (MGM+FT): R2={g_r2:.4f}, RMSE={g_rmse:.0f}, MAE={g_mae:.0f}")
print(f"Random Forest: R2={rf_r2:.4f}, RMSE={rf_rmse:.0f}, MAE={rf_mae:.0f}")

print()
print("=== Sample Predictions ===")
for i in range(min(10, len(true_yield))):
    print(f"  Actual: {true_yield[i]:.0f}  Predicted: {pred_yield[i]:.0f}  Error: {abs(true_yield[i]-pred_yield[i]):.0f}")
