"""Future prediction: this year's microbiome → next year's yield/pH"""

import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

BASE = "data/raw/longterm/bonares_data"

# 1. Build microbiome data
print("=== Building time-series pairs ===")
bac = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_BACTERIA.csv", low_memory=False)
genus_ref = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_GENUS.csv")
genus_map = dict(zip(genus_ref["Genus_ID"], genus_ref["Name"]))
bac["Genus_Name"] = bac["Genus_ID"].map(genus_map)
grouped = bac.groupby(["Plot_ID", "Experimental_Year", "Genus_Name"])["Value"].sum().reset_index()
pivot = grouped.pivot_table(index=["Plot_ID", "Experimental_Year"], columns="Genus_Name", values="Value", fill_value=0).reset_index()
genus_cols = [c for c in pivot.columns if c not in ["Plot_ID", "Experimental_Year"]]

# 2. Future yield
harvest = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_HARVEST.csv")
yld = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_YIELD.csv")
hy = harvest.merge(yld[["Harvest_ID", "Yield_Total"]], on="Harvest_ID", how="inner").dropna(subset=["Yield_Total"])
hy_yearly = hy.groupby(["Plot_ID", "Experimental_Year"])["Yield_Total"].mean().reset_index()
yield_lookup = dict(zip(zip(hy_yearly["Plot_ID"], hy_yearly["Experimental_Year"]), hy_yearly["Yield_Total"]))

# 3. Build pairs: current microbiome → next year yield
data_rows = []
for _, row in pivot.iterrows():
    plot, year = row["Plot_ID"], row["Experimental_Year"]
    future_yield = yield_lookup.get((plot, year + 1))
    if future_yield is not None:
        data_row = row.to_dict()
        data_row["Future_Yield"] = future_yield
        data_rows.append(data_row)

data = pd.DataFrame(data_rows)
print(f"Time-series pairs: {len(data)}")
print(f"Current years: {sorted(data['Experimental_Year'].unique())}")
print(f"Future yield range: {data['Future_Yield'].min():.0f} ~ {data['Future_Yield'].max():.0f}")

# 4. Load model
model = GPT2LMHeadModel.from_pretrained("checkpoints/gaia_v4/best")
with open("checkpoints/gaia_v4/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print(f"Model loaded: vocab={len(tokenizer.vocab)}")


class FutureDataset(Dataset):
    def __init__(self, df, genus_cols, tokenizer, labels):
        self.samples, self.labels = [], []
        bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
        for i, (_, row) in enumerate(df.iterrows()):
            nonzero = row[genus_cols][row[genus_cols] > 0].sort_values(ascending=False)
            tokens = [bos]
            for genus in nonzero.index:
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
                self.labels.append(labels[i])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class FutureRegressor(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        for p in self.gpt.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        with torch.no_grad():
            h = self.gpt(x, output_hidden_states=True).hidden_states[-1]
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


# 5. Train future yield prediction
print("\n=== Future Yield Prediction ===")
print("(This year's microbiome → Next year's yield)")

ym, ys = data["Future_Yield"].mean(), data["Future_Yield"].std()
labels_norm = ((data["Future_Yield"] - ym) / ys).tolist()
ds = FutureDataset(data, genus_cols, tokenizer, labels_norm)
print(f"Tokenized: {len(ds)} samples")

tr_n = int(0.8 * len(ds))
tr, te = random_split(ds, [tr_n, len(ds) - tr_n], generator=torch.Generator().manual_seed(42))

reg = FutureRegressor(model).cuda()
opt = torch.optim.Adam(reg.head.parameters(), lr=1e-3)

for ep in range(30):
    reg.train()
    for bx, by in DataLoader(tr, batch_size=16, shuffle=True):
        loss = nn.MSELoss()(reg(bx.cuda()), torch.tensor(by, dtype=torch.float).cuda())
        opt.zero_grad()
        loss.backward()
        opt.step()

    if (ep + 1) % 10 == 0:
        reg.eval()
        p, l = [], []
        with torch.no_grad():
            for bx, by in DataLoader(te, batch_size=16):
                p.extend(reg(bx.cuda()).cpu().tolist())
                l.extend(by)
        po = [v * ys + ym for v in p]
        lo = [v * ys + ym for v in l]
        r2 = r2_score(lo, po)
        rmse = np.sqrt(mean_squared_error(lo, po))
        print(f"Epoch {ep+1}: R2={r2:.4f}, RMSE={rmse:.0f}")

# Final Gaia results
reg.eval()
p, l = [], []
with torch.no_grad():
    for bx, by in DataLoader(te, batch_size=16):
        p.extend(reg(bx.cuda()).cpu().tolist())
        l.extend(by)
po = [v * ys + ym for v in p]
lo = [v * ys + ym for v in l]
g_r2 = r2_score(lo, po)
g_rmse = np.sqrt(mean_squared_error(lo, po))
g_mae = mean_absolute_error(lo, po)

# RF comparison
X = data[genus_cols].values
y = data["Future_Yield"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(200, random_state=42, n_jobs=-1).fit(Xtr, ytr)
rf_pred = rf.predict(Xte)
rf_r2 = r2_score(yte, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(yte, rf_pred))

print(f"\n=== Future Yield Prediction Results ===")
print(f"Gaia: R2={g_r2:.4f}, RMSE={g_rmse:.0f}, MAE={g_mae:.0f}")
print(f"RF:   R2={rf_r2:.4f}, RMSE={rf_rmse:.0f}")

print(f"\n=== Sample Predictions (This year → Next year yield) ===")
for i in range(min(10, len(lo))):
    print(f"  Actual: {lo[i]:.0f}  Predicted: {po[i]:.0f}  Error: {abs(lo[i]-po[i]):.0f}")
