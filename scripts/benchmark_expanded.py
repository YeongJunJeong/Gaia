"""Expanded vocab model: re-run all benchmarks"""

import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load expanded model + tokenizer
model = GPT2LMHeadModel.from_pretrained("checkpoints/gaia_v3/best")
model.eval()
with open("checkpoints/gaia_v3/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print(f"Model loaded: vocab={len(tokenizer.vocab)}")


class GenusDataset(Dataset):
    def __init__(self, df, genus_cols, tokenizer, labels, genus_prefix="g__"):
        self.samples, self.labels = [], []
        bos, eos, pad = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id
        for i, (_, row) in enumerate(df.iterrows()):
            nonzero = row[genus_cols][row[genus_cols] > 0].sort_values(ascending=False)
            tokens = [bos]
            for genus in nonzero.index:
                tid = tokenizer.vocab.get(f"{genus_prefix}{genus}")
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


class Clf(nn.Module):
    def __init__(self, gpt, n):
        super().__init__()
        self.gpt = gpt
        for p in self.gpt.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, n))

    def forward(self, x):
        with torch.no_grad():
            h = self.gpt(x, output_hidden_states=True).hidden_states[-1]
        mask = (x != 0).unsqueeze(-1).float()
        return self.head((h * mask).sum(1) / mask.sum(1).clamp(min=1))


class Reg(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt
        for p in self.gpt.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):
        with torch.no_grad():
            h = self.gpt(x, output_hidden_states=True).hidden_states[-1]
        mask = (x != 0).unsqueeze(-1).float()
        return self.head((h * mask).sum(1) / mask.sum(1).clamp(min=1)).squeeze(-1)


def run_clf(data, genus_cols, label_col, name, prefix="g__"):
    ds = GenusDataset(data, genus_cols, tokenizer, data[label_col].tolist(), prefix)
    tr_n = int(0.8 * len(ds))
    tr, te = random_split(ds, [tr_n, len(ds) - tr_n], generator=torch.Generator().manual_seed(42))
    clf = Clf(model, 2).cuda()
    opt = torch.optim.Adam(clf.head.parameters(), lr=1e-3)
    for ep in range(20):
        clf.train()
        for bx, by in DataLoader(tr, batch_size=16, shuffle=True):
            loss = nn.CrossEntropyLoss()(clf(bx.cuda()), torch.tensor(by).cuda())
            opt.zero_grad(); loss.backward(); opt.step()
    clf.eval()
    p, l = [], []
    with torch.no_grad():
        for bx, by in DataLoader(te, batch_size=16):
            p.extend(clf(bx.cuda()).argmax(1).cpu().tolist()); l.extend(by)
    g = accuracy_score(l, p)
    X, y = data[genus_cols].values, data[label_col].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    r = accuracy_score(yte, RandomForestClassifier(200, random_state=42, n_jobs=-1).fit(Xtr, ytr).predict(Xte))
    print(f"{name}: Gaia {g:.1%} vs RF {r:.1%}")
    return g, r


def run_reg(data, genus_cols, label_col, name, prefix="g__"):
    ym, ys = data[label_col].mean(), data[label_col].std()
    labels_n = ((data[label_col] - ym) / ys).tolist()
    ds = GenusDataset(data, genus_cols, tokenizer, labels_n, prefix)
    tr_n = int(0.8 * len(ds))
    tr, te = random_split(ds, [tr_n, len(ds) - tr_n], generator=torch.Generator().manual_seed(42))
    reg = Reg(model).cuda()
    opt = torch.optim.Adam(reg.head.parameters(), lr=1e-3)
    for ep in range(30):
        reg.train()
        for bx, by in DataLoader(tr, batch_size=16, shuffle=True):
            loss = nn.MSELoss()(reg(bx.cuda()), torch.tensor(by, dtype=torch.float).cuda())
            opt.zero_grad(); loss.backward(); opt.step()
    reg.eval()
    p, l = [], []
    with torch.no_grad():
        for bx, by in DataLoader(te, batch_size=16):
            p.extend(reg(bx.cuda()).cpu().tolist()); l.extend(by)
    po = [v * ys + ym for v in p]
    lo = [v * ys + ym for v in l]
    g = r2_score(lo, po)
    X, y = data[genus_cols].values, data[label_col].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    r = r2_score(yte, RandomForestRegressor(200, random_state=42, n_jobs=-1).fit(Xtr, ytr).predict(Xte))
    print(f"{name}: Gaia R2={g:.3f} vs RF R2={r:.3f}")
    return g, r


results = {}

# === 1. Biome Classification (MGnify) ===
print("\n=== MGnify Benchmarks ===")
mgnify_ab = pd.read_csv("data/raw/mgnify/mgnify_abundance.csv")
mgnify_meta = pd.read_csv("data/raw/mgnify/mgnify_metadata.csv")
mgnify_genus = [c for c in mgnify_ab.columns if c not in ["sample_id", "analysis_id"]]
merged = mgnify_ab.merge(mgnify_meta[["sample_id", "biome"]], on="sample_id", how="inner")


def clean_biome(b):
    if pd.isna(b):
        return None
    b = str(b).lower()
    if "forest" in b:
        return "forest"
    if "grassland" in b or "prairie" in b:
        return "grassland"
    return None


merged["label"] = merged["biome"].apply(clean_biome)
valid = merged.dropna(subset=["label"]).reset_index(drop=True)
valid["label_id"] = (valid["label"] == "grassland").astype(int)
results["biome"] = run_clf(valid, mgnify_genus, "label_id", "Biome (forest vs grassland)")

# === 2. Drought Detection (Naylor) ===
print("\n=== Naylor Benchmark ===")
naylor = pd.read_csv("data/raw/naylor/naylor_genus_with_labels.csv")
naylor_genus = [c for c in naylor.columns if c not in ["sample_id", "run_id", "treatment", "host"]]
naylor["drought_label"] = (naylor["treatment"] == "drought").astype(int)
results["drought"] = run_clf(naylor, naylor_genus, "drought_label", "Drought detection")

# === 3. BonaRes Benchmarks ===
print("\n=== BonaRes Benchmarks ===")
BASE = "data/raw/longterm/bonares_data"
bac = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_BACTERIA.csv", low_memory=False)
genus_ref = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_GENUS.csv")
plot_df = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_PLOT.csv")
treatment = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_TREATMENT.csv")
f1_level = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_FACTOR_1_LEVEL.csv")
f2_level = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_FACTOR_2_LEVEL.csv")

genus_map = dict(zip(genus_ref["Genus_ID"], genus_ref["Name"]))
bac["Genus_Name"] = bac["Genus_ID"].map(genus_map)
grouped = bac.groupby(["Plot_ID", "Experimental_Year", "Genus_Name"])["Value"].sum().reset_index()
pivot = grouped.pivot_table(index=["Plot_ID", "Experimental_Year"], columns="Genus_Name", values="Value", fill_value=0).reset_index()
b_genus = [c for c in pivot.columns if c not in ["Plot_ID", "Experimental_Year"]]

plot_treat = plot_df[["Plot_ID", "Treatment_ID"]].merge(treatment, on="Treatment_ID")
plot_treat = plot_treat.merge(f1_level[["Factor_1_Level_ID", "Name_EN"]].rename(columns={"Name_EN": "tillage"}), on="Factor_1_Level_ID")
plot_treat = plot_treat.merge(f2_level[["Factor_2_Level_ID", "Name_EN"]].rename(columns={"Name_EN": "fert"}), on="Factor_2_Level_ID")
bdata = pivot.merge(plot_treat[["Plot_ID", "tillage", "fert"]], on="Plot_ID", how="left").dropna(subset=["tillage"]).reset_index(drop=True)

bdata["till_label"] = (bdata["tillage"] == "Plough").astype(int)
results["tillage"] = run_clf(bdata, b_genus, "till_label", "Tillage (cultivator vs plough)")

bdata["fert_label"] = (bdata["fert"] == "intensive").astype(int)
results["fert"] = run_clf(bdata, b_genus, "fert_label", "Fertilization (ext vs int)")

# pH
soil = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_SOIL_LAB.csv")
soil_samp = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_SOIL_SAMPLING.csv")
soil = soil.merge(soil_samp[["Soil_Sampling_ID", "Plot_ID", "Experimental_Year"]], on="Soil_Sampling_ID", how="left")
soil_ph = soil.dropna(subset=["pH"]).groupby(["Plot_ID", "Experimental_Year"])["pH"].mean().reset_index()
data_ph = pivot.merge(soil_ph, on=["Plot_ID", "Experimental_Year"], how="inner").dropna(subset=["pH"]).reset_index(drop=True)
if len(data_ph) > 10:
    results["ph_bonares"] = run_reg(data_ph, b_genus, "pH", "pH (BonaRes)")

# Yield
harvest = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_HARVEST.csv")
yld = pd.read_csv(f"{BASE}/lte_westerfeld.V1_0_YIELD.csv")
hy = harvest.merge(yld[["Harvest_ID", "Yield_Total"]], on="Harvest_ID", how="inner").dropna(subset=["Yield_Total"])
hy = hy.groupby(["Plot_ID", "Experimental_Year"])["Yield_Total"].mean().reset_index()
data_yld = pivot.merge(hy, on=["Plot_ID", "Experimental_Year"], how="inner").dropna(subset=["Yield_Total"]).reset_index(drop=True)
if len(data_yld) > 10:
    results["yield_bonares"] = run_reg(data_yld, b_genus, "Yield_Total", "Yield (BonaRes)")

# === Summary ===
print("\n" + "=" * 60)
print("EXPANDED MODEL (100% MATCHING) vs PREVIOUS (47-58%)")
print("=" * 60)
for k, (g, r) in results.items():
    winner = "GAIA" if g > r else "RF"
    print(f"  {k:20s}: Gaia={g:.3f}  RF={r:.3f}  [{winner}]")
