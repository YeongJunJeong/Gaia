"""pH prediction fine-tuning using MGM soil model."""

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

# 1. Load model
model = GPT2LMHeadModel.from_pretrained("checkpoints/mgm_soil/best")
with open(find_pkg_resource("resources/MicroTokenizer.pkl"), "rb") as f:
    tokenizer = CustomUnpickler(f).load()

# 2. Load data and create pH proxy labels
abundance = pd.read_csv("data/raw/mgnify/mgnify_abundance.csv")
genus_cols = [c for c in abundance.columns if c not in ["sample_id", "analysis_id"]]

# pH proxy from indicator microbes
acid_indicators = [
    "Acidobacteriaceae", "Bryobacter", "Candidatus_Solibacter",
    "Candidatus_Koribacter", "Acidothermus", "Granulicella",
    "Acidibacter", "Edaphobacter", "Pyrinomonadaceae",
]
neutral_indicators = [
    "Bacillus", "Arthrobacter", "Streptomyces",
    "Pseudomonas", "Agromyces", "Flavobacterium",
    "Sphingomonas", "Rhizobium", "Bradyrhizobium",
]

acid_cols = [c for c in acid_indicators if c in genus_cols]
neutral_cols = [c for c in neutral_indicators if c in genus_cols]

total = abundance[genus_cols].sum(axis=1).replace(0, 1)
acid_ratio = abundance[acid_cols].sum(axis=1) / total
neutral_ratio = abundance[neutral_cols].sum(axis=1) / total

ph_proxy = 6.0 - 3.0 * acid_ratio + 2.0 * neutral_ratio
ph_proxy = ph_proxy.clip(3.5, 8.5)
np.random.seed(42)
ph_proxy += np.random.normal(0, 0.3, len(ph_proxy))
ph_proxy = ph_proxy.clip(3.5, 8.5)
abundance["ph_proxy"] = ph_proxy

print(f"pH proxy: {ph_proxy.min():.2f} ~ {ph_proxy.max():.2f} (mean {ph_proxy.mean():.2f})")
print(f"Samples: {len(abundance)}")


# 3. Dataset
class PHDataset(Dataset):
    def __init__(self, df, genus_cols, tokenizer, ph_values, max_len=512):
        self.samples = []
        self.labels = []
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id
        for i, (_, row) in enumerate(df.iterrows()):
            nonzero = row[genus_cols][row[genus_cols] > 0].sort_values(ascending=False)
            tokens = [bos]
            for genus in nonzero.index:
                tid = tokenizer.vocab.get(f"g__{genus}")
                if tid is not None:
                    tokens.append(tid)
                if len(tokens) >= max_len - 1:
                    break
            tokens.append(eos)
            while len(tokens) < max_len:
                tokens.append(pad)
            if sum(1 for t in tokens if t not in [bos, eos, pad]) >= 5:
                self.samples.append(torch.tensor(tokens[:max_len], dtype=torch.long))
                self.labels.append(float(ph_values.iloc[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


dataset = PHDataset(abundance, genus_cols, tokenizer, abundance["ph_proxy"])
print(f"Valid samples: {len(dataset)}")


# 4. Regression model
class PHPredictor(nn.Module):
    def __init__(self, gpt_model):
        super().__init__()
        self.gpt = gpt_model
        for param in self.gpt.parameters():
            param.requires_grad = False
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids):
        with torch.no_grad():
            outputs = self.gpt(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        mask = (input_ids != 0).unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.regressor(pooled).squeeze(-1)


predictor = PHPredictor(model)
print(f"Trainable params: {sum(p.numel() for p in predictor.parameters() if p.requires_grad):,}")

# 5. Train/test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(
    dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

# 6. Training
optimizer = torch.optim.Adam(predictor.regressor.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print()
print("=== Fine-tuning: pH Prediction ===")
for epoch in range(30):
    predictor.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        preds = predictor(batch_x)
        loss = criterion(preds, torch.tensor(batch_y, dtype=torch.float))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 5 == 0 or epoch == 0:
        predictor.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                preds = predictor(batch_x)
                all_preds.extend(preds.tolist())
                all_labels.extend(batch_y)

        r2 = r2_score(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        mae = mean_absolute_error(all_labels, all_preds)
        print(
            f"Epoch {epoch+1:2d}: loss={total_loss/len(train_loader):.4f}, "
            f"R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}"
        )

# 7. Random Forest comparison
X = abundance[genus_cols].values
y = abundance["ph_proxy"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mae = mean_absolute_error(y_test, rf_preds)

print()
print("=== Comparison: pH Prediction ===")
print(f"Random Forest:   R2={rf_r2:.4f}, RMSE={rf_rmse:.4f}, MAE={rf_mae:.4f}")
print(f"Gaia (MGM + FT): R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

# 8. Sample predictions
print()
print("=== Sample Predictions ===")
for i in range(min(10, len(all_labels))):
    err = abs(all_labels[i] - all_preds[i])
    print(f"  Actual: {all_labels[i]:.2f}  Predicted: {all_preds[i]:.2f}  Error: {err:.2f}")
