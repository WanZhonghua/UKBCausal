import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'   

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR   = "/home/wzh/UKB-SJTU/preprocess/MAE-Train-Data"
SAVE_DIR   = "/home/wzh/UKB-SJTU/preprocess/MAE-Weights"

BATCH_SIZE = 4090
EPOCHS     = 100
LR         = 1e-3
MASK_MIN   = 0.1    
MASK_MAX   = 0.4     
LATENT_DIM = 3
HIDDEN_DIM = 64
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    print(f"Use device: {DEVICE}")
    torch.cuda.manual_seed_all(SEED)

os.makedirs(SAVE_DIR, exist_ok=True)

class OrganDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        X = df.iloc[:, 1:].values.astype(np.float32)

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X).astype(np.float32)

        self.X = torch.from_numpy(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]



class MaskedAE(nn.Module):
    def __init__(self, input_dim, latent_dim=3, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, mask):
        x_masked = x.clone()
        x_masked[mask.bool()] = 0.0
        inp = torch.cat([x_masked, mask], dim=-1)
        z = self.encoder(inp)
        x_rec = self.decoder(z)
        return x_rec, z


def train_one(csv_path, save_path):
    print(f"\n=== Training on {csv_path} ===")

    dataset = OrganDataset(csv_path)
    n_samples = len(dataset)
    if n_samples == 0:
        print("  [skip] empty dataset")
        return

    batch_size = min(BATCH_SIZE, n_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    input_dim = dataset.X.shape[1]
    model = MaskedAE(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for x in dataloader:
            x = x.to(DEVICE)
            B, D = x.shape

            mask_ratio = np.random.uniform(MASK_MIN, MASK_MAX)

            rand = torch.rand(B, D, device=DEVICE)
            mask = (rand < mask_ratio).float()

            x_rec, z = model(x, mask)

            diff2 = (x_rec - x) ** 2
            loss = (diff2 * mask).sum() / (mask.sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        print(f"  Epoch {epoch}/{EPOCHS}  loss={avg_loss:.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "latent_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIM,
        },
        save_path,
    )
    print(f"  Saved to {save_path}")



if __name__ == "__main__":
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".csv"):
            continue
        csv_path = os.path.join(DATA_DIR, fname)
        organ_name = os.path.splitext(fname)[0]
        save_path = os.path.join(SAVE_DIR, f"{organ_name}_MAE.pt")
        train_one(csv_path, save_path)

