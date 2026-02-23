import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

WEIGHT_DIR   = "/home/wzh/UKB-SJTU/preprocess/MAE-Weights"
TRAIN_DIR    = "/home/wzh/UKB-SJTU/preprocess/MAE-Train-Data"
DATA_DIR     = "/home/wzh/UKB-SJTU/preprocess/data_nodes/by_body-organ"
SAVE_LATENT_DIR = "/home/wzh/UKB-SJTU/preprocess/Organ-Latent"

BATCH_SIZE   = 1024
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_LATENT_DIR, exist_ok=True)



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


def process_one_weight(weight_path: str):
    fname = os.path.basename(weight_path)           # Hepatic_MAE.pt
    organ = fname.replace("_MAE.pt", "")           # Hepatic

    train_csv = os.path.join(TRAIN_DIR, f"{organ}.csv")
    data_csv  = os.path.join(DATA_DIR,  f"{organ}.csv")
    save_npy  = os.path.join(SAVE_LATENT_DIR, f"{organ}.npy")

    if not os.path.exists(train_csv):
        print(f"[skip] TRAIN csv not found for {organ}: {train_csv}")
        return
    if not os.path.exists(data_csv):
        print(f"[skip] DATA csv not found for {organ}: {data_csv}")
        return

    print(f"\n=== Processing organ: {organ} ===")
    print(f"  weight: {weight_path}")
    print(f"  train_csv: {train_csv}")
    print(f"  data_csv : {data_csv}")
    print(f"  save_npy : {save_npy}")

    ckpt = torch.load(weight_path, map_location=DEVICE)
    input_dim  = ckpt["input_dim"]
    latent_dim = ckpt["latent_dim"]
    hidden_dim = ckpt["hidden_dim"]

    model = MaskedAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    df_train = pd.read_csv(train_csv)
    X_train = df_train.iloc[:, 1:].values.astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(X_train)

    df_data = pd.read_csv(data_csv)
    X = df_data.iloc[:, 1:].values.astype(np.float32)

    X_std = scaler.transform(X).astype(np.float32)

    mask_np = np.isnan(X_std).astype(np.float32)

    X_std[np.isnan(X_std)] = 0.0

    tensor_X = torch.from_numpy(X_std)
    tensor_mask = torch.from_numpy(mask_np)

    dataset = TensorDataset(tensor_X, tensor_mask)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_z = []
    with torch.no_grad():
        for x_batch, m_batch in loader:
            x_batch = x_batch.to(DEVICE)
            m_batch = m_batch.to(DEVICE)
            _, z = model(x_batch, m_batch)
            all_z.append(z.cpu().numpy())

    Z_latent = np.concatenate(all_z, axis=0)  


    np.save(save_npy, Z_latent)
    print(f"  Saved latent to: {save_npy}")


if __name__ == "__main__":
    for fname in os.listdir(WEIGHT_DIR):
        if not fname.endswith("_MAE.pt"):
            continue
        weight_path = os.path.join(WEIGHT_DIR, fname)
        process_one_weight(weight_path)
