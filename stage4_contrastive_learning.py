"""
stage4_contrastive_learning.py
==============================
Stage 4: Supervised Contrastive Learning (SupCon) & Cross-Modal Alignment
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import silhouette_score
import rasterio

from seed_lock import set_seed  # <-- ADDED

# --- DATASET & ENCODER ---

class MultiModalSAROptDataset(Dataset):
    def __init__(self, df, patch_dir, target_size=(224, 224), transform=None):
        self.df = df.reset_index(drop=True)
        self.patch_dir = patch_dir
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _resolve_file_path(self, row):
        for col in ["filepath", "local_path", "export_name"]:
            if col in row and pd.notna(row[col]):
                val = str(row[col])
                if os.path.exists(val):
                    return val
                rel_path = os.path.join(self.patch_dir, os.path.basename(val))
                if os.path.exists(rel_path):
                    return rel_path
        raise FileNotFoundError(f"Could not locate image file for row index {row.name}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_file_path(row)

        with rasterio.open(img_path) as src:
            img_data = src.read().astype(np.float32)
            img_data = np.nan_to_num(img_data, nan=0.0, posinf=0.0, neginf=0.0)

        if img_data.shape[0] >= 5:
            sar_img = img_data[:2, :, :]
            opt_img = img_data[2:5, :, :]
        elif img_data.shape[0] >= 2:
            sar_img = img_data[:2, :, :]
            opt_img = img_data[:3, :, :] if img_data.shape[0] >= 3 else img_data[:2, :, :]
        else:
            sar_img = img_data
            opt_img = img_data

        sar_tensor = torch.nan_to_num(torch.from_numpy(sar_img), nan=0.0, posinf=0.0, neginf=0.0)
        opt_tensor = torch.nan_to_num(torch.from_numpy(opt_img), nan=0.0, posinf=0.0, neginf=0.0)

        if sar_tensor.shape[0] < 3:
            sar_tensor = sar_tensor.repeat(3, 1, 1)[:3, :, :]
        if opt_tensor.shape[0] < 3:
            opt_tensor = opt_tensor.repeat(3, 1, 1)[:3, :, :]

        sar_tensor = F.interpolate(
            sar_tensor.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False
        ).squeeze(0)
        opt_tensor = F.interpolate(
            opt_tensor.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False
        ).squeeze(0)

        pseudo_label = int(row.get("pseudo_label", row.get("gmm_cluster", 0)))
        return sar_tensor, opt_tensor, pseudo_label


class ModalityEncoder(nn.Module):
    def __init__(self, proj_dim=64):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )

    def forward(self, x):
        feat = self.features(x).squeeze(-1).squeeze(-1)
        proj = self.projection_head(feat)
        norm_proj = F.normalize(proj, p=2, dim=1)
        return torch.nan_to_num(norm_proj, nan=0.0, posinf=0.0, neginf=0.0)

# --- LOSS FUNCTIONS ---

class HardNegativeWeightedSupCon(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cluster_proximity = None

    def set_cluster_proximity(self, proximity_matrix):
        self.cluster_proximity = proximity_matrix

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        if self.cluster_proximity is not None:
            labels_flat = labels.squeeze()
            prox_weights = self.cluster_proximity[labels_flat][:, labels_flat].to(device)
            weight_mask = torch.where(mask == 0, prox_weights, torch.ones_like(mask))
            exp_logits = torch.exp(logits) * weight_mask * logits_mask
        else:
            exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return torch.nan_to_num(loss, nan=0.0)


class CrossModalInfoNCE(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_sar, z_opt):
        batch_size = z_sar.shape[0]
        device = z_sar.device

        sim_matrix = torch.matmul(z_sar, z_opt.T) / self.temperature
        labels = torch.arange(batch_size, device=device)

        loss_sar2opt = F.cross_entropy(sim_matrix, labels)
        loss_opt2sar = F.cross_entropy(sim_matrix.T, labels)
        loss = (loss_sar2opt + loss_opt2sar) / 2.0
        return torch.nan_to_num(loss, nan=0.0)

# --- EVALUATION METRICS ---

def compute_test_silhouette(sar_enc, opt_enc, test_df, patch_dir, device):
    sar_enc.eval()
    opt_enc.eval()
    ds = MultiModalSAROptDataset(test_df, patch_dir)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    fused_embeds, labels = [], []
    with torch.no_grad():
        for sar, opt, lbl in loader:
            sar, opt = sar.to(device), opt.to(device)
            z_s = sar_enc(sar)
            z_o = opt_enc(opt)
            z_fused = F.normalize((z_s + z_o) / 2.0, p=2, dim=1)
            fused_embeds.append(z_fused.cpu().numpy())
            labels.append(lbl.numpy() if isinstance(lbl, torch.Tensor) else lbl)

    X = np.nan_to_num(np.vstack(fused_embeds), nan=0.0)
    y = np.concatenate(labels)
    if len(np.unique(y)) < 2:
        return 0.0
    return float(silhouette_score(X, y))

# --- STAGE 4 EXECUTION ---

def run_stage4(train_df, test_df, patch_dir="./downloaded_patches", output_dir="./saved_models", num_epochs=100, lr=1e-3, device="cuda", seed=42):
    set_seed(seed)  # <-- ADDED: must be first, before model/dataloader creation

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    sar_enc = ModalityEncoder(proj_dim=64).to(device)
    opt_enc = ModalityEncoder(proj_dim=64).to(device)

    optimizer = torch.optim.Adam(
        list(sar_enc.parameters()) + list(opt_enc.parameters()), lr=lr
    )

    supcon_fn = HardNegativeWeightedSupCon(temperature=0.07)
    infonce_fn = CrossModalInfoNCE(temperature=0.07)

    train_ds = MultiModalSAROptDataset(train_df, patch_dir)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)

    best_sil = -1.0

    for epoch in range(1, num_epochs + 1):
        sar_enc.train()
        opt_enc.train()
        total_loss = 0.0

        for sar, opt, lbls in train_loader:
            sar, opt, lbls = sar.to(device), opt.to(device), lbls.to(device)

            optimizer.zero_grad()
            z_sar = sar_enc(sar)
            z_opt = opt_enc(opt)

            loss_supcon_sar = supcon_fn(z_sar, lbls)
            loss_supcon_opt = supcon_fn(z_opt, lbls)
            loss_align = infonce_fn(z_sar, z_opt)

            loss = loss_supcon_sar + loss_supcon_opt + (0.5 * loss_align)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        test_sil = compute_test_silhouette(sar_enc, opt_enc, test_df, patch_dir, device)
        print(f"Epoch {epoch}/{num_epochs} - Loss: {total_loss/len(train_loader):.4f} - Test Silhouette: {test_sil:.4f}")

        checkpoint = {
            "sar_state": sar_enc.state_dict(),
            "opt_state": opt_enc.state_dict(),
            "epoch": epoch,
            "silhouette": test_sil
        }
        torch.save(checkpoint, os.path.join(output_dir, "stage4_latest.pth"))

        if test_sil >= best_sil:
            best_sil = test_sil
            torch.save(checkpoint, os.path.join(output_dir, "stage4_best.pth"))
            torch.save(sar_enc.state_dict(), os.path.join(output_dir, "sar_encoder_best.pth"))
            torch.save(opt_enc.state_dict(), os.path.join(output_dir, "optical_encoder_best.pth"))

    print(f"Stage 4 Complete. Best Test Silhouette Score: {best_sil:.4f}")

if __name__ == "__main__":
    input_csv = "./patch_metadata_clustered.csv"
    if os.path.exists(input_csv):
        df = pd.read_csv(input_csv)
        clean_df = df[df.get("valid", True) == True].copy().reset_index(drop=True)
        train_df = clean_df[clean_df["split"] == "train"].reset_index(drop=True)
        test_df = clean_df[clean_df["split"] == "test"].reset_index(drop=True)
        run_stage4(train_df, test_df, num_epochs=100)
    else:
        print(f"❌ Input CSV missing: {input_csv}")