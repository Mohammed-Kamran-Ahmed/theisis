"""
train_cahn_ierc.py - Complete Training & Evaluation Pipeline for CAHN / IERC Dual-Encoders on reBEN
===================================================================================================
Executes:
1. Multi-modal training on reBEN official splits.
2. CAHN (Cluster-Aware Hard-Negative) SupCon representation learning.
3. Cross-modal InfoNCE alignment between SAR and Optical towers.
4. Periodic Iterative Embedding-Anchored Re-Clustering (IERC).
5. Multi-label classification evaluation (Macro AP, Micro AP, Macro F1, Micro F1).
6. Cross-modal Canonical Correlation Analysis (CCA) with 200 permutation null runs.
7. Inter-cluster geometry validation (Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, NMI).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)


from reben_pipeline.dataset import ReBENDataset, SynchronizedTransforms, create_synthetic_reben_dataframe
from reben_pipeline.models import CAHNIERCDualEncoder
from reben_pipeline.losses import JointDualEncoderLoss, ClusterAwareHardNegativeSupCon
from reben_pipeline.metrics import MultiLabelMetricsCalculator, BEN_CLASSES_19
from reben_pipeline.cca_analysis import run_cca_analysis
from reben_pipeline.cluster_metrics import compute_clustering_metrics
from reben_pipeline.ierc import IERCController, extract_latent_features


def set_all_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: JointDualEncoderLoss,
    device: torch.device,
    use_cahn: bool = True,
    use_align: bool = True
) -> dict:
    model.train()
    total_loss, total_cahn, total_align, total_cls = 0.0, 0.0, 0.0, 0.0
    n_batches = len(loader)

    for batch in loader:
        s1 = batch["s1"].to(device)
        s2 = batch["s2"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        out = model(sar_img=s1, opt_img=s2)

        # Loss calculation
        loss, loss_dict = loss_fn(
            z_sar=out["z_sar"],
            z_opt=out["z_opt"],
            logits_fused=out["logits_fused"],
            targets=targets,
            logits_sar=out["logits_sar"],
            logits_opt=out["logits_opt"],
            multi_label=True
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_dict["total_loss"]
        total_cahn += loss_dict["cahn_loss"]
        total_align += loss_dict["align_loss"]
        total_cls += loss_dict["cls_loss"]

    return {
        "train_loss": total_loss / max(n_batches, 1),
        "train_cahn": total_cahn / max(n_batches, 1),
        "train_align": total_align / max(n_batches, 1),
        "train_cls": total_cls / max(n_batches, 1)
    }


def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    metric_calc: MultiLabelMetricsCalculator
) -> dict:
    model.eval()
    all_fused_logits, all_sar_logits, all_opt_logits, all_targets = [], [], [], []
    all_z_sar, all_z_opt, all_z_fused = [], [], []

    with torch.no_grad():
        for batch in loader:
            s1 = batch["s1"].to(device)
            s2 = batch["s2"].to(device)
            targets = batch["targets"]

            out = model(sar_img=s1, opt_img=s2)

            all_fused_logits.append(out["logits_fused"].cpu())
            all_sar_logits.append(out["logits_sar"].cpu())
            all_opt_logits.append(out["logits_opt"].cpu())
            all_targets.append(targets)

            all_z_sar.append(out["z_sar"].cpu().numpy())
            all_z_opt.append(out["z_opt"].cpu().numpy())
            all_z_fused.append(out["z_fused"].cpu().numpy())

    fused_logits = torch.cat(all_fused_logits, dim=0)
    sar_logits = torch.cat(all_sar_logits, dim=0)
    opt_logits = torch.cat(all_opt_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Multi-label metrics for Fusion (S1+S2), SAR-only (S1), Optical-only (S2)
    fusion_metrics = metric_calc.compute_all(fused_logits, targets)
    sar_metrics = metric_calc.compute_all(sar_logits, targets)
    opt_metrics = metric_calc.compute_all(opt_logits, targets)

    return {
        "fusion": fusion_metrics,
        "s1": sar_metrics,
        "s2": opt_metrics,
        "z_sar": np.vstack(all_z_sar),
        "z_opt": np.vstack(all_z_opt),
        "z_fused": np.vstack(all_z_fused),
        "targets": targets.numpy()
    }


def run_pipeline(
    df: pd.DataFrame = None,
    s1_dir: str = None,
    s2_dir: str = None,
    backbone: str = "resnet50",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    use_cahn: bool = True,
    use_ierc: bool = True,
    ierc_interval: int = 3,
    output_dir: str = "./results",
    seed: int = 42,
    is_synthetic: bool = False
):
    set_all_seeds(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device} | Seed: {seed} | Backbone: {backbone}")

    # 1. Dataset Preparation
    if df is None:
        print("[*] Generating benchmark dataset structure...")
        df = create_synthetic_reben_dataframe(num_samples=1200, seed=seed)
        is_synthetic = True

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"[*] Dataset Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    transform_train = SynchronizedTransforms(p_flip=0.5, p_rot=0.5)
    train_ds = ReBENDataset(train_df, s1_dir, s2_dir, normalize=True, transform=transform_train, is_synthetic=is_synthetic)
    val_ds = ReBENDataset(val_df, s1_dir, s2_dir, normalize=True, is_synthetic=is_synthetic)
    test_ds = ReBENDataset(test_df, s1_dir, s2_dir, normalize=True, is_synthetic=is_synthetic)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Model & Loss Setup
    model = CAHNIERCDualEncoder(
        sar_channels=2,
        opt_channels=10,
        num_classes=19,
        backbone=backbone,
        proj_dim=128,
        fusion_type="concat",
        pretrained=True
    ).to(device)

    # Compute pos_weight from training label frequencies to prevent trivial all-zero predictions
    print("[*] Computing BCE pos_weight from training label frequencies...")
    all_targets_list = []
    for i in range(len(train_ds)):
        all_targets_list.append(train_ds[i]["targets"])
    all_targets_stack = torch.stack(all_targets_list, dim=0)  # (N_train, 19)
    pos_freq = all_targets_stack.mean(dim=0).clamp(min=0.01, max=0.99)
    pos_weight = ((1.0 - pos_freq) / pos_freq).to(device)
    print(f"   pos_weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")

    loss_fn = JointDualEncoderLoss(
        temperature=0.07,
        lambda_cahn=1.0 if use_cahn else 0.0,
        lambda_align=0.5,
        lambda_cls=1.0,
        pos_weight=pos_weight
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    metric_calc = MultiLabelMetricsCalculator(num_classes=19)

    ierc_controller = IERCController(n_clusters=16, seed=seed) if use_ierc else None

    # Trackers
    best_val_ap_macro = -1.0
    history = []

    print("\n" + "="*80)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val AP(Macro)':<14} | {'Val AP(Micro)':<14} | {'Val F1(Macro)':<14} | {'Val F1(Micro)':<14}")
    print("="*80)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        # IERC step at specified intervals
        if use_ierc and epoch > 1 and epoch % ierc_interval == 0:
            print(f"[*] Running IERC Re-Clustering Step at Epoch {epoch}...")
            _, prox_matrix, _ = ierc_controller.step(model, train_loader, device)
            loss_fn.cahn_loss.set_proximity_matrix(prox_matrix)

        train_res = train_epoch(model, train_loader, optimizer, loss_fn, device, use_cahn=use_cahn)
        scheduler.step()

        # Validation
        val_eval = evaluate_split(model, val_loader, device, metric_calc)
        val_macro_ap = val_eval["fusion"]["ap_macro"]
        val_micro_ap = val_eval["fusion"]["ap_micro"]
        val_macro_f1 = val_eval["fusion"]["f1_macro"]
        val_micro_f1 = val_eval["fusion"]["f1_micro"]

        print(f"{epoch:<6d} | {train_res['train_loss']:<10.4f} | {val_macro_ap:<14.2f}% | {val_micro_ap:<14.2f}% | {val_macro_f1:<14.2f}% | {val_micro_f1:<14.2f}%  ({time.time()-t0:.1f}s)")

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_res["train_loss"],
            "val_ap_macro": val_macro_ap,
            "val_ap_micro": val_micro_ap,
            "val_f1_macro": val_macro_f1,
            "val_f1_micro": val_micro_f1
        }
        history.append(epoch_record)

        if val_macro_ap > best_val_ap_macro:
            best_val_ap_macro = val_macro_ap
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_val_ap_macro": best_val_ap_macro,
                "config": {
                    "backbone": backbone,
                    "use_cahn": use_cahn,
                    "use_ierc": use_ierc,
                    "lr": lr,
                    "seed": seed
                }
            }, os.path.join(output_dir, "best_dual_encoder_model.pth"))

    print("="*80)
    print(f"[*] Training Complete. Best Validation Macro AP: {best_val_ap_macro:.2f}%")

    # 3. Final Evaluation on Held-Out Test Split
    print("\n[*] Loading best checkpoint for Test Evaluation...")
    best_ckpt = torch.load(os.path.join(output_dir, "best_dual_encoder_model.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    test_eval = evaluate_split(model, test_loader, device, metric_calc)

    # 4. Canonical Correlation Analysis (CCA) Benchmark with 200 Permutations
    print("[*] Running Permutation-Tested Canonical Correlation Analysis (CCA)...")
    cca_res = run_cca_analysis(test_eval["z_sar"], test_eval["z_opt"], n_components=8, n_permutations=200, seed=seed)
    print(f"  -> First Canonical Correlation r1: {cca_res['r1']:.4f}")
    print(f"  -> Mean Canonical Correlation r_bar: {cca_res['r_bar']:.4f}")
    print(f"  -> Null Mean: {cca_res['r1_null_mean']:.4f} | Alignment Gain Delta_null: {cca_res['delta_null']:.4f} (p = {cca_res['p_value']:.4f})")

    # 5. Latent Clustering Metrics
    print("[*] Computing Latent Clustering Quality Metrics...")
    if use_ierc and ierc_controller and ierc_controller.gmm is not None:
        cluster_labels = ierc_controller.gmm.predict(test_eval["z_fused"])
    else:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=16, random_state=seed, n_init=5).fit(test_eval["z_fused"])
        cluster_labels = kmeans.labels_

    cluster_res = compute_clustering_metrics(test_eval["z_fused"], cluster_labels, reference_labels=test_eval["targets"])
    print(f"  -> Silhouette Score: {cluster_res['silhouette']:.4f}")
    print(f"  -> Davies-Bouldin Index: {cluster_res['davies_bouldin']:.4f}")
    print(f"  -> Calinski-Harabasz Index: {cluster_res['calinski_harabasz']:.2f}")
    print(f"  -> Adjusted Rand Index (ARI): {cluster_res.get('ari', 0.0):.4f}")
    print(f"  -> Normalized Mutual Info (NMI): {cluster_res.get('nmi', 0.0):.4f}")

    # Compile Final Benchmark Summary
    summary = {
        "config": {
            "backbone": backbone,
            "use_cahn": use_cahn,
            "use_ierc": use_ierc,
            "epochs": epochs,
            "lr": lr,
            "seed": seed
        },
        "results": {
            "s1_only": {
                "ap_macro": test_eval["s1"]["ap_macro"],
                "ap_micro": test_eval["s1"]["ap_micro"],
                "f1_macro": test_eval["s1"]["f1_macro"],
                "f1_micro": test_eval["s1"]["f1_micro"],
            },
            "s2_only": {
                "ap_macro": test_eval["s2"]["ap_macro"],
                "ap_micro": test_eval["s2"]["ap_micro"],
                "f1_macro": test_eval["s2"]["f1_macro"],
                "f1_micro": test_eval["s2"]["f1_micro"],
            },
            "s1_s2_fusion": {
                "ap_macro": test_eval["fusion"]["ap_macro"],
                "ap_micro": test_eval["fusion"]["ap_micro"],
                "f1_macro": test_eval["fusion"]["f1_macro"],
                "f1_micro": test_eval["fusion"]["f1_micro"],
            }
        },
        "cca_benchmark": cca_res,
        "cluster_metrics": cluster_res,
        "training_history": history
    }

    results_file = os.path.join(output_dir, "reben_experiment_summary.json")
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50", "resnet101"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--no_cahn", action="store_true")
    p.add_argument("--no_ierc", action="store_true")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run_pipeline(
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_cahn=not args.no_cahn,
        use_ierc=not args.no_ierc,
        output_dir=args.output_dir,
        seed=args.seed
    )
