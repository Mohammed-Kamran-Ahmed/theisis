"""
overfit_test.py — §2 Overfit Sanity Check
==========================================
Takes exactly 50 fixed synthetic samples, trains backbone + classification head
+ BCE only (no CAHN, no IERC, no augmentation) for 200 epochs.

Expected: training F1(macro) climbs toward ~100% on those 50 samples.
If it stays near 0% → bug is still in data loading or loss.
If it reaches ~100% → pipeline is fundamentally healthy.

Run from d:\\BTP\\theisis_backup\\ with:
    D:\\BTP\\theisis\\thesis_env\\Scripts\\python.exe -m reben_pipeline.overfit_test
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, "D:/BTP/theisis_backup")

from reben_pipeline.dataset import ReBENDataset, create_synthetic_reben_dataframe
from reben_pipeline.models import CAHNIERCDualEncoder
from reben_pipeline.metrics import MultiLabelMetricsCalculator, BEN_CLASSES_19

NUM_SAMPLES = 50
EPOCHS = 200
BATCH_SIZE = 50   # single batch for simplicity
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 1. Create 50 fixed samples (all marked as 'train') ──────────────────────
df = create_synthetic_reben_dataframe(num_samples=NUM_SAMPLES, split_ratios=(1.0, 0.0, 0.0), seed=SEED)
print(f"\nLabel example (raw):  {df['labels'].iloc[0]}")

ds = ReBENDataset(df, normalize=False, is_synthetic=True)   # no normalise for speed
item0 = ds[0]
print(f"Target vector:        {item0['targets']}")
print(f"Target sum (should>0): {item0['targets'].sum().item()}")

# Verify labels are non-zero
label_sums = torch.stack([ds[i]["targets"] for i in range(len(ds))]).sum(dim=1)
print(f"Label sum statistics — min={label_sums.min().item():.0f}  "
      f"max={label_sums.max().item():.0f}  "
      f"mean={label_sums.mean().item():.2f}")
assert label_sums.min().item() > 0, "FAIL: at least one sample has all-zero labels"
print("[PASS] All 50 samples have at least one active label\n")

loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

# ── 2. Model — small backbone, classification head only ─────────────────────
model = CAHNIERCDualEncoder(
    sar_channels=2,
    opt_channels=10,
    num_classes=19,
    backbone="resnet50",
    proj_dim=128,
    fusion_type="concat",
    pretrained=False        # random init — overfit test
).to(device)

# Compute pos_weight for balanced BCE
all_t = torch.stack([ds[i]["targets"] for i in range(len(ds))], dim=0)
pos_freq = all_t.mean(dim=0).clamp(0.01, 0.99)
pos_weight = ((1.0 - pos_freq) / pos_freq).to(device)

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"{'Epoch':<6}  {'BCE Loss':<12}  {'F1 Macro':<12}  {'F1 Micro':<12}  {'Active preds'}")
print("-" * 65)

# ── 3. Training loop ─────────────────────────────────────────────────────────
best_f1 = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    for batch in loader:
        s1 = batch["s1"].to(device)
        s2 = batch["s2"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        out = model(sar_img=s1, opt_img=s2)
        loss = loss_fn(out["logits_fused"], targets)
        loss.backward()
        optimizer.step()

    # ── Evaluate on same 50 samples ──────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        all_logits, all_targets = [], []
        for batch in loader:
            s1 = batch["s1"].to(device)
            s2 = batch["s2"].to(device)
            out = model(sar_img=s1, opt_img=s2)
            all_logits.append(out["logits_fused"].cpu())
            all_targets.append(batch["targets"])

    logits_cat = torch.cat(all_logits, dim=0)
    tgt_cat    = torch.cat(all_targets, dim=0)
    probs      = torch.sigmoid(logits_cat)
    preds_bin  = (probs >= 0.5).numpy().astype(int)
    tgt_np     = tgt_cat.numpy().astype(int)

    f1_mac = f1_score(tgt_np, preds_bin, average="macro", zero_division=0) * 100
    f1_mic = f1_score(tgt_np, preds_bin, average="micro", zero_division=0) * 100
    n_active = preds_bin.sum(axis=1).mean()
    best_f1  = max(best_f1, f1_mac)

    if epoch % 20 == 0 or epoch == 1:
        print(f"{epoch:<6d}  {loss.item():<12.5f}  {f1_mac:<12.2f}  {f1_mic:<12.2f}  {n_active:.1f}")

print("\n" + "="*65)
print(f"Best Train F1(macro): {best_f1:.2f}%")
if best_f1 >= 80.0:
    print("[PASS] Model can overfit 50 samples — pipeline is healthy")
elif best_f1 >= 40.0:
    print("[PARTIAL] Model learns but slowly — check LR, label balance")
else:
    print("[FAIL] Model cannot overfit 50 samples — deeper bug remains")
