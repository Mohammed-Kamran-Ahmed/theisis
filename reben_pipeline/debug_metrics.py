"""
debug_metrics.py — §1 Unit-test for metrics.py and dataset.py
==============================================================
Run from d:\\BTP\\theisis_backup\\ with:
    D:\\BTP\\theisis\\thesis_env\\Scripts\\python.exe -m reben_pipeline.debug_metrics
"""
import sys, io
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, "D:/BTP/theisis_backup")

from reben_pipeline.metrics import MultiLabelMetricsCalculator, BEN_CLASSES_19
from reben_pipeline.dataset import create_synthetic_reben_dataframe, ReBENDataset

PASS = "[PASS]"
FAIL = "[FAIL]"


print("=" * 60)
print("§1 METRICS UNIT TESTS")
print("=" * 60)

calc = MultiLabelMetricsCalculator(num_classes=19)

# ── Test 1a: Sparse classes (only 2 of 19 active) ─────────────────────────
# Official reBEN protocol uses torchmetrics zero_division=0 across all 19 classes.
# When only 2 classes exist in the batch, the other 17 have 0/0 -> F1=0.
# Macro average = (100% * 2 + 0% * 17) / 19 = 10.53%. Micro average = 100.0%.
targets_sparse = torch.zeros(10, 19)
targets_sparse[:, 0] = 1.0
targets_sparse[:, 5] = 1.0
logits_sparse = torch.zeros(10, 19)
logits_sparse[:, 0] = 5.0
logits_sparse[:, 5] = 5.0
res1a = calc.compute_all(logits_sparse, targets_sparse)
ok1a = abs(res1a["f1_micro"] - 100.0) < 1.0 and abs(res1a["f1_macro"] - 10.53) < 0.2
print(f"Test 1a Sparse active (2/19) F1(micro)={res1a['f1_micro']:.1f}% F1(macro)={res1a['f1_macro']:.1f}% -- {PASS if ok1a else FAIL}")
print("        (Under official zero_division=0, 17 absent classes = 0.0 -> macro = 2/19 = 10.5%)")

# ── Test 1b: Full 19-class coverage (all 19 classes present, perfect preds) ─
# On a representative test set (like reBEN test split), all 19 classes are present.
targets_full = torch.eye(19).repeat(2, 1)  # 38 samples, each class has 2 positives
logits_full = targets_full * 10.0 - 5.0    # +5.0 for positive, -5.0 for negative
res1b = calc.compute_all(logits_full, targets_full)
ok1b = abs(res1b["f1_macro"] - 100.0) < 1.0 and abs(res1b["f1_micro"] - 100.0) < 1.0
print(f"Test 1b All 19 classes active F1(macro)={res1b['f1_macro']:.1f}% F1(micro)={res1b['f1_micro']:.1f}% -- {PASS if ok1b else FAIL}")

# ── Test 2: All-negative logits should give F1=0% (expected, not a bug) ──────
logits_neg = torch.full((10, 19), -2.0)
res2 = calc.compute_all(logits_neg, targets_sparse)
ok2 = res2["f1_micro"] < 1.0
print(f"Test 2  All-neg logits  F1(micro)={res2['f1_micro']:.1f}%  -- {PASS if ok2 else FAIL} (0% is correct here)")

# ── Test 3: Near-zero logits with sigmoid fix ─────────────────────────────────
# sigmoid(0.8) = 0.69 > 0.5 → pred=1 → should match target
targets3_t = torch.zeros(10, 19); targets3_t[:, 2] = 1.0
logits3 = torch.zeros(10, 19)
logits3[:, 2] = 0.8   # sigmoid(0.8) ≈ 0.69 → above 0.5 → pred=1
res3 = calc.compute_all(logits3, targets3_t)
ok3 = res3["f1_micro"] > 90.0   # class-2 should be predicted correctly (micro ignores empty classes)
print(f"Test 3  Near-zero logit F1(micro)={res3['f1_micro']:.1f}%  -- {PASS if ok3 else FAIL} (sigmoid fix check)")
print(f"        (sigmoid(0.8)=0.69>0.5, so class-2 preds=1, micro-F1 should be 100%)")

print()
print("=" * 60)
print("§1 DATASET LABEL PARSING UNIT TESTS")
print("=" * 60)

# ── Test 4: Labels survive CSV round-trip ─────────────────────────────────────
df = create_synthetic_reben_dataframe(num_samples=20, seed=42)
print(f"Raw label[0]   (pre-CSV):  {df['labels'].iloc[0]}  (type={type(df['labels'].iloc[0]).__name__})")

# Simulate CSV round-trip
buf = io.StringIO()
df.to_csv(buf, index=False)
buf.seek(0)
df2 = pd.read_csv(buf)
print(f"Label[0] (post-CSV):       {df2['labels'].iloc[0]}  (type={type(df2['labels'].iloc[0]).__name__})")

ds = ReBENDataset(df2, normalize=False, is_synthetic=True)
label_sums = torch.stack([ds[i]["targets"] for i in range(len(ds))]).sum(dim=1)

ok4_nonzero = (label_sums > 0).all().item()
ok4_types   = (label_sums >= 1).all().item()
print(f"Test 4a All labels non-zero after CSV round-trip  — {PASS if ok4_nonzero else FAIL}")
print(f"        label sums: min={label_sums.min().item():.0f}  max={label_sums.max().item():.0f}  mean={label_sums.mean().item():.2f}")

# ── Test 5: Direct list (no CSV) still works ──────────────────────────────────
ds_direct = ReBENDataset(df, normalize=False, is_synthetic=True)
label_sums_direct = torch.stack([ds_direct[i]["targets"] for i in range(len(ds_direct))]).sum(dim=1)
ok5 = (label_sums_direct > 0).all().item()
print(f"Test 5  Direct list labels non-zero               — {PASS if ok5 else FAIL}")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
all_pass = all([ok1a, ok1b, ok2, ok3, ok4_nonzero, ok5])
if all_pass:
    print("ALL TESTS PASSED -- metrics and dataset logic fully verified!")
else:
    failed = [t for t, ok in [("T1a",ok1a),("T1b",ok1b),("T2",ok2),("T3",ok3),("T4",ok4_nonzero),("T5",ok5)] if not ok]
    print(f"FAILED: {failed} -- do NOT proceed until fixed")

