"""
metrics.py - Multi-label evaluation metrics for the reBEN benchmark
===================================================================
Stateless, sklearn-only implementation (no torchmetrics objects, so nothing
accumulates between epochs).

Conventions (identical for the dual encoder and the baseline):
  * Inputs are RAW LOGITS. Sigmoid is always applied here.
  * AP (macro)  : mean of per-class AP over classes that have >= 1 positive in
                  the evaluated split ("present" classes). `n_classes_present`
                  is returned so the number is never reported without its
                  denominator.
  * AP (micro)  : sklearn micro-averaged AP over all (sample, class) pairs.
  * F1          : fixed decision threshold (default 0.5, strict '>').
                  `f1_macro`         = mean over ALL classes, absent classes
                                       scored 0 (zero_division=0). This is the
                                       torchmetrics/reBEN-style convention and
                                       equals `f1_macro_present` only when all
                                       19 classes occur in the split.
                  `f1_macro_present` = mean over present classes only.
  * Per-class values for absent classes are None (not 0.0).
"""

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

BEN_CLASSES_19 = [
    'Agro-forestry areas',
    'Arable land',
    'Beaches, dunes, sands',
    'Broad-leaved forest',
    'Coastal wetlands',
    'Complex cultivation patterns',
    'Coniferous forest',
    'Industrial or commercial units',
    'Inland waters',
    'Inland wetlands',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Marine waters',
    'Mixed forest',
    'Moors, heathland and sclerophyllous vegetation',
    'Natural grassland and sparsely vegetated areas',
    'Pastures',
    'Permanent crops',
    'Transitional woodland, shrub',
    'Urban fabric',
]


class MultiLabelMetricsCalculator:
    def __init__(self, num_classes: int = 19, threshold: float = 0.5):
        self.num_classes = num_classes
        self.threshold = threshold

    def compute_all(self, logits, targets) -> dict:
        """
        Args:
            logits:  (N, C) raw logits (torch tensor or ndarray).
            targets: (N, C) binary ground truth.
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)

        probs = torch.sigmoid(logits.detach().float()).cpu().numpy()
        t = (targets.detach().cpu().numpy() > 0.5).astype(int)

        if probs.shape != t.shape:
            raise ValueError(f"logits {probs.shape} and targets {t.shape} differ in shape")
        if not np.isfinite(probs).all():
            raise FloatingPointError("Non-finite predictions passed to metrics.")

        n_cls = t.shape[1]
        present = t.sum(axis=0) > 0
        n_present = int(present.sum())
        if n_present == 0:
            raise ValueError("No positive labels at all in this split.")

        # ---- Average precision ----
        class_ap = np.full(n_cls, np.nan)
        for c in np.where(present)[0]:
            class_ap[c] = average_precision_score(t[:, c], probs[:, c])
        ap_macro = float(np.nanmean(class_ap[present])) * 100.0
        ap_micro = float(average_precision_score(t, probs, average="micro")) * 100.0

        # ---- Thresholded metrics (strict '>' so sigmoid(0)=0.5 is negative) ----
        preds = (probs > self.threshold).astype(int)
        class_f1 = f1_score(t, preds, average=None, zero_division=0)
        f1_macro = float(class_f1.mean()) * 100.0
        f1_macro_present = float(class_f1[present].mean()) * 100.0
        f1_micro = float(f1_score(t, preds, average="micro", zero_division=0)) * 100.0

        prec_macro = float(precision_score(t, preds, average="macro", zero_division=0)) * 100.0
        prec_micro = float(precision_score(t, preds, average="micro", zero_division=0)) * 100.0
        rec_macro = float(recall_score(t, preds, average="macro", zero_division=0)) * 100.0
        rec_micro = float(recall_score(t, preds, average="micro", zero_division=0)) * 100.0

        names = BEN_CLASSES_19[:n_cls]
        classwise_ap = {
            n: (float(class_ap[i]) * 100.0 if present[i] else None) for i, n in enumerate(names)
        }
        classwise_f1 = {
            n: (float(class_f1[i]) * 100.0 if present[i] else None) for i, n in enumerate(names)
        }

        return {
            "ap_macro": ap_macro,
            "ap_micro": ap_micro,
            "f1_macro": f1_macro,
            "f1_macro_present": f1_macro_present,
            "f1_micro": f1_micro,
            "precision_macro": prec_macro,
            "precision_micro": prec_micro,
            "recall_macro": rec_macro,
            "recall_micro": rec_micro,
            "hamming_loss": float(hamming_loss(t, preds)),
            "n_classes_present": n_present,
            "n_samples": int(t.shape[0]),
            "classwise_ap": classwise_ap,
            "classwise_f1": classwise_f1,
        }