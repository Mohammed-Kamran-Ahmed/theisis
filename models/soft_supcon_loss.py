"""
models/soft_supcon_loss.py
=============================
Fix #3 for the overfitting gap: soft-label Supervised Contrastive Loss.

Why: your current SupCon loss (Eq. 2 in the paper) treats GMM cluster
assignment as ground truth -- a hard 0/1 label. But your GMM clusters are
themselves noisy pseudo-labels from an 8-component mixture fit on
hand-crafted features, with severe imbalance (cluster 3: n=5, cluster 2:
n=111) and (per your PCA plot) visibly overlapping, non-convex boundaries
between clusters 0/1/5/7. Forcing hard positive/negative splits on
ambiguous points teaches the encoder to memorize arbitrary GMM boundary
decisions rather than real structure -- a likely contributor to the
0.855 (train) -> 0.153 (test) silhouette gap.

This soft variant weights each potential positive pair by the GMM
posterior similarity between two samples' cluster-probability vectors,
instead of a hard same-cluster / different-cluster split. Points near a
GMM decision boundary contribute partial, down-weighted positive/negative
signal instead of full-strength contradictory signal.

Drop-in replacement for your existing SupCon loss call in
stage4_contrastive_learning.py -- same inputs/shapes, one extra argument.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSupConLoss(nn.Module):
    """
    Soft-label Supervised Contrastive Loss.

    Standard SupCon (Khosla et al. 2020) needs a hard label per sample and
    builds P(i) = {samples sharing i's label}. This version instead takes
    the full GMM posterior probability vector per sample (soft cluster
    membership, shape [N, K]) and computes a continuous "positiveness"
    weight for every pair as the cosine similarity (or dot product, since
    posteriors are already non-negative and sum to 1) between their
    posterior vectors. Two samples that are both ~90% cluster 2 get a
    strong positive weight; two samples split 50/50 between clusters 2
    and 5 get a much weaker, partial-credit weight instead of being
    forced into a hard bucket.

    Args:
        temperature: SupCon temperature (tau). Same default as standard SupCon.
        base_temperature: for loss scaling, matches the original SupCon paper's
            convention; leave at default unless you know you want to change it.
    """

    def __init__(self, temperature: float = 0.1, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, gmm_posteriors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: L2-normalized projection embeddings, shape [N, D]
                (same `z` you already pass to your current SupCon loss --
                e.g. the fused/averaged 64-dim SAR+optical projection, or
                call this once per modality if you keep them separate).
            gmm_posteriors: soft GMM cluster-membership probabilities per
                sample, shape [N, K] (K=8 in your case), rows summing to 1.
                This is `gmm.predict_proba(features)` output -- NOT
                `gmm.predict(features)` (which gives you the hard label
                you're using today).

        Returns:
            scalar loss tensor.
        """
        device = features.device
        N = features.shape[0]

        # Positive-pair weight matrix: how similar are sample i and j's
        # soft cluster memberships? Using dot product of posteriors is a
        # natural "soft co-membership" score bounded in [0, 1] since rows
        # are probability distributions (Cauchy-Schwarz-ish upper bound
        # reached only when i, j have identical posteriors).
        pos_weight = gmm_posteriors @ gmm_posteriors.T  # [N, N], in [0, 1]

        # Remove self-pairs from consideration
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        pos_weight = pos_weight.masked_fill(self_mask, 0.0)

        # Cosine similarity of embeddings (features assumed pre-normalized;
        # normalize defensively in case they aren't)
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature  # [N, N]

        # For numerical stability, subtract row-wise max before exponentiating
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim)
        exp_sim = exp_sim.masked_fill(self_mask, 0.0)

        # Denominator: sum over ALL other samples (standard SupCon denominator,
        # unweighted -- every sample is still a valid "candidate" in the softmax)
        denom = exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12)

        log_prob = sim - torch.log(denom)  # [N, N]

        # Weighted mean over positives, per-anchor, using soft weights
        # instead of a hard P(i) set
        weight_sum = pos_weight.sum(dim=1).clamp_min(1e-12)
        weighted_log_prob = (pos_weight * log_prob).sum(dim=1) / weight_sum

        loss = -(self.temperature / self.base_temperature) * weighted_log_prob

        # Anchors with ~zero total positive weight (e.g. a genuinely
        # isolated outlier with no similar posterior anywhere in the
        # batch) contribute ~0 signal rather than blowing up; guard anyway:
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        return loss.mean()


# ----------------------------------------------------------------------
# INTEGRATION NOTES for stage4_contrastive_learning.py
# ----------------------------------------------------------------------
# 1. You need GMM posteriors, not just hard labels. Wherever your current
#    code does something like:
#
#       cluster_labels = gmm.predict(feature_matrix)
#
#    also compute:
#
#       cluster_posteriors = gmm.predict_proba(feature_matrix)  # [N_total, 8]
#
#    and make sure your Dataset/DataLoader returns `cluster_posteriors[idx]`
#    (a length-8 float vector) alongside each sample, the same way it
#    currently returns the hard `cluster_labels[idx]`.
#
# 2. Swap the loss:
#
#       from models.soft_supcon_loss import SoftSupConLoss
#       criterion = SoftSupConLoss(temperature=0.1)
#       ...
#       loss = criterion(z_batch, posteriors_batch)
#
#    replacing your current hard-label SupCon call. Keep everything else
#    (optimizer, scheduler, hard-negative sampler if you've added it,
#    cross-modal alignment term) unchanged -- this only replaces how the
#    positive/negative weighting works, not the rest of the training loop.
#
# 3. This is a genuinely different run from your existing "Baseline" and
#    "Re-clustered" runs -- log it as its own row in Table 2 of the paper
#    (e.g. "Soft-label SupCon") so it slots into the ablation table your
#    reviewer feedback asked for, rather than overwriting a completed run.