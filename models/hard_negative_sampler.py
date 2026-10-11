"""
models/hard_negative_sampler.py
================================
Novelty 1 — Cluster-Aware Hard Negative Mining for Cross-Modal SupCon.

Two complementary improvements over vanilla SupCon:

  (a) CrossModalAlignmentLoss
      Symmetric InfoNCE (CLIP-style) between paired SAR and optical projections.
      Explicitly pulls each SAR patch closer to its PAIRED optical patch,
      independent of cluster labels. This directly optimizes cross-modal
      alignment at the instance level — something standard SupCon does not do,
      since SupCon only groups same-cluster samples without caring which SAR
      embedding matches which optical embedding within a cluster.

      Reference: Radford et al., "Learning Transferable Visual Models From
      Natural Language Supervision", ICML 2021. Adapted for SAR-Optical pairs.

  (b) HardNegativeWeightedSupCon
      Reweights the standard SupCon negative contribution by GMM cluster
      proximity. Negatives drawn from clusters that are geometrically close
      to the anchor's cluster in hand-crafted feature space receive higher
      weight, forcing the encoder to actively maintain separation between
      structurally similar but semantically distinct agricultural land types.

      When proximity weights are uniform (alpha=1.0 or no GMM means set),
      this loss degenerates exactly to standard SupCon — fully backwards
      compatible.

      Reference: Robinson et al., "Contrastive Learning with Hard Negative
      Samples", ICLR 2021. Adapted for multi-modal cluster-label setting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# (a) CROSS-MODAL ALIGNMENT LOSS  (InfoNCE / CLIP-style)
# ---------------------------------------------------------------------------

class CrossModalAlignmentLoss(nn.Module):
    """
    Symmetric InfoNCE cross-modal alignment loss for SAR-Optical pairs.

    For a batch of B paired patches, constructs a B×B similarity matrix
    between SAR and optical projections. The diagonal entries are the true
    pairs; all off-diagonal entries are treated as negatives. Minimising
    cross-entropy in both directions (SAR→Opt and Opt→SAR) is equivalent
    to maximising mutual information between the two modalities.

    Args:
        temperature: Softmax temperature (default 0.07, same as SupCon).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_sar: torch.Tensor, z_opt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_sar : [B, D] L2-normalised SAR projections.
            z_opt : [B, D] L2-normalised Optical projections.
        Returns:
            Scalar alignment loss.
        """
        device = z_sar.device
        B = z_sar.shape[0]

        # B×B cosine-similarity matrix (already L2-normalised → matmul = cosine)
        sim = torch.matmul(z_sar, z_opt.T) / self.temperature   # [B, B]

        # Ground truth: pair (i, i) is the positive for query i
        labels = torch.arange(B, device=device)

        # Symmetric cross-entropy: SAR→Optical and Optical→SAR
        loss_fwd = F.cross_entropy(sim,   labels)   # each SAR finds its optical
        loss_bwd = F.cross_entropy(sim.T, labels)   # each optical finds its SAR

        return (loss_fwd + loss_bwd) * 0.5


# ---------------------------------------------------------------------------
# (b) HARD-NEGATIVE-WEIGHTED SUPERVISED CONTRASTIVE LOSS
# ---------------------------------------------------------------------------

class HardNegativeWeightedSupCon(nn.Module):
    """
    SupCon loss with cluster-proximity hard-negative reweighting.

    Standard SupCon treats all in-batch negatives (different-cluster) equally.
    This variant builds a cluster proximity matrix P from GMM centroid cosine
    similarities and upweights negatives from nearby clusters:

        neg_weight(i, j) = alpha + (1 - alpha) * P[cluster_i, cluster_j]

    where alpha ∈ [0, 1] is the baseline weight.  Setting alpha=1 or not
    calling set_cluster_proximity() recovers exact vanilla SupCon.

    Args:
        temperature : Softmax temperature.
        alpha       : Baseline negative weight (1.0 = vanilla SupCon).
    """

    def __init__(self, temperature: float = 0.07, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.cluster_proximity: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def set_cluster_proximity(self, gmm_means: np.ndarray, device: torch.device):
        """
        Pre-compute cluster proximity matrix from GMM centroids.

        Args:
            gmm_means : [K, F] array of GMM cluster means in *feature space*.
            device    : Torch device to store the matrix on.
        """
        means = torch.tensor(gmm_means, dtype=torch.float32)
        means_n = F.normalize(means, dim=1)           # cosine-sim ready
        prox = torch.matmul(means_n, means_n.T)        # [K, K] ∈ [-1, 1]
        prox.fill_diagonal_(0.0)                       # self-proximity irrelevant
        prox = torch.clamp(prox, min=0.0)              # only upweight, never down
        self.cluster_proximity = prox.to(device)

    # ------------------------------------------------------------------
    def forward(
        self,
        z_sar: torch.Tensor,
        z_opt: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_sar   : [B, D] L2-normalised SAR projections.
            z_opt   : [B, D] L2-normalised Optical projections.
            labels  : [B] LongTensor of GMM cluster labels.
        Returns:
            Scalar hard-negative-weighted SupCon loss.
        """
        device = z_sar.device
        B = z_sar.shape[0]

        # Concatenate both modalities → [2B, D]
        features = torch.cat([z_sar, z_opt], dim=0)          # [2B, D]
        labels_cat = torch.cat([labels, labels]).view(-1, 1)  # [2B, 1]

        # ---- Positive / self masks ----------------------------------------
        label_mask = torch.eq(labels_cat, labels_cat.T).float()  # [2B, 2B]
        self_mask  = torch.eye(2 * B, device=device)
        positive_mask = label_mask - self_mask                   # [2B, 2B]

        # ---- Similarity matrix -------------------------------------------
        sim = torch.matmul(features, features.T) / self.temperature  # [2B, 2B]
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - sim_max.detach()  # numerical stability via max subtraction

        # ---- Hard-negative weighting -------------------------------------
        logits_mask = 1.0 - self_mask   # exclude self from denominator

        if self.cluster_proximity is not None:
            lbl = labels_cat.squeeze(1)  # [2B]
            # [2B, 2B] proximity between each pair's clusters
            prox_w = self.cluster_proximity[lbl][:, lbl]
            # neg_weight: positives + self keep weight 1; negatives get alpha+(1-a)*p
            neg_mask   = 1.0 - label_mask
            hard_w     = self.alpha + (1.0 - self.alpha) * prox_w
            hard_w     = hard_w * neg_mask + label_mask          # pos stays 1
            exp_logits = torch.exp(logits) * logits_mask * hard_w
        else:
            exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # ---- Average over valid positives --------------------------------
        sum_pos = positive_mask.sum(1)
        valid   = sum_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        sum_pos = torch.where(valid, sum_pos, torch.ones_like(sum_pos))
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / sum_pos
        return -mean_log_prob_pos[valid].mean()
