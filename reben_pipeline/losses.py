"""Losses for the CAHN/IERC dual-encoder framework.

Changes vs. previous version
----------------------------
* CAHN is computed in the log domain (logsumexp with an explicit -inf self-mask)
  instead of exp(...).clamp_min(1e-8); same maths, no silent clamping.
* nan_to_num() removed: a non-finite loss now raises FloatingPointError instead of
  being turned into 0 and silently ignored.
* `last_valid_anchor_frac` records which fraction of anchors in the batch had at
  least one positive partner (Jaccard >= threshold). With tiny batches this is
  often low, and then the contrastive term is barely doing anything.
* Terms with lambda == 0 are skipped entirely (needed for clean ablations).
* The 0.25 weight of the per-modality classification heads is now a parameter.

Loss definition (multi-label mode)
----------------------------------
For anchor i, with cosine similarities s_ij / tau over j != i:
    positives P(i) = { j : Jaccard(y_i, y_j) >= affinity_threshold }
    negatives N(i) = the remaining j != i
    w_ij = 1 + Jaccard(y_i, y_j)                 for j in N(i)   (1 for positives)
    w_ij *= 1 + proximity_scale * prox[c_i, c_j]  for j in N(i), if IERC labels given
    L_i  = - mean_{p in P(i)}  [ s_ip - log sum_{j != i} w_ij exp(s_ij) ]
So IERC only re-weights the negatives of the denominator; positives are untouched.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusterAwareHardNegativeSupCon(nn.Module):
    """Supervised contrastive loss with optional IERC cluster-proximity weighting."""

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        proximity_matrix: Optional[torch.Tensor] = None,
        affinity_threshold: float = 0.3,
        proximity_scale: float = 1.0,
    ):
        super().__init__()
        if temperature <= 0 or base_temperature <= 0:
            raise ValueError("Temperatures must be positive.")
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.affinity_threshold = affinity_threshold
        self.proximity_scale = proximity_scale
        self.proximity_matrix = None
        self.last_valid_anchor_frac = 0.0
        if proximity_matrix is not None:
            self.set_proximity_matrix(proximity_matrix)

    def set_proximity_matrix(self, matrix: Optional[torch.Tensor]) -> None:
        """Set or clear a square (num_clusters, num_clusters) proximity matrix."""
        if matrix is None:
            self.proximity_matrix = None
            return
        matrix = torch.as_tensor(matrix).detach().float()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("proximity_matrix must be square.")
        self.proximity_matrix = matrix

    def _pair_proximity(self, labels: torch.Tensor, device, dtype) -> torch.Tensor:
        labels = torch.as_tensor(labels, device=device).long().view(-1)
        prox = self.proximity_matrix.to(device=device, dtype=dtype)
        if labels.numel() and (labels.min().item() < 0 or labels.max().item() >= prox.shape[0]):
            raise ValueError("A cluster label is outside the proximity matrix.")
        return prox[labels[:, None], labels[None, :]].clamp(0.0, 1.0)

    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        multi_label: bool = False,
        cluster_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1:
            self.last_valid_anchor_frac = 0.0
            return features.sum() * 0.0

        features = features.float()
        sim = torch.matmul(features, features.T) / self.temperature
        eye = torch.eye(batch_size, dtype=torch.bool, device=device)
        not_self = (~eye).to(sim.dtype)

        if multi_label:
            targets = targets.to(device=device, dtype=sim.dtype)
            if targets.ndim != 2 or targets.shape[0] != batch_size:
                raise ValueError("multi-label targets must have shape (B, C).")
            intersection = torch.matmul(targets, targets.T)
            cardinality = targets.sum(dim=1, keepdim=True)
            union = cardinality + cardinality.T - intersection
            affinity = intersection / union.clamp_min(1e-7)

            pos_mask = (affinity >= self.affinity_threshold).to(sim.dtype) * not_self
            neg_mask = (affinity < self.affinity_threshold).to(sim.dtype) * not_self
            weight = torch.where(neg_mask > 0, 1.0 + affinity, torch.ones_like(affinity))

            if self.proximity_matrix is not None and cluster_labels is not None:
                if torch.as_tensor(cluster_labels).numel() != batch_size:
                    raise ValueError("Provide one cluster label per batch sample.")
                pair_prox = self._pair_proximity(cluster_labels, device, sim.dtype)
                weight = torch.where(
                    neg_mask > 0, weight * (1.0 + self.proximity_scale * pair_prox), weight
                )
        else:
            labels = targets.reshape(-1).to(device=device)
            if labels.numel() != batch_size:
                raise ValueError("Single-label targets must have shape (B,).")
            pos_mask = torch.eq(labels[:, None], labels[None, :]).to(sim.dtype) * not_self
            neg_mask = (1.0 - pos_mask) * not_self
            weight = torch.ones_like(sim)
            if self.proximity_matrix is not None:
                pair_prox = self._pair_proximity(labels, device, sim.dtype)
                weight = torch.where(neg_mask > 0, 1.0 + self.proximity_scale * pair_prox, weight)

        # log sum_{j != i} w_ij exp(s_ij)
        masked = sim.masked_fill(eye, float("-inf")) + torch.log(weight.clamp_min(1e-12))
        log_den = torch.logsumexp(masked, dim=1, keepdim=True)
        log_prob = sim - log_den

        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        self.last_valid_anchor_frac = float(valid.float().mean().item())
        if not valid.any():
            return features.sum() * 0.0  # differentiable zero

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count.clamp_min(1.0)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos[valid].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite CAHN loss (check embeddings / learning rate).")
        return loss


class CrossModalInfoNCE(nn.Module):
    """Symmetric bidirectional InfoNCE between paired SAR and optical embeddings."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature

    def forward(self, z_sar: torch.Tensor, z_opt: torch.Tensor) -> torch.Tensor:
        if z_sar.shape != z_opt.shape:
            raise ValueError("z_sar and z_opt must have the same shape.")
        batch_size = z_sar.shape[0]
        if batch_size <= 1:
            return (z_sar.sum() + z_opt.sum()) * 0.0
        sims = torch.matmul(z_sar.float(), z_opt.float().T) / self.temperature
        labels = torch.arange(batch_size, device=z_sar.device)
        return 0.5 * (F.cross_entropy(sims, labels) + F.cross_entropy(sims.T, labels))


class JointDualEncoderLoss(nn.Module):
    """CAHN + cross-modal alignment + multi-label classification."""

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_cahn: float = 1.0,
        lambda_align: float = 0.5,
        lambda_cls: float = 1.0,
        lambda_modality_cls: float = 0.25,
        pos_weight: Optional[torch.Tensor] = None,
        affinity_threshold: float = 0.3,
        proximity_scale: float = 1.0,
    ):
        super().__init__()
        self.cahn_loss = ClusterAwareHardNegativeSupCon(
            temperature=temperature,
            affinity_threshold=affinity_threshold,
            proximity_scale=proximity_scale,
        )
        self.align_loss = CrossModalInfoNCE(temperature=temperature)
        self.cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.lambda_cahn = lambda_cahn
        self.lambda_align = lambda_align
        self.lambda_cls = lambda_cls
        self.lambda_modality_cls = lambda_modality_cls

    def forward(
        self,
        z_sar: torch.Tensor,
        z_opt: torch.Tensor,
        logits_fused: torch.Tensor,
        targets: torch.Tensor,
        logits_sar: Optional[torch.Tensor] = None,
        logits_opt: Optional[torch.Tensor] = None,
        multi_label: bool = True,
        cluster_labels: Optional[torch.Tensor] = None,
    ):
        """Returns (total_loss, dict of python floats)."""
        zero = logits_fused.sum() * 0.0
        targets = targets.float()

        loss_cahn, anchor_frac = zero, 0.0
        if self.lambda_cahn > 0:
            l_s = self.cahn_loss(z_sar, targets, multi_label=multi_label,
                                 cluster_labels=cluster_labels)
            frac_s = self.cahn_loss.last_valid_anchor_frac
            l_o = self.cahn_loss(z_opt, targets, multi_label=multi_label,
                                 cluster_labels=cluster_labels)
            frac_o = self.cahn_loss.last_valid_anchor_frac
            loss_cahn = 0.5 * (l_s + l_o)
            anchor_frac = 0.5 * (frac_s + frac_o)

        loss_align = self.align_loss(z_sar, z_opt) if self.lambda_align > 0 else zero

        loss_cls = self.cls_loss(logits_fused, targets)
        if logits_sar is not None and logits_opt is not None and self.lambda_modality_cls > 0:
            loss_cls = loss_cls + self.lambda_modality_cls * (
                self.cls_loss(logits_sar, targets) + self.cls_loss(logits_opt, targets)
            )

        total = (
            self.lambda_cahn * loss_cahn
            + self.lambda_align * loss_align
            + self.lambda_cls * loss_cls
        )
        loss_dict = {
            "total_loss": float(total.detach().item()),
            "cahn_loss": float(loss_cahn.detach().item()),
            "align_loss": float(loss_align.detach().item()),
            "cls_loss": float(loss_cls.detach().item()),
            "cahn_valid_anchor_frac": float(anchor_frac),
        }
        return total, loss_dict