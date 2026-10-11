"""
ierc.py - Iterative Embedding-Anchored Re-Clustering (IERC)
===========================================================
Extract fused embeddings of the TRAINING set -> fit a GMM -> build a cluster
proximity matrix -> hand it to the CAHN loss. Re-done every few epochs.

Changes vs. previous version
----------------------------
* Default covariance_type="diag" (was "full"). 16 full-covariance components in
  128-D is ~134k parameters fit on ~4k points; diag is ~4k. Optional PCA first.
* The cluster model is wrapped in LatentClusterer so that PCA (if used) is applied
  consistently at fit time AND predict time.
* same_cluster_proximity (default 1.0, was an implicit 0.0): a pair with DIFFERENT
  labels that falls in the SAME GMM cluster is the hardest negative, so it now gets
  the largest weight instead of the smallest. Ablate 0.0 vs 1.0.
* Feature extraction restores the model's train/eval mode and can be capped.
* Controller state can be saved/restored (needed for exact resume).
"""

import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


@torch.no_grad()
def extract_latent_features(model, dataloader, device, max_batches=None):
    """Normalised SAR / optical / fused embeddings for (x, y) batches (x: B x 12 x H x W)."""
    was_training = model.training
    model.eval()
    z_sar, z_opt, z_fused, targets = [], [], [], []
    try:
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            if not isinstance(batch, (tuple, list)):
                raise TypeError("Expected (x, y) batches from the loader.")
            x, y = batch[0], batch[1]
            x = x.to(device).float()
            out = model(sar_img=x[:, :2], opt_img=x[:, 2:])
            z_sar.append(out["z_sar"].float().cpu().numpy())
            z_opt.append(out["z_opt"].float().cpu().numpy())
            z_fused.append(out["z_fused"].float().cpu().numpy())
            targets.append(y.cpu().numpy())
    finally:
        model.train(was_training)
    return {
        "z_sar": np.vstack(z_sar),
        "z_opt": np.vstack(z_opt),
        "z_fused": np.vstack(z_fused),
        "targets": np.vstack(targets),
    }


class LatentClusterer:
    """GMM (optionally after PCA) with a consistent fit/predict pipeline."""

    def __init__(self, n_clusters=16, covariance_type="diag", pca_dim=None, seed=42,
                 reg_covar=1e-4, max_iter=200):
        self.n_clusters = n_clusters
        self.covariance_type = covariance_type
        self.pca_dim = pca_dim
        self.seed = seed
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.pca = None
        self.gmm = None

    def _project(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.pca.transform(X) if self.pca is not None else X

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if len(X) < self.n_clusters:
            raise ValueError(f"Need >= {self.n_clusters} samples to fit, got {len(X)}")
        self.pca = None
        if self.pca_dim:
            self.pca = PCA(n_components=min(self.pca_dim, X.shape[1]),
                           random_state=self.seed).fit(X)
        self.gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            random_state=self.seed,
        ).fit(self._project(X))
        return self

    def predict(self, X):
        return self.gmm.predict(self._project(X))

    def predict_proba(self, X):
        return self.gmm.predict_proba(self._project(X))

    @property
    def centers_(self):
        return self.gmm.means_

    means_ = centers_  # alias for older code

    @property
    def converged_(self):
        return bool(self.gmm.converged_)


def fit_latent_gmm(embeddings, n_clusters=16, seed=42, covariance_type="diag", pca_dim=None):
    """Convenience wrapper: returns (clusterer, hard labels, responsibilities)."""
    c = LatentClusterer(n_clusters, covariance_type, pca_dim, seed).fit(embeddings)
    return c, c.predict(embeddings), c.predict_proba(embeddings)


def compute_cluster_proximity_matrix(clusterer_or_centers, same_cluster_value: float = 1.0):
    """
    Proximity in [0, 1] between clusters: 1 - d/d_max on L2-normalised centres.
    Diagonal = same_cluster_value (see module docstring).
    Accepts a LatentClusterer, a fitted sklearn GMM, or a (K, D) array of centres.
    """
    centers = getattr(clusterer_or_centers, "centers_", None)
    if centers is None:
        centers = getattr(clusterer_or_centers, "means_", None)
    if centers is None:
        centers = clusterer_or_centers
    centers = np.asarray(centers, dtype=np.float64)

    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    c = centers / np.maximum(norms, 1e-12)
    dist = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    d_max = dist.max()
    prox = np.ones_like(dist) if d_max < 1e-9 else 1.0 - dist / d_max
    np.fill_diagonal(prox, same_cluster_value)
    return torch.tensor(prox, dtype=torch.float32)


class IERCController:
    """Extract -> re-cluster -> proximity matrix. Keeps the fitted clusterer for per-batch prediction."""

    def __init__(self, n_clusters=16, seed=42, covariance_type="diag", pca_dim=None,
                 same_cluster_proximity=1.0, max_batches=None):
        self.n_clusters = n_clusters
        self.seed = seed
        self.covariance_type = covariance_type
        self.pca_dim = pca_dim
        self.same_cluster_proximity = same_cluster_proximity
        self.max_batches = max_batches
        self.clusterer = None
        self.proximity_matrix = None
        self.last_stats = {}

    @property
    def ready(self):
        return self.clusterer is not None

    @property
    def gmm(self):  # backwards-compatible name
        return self.clusterer

    def predict(self, z):
        return self.clusterer.predict(z)

    def step(self, model, train_loader, device):
        """One IERC round on TRAINING data only. Returns (train_labels, proximity, clusterer)."""
        t0 = time.time()
        feats = extract_latent_features(model, train_loader, device, self.max_batches)
        z = feats["z_fused"]
        clusterer = LatentClusterer(self.n_clusters, self.covariance_type, self.pca_dim,
                                    self.seed).fit(z)
        labels = clusterer.predict(z)
        sizes = np.bincount(labels, minlength=self.n_clusters)
        self.clusterer = clusterer
        self.proximity_matrix = compute_cluster_proximity_matrix(
            clusterer, self.same_cluster_proximity)
        self.last_stats = {
            "n_fit_samples": int(len(z)),
            "converged": clusterer.converged_,
            "n_iter": int(clusterer.gmm.n_iter_),
            "cluster_size_min": int(sizes.min()),
            "cluster_size_median": float(np.median(sizes)),
            "cluster_size_max": int(sizes.max()),
            "n_empty_clusters": int((sizes == 0).sum()),
            "seconds": round(time.time() - t0, 1),
        }
        return labels, self.proximity_matrix, clusterer

    def state_dict(self):
        return {"clusterer": self.clusterer, "proximity_matrix": self.proximity_matrix,
                "last_stats": self.last_stats}

    def load_state_dict(self, state):
        self.clusterer = state.get("clusterer")
        self.proximity_matrix = state.get("proximity_matrix")
        self.last_stats = state.get("last_stats", {})