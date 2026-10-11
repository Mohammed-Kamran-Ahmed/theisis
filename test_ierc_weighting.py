
import torch
from reben_pipeline.losses import ClusterAwareHardNegativeSupCon

torch.manual_seed(42)

# Four samples, with two-dimensional embeddings
features = torch.nn.functional.normalize(
    torch.randn(4, 8), dim=1
)
targets = torch.tensor([
    [1., 0., 0.],
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 1., 0.],
])

# Two GMM clusters
cluster_labels = torch.tensor([0, 0, 1, 1])

loss_fn = ClusterAwareHardNegativeSupCon(temperature=0.2)

# CAHN without IERC proximity weighting
loss_without_ierc = loss_fn(
    features, targets,
    multi_label=True,
    cluster_labels=cluster_labels,
)

# Add a proximity matrix between the two clusters
proximity = torch.tensor([
    [0.0, 1.0],
    [1.0, 0.0],
])
loss_fn.set_proximity_matrix(proximity)

# CAHN with IERC proximity weighting
loss_with_ierc = loss_fn(
    features, targets,
    multi_label=True,
    cluster_labels=cluster_labels,
)

print(f"Without IERC: {loss_without_ierc.item():.6f}")
print(f"With IERC:    {loss_with_ierc.item():.6f}")
print("Finite losses:", torch.isfinite(loss_without_ierc).item()
      and torch.isfinite(loss_with_ierc).item())
print("Loss changed:", not torch.isclose(
    loss_without_ierc, loss_with_ierc
).item())

