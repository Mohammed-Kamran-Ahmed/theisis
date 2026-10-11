
import json
from pathlib import Path

import numpy as np
import torch

from configilm.extra.BENv2_utils import NEW_LABELS

from train_baseline import (
    PROJECT_ROOT,
    NUM_CLASSES,
    build_data_module,
    ResNet50Baseline,
)


# ------------------------------------------------------------
# Paths and settings
# ------------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reben_baseline_10epoch"
CHECKPOINT_PATH = OUTPUT_DIR / "best_resnet50_12ch.pt"

REPORT_PATH = OUTPUT_DIR / "classwise_test_metrics.json"

BATCH_SIZE = 2
NUM_WORKERS = 0


# ------------------------------------------------------------
# Average precision
# ------------------------------------------------------------

def average_precision(
    targets: np.ndarray,
    scores: np.ndarray,
) -> float:
    """
    Calculate average precision from binary targets and scores.

    Equal scores are handled as a group rather than being
    arbitrarily ordered.
    """

    targets = np.asarray(targets, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    positives = int(targets.sum())

    if positives == 0:
        return float("nan")

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_targets = targets[order]

    # Evaluate precision at the end of each group of tied scores.
    group_ends = np.r_[
        np.where(np.diff(sorted_scores) != 0)[0],
        len(sorted_scores) - 1,
    ]

    cumulative_tp = np.cumsum(sorted_targets)[group_ends]
    cumulative_fp = (
        group_ends + 1
    ) - cumulative_tp

    precision = cumulative_tp / (
        cumulative_tp + cumulative_fp
    )

    recall = cumulative_tp / positives

    previous_recall = np.r_[0.0, recall[:-1]]

    ap = np.sum(
        (recall - previous_recall) * precision
    )

    return float(ap)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

@torch.no_grad()
def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run train_baseline.py first."
        )

    if len(NEW_LABELS) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} class names, "
            f"but ConfigILM provides {len(NEW_LABELS)}."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("CLASS-WISE RESNET-50 BASELINE EVALUATION")
    print("=" * 65)
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # Build the verified data module.
    dm = build_data_module(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Initialize the test split.
    dm.setup("test")
    test_loader = dm.test_dataloader()

    # No pretrained download is needed because the checkpoint
    # contains the complete trained model state.
    model = ResNet50Baseline(
        num_classes=NUM_CLASSES,
        pretrained=False,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        "Best validation macro AP: "
        f"{checkpoint['val_macro_ap']:.6f}"
    )

    all_targets = []
    all_scores = []
    total_loss = 0.0
    total_samples = 0

    criterion = torch.nn.BCEWithLogitsLoss()

    # Inference on the test set.
    for batch_idx, (images, targets) in enumerate(
        test_loader,
        start=1,
    ):
        images = images.to(
            device=device,
            dtype=torch.float32,
        )

        targets = targets.to(
            device=device,
            dtype=torch.float32,
        )

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_targets.append(
            targets.cpu().numpy()
        )

        # AP is ranking-based, so logits can be used directly.
        all_scores.append(
            logits.cpu().numpy()
        )

        if batch_idx % 100 == 0:
            print(
                f"Processed {total_samples} test patches",
                flush=True,
            )

    if total_samples == 0:
        raise RuntimeError("Test loader returned no samples.")

    targets_array = np.concatenate(
        all_targets,
        axis=0,
    )

    scores_array = np.concatenate(
        all_scores,
        axis=0,
    )

    if targets_array.shape[1] != len(NEW_LABELS):
        raise ValueError(
            f"Expected {len(NEW_LABELS)} target columns, "
            f"got {targets_array.shape[1]}."
        )

    # --------------------------------------------------------
    # Per-class report
    # --------------------------------------------------------

    class_results = []
    valid_class_aps = []

    print("\nCLASS-WISE TEST RESULTS")
    print("-" * 100)
    print(
        f"{'Class':<65} {'Positives':>10} {'AP':>12}"
    )
    print("-" * 100)

    for class_idx, class_name in enumerate(NEW_LABELS):
        class_targets = targets_array[:, class_idx]
        class_scores = scores_array[:, class_idx]

        positive_count = int(class_targets.sum())

        if positive_count == 0:
            ap = None
            ap_display = "N/A"
        else:
            ap = average_precision(
                class_targets,
                class_scores,
            )

            valid_class_aps.append(ap)
            ap_display = f"{ap:.6f}"

        print(
            f"{class_name:<65} "
            f"{positive_count:>10} "
            f"{ap_display:>12}"
        )

        class_results.append(
            {
                "class_index": class_idx,
                "class_name": class_name,
                "positive_samples": positive_count,
                "average_precision": ap,
            }
        )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    macro_ap = (
        float(np.mean(valid_class_aps))
        if valid_class_aps
        else None
    )

    micro_ap = average_precision(
        targets_array.reshape(-1),
        scores_array.reshape(-1),
    )

    test_loss = total_loss / total_samples

    print("-" * 100)
    print(f"Test samples: {total_samples}")
    print(f"Test BCE loss: {test_loss:.6f}")
    print(
        f"Macro AP: "
        f"{macro_ap:.6f}" if macro_ap is not None
        else "Macro AP: N/A"
    )
    print(f"Micro AP: {micro_ap:.6f}")
    print(
        f"Classes with positive test samples: "
        f"{len(valid_class_aps)}/{NUM_CLASSES}"
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report = {
        "model": "ResNet-50",
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_macro_ap": float(
            checkpoint["val_macro_ap"]
        ),
        "test_samples": total_samples,
        "test_bce_loss": test_loss,
        "test_macro_ap": macro_ap,
        "test_micro_ap": micro_ap,
        "classes_with_positive_test_samples": len(
            valid_class_aps
        ),
        "num_classes": NUM_CLASSES,
        "classwise_results": class_results,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            allow_nan=False,
        )

    print(f"\nReport saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()
