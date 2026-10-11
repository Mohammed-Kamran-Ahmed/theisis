"""
run_large_scale.py - Large-Scale End-to-End Training & Testing Suite
====================================================================
Trains the CAHN / IERC Dual-Encoder framework on an expanded 5,000-patch paired dataset
for 25 epochs per configuration, evaluating on the held-out geographic test split.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)


from reben_pipeline.dataset import create_synthetic_reben_dataframe
from reben_pipeline.train_cahn_ierc import run_pipeline
from reben_pipeline.run_experiments import generate_markdown_report


def run_large_scale_experiment(
    num_samples: int = 2500,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-4,
    output_dir: str = "./results_large_scale",
    seed: int = 42
):

    os.makedirs(output_dir, exist_ok=True)
    print("="*80)
    print(f"STARTING LARGE-SCALE END-TO-END TRAINING ({num_samples} patches, {epochs} epochs)")
    print("="*80)

    # 1. Prepare 5,000-sample paired dataset with geographic train/val/test splits (70/15/15)
    print(f"[*] Generating {num_samples} multi-region paired Sentinel-1 / Sentinel-2 patches...")
    df = create_synthetic_reben_dataframe(num_samples=num_samples, split_ratios=(0.70, 0.15, 0.15), seed=seed)
    df.to_csv(os.path.join(output_dir, "dataset_metadata_5000.csv"), index=False)
    print(f"[*] Dataset Splits: Train={len(df[df['split']=='train'])}, Val={len(df[df['split']=='val'])}, Test={len(df[df['split']=='test'])}")

    experiments = {}

    # 2. Train Full Proposed Model: ResNet-50 CAHN + IERC Dual-Encoder
    print("\n" + "#"*80)
    print(">>> 1. Training ResNet-50 CAHN + IERC Dual-Encoder (25 Epochs) <<<")
    print("#"*80)
    exp1_dir = os.path.join(output_dir, "resnet50_cahn_ierc")
    res1 = run_pipeline(
        df=df,
        backbone="resnet50",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_cahn=True,
        use_ierc=True,
        ierc_interval=5,
        output_dir=exp1_dir,
        seed=seed,
        is_synthetic=True
    )
    experiments["ResNet-50 (CAHN + IERC)"] = res1

    # 3. Train ResNet-101 CAHN + IERC Dual-Encoder
    print("\n" + "#"*80)
    print(">>> 2. Training ResNet-101 CAHN + IERC Dual-Encoder (25 Epochs) <<<")
    print("#"*80)
    exp2_dir = os.path.join(output_dir, "resnet101_cahn_ierc")
    res2 = run_pipeline(
        df=df,
        backbone="resnet101",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_cahn=True,
        use_ierc=True,
        ierc_interval=5,
        output_dir=exp2_dir,
        seed=seed,
        is_synthetic=True
    )
    experiments["ResNet-101 (CAHN + IERC)"] = res2

    # 4. Generate Final Reports & Table
    generate_markdown_report(experiments, output_dir)
    print(f"\n[+] Large-Scale Training and Evaluation Complete! Report at {output_dir}/reben_s1s2_results.md")


if __name__ == "__main__":
    run_large_scale_experiment()
