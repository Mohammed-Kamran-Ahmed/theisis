"""
run_experiments.py - Automated Benchmark & Ablation Suite
=========================================================
Runs:
1. Proposed CAHN + IERC Dual-Encoder (S1+S2 Fusion)
2. Ablation: SupCon Dual-Encoder without CAHN
3. Ablation: Dual-Encoder without IERC Re-Clustering
4. Modality breakdowns: S1-only, S2-only, S1+S2 Fusion
5. Canonical Correlation Analysis (CCA) Benchmark at each configuration
6. Inter-cluster geometry analysis (Silhouette, ARI, NMI)
7. Generates results/reben_s1s2_results.md and comprehensive comparison table
"""

import os
import sys
import json
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)


from reben_pipeline.train_cahn_ierc import run_pipeline
from reben_pipeline.dataset import create_synthetic_reben_dataframe


def run_full_suite(
    epochs: int = 12,
    batch_size: int = 32,
    lr: float = 1e-4,
    backbone: str = "resnet50",
    output_dir: str = "./results",
    seed: int = 42
):
    os.makedirs(output_dir, exist_ok=True)
    print("="*80)
    print("STARTING FULL REBEN BENCHMARK & ABLATION EXPERIMENTS")
    print("="*80)

    # Base dataset (consistent across all ablations)
    df = create_synthetic_reben_dataframe(num_samples=1500, split_ratios=(0.7, 0.15, 0.15), seed=seed)

    all_experiments = {}

    # 1. Proposed Model: CAHN + IERC Dual-Encoder
    print("\n>>> Running Experiment 1: CAHN + IERC Dual-Encoder (Proposed) <<<")
    exp1_dir = os.path.join(output_dir, "exp1_cahn_ierc")
    res1 = run_pipeline(
        df=df,
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_cahn=True,
        use_ierc=True,
        ierc_interval=3,
        output_dir=exp1_dir,
        seed=seed,
        is_synthetic=True
    )
    all_experiments["Proposed (CAHN + IERC)"] = res1

    # 2. Ablation 1: Dual-Encoder without CAHN (Plain SupCon)
    print("\n>>> Running Experiment 2: Ablation w/o CAHN (Plain SupCon) <<<")
    exp2_dir = os.path.join(output_dir, "exp2_no_cahn")
    res2 = run_pipeline(
        df=df,
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_cahn=False,
        use_ierc=True,
        ierc_interval=3,
        output_dir=exp2_dir,
        seed=seed,
        is_synthetic=True
    )
    all_experiments["Ablation (w/o CAHN)"] = res2

    # 3. Ablation 2: Dual-Encoder without IERC Re-Clustering
    print("\n>>> Running Experiment 3: Ablation w/o IERC <<<")
    exp3_dir = os.path.join(output_dir, "exp3_no_ierc")
    res3 = run_pipeline(
        df=df,
        backbone=backbone,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        use_cahn=True,
        use_ierc=False,
        output_dir=exp3_dir,
        seed=seed,
        is_synthetic=True
    )
    all_experiments["Ablation (w/o IERC)"] = res3

    # Generate Markdown Report & Table
    generate_markdown_report(all_experiments, output_dir)
    print(f"\n[+] Experiments Complete! Results saved in {output_dir}")


def generate_markdown_report(experiments: dict, output_dir: str):
    """Formats comparison table matching reBEN Table I with our additions."""
    out_path = os.path.join(output_dir, "reben_s1s2_results.md")

    # Published numbers from reBEN Paper Table I (Clasen et al., 2024)
    published_baselines = [
        {"Model": "ResNet-50", "Modality": "S1", "AP_macro": 48.91, "AP_micro": 75.37, "F1_macro": 42.92, "F1_micro": 62.77, "Source": "Published (reBEN Table I)"},
        {"Model": "ResNet-50", "Modality": "S2", "AP_macro": 68.80, "AP_micro": 85.07, "F1_macro": 62.48, "F1_micro": 75.05, "Source": "Published (reBEN Table I)"},
        {"Model": "ResNet-50", "Modality": "S1+S2", "AP_macro": 70.76, "AP_micro": 86.06, "F1_macro": 64.90, "F1_micro": 76.50, "Source": "Published (reBEN Table I)"},
        {"Model": "ResNet-101", "Modality": "S1", "AP_macro": 49.34, "AP_micro": 75.66, "F1_macro": 43.15, "F1_micro": 63.02, "Source": "Published (reBEN Table I)"},
        {"Model": "ResNet-101", "Modality": "S2", "AP_macro": 69.12, "AP_micro": 85.29, "F1_macro": 62.85, "F1_micro": 75.31, "Source": "Published (reBEN Table I)"},
        {"Model": "ResNet-101", "Modality": "S1+S2", "AP_macro": 70.93, "AP_micro": 86.21, "F1_macro": 64.69, "F1_micro": 76.59, "Source": "Published (reBEN Table I)"},
    ]

    lines = [
        "# reBEN Benchmark Results: SAR-Optical Dual Encoders with CAHN & IERC",
        "",
        "## 1. Benchmark Comparison Table (reBEN Protocol Table I Format)",
        "",
        "| Model | Modality | AP (macro) | AP (micro) | F1 (macro) | F1 (micro) | Source / Note |",
        "|---|---|---|---|---|---|---|"
    ]

    # Add published baselines
    for b in published_baselines:
        lines.append(f"| {b['Model']} | {b['Modality']} | {b['AP_macro']:.2f}% | {b['AP_micro']:.2f}% | {b['F1_macro']:.2f}% | {b['F1_micro']:.2f}% | {b['Source']} |")

    lines.append("|---|---|---|---|---|---|---|")

    # Add our experiment rows
    for name, exp in experiments.items():
        res = exp["results"]
        # S1 only
        s1_res = res.get("s1_only", res.get("s1", {}))
        lines.append(f"| **Dual-Encoder ({name})** | S1 | {s1_res.get('ap_macro', 0.0):.2f}% | {s1_res.get('ap_micro', 0.0):.2f}% | {s1_res.get('f1_macro', 0.0):.2f}% | {s1_res.get('f1_micro', 0.0):.2f}% | Our Implementation |")
        # S2 only
        s2_res = res.get("s2_only", res.get("s2", {}))
        lines.append(f"| **Dual-Encoder ({name})** | S2 | {s2_res.get('ap_macro', 0.0):.2f}% | {s2_res.get('ap_micro', 0.0):.2f}% | {s2_res.get('f1_macro', 0.0):.2f}% | {s2_res.get('f1_micro', 0.0):.2f}% | Our Implementation |")
        # S1+S2 Fusion
        fus_res = res.get("s1_s2_fusion", res.get("fusion", {}))
        lines.append(f"| **Dual-Encoder ({name})** | **S1+S2** | **{fus_res.get('ap_macro', 0.0):.2f}%** | **{fus_res.get('ap_micro', 0.0):.2f}%** | **{fus_res.get('f1_macro', 0.0):.2f}%** | **{fus_res.get('f1_micro', 0.0):.2f}%** | **Fusion Target (Goal ≥ 72.0%)** |")


    lines.extend([
        "",
        "---",
        "",
        "## 2. Canonical Correlation Analysis (CCA) Cross-Modal Verification",
        "",
        "| Configuration | $r_1$ (1st Canon. Corr) | $\\bar{r}$ (Mean Corr) | $r_{1,\\text{null}}$ | $\\Delta_{\\text{null}}$ Gain | $p$-value |",
        "|---|---|---|---|---|---|"
    ])

    for name, exp in experiments.items():
        cca = exp["cca_benchmark"]
        lines.append(f"| {name} | {cca['r1']:.4f} | {cca['r_bar']:.4f} | {cca['r1_null_mean']:.4f} | +{cca['delta_null']:.4f} | {cca['p_value']:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Latent Clustering Quality & Semantic Alignment",
        "",
        "| Configuration | Silhouette Score $\\uparrow$ | Davies-Bouldin $\\downarrow$ | Calinski-Harabasz $\\uparrow$ | ARI | NMI |",
        "|---|---|---|---|---|---|"
    ])

    for name, exp in experiments.items():
        cl = exp["cluster_metrics"]
        lines.append(f"| {name} | {cl['silhouette']:.4f} | {cl['davies_bouldin']:.4f} | {cl['calinski_harabasz']:.1f} | {cl.get('ari', 0.0):.4f} | {cl.get('nmi', 0.0):.4f} |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also save as JSON summary
    with open(os.path.join(output_dir, "all_experiments_summary.json"), "w") as f:
        json.dump(experiments, f, indent=2)


if __name__ == "__main__":
    run_full_suite()
