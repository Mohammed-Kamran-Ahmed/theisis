"""
run_fixed_pipeline.py - Run the fully-debugged CAHN/IERC pipeline
==================================================================
All three bugs from DEBUG_PRIORITIES.md are now fixed:
  §1 F1=0%:  ast.literal_eval labels, sigmoid always applied, threshold=0.5001, pos_weight
  §2 CCA:    PCA pre-reduction to N//10 dims before CCA
  §3 no data: still synthetic — see comments below on getting real reBEN data

Usage (local GPU, quick sanity run):
    D:\BTP\theisis\thesis_env\Scripts\python.exe run_fixed_pipeline.py --backbone resnet50 --epochs 25 --num_samples 2500

Usage (full run for thesis, ResNet-101):
    D:\BTP\theisis\thesis_env\Scripts\python.exe run_fixed_pipeline.py --backbone resnet101 --epochs 100 --num_samples 2500 --output_dir ./results_fixed_101

NOTE on getting real reBEN data:
    The full 549,488-patch dataset is ~118 GB from Zenodo record 10891137.
    For a GPU with 8+ GB VRAM and 16+ hours compute, download via:
        pip install zenodo-get
        zenodo_get 10891137
    Or on Kaggle/Colab: load earthnets/BigEarthNetV2 from HuggingFace.
    Then pass --s1_dir and --s2_dir pointing to extracted patch folders,
    and --metadata_csv pointing to the reBEN split CSV.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reben_pipeline.train_cahn_ierc import run_pipeline
from reben_pipeline.dataset import create_synthetic_reben_dataframe


def main():
    p = argparse.ArgumentParser(description="Run CAHN/IERC dual-encoder on reBEN (fixed pipeline)")
    p.add_argument("--backbone",     default="resnet50",  choices=["resnet18", "resnet50", "resnet101"])
    p.add_argument("--epochs",       type=int,   default=25)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--num_samples",  type=int,   default=2500,  help="Synthetic samples (ignored if --metadata_csv set)")
    p.add_argument("--output_dir",   default="./results_fixed")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--no_cahn",      action="store_true")
    p.add_argument("--no_ierc",      action="store_true")
    p.add_argument("--metadata_csv", default=None, help="Path to real reBEN metadata CSV (optional)")
    p.add_argument("--s1_dir",       default=None, help="Root dir for S1 GeoTIFFs (optional)")
    p.add_argument("--s2_dir",       default=None, help="Root dir for S2 GeoTIFFs (optional)")
    args = p.parse_args()

    # Load real or synthetic data
    if args.metadata_csv:
        import pandas as pd
        print(f"[*] Loading real reBEN metadata from: {args.metadata_csv}")
        df = pd.read_csv(args.metadata_csv)
        is_synthetic = False
    else:
        print(f"[*] Generating {args.num_samples} synthetic reBEN patches (no real GeoTIFFs)")
        df = create_synthetic_reben_dataframe(num_samples=args.num_samples, seed=args.seed)
        is_synthetic = True

    summary = run_pipeline(
        df=df,
        s1_dir=args.s1_dir,
        s2_dir=args.s2_dir,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_cahn=not args.no_cahn,
        use_ierc=not args.no_ierc,
        ierc_interval=5,
        output_dir=args.output_dir,
        seed=args.seed,
        is_synthetic=is_synthetic
    )

    # Print final benchmark table
    print("\n" + "=" * 70)
    print("FINAL RESULTS (Test Split)")
    print("=" * 70)
    r = summary["results"]
    print(f"{'Modality':<20} {'AP(macro)':<12} {'AP(micro)':<12} {'F1(macro)':<12} {'F1(micro)'}")
    print("-" * 70)
    for key, label in [("s1_only", "S1 only"), ("s2_only", "S2 only"), ("s1_s2_fusion", "S1+S2 Fusion")]:
        v = r[key]
        print(f"{label:<20} {v['ap_macro']:<12.2f} {v['ap_micro']:<12.2f} {v['f1_macro']:<12.2f} {v['f1_micro']:.2f}")
    print("=" * 70)

    cca = summary["cca_benchmark"]
    print(f"\nCCA  r1={cca['r1']:.4f}  null_mean={cca['r1_null_mean']:.4f}  "
          f"delta={cca['delta_null']:.4f}  p={cca['p_value']:.4f}  "
          f"(dim used: {cca.get('cca_dim_used', '?')})")
    print(f"Results saved to: {args.output_dir}/reben_experiment_summary.json")


if __name__ == "__main__":
    main()
