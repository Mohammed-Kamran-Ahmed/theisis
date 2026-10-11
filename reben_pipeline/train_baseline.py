"""
train_lmdb.py - CAHN/IERC dual-encoder on the official BENv2 LMDB (ConfigILM)
=============================================================================
Band layout of the loader output (B, 12, 120, 120):
    ch 0-1  : Sentinel-1 VV, VH        ch 2-11 : Sentinel-2 (10 bands)

Ablation ladder (--variant):
    cls        classification only (dual encoder, no contrastive terms)
    align      + cross-modal InfoNCE
    cahn       + CAHN contrastive loss (no IERC)
    cahn_ierc  + IERC re-clustering / proximity weighting   (full method)

Examples
--------
  python -m reben_pipeline.train_lmdb --variant cahn_ierc --epochs 30 --batch_size 32 --seed 1
  python -m reben_pipeline.train_lmdb --variant cahn_ierc --epochs 30 --batch_size 32 --seed 1 --resume

Protocol notes
--------------
* Checkpoint selection uses VALIDATION macro AP only; the test split is set up and
  evaluated once, at the end, with the selected checkpoint.
* best_model.pth = best-by-validation weights. last.pt = full training state
  (model, optimizer, scheduler, scaler, RNG, IERC state, history) for exact --resume.
* An existing output dir is never overwritten silently: pass --resume or a new dir.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from reben_pipeline.common import (
    atomic_save,
    build_data_module,
    capture_rng_state,
    check_label_alignment,
    class_frequencies_from_parquet,
    load_checkpoint,
    make_pos_weight,
    restore_rng_state,
    set_all_seeds,
    split_bands,
    to_jsonable,
)
from reben_pipeline.ierc import IERCController
from reben_pipeline.losses import JointDualEncoderLoss
from reben_pipeline.metrics import MultiLabelMetricsCalculator
from reben_pipeline.models import CAHNIERCDualEncoder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

VARIANTS = {
    "cls":       dict(lambda_cahn=0.0, lambda_align=0.0, ierc=False),
    "align":     dict(lambda_cahn=0.0, lambda_align=0.5, ierc=False),
    "cahn":      dict(lambda_cahn=1.0, lambda_align=0.5, ierc=False),
    "cahn_ierc": dict(lambda_cahn=1.0, lambda_align=0.5, ierc=True),
}
RESUME_CHECK_KEYS = ("variant", "backbone", "epochs", "batch_size", "lr", "seed")


def should_recluster(epoch: int, start: int, interval: int) -> bool:
    return epoch >= start and (epoch - start) % interval == 0


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, loss_fn, device, ierc, scaler, amp, grad_clip,
                limit_batches=None):
    model.train()
    sums = defaultdict(float)
    n_batches = 0

    for i, (x, y) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        if x.shape[0] < 2:  # BatchNorm cannot train on a single sample
            continue
        s1, s2 = split_bands(x.to(device).float())
        targets = y.to(device).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(sar_img=s1, opt_img=s2)
        out = {k: v.float() for k, v in out.items()}

        cluster_labels = None
        if ierc is not None and ierc.ready:
            # IERC cluster id of every sample in this batch, from its current fused embedding.
            cluster_labels = torch.as_tensor(
                ierc.predict(out["z_fused"].detach().cpu().numpy()),
                device=device, dtype=torch.long)

        loss, ld = loss_fn(
            z_sar=out["z_sar"], z_opt=out["z_opt"],
            logits_fused=out["logits_fused"], targets=targets,
            logits_sar=out["logits_sar"], logits_opt=out["logits_opt"],
            multi_label=True, cluster_labels=cluster_labels,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at batch {i}: {ld}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        for k in ("total_loss", "cahn_loss", "align_loss", "cls_loss", "cahn_valid_anchor_frac"):
            sums[k] += ld[k]
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("Training loader produced no usable batches.")
    return {
        "train_loss": sums["total_loss"] / n_batches,
        "train_cahn": sums["cahn_loss"] / n_batches,
        "train_align": sums["align_loss"] / n_batches,
        "train_cls": sums["cls_loss"] / n_batches,
        "train_anchor_frac": sums["cahn_valid_anchor_frac"] / n_batches,
        "n_batches": n_batches,
    }


@torch.no_grad()
def evaluate_split(model, loader, device, calc, limit_batches=None, return_embeddings=False):
    model.eval()
    fused, sar, opt, tg = [], [], [], []
    zs, zo, zf = [], [], []
    for i, (x, y) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        s1, s2 = split_bands(x.to(device).float())
        out = model(sar_img=s1, opt_img=s2)
        fused.append(out["logits_fused"].float().cpu())
        sar.append(out["logits_sar"].float().cpu())
        opt.append(out["logits_opt"].float().cpu())
        tg.append(y.float())
        if return_embeddings:
            zs.append(out["z_sar"].float().cpu().numpy())
            zo.append(out["z_opt"].float().cpu().numpy())
            zf.append(out["z_fused"].float().cpu().numpy())

    targets = torch.cat(tg)
    res = {
        "fusion": calc.compute_all(torch.cat(fused), targets),
        "s1": calc.compute_all(torch.cat(sar), targets),
        "s2": calc.compute_all(torch.cat(opt), targets),
        "n_patches": int(len(targets)),
        "n_classes_present": int((targets.sum(0) > 0).sum().item()),
    }
    if return_embeddings:
        res.update(z_sar=np.vstack(zs), z_opt=np.vstack(zo), z_fused=np.vstack(zf),
                   targets=targets.numpy())
    return res


def kmeans_cluster_metrics(z, k=16, seed=42):
    """Same clustering procedure for EVERY variant, so numbers are comparable."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    if len(z) <= k:
        return {"skipped": f"only {len(z)} samples for k={k}"}
    labels = KMeans(n_clusters=k, random_state=seed, n_init=5).fit_predict(z)
    if len(np.unique(labels)) < 2:
        return {"skipped": "k-means returned a single cluster"}
    return {
        "k": k,
        "silhouette": float(silhouette_score(z, labels)),
        "davies_bouldin": float(davies_bouldin_score(z, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(z, labels)),
    }


def run_diagnostics(test_eval, seed):
    """Secondary analyses. Never allowed to crash a finished run."""
    diag = {}
    try:
        from reben_pipeline.cca_analysis import run_cca_analysis
        diag["cca"] = run_cca_analysis(test_eval["z_sar"], test_eval["z_opt"],
                                       n_components=8, n_permutations=200, seed=seed)
    except Exception as exc:  # noqa: BLE001
        diag["cca"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        diag["clustering"] = kmeans_cluster_metrics(test_eval["z_fused"], 16, seed)
    except Exception as exc:  # noqa: BLE001
        diag["clustering"] = {"error": f"{type(exc).__name__}: {exc}"}
    return diag


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_lmdb_pipeline(args):
    cfg = vars(args).copy()
    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")

    out_dir = Path(args.output_dir)
    best_path, last_path = out_dir / "best_model.pth", out_dir / "last.pt"
    summary_path = out_dir / "lmdb_experiment_summary.json"
    if args.resume:
        if not last_path.is_file():
            raise SystemExit(f"--resume given but {last_path} does not exist.")
    elif best_path.exists() or last_path.exists():
        raise SystemExit(f"{out_dir} already contains checkpoints. Use --resume, or a new --output_dir.")
    out_dir.mkdir(parents=True, exist_ok=True)

    vp = VARIANTS[args.variant]
    lambda_cahn = vp["lambda_cahn"] if args.lambda_cahn is None else args.lambda_cahn
    lambda_align = vp["lambda_align"] if args.lambda_align is None else args.lambda_align
    use_ierc = vp["ierc"] and lambda_cahn > 0
    cfg.update(lambda_cahn=lambda_cahn, lambda_align=lambda_align, use_ierc=use_ierc, amp_active=amp)

    print(f"[*] Device: {device} | AMP: {amp} | variant: {args.variant} | backbone: {args.backbone}"
          f" | epochs: {args.epochs} | batch: {args.batch_size} | seed: {args.seed}")
    print(f"[*] lambda_cahn={lambda_cahn} lambda_align={lambda_align} IERC={use_ierc}")

    # ---- Data ----
    dm = build_data_module(args.lmdb_dir, args.parquet, args.batch_size, args.num_workers)
    train_loader, val_loader = dm.train_dataloader(), dm.val_dataloader()
    print(f"[*] Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    freq = class_frequencies_from_parquet(args.parquet, "train")
    check_label_alignment(freq, train_loader)
    pos_weight = make_pos_weight(freq, args.pos_weight_cap, device) if args.pos_weight else None
    print(f"[*] pos_weight: {'ON (cap %.1f)' % args.pos_weight_cap if args.pos_weight else 'OFF (plain BCE)'}")

    # ---- Model / loss / optim ----
    model = CAHNIERCDualEncoder(
        sar_channels=2, opt_channels=10, num_classes=19, backbone=args.backbone,
        proj_dim=128, fusion_type="concat", pretrained=not args.no_pretrained,
    ).to(device)
    loss_fn = JointDualEncoderLoss(
        temperature=args.temperature, lambda_cahn=lambda_cahn, lambda_align=lambda_align,
        lambda_cls=1.0, lambda_modality_cls=0.25, pos_weight=pos_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    calc = MultiLabelMetricsCalculator(num_classes=19)
    ierc = None
    if use_ierc:
        ierc = IERCController(
            n_clusters=args.ierc_clusters, seed=args.seed, covariance_type=args.ierc_cov,
            pca_dim=args.ierc_pca or None, same_cluster_proximity=args.ierc_same_cluster,
            max_batches=args.ierc_max_batches or None)

    start_epoch, best_val, best_epoch, history = 1, -1.0, 0, []

    # ---- Exact resume ----
    if args.resume:
        ck = load_checkpoint(last_path)
        for key in RESUME_CHECK_KEYS:
            if ck["config"].get(key) != cfg.get(key):
                raise SystemExit(f"--resume config mismatch on '{key}': "
                                 f"checkpoint={ck['config'].get(key)!r} now={cfg.get(key)!r}")
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        scaler.load_state_dict(ck["scaler_state"])
        start_epoch = ck["epoch"] + 1
        best_val, best_epoch, history = ck["best_val_ap_macro"], ck["best_epoch"], ck["history"]
        if ierc is not None and ck.get("ierc_state") is not None:
            ierc.load_state_dict(ck["ierc_state"])
            if ierc.proximity_matrix is not None:
                loss_fn.cahn_loss.set_proximity_matrix(ierc.proximity_matrix)
        restore_rng_state(ck["rng_state"])
        print(f"[resume] restored full state after epoch {ck['epoch']}; continuing at "
              f"epoch {start_epoch} (best val AP so far {best_val:.2f}% @ epoch {best_epoch})")

    hdr = (f"{'Ep':>3} | {'Loss':>7} | {'CAHN':>6} | {'Align':>6} | {'Cls':>6} | {'Anch%':>5} |"
           f" {'ValAPma':>8} | {'ValAPmi':>8} | {'ValF1ma':>8} | {'ValF1mi':>8} | {'Time':>6}")
    print("\n" + "-" * len(hdr) + "\n" + hdr + "\n" + "-" * len(hdr))

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

        if ierc is not None and should_recluster(epoch, args.ierc_start, args.ierc_interval):
            _, prox, _ = ierc.step(model, train_loader, device)
            loss_fn.cahn_loss.set_proximity_matrix(prox)
            s = ierc.last_stats
            print(f"  [IERC] epoch {epoch}: refit on {s['n_fit_samples']} train samples in "
                  f"{s['seconds']}s | converged={s['converged']} | cluster sizes "
                  f"min/med/max = {s['cluster_size_min']}/{s['cluster_size_median']:.0f}/"
                  f"{s['cluster_size_max']} | empty={s['n_empty_clusters']}")

        tr = train_epoch(model, train_loader, optimizer, loss_fn, device, ierc, scaler, amp,
                         args.grad_clip, args.limit_train_batches)
        scheduler.step()
        val = evaluate_split(model, val_loader, device, calc, args.limit_val_batches)
        vf = val["fusion"]
        if not np.isfinite(vf["ap_macro"]):
            raise RuntimeError("Validation macro AP is not finite.")

        print(f"{epoch:>3d} | {tr['train_loss']:>7.4f} | {tr['train_cahn']:>6.4f} |"
              f" {tr['train_align']:>6.4f} | {tr['train_cls']:>6.4f} |"
              f" {100 * tr['train_anchor_frac']:>5.1f} | {vf['ap_macro']:>7.2f}% |"
              f" {vf['ap_micro']:>7.2f}% | {vf['f1_macro']:>7.2f}% | {vf['f1_micro']:>7.2f}% |"
              f" {time.time() - t0:>5.0f}s")

        history.append({
            "epoch": epoch, "lr": lr_now, **tr,
            "val_ap_macro": vf["ap_macro"], "val_ap_micro": vf["ap_micro"],
            "val_f1_macro": vf["f1_macro"], "val_f1_micro": vf["f1_micro"],
            "val_s1_ap_macro": val["s1"]["ap_macro"], "val_s2_ap_macro": val["s2"]["ap_macro"],
            "val_n_patches": val["n_patches"], "val_n_classes_present": val["n_classes_present"],
        })

        if vf["ap_macro"] > best_val:
            best_val, best_epoch = vf["ap_macro"], epoch
            atomic_save({"model_state": model.state_dict(), "epoch": epoch,
                         "best_val_ap_macro": best_val, "config": cfg}, best_path)
            print(f"  [ckpt] new best (val AP-macro {best_val:.2f}%) -> {best_path.name}")

        atomic_save({
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
            "epoch": epoch, "best_val_ap_macro": best_val, "best_epoch": best_epoch,
            "history": history, "rng_state": capture_rng_state(),
            "ierc_state": ierc.state_dict() if ierc is not None else None, "config": cfg,
        }, last_path)

        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            print(f"[debug] --stop_after_epoch {args.stop_after_epoch}: stopping before test.")
            return None

    print(f"[*] Training done. Best val AP(macro) {best_val:.2f}% at epoch {best_epoch}")

    # ---- Test: set up and evaluate ONCE, with the validation-selected checkpoint ----
    if not best_path.is_file():
        raise RuntimeError("No best checkpoint was saved.")
    model.load_state_dict(load_checkpoint(best_path)["model_state"])
    dm.setup("test")
    test_eval = evaluate_split(model, dm.test_dataloader(), device, calc,
                               args.limit_test_batches, return_embeddings=True)
    te, se, oe = test_eval["fusion"], test_eval["s1"], test_eval["s2"]
    print("\n" + "=" * 70)
    print(f"  TEST (n={test_eval['n_patches']}, {test_eval['n_classes_present']}/19 classes present)"
          f"  checkpoint: epoch {best_epoch}")
    for name, m in (("Fusion S1+S2", te), ("S1 only", se), ("S2 only", oe)):
        print(f"  {name:<13} AP-macro={m['ap_macro']:.2f}%  AP-micro={m['ap_micro']:.2f}%  "
              f"F1-macro={m['f1_macro']:.2f}%  F1-micro={m['f1_micro']:.2f}%")
    if test_eval["n_classes_present"] < 19:
        print(f"  NOTE: AP-macro averages the {test_eval['n_classes_present']} present classes; "
              f"F1-macro counts the {19 - test_eval['n_classes_present']} absent ones as 0.")
    print("=" * 70)

    summary = {
        "config": cfg,
        "best_epoch": best_epoch,
        "best_val_ap_macro": best_val,
        "test": {"n_patches": test_eval["n_patches"],
                 "n_classes_present": test_eval["n_classes_present"],
                 "fusion": te, "s1": se, "s2": oe},
        "ierc_last_stats": ierc.last_stats if ierc is not None else None,
        "history": history,
    }
    write_json(summary, summary_path)  # saved BEFORE diagnostics, so a diagnostics crash loses nothing

    summary.update(run_diagnostics(test_eval, args.seed))
    write_json(summary, summary_path)
    print(f"[*] Summary saved -> {summary_path}")
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CAHN/IERC dual encoder on reBEN LMDB")
    p.add_argument("--lmdb_dir", default="data/reben_lithuania/BENv2_lithuania_summer.lmdb")
    p.add_argument("--parquet", default="data/reben_lithuania/metadata_lithuania_summer.parquet")
    p.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50", "resnet101"])
    p.add_argument("--variant", default="cahn_ierc", choices=list(VARIANTS))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", "--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", "--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--lambda_cahn", type=float, default=None, help="override the variant default")
    p.add_argument("--lambda_align", type=float, default=None, help="override the variant default")
    p.add_argument("--pos_weight", action="store_true", help="BCE pos_weight (default: plain BCE)")
    p.add_argument("--pos_weight_cap", type=float, default=10.0)
    p.add_argument("--ierc_start", type=int, default=3, help="first epoch that (re)fits the GMM")
    p.add_argument("--ierc_interval", type=int, default=3)
    p.add_argument("--ierc_clusters", type=int, default=16)
    p.add_argument("--ierc_cov", default="diag", choices=["diag", "full", "spherical", "tied"])
    p.add_argument("--ierc_pca", type=int, default=0, help="PCA dim before GMM (0 = none)")
    p.add_argument("--ierc_same_cluster", type=float, default=1.0,
                   help="proximity given to pairs inside the same GMM cluster (old behaviour: 0.0)")
    p.add_argument("--ierc_max_batches", type=int, default=0, help="cap batches used to fit GMM (0 = all)")
    p.add_argument("--amp", action="store_true", help="fp16 autocast (CUDA only)")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", "--output-dir", default=None)
    p.add_argument("--num_workers", "--num-workers", type=int, default=0)
    p.add_argument("--limit_train_batches", type=int, default=None)
    p.add_argument("--limit_val_batches", type=int, default=None)
    p.add_argument("--limit_test_batches", type=int, default=None)
    p.add_argument("--stop_after_epoch", type=int, default=0, help="debug: stop (no test) after N epochs")
    args = p.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = f"results/{args.variant}_seed{args.seed}"
    return args


def main(argv=None):
    return run_lmdb_pipeline(parse_args(argv))


if __name__ == "__main__":
    main()