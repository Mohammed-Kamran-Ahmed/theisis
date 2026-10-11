import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist

from stage4_contrastive_learning import (
    ModalityEncoder,
    MultiModalSAROptDataset,
    HardNegativeWeightedSupCon,
    CrossModalInfoNCE,
    compute_test_silhouette
)
from seed_lock import set_seed  # <-- ADDED


def extract_fused_features(sar_enc, opt_enc, df, patch_dir, device):
    sar_enc.eval()
    opt_enc.eval()

    dataset = MultiModalSAROptDataset(df, patch_dir)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    features = []

    with torch.no_grad():
        for sar, opt, _ in loader:
            sar = sar.to(device)
            opt = opt.to(device)

            z_s = sar_enc(sar)
            z_o = opt_enc(opt)

            fused = F.normalize((z_s + z_o) / 2.0, p=2, dim=1)
            features.append(fused.cpu().numpy())

    return np.vstack(features)


def generate_clusters(embeddings, n_clusters, seed=42):
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=seed  # <-- was hardcoded 42, now tied to the run seed
    )

    gmm.fit(embeddings)
    labels = gmm.predict(embeddings)

    return gmm, labels


def build_proximity_matrix(gmm):
    centers = gmm.means_

    norms = np.linalg.norm(
        centers,
        axis=1,
        keepdims=True
    )

    normalized = centers / np.maximum(norms, 1e-12)

    distances = cdist(
        normalized,
        normalized,
        metric="euclidean"
    )

    max_distance = max(distances.max(), 1e-12)

    return torch.tensor(
        distances / max_distance,
        dtype=torch.float32
    )


def save_checkpoint(path, sar_enc, opt_enc, epoch, score):
    torch.save(
        {
            "sar_state": sar_enc.state_dict(),
            "opt_state": opt_enc.state_dict(),
            "epoch": epoch,
            "silhouette": score
        },
        path
    )


def run_stage4b_ierc(
    train_df,
    test_df,
    patch_dir="./downloaded_patches",
    model_dir="./saved_models",
    num_clusters=8,
    refinement_rounds=5,
    epochs_per_round=3,
    lr=1e-4,
    device="cuda",
    seed=42
):
    set_seed(seed)  # <-- ADDED: must be first, before checkpoint load / dataloader creation

    os.makedirs(model_dir, exist_ok=True)

    device = torch.device(
        device if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = os.path.join(
        model_dir,
        "stage4_best.pth"
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "Stage 4 checkpoint not found."
        )

    sar_enc = ModalityEncoder(proj_dim=64).to(device)
    opt_enc = ModalityEncoder(proj_dim=64).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    sar_enc.load_state_dict(checkpoint["sar_state"])
    opt_enc.load_state_dict(checkpoint["opt_state"])

    train_df = train_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    best_train_sil = -1.0
    best_epoch = 0

    optimizer = torch.optim.Adam(
        list(sar_enc.parameters()) +
        list(opt_enc.parameters()),
        lr=lr
    )

    infonce_fn = CrossModalInfoNCE(
        temperature=0.07
    )

    global_epoch = 0

    for round_id in range(1, refinement_rounds + 1):

        print(
            f"\n========== IERC ROUND "
            f"{round_id}/{refinement_rounds} =========="
        )

        z_train = extract_fused_features(
            sar_enc,
            opt_enc,
            train_df,
            patch_dir,
            device
        )

        gmm, train_labels = generate_clusters(
            z_train,
            num_clusters,
            seed=seed
        )

        train_df["pseudo_label"] = train_labels

        z_test = extract_fused_features(
            sar_enc,
            opt_enc,
            test_df,
            patch_dir,
            device
        )

        test_labels = gmm.predict(z_test)
        test_df["pseudo_label"] = test_labels

        proximity = build_proximity_matrix(gmm)

        supcon_fn = HardNegativeWeightedSupCon(
            temperature=0.07
        )

        supcon_fn.set_cluster_proximity(
            proximity
        )

        train_dataset = MultiModalSAROptDataset(
            train_df,
            patch_dir
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            drop_last=True
        )

        for local_epoch in range(1, epochs_per_round + 1):

            global_epoch += 1

            sar_enc.train()
            opt_enc.train()

            total_loss = 0.0

            for sar, opt, labels in train_loader:

                sar = sar.to(device)
                opt = opt.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                z_sar = sar_enc(sar)
                z_opt = opt_enc(opt)

                loss_sar = supcon_fn(
                    z_sar,
                    labels
                )

                loss_opt = supcon_fn(
                    z_opt,
                    labels
                )

                loss_align = infonce_fn(
                    z_sar,
                    z_opt
                )

                loss = (
                    loss_sar +
                    loss_opt +
                    0.5 * loss_align
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            train_features = extract_fused_features(
                sar_enc,
                opt_enc,
                train_df,
                patch_dir,
                device
            )

            train_sil = compute_test_silhouette(
                sar_enc,
                opt_enc,
                train_df,
                patch_dir,
                device
            )

            print(
                f"Epoch {global_epoch} | "
                f"Loss: {total_loss / len(train_loader):.4f} | "
                f"Train Silhouette: {train_sil:.4f}"
            )

            if train_sil > best_train_sil:

                best_train_sil = train_sil
                best_epoch = global_epoch

                save_checkpoint(
                    os.path.join(
                        model_dir,
                        "stage4b_ierc_best.pth"
                    ),
                    sar_enc,
                    opt_enc,
                    global_epoch,
                    train_sil
                )

                torch.save(
                    sar_enc.state_dict(),
                    os.path.join(
                        model_dir,
                        "sar_encoder_reclustered.pth"
                    )
                )

                torch.save(
                    opt_enc.state_dict(),
                    os.path.join(
                        model_dir,
                        "optical_encoder_reclustered.pth"
                    )
                )

        print(
            f"Round {round_id} completed | "
            f"Best Train Silhouette: {best_train_sil:.4f}"
        )

    best_checkpoint = torch.load(
        os.path.join(
            model_dir,
            "stage4b_ierc_best.pth"
        ),
        map_location=device,
        weights_only=False
    )

    sar_enc.load_state_dict(
        best_checkpoint["sar_state"]
    )

    opt_enc.load_state_dict(
        best_checkpoint["opt_state"]
    )

    final_train_features = extract_fused_features(
        sar_enc,
        opt_enc,
        train_df,
        patch_dir,
        device
    )

    final_test_sil = compute_test_silhouette(
        sar_enc,
        opt_enc,
        test_df,
        patch_dir,
        device
    )

    final_gmm, final_train_labels = generate_clusters(
        final_train_features,
        num_clusters,
        seed=seed
    )

    final_test_features = extract_fused_features(
        sar_enc,
        opt_enc,
        test_df,
        patch_dir,
        device
    )

    final_test_labels = final_gmm.predict(
        final_test_features
    )

    train_df["pseudo_label"] = final_train_labels
    test_df["pseudo_label"] = final_test_labels

    combined = pd.concat(
        [train_df, test_df],
        ignore_index=True
    )

    combined.to_csv(
        "patch_metadata_reclustered.csv",
        index=False
    )

    np.save(
        "embedding_labels_train.npy",
        final_train_labels
    )

    np.save(
        "embedding_labels_test.npy",
        final_test_labels
    )

    print("\n==============================================")
    print("        STAGE 4B IERC COMPLETE")
    print("==============================================")
    print(f"Best Train Silhouette : {best_train_sil:.4f}")
    print(f"Final Test Silhouette : {final_test_sil:.4f}")
    print(f"Best Epoch            : {best_epoch}")
    print("==============================================")

    return train_df, test_df


if __name__ == "__main__":

    input_csv = "./patch_metadata_clustered.csv"

    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            "patch_metadata_clustered.csv not found."
        )

    df = pd.read_csv(input_csv)

    clean_df = df[
        df.get("valid", True) == True
    ].copy().reset_index(drop=True)

    train_df = clean_df[
        clean_df["split"] == "train"
    ].reset_index(drop=True)

    test_df = clean_df[
        clean_df["split"] == "test"
    ].reset_index(drop=True)

    run_stage4b_ierc(
        train_df,
        test_df,
        num_clusters=8,
        refinement_rounds=5,
        epochs_per_round=3
    )