import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import get_datasets
from src.metrics import compute_recon_metrics
from src.models.mlp_classifier import LatentDecoder, MLPClassifier
from src.models.vanilla_ae import VanillaAE
from src.utils import set_seed

CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp_ckpt", type=str, default="checkpoints/mlp_classifier.pt")
    parser.add_argument("--ae_ckpt", type=str, default="checkpoints/vanilla_ae.pt")
    parser.add_argument("--decoder_epochs", type=int, default=30)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--fig_dir", type=str, default="figures")
    return parser.parse_args()


def load_model(model: nn.Module, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()


def recon_metrics(originals: torch.Tensor, recons: torch.Tensor) -> dict:
    mse, psnr, ssim = compute_recon_metrics(originals, recons)
    return {"mse": mse, "psnr": psnr, "ssim": ssim}


def orthogonality_error(W: torch.Tensor) -> float:
    I = torch.eye(W.shape[1])
    return (torch.linalg.norm(W.T @ W - I) / torch.linalg.norm(I)).item()


def intra_class_variance(images: torch.Tensor, labels: torch.Tensor) -> float:
    flat = images.view(len(images), -1)
    return torch.stack([flat[labels == c].var(dim=0).mean() for c in range(10)]).mean().item()


def train_decoder(mlp: MLPClassifier, train_loader, epochs: int) -> LatentDecoder:
    for p in mlp.parameters():
        p.requires_grad = False

    decoder = LatentDecoder(latent_dim=64)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        total, n = 0.0, 0
        for x, _ in train_loader:
            with torch.no_grad():
                z = mlp.encode(x)
            loss = loss_fn(decoder(z), x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        print(f"  [decoder] Epoch {epoch}/{epochs} | train {total / n:.5f}")
    return decoder.eval()


def save_figure(rows: list, path: str) -> None:
    n_cols = max(len(imgs) for _, imgs, _ in rows)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(n_cols * 1.15, len(rows) * 1.35))
    for r, (title, imgs, labels) in enumerate(rows):
        for c in range(n_cols):
            axes[r, c].axis("off")
            if c < len(imgs):
                axes[r, c].imshow(imgs[c, 0], cmap="gray", vmin=0, vmax=1)
                if labels:
                    axes[r, c].set_title(labels[c], fontsize=7)
        axes[r, 0].text(-0.12, 0.5, title, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(42)

    train_loader, test_loader, _ = get_datasets(batch_size=128)
    x_test = torch.cat([x for x, _ in test_loader])
    y_test = torch.cat([y for _, y in test_loader])

    mlp, ae = MLPClassifier(), VanillaAE()
    load_model(mlp, args.mlp_ckpt)
    load_model(ae, args.ae_ckpt)
    W1, W2, W3 = (w.detach() for w in mlp.weights())

    with torch.no_grad():
        accuracy = (mlp(x_test).argmax(dim=1) == y_test).float().mean().item() * 100
        z_test = mlp.encode(x_test)

        y_onehot = torch.eye(10)
        x_class = torch.sigmoid(F.relu(F.relu(y_onehot @ W3) @ W2) @ W1).view(-1, 1, 28, 28)

        x_transpose = torch.sigmoid(F.relu(z_test @ W2) @ W1).view(-1, 1, 28, 28)

    decoder = train_decoder(mlp, train_loader, args.decoder_epochs)

    with torch.no_grad():
        x_decoder = decoder(z_test)
        x_ae = ae(x_test)

        z_rand = z_test.mean(dim=0) + z_test.std(dim=0) * torch.randn(10, 64)
        x_random = decoder(z_rand)

    var_asli = intra_class_variance(x_test, y_test)
    results = {
        "classifier_test_accuracy": accuracy,
        "orthogonality_error": {"W1": orthogonality_error(W1),
                                "W2": orthogonality_error(W2),
                                "W3": orthogonality_error(W3)},
        "tahap3_transpose_from_z_mlp": recon_metrics(x_test, x_transpose),
        "tahap4_decoder_from_z_mlp": recon_metrics(x_test, x_decoder),
        "tahap4_vanilla_ae": recon_metrics(x_test, x_ae),
        "intra_class_variance_kept": {
            "decoder_from_z_mlp": intra_class_variance(x_decoder, y_test) / var_asli,
            "vanilla_ae": intra_class_variance(x_ae, y_test) / var_asli,
        },
    }
    print(json.dumps(results, indent=2))

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "mlp_experiment.json"), "w") as f:
        json.dump(results, f, indent=2)

    n = 8
    save_figure([
        ("(1) Citra asli", x_test[:n], [CLASS_NAMES[i] for i in y_test[:n]]),
        ("(2a) $W^T$ dari one-hot kelas", x_class, CLASS_NAMES),
        ("(2b) $W^T$ dari $z_{MLP}$", x_transpose[:n], None),
        ("(3a) Decoder dari $z_{MLP}$ (freeze)", x_decoder[:n], None),
        ("(3b) Vanilla AE ($z_{AE}$)", x_ae[:n], None),
        ("(4) Sampling $z \\sim N(\\mu_z, \\sigma_z^2)$", x_random, None),
    ], os.path.join(args.fig_dir, "fig3_mlp_limits.png"))
    print("selesai, hasil di", args.results_dir, "dan", args.fig_dir)


if __name__ == "__main__":
    main()
