import argparse

import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.metrics import  compute_recon_metrics, compute_fid, measure_latency
from src.utils import save_checkpoint, set_seed
from src.data import get_datasets
from src.models.vae import VAE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--fid_samples", type=int, default=5000)
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kl_weight: float = 1.0
)  -> tuple[float, float, float]:
    model.train()
    total_loss, total_recon, total_kl = 0.0 , 0.0, 0.0
    num_samples = len(loader.dataset) #type: ignore

    for x, _ in loader: 
        x = x.to(device)
        #forward pass
        x_hat, mu, logvar = model(x)

        #hitung dua loss
        #recon loss / mse rata rata eror perpiksel
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        # KL Loss: Jarak mu dan logvar dari distribusi normal standar N(0, 1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()   

        loss = recon_loss + (kl_weight / 784.0) * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_recon += recon_loss.item() * x.size(0)
        total_kl += kl_loss.item() * x.size(0)

    return total_loss / num_samples, total_recon / num_samples, total_kl / num_samples

def evaluate_loss(
        model: nn.Module,
        loader: DataLoader, 
        device: torch.device,
        kl_weight: float = 1.0, 
) -> tuple[float, float, float]: 
    model.eval()
    total_loss, total_recon, total_kl = 0.0 , 0.0, 0.0
    num_samples = len(loader.dataset) #type: ignore
    with torch.no_grad(): 
        for x ,_ in loader: 
            x= x.to(device)
            x_hat, mu, logvar = model(x)

            recon_loss = F.mse_loss(x_hat, x, reduction="mean")
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            loss = recon_loss + (kl_weight / 784.0) * kl_loss

            total_loss += loss.item() * x.size(0)
            total_recon += recon_loss.item() * x.size(0)
            total_kl += kl_loss.item() * x.size(0)
    return total_loss / num_samples, total_recon / num_samples, total_kl / num_samples

def evalute_test_metrics(model, test_loader, device, num_fid_samples= 5000): 
    model.eval()
    originals, recons = [], []

    with torch.no_grad(): 
        for x , _ in test_loader: 
            x = x.to(device)
            x_hat, _, _ = model(x)
            originals.append(x.cpu())
            recons.append(x_hat.cpu())
    originals = torch.cat(originals)
    recons = torch.cat(recons)

    mse, psnr, ssim = compute_recon_metrics(originals, recons)

    latency_ms = measure_latency(lambda: model.sample(64, device=device), device=device)

    real_samples = originals[:num_fid_samples]
    fake_sample = model.sample(num_fid_samples, device=device).cpu()
    fid_score = compute_fid(real_samples, fake_sample, device=device)

    return {
        "test_mse": mse,
        "test_psnr": psnr,
        "test_ssim": ssim,
        "latency_ms": latency_ms,
        "fid_score": fid_score,
    }

def main():
    args = parse_args()
    set_seed(42)  # Seed wajib 42 sesuai panduan PDF

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, val_loader = get_datasets(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_losses": [], "val_losses": [], "recon_losses": [], "kl_losses": [], "epoch_times": []}

    # Loop 30 Epoch
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_recon, tr_kl = train_one_epoch(model, train_loader, optimizer, device, args.kl_weight)
        epoch_time = time.perf_counter() - t0
        val_loss, _, _ = evaluate_loss(model, val_loader, device, args.kl_weight)

        history["train_losses"].append(tr_loss)
        history["val_losses"].append(val_loss)
        history["recon_losses"].append(tr_recon)
        history["kl_losses"].append(tr_kl)
        history["epoch_times"].append(epoch_time)

        print(f"Epoch {epoch:02d}/{args.epochs} | Train: {tr_loss:.5f} (recon: {tr_recon:.5f}, kl: {tr_kl:.2f}) | Val: {val_loss:.5f} | {epoch_time:.1f}s")

    # Evaluasi Test Set Lengkap
    metrics = evalute_test_metrics(model, test_loader, device, args.fid_samples)

    # Simpan Bobot Model (.pt) dan Hasil Metrik (.json)
    save_checkpoint(model, optimizer, args.epochs, os.path.join(args.out_dir, "vae.pt"))
    
    # Format JSON disamakan dengan vanilla_ae.json
    results = {
        "model": "Variational Autoencoder (VAE)",
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "time_per_epoch_s": sum(history["epoch_times"]) / len(history["epoch_times"]),
        **metrics,
        "train_losses": history["train_losses"],
        "val_losses": history["val_losses"],
    }
    with open(os.path.join(args.results_dir, "vae.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__": 
    main()
