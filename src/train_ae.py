import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import set_seed, save_checkpoint, load_checkpoint
from src.data import get_datasets
from src.models.vanilla_ae import VanillaAE
from src.metrics import compute_recon_metrics, measure_latency

from src.models.sparse_ae import SparseAE


#flag parser
def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", type=str, default="vanilla", choices=["vanilla", "sparse"])
    parser.add_argument("--sparsity_lambda", type=float, default=1e-3)
    return parser.parse_args()

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    sparsity_lambda: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    for x, _ in loader: 
        x = x.to(device)
        #sparsity ae
        if sparsity_lambda > 0: 
            x_hat, h = model.forward_with_hidden(x) #type: ignore
            recon_loss = loss_fn(x_hat, x)
            loss = recon_loss + sparsity_lambda * h.abs().mean()
        else:  
            x_hat = model(x)
            recon_loss = loss_fn(x_hat, x)
            loss = recon_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += recon_loss.item() * x.size(0)
    
    return total_loss / len(loader.dataset)  # type: ignore

def evaluate_loss(
        model: nn.Module, 
        loader: DataLoader, 
        loss_fn: nn.Module, 
        device: torch.device
) -> float: 
    model.eval()
    total_loss =0.0
    with torch.no_grad(): 
        for x, _ in loader: 
            x = x.to(device)
            x_hat = model(x)
            loss = loss_fn(x_hat, x)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)  # type: ignore

def measure_sparsity(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    #berapa persen neuron di h yang mati 
    model.eval()
    zeros, total = 0, 0
    with torch.no_grad():
        for x, _ in loader:
            h = model.encoder[:3](x.to(device))  # type: ignore
            zeros += (h == 0).sum().item()
            total += h.numel()
    return zeros / total * 100



def main(): 
    args = parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader, test_loader, val_loader = get_datasets(batch_size=128, num_workers=args.num_workers)
    if args.model == "sparse": 
        model = SparseAE(latent_dim=64).to(device)
        sparsity_lambda = args.sparsity_lambda
        name, display_name = "sparse_ae", "Sparse AE"
    else: 
        model = VanillaAE(latent_dim=64).to(device)
        sparsity_lambda = 0.0
        name, display_name = "vanilla_ae", "Vanilla AE"
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    ckpt_path = os.path.join(args.out_dir, f"{name}.pt")
    hist_path = os.path.join(args.out_dir, f"{name}_history.json")
    

    start_epoch= 1
    history = {"train_losses": [], "val_losses": [], "epoch_times": []}

    if args.resume and os.path.exists(ckpt_path): 
        last_epoch = load_checkpoint(model, optimizer, ckpt_path, device)
        start_epoch = last_epoch + 1
        with open(hist_path) as f: 
            history = json.load(f)
        print(f"Resume dari epoch: {start_epoch}")
    
    for epoch in range(start_epoch, args.epochs + 1): 
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, sparsity_lambda)
        epoch_time = time.perf_counter() - t0
        val_loss = evaluate_loss(model, val_loader, loss_fn, device)

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["epoch_times"].append(epoch_time)
        print(f"Epoch {epoch}/{args.epochs} | train {train_loss:.5f} | "
              f"val {val_loss:.5f} | {epoch_time:.1f}s")

        # save, biar kalau sesi putus progresnya nggak hilang
        save_checkpoint(model, optimizer, epoch, ckpt_path)
        with open(hist_path, "w") as f:
            json.dump(history, f)
        

    #evaluasi di test set
    model.eval()
    originals, recons = [], []
    with torch.no_grad(): 
        for x, _ in test_loader: 
            x = x.to(device)
            originals.append(x.cpu())
            recons.append(model(x).cpu())
    originals = torch.cat(originals)
    recons = torch.cat(recons)
    mse, psnr, ssim = compute_recon_metrics(originals, recons)
    x64 = next(iter(test_loader))[0][:64].to(device)
    latency_ms = measure_latency(lambda: model(x64), device)

    sparsity_pct = measure_sparsity(model, test_loader, device)

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"MSE {mse:.5f} | PSNR {psnr:.2f} dB | SSIM {ssim:.4f} | "
        f"sparsity {sparsity_pct:.1f}% | latency {latency_ms:.2f} ms | {gpu_name}")


    #save result to json
    results = {
        "model": display_name,
        "latent_dim": 64,
        "sparsity_lambda": sparsity_lambda,   
        "epochs": args.epochs,
        "gpu": gpu_name,
        "time_per_epoch_s": sum(history["epoch_times"]) / len(history["epoch_times"]),
        "latency_ms": latency_ms,
        "test_mse": mse,
        "test_psnr": psnr,
        "test_ssim": ssim,
        "test_sparsity_pct": sparsity_pct,     
        "train_losses": history["train_losses"],
        "val_losses": history["val_losses"],
    }
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, f"{name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("hasil disimpan ke", results_path)


if __name__ == "__main__": 
    main()
