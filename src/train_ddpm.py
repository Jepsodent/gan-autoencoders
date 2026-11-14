import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import get_datasets
from src.metrics import compute_fid, measure_latency
from src.models.ddpm import DDPM
from src.utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=300)  # 300 langkah 
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--fid_samples", type=int, default=5000)
    return parser.parse_args()


def train_one_epoch(
    model: DDPM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss, total_samples = 0.0, 0

    for x, _ in loader:
        b = x.size(0)
        total_samples += b

        # 1. Normalisasi gambar asli dari [0, 1] ke [-1, 1]
        x_0 = (x.to(device) * 2.0) - 1.0

        # 2. Pilih langkah waktu t secara acak untuk tiap gambar di batch (0 s/d 299)
        t = torch.randint(0, model.timesteps, (b,), device=device).long()

        # 3. noise acak ke gambar asli pada langkah t (Forward Process)
        noise = torch.randn_like(x_0)
        x_t, _ = model.q_sample(x_0, t, noise)

        # 4. U-Net tebak noise trsbt
        pred_noise = model.unet(x_t, t)

        # 5. Loss: MSE dari noise asli vs noise tebakan U-Net
        loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * b

    return total_loss / total_samples

def evaluate_test_metrics(
    model: DDPM,
    test_loader: DataLoader,
    device: torch.device,
    num_fid_samples: int = 5000,
) -> dict:
    model.eval()

    # 1. Ambil gambar asli dari Test Set
    real_imgs = []
    with torch.no_grad():
        for x, _ in test_loader:
            real_imgs.append(x)
            if len(torch.cat(real_imgs)) >= num_fid_samples:
                break
    real_imgs = torch.cat(real_imgs)[:num_fid_samples]

    # 2. Generator bersihin debu secara iteratif per batch (256) ->  jgn lgsg 5k
    fake_imgs = []
    batch_gen = 256
    with torch.no_grad():
        for i in range(0, num_fid_samples, batch_gen):
            cur_b = min(batch_gen, num_fid_samples - i)
            s = model.sample(cur_b, device=device)
            # Kembalikan dari [-1, 1] ke [0, 1] untuk penilaian FID
            fake_imgs.append(((s + 1.0) / 2.0).clamp(0.0, 1.0).cpu())
    fake_imgs = torch.cat(fake_imgs)

    # 3. Hitung skor FID
    fid_score = compute_fid(real_imgs, fake_imgs, device=device)

    # 4. Ukur Latensi inferensi untuk 1 batch (64 citra)
    latency_ms = measure_latency(
        lambda: model.sample(64, device=device),
        device=device,
        n_runs=5, #5 dh cukup , 20 kelamaan buat (karena tiap run harus ngulang 300 langkah untuk 64 gambar -> GPU -> sekaligus bersamaan)
    )

    return {
        "latency_ms": latency_ms,
        "fid_score": fid_score,
    }


def main():
    args = parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, _ = get_datasets(
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = DDPM(timesteps=args.timesteps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_losses": [], "epoch_times": []}

    print(f"=== Mulai Training DDPM ({args.epochs} Epochs, {args.timesteps} Timesteps) ===")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        loss = train_one_epoch(model, train_loader, optimizer, device)
        epoch_time = time.perf_counter() - t0

        history["train_losses"].append(loss)
        history["epoch_times"].append(epoch_time)

        print(f"Epoch {epoch:02d}/{args.epochs} | Noise MSE Loss: {loss:.5f} | {epoch_time:.1f}s")

    # Evaluasi metrik
    metrics = evaluate_test_metrics(model, test_loader, device, args.fid_samples)

    # save checkpoint model
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "ddpm.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, ckpt_path)

    # save hasil metrik JSON
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, "ddpm.json")
    results = {
        "model": "Denoising Diffusion Probabilistic Models (DDPM)",
        "latent_dim": "28x28 (Dimensi Citra)",
        "timesteps": args.timesteps,
        "epochs": args.epochs,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "time_per_epoch_s": sum(history["epoch_times"]) / len(history["epoch_times"]),
        **metrics,
        "train_losses": history["train_losses"],
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Selesai! Hasil disimpan ke {results_path} dan {ckpt_path}")


if __name__ == "__main__":
    main()