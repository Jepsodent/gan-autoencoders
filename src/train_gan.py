import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import get_datasets
from src.metrics import compute_fid, measure_latency
from src.utils import set_seed

from src.models.vanilla_gan import VanillaGenerator, VanillaDiscriminator
from src.models.dcgan import DCGANGenerator, DCGANDiscriminator


def weights_init_dcgan(m: nn.Module) -> None:
    # Inisialisasi bobot konvolusi normal(0, 0.02) agar training stabil
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

def parse_args():
    parser = argparse.ArgumentParser()
    # Pilihan model: mau latih vanilla atau dcgan
    parser.add_argument("--model", type=str, default="vanilla", choices=["vanilla", "dcgan"])
    parser.add_argument("--epochs", type=int, default=30)
    # LR standar GAN: 0.0002 dengan Adam beta1=0.5
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--fid_samples", type=int, default=5000)
    return parser.parse_args()

def train_one_epoch(
        generator: nn.Module, 
        discriminator: nn.Module,
        loader: DataLoader,
        opt_g: torch.optim.Optimizer,
        opt_d: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        latent_dim=  100
) -> tuple[float,float]: 
    generator.train()
    discriminator.train()
    total_d_loss = 0.0
    total_g_loss = 0.0
    total_samples = 0

    for x_real , _ in loader: 
        b_size = x_real.size(0)
        total_samples += b_size
        # 1. Skala gambar asli dari [0, 1] menjadi [-1, 1] agar sepadan dengan Tanh
        x_real = (x_real.to(device) * 2.0) - 1.0
        real_labels = torch.ones(b_size, 1, device=device)   # Label 1 = Asli
        fake_labels = torch.zeros(b_size, 1, device=device)  # Label 0 = Palsu

        # latih discriminator
        opt_d.zero_grad()
        # gammbar asli
        pred_real = discriminator(x_real)
        loss_d_real = criterion(pred_real, real_labels)

        #gambar palsu
        z = torch.randn(b_size, latent_dim, device=device)
        x_fake = generator(z)

        pred_fake = discriminator(x_fake.detach())
        loss_d_fake = criterion(pred_fake, fake_labels) #acuan nya 0 , jadi loss nya |0.1 - 0| = 0.1 (belajar dikit) -> 0 - 1 -> fake - real

        loss_d = (loss_d_real + loss_d_fake) / 2.0
        loss_d.backward()
        opt_d.step()


        opt_g.zero_grad()
        pred_fake_for_g = discriminator(x_fake)
        loss_g = criterion(pred_fake_for_g, real_labels) #acuannya 1 , jadi lossnya | 0.1 - 1| = 0.9 (belajar banyak)
        loss_g.backward()
        opt_g.step()

        total_d_loss += loss_d.item() * b_size
        total_g_loss += loss_g.item() * b_size

    return total_d_loss/total_samples, total_g_loss/total_samples

def evaluate_test_metrics(
    generator: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    latent_dim = 100,
    num_fid_samples = 5000,
) -> dict:
    generator.eval()

    # 1. Kumpulkan sampel asli dari test set (skala 0 - 1)
    real_imgs = []
    with torch.no_grad():
        for x, _ in test_loader:
            real_imgs.append(x)
            if len(torch.cat(real_imgs)) >= num_fid_samples:
                break
    real_imgs = torch.cat(real_imgs)[:num_fid_samples]

    # 2. Generator buat gambar baru dari noise acak
    with torch.no_grad():
        z = torch.randn(num_fid_samples, latent_dim, device=device)
        fake_tanh = generator(z)
        fake_imgs = ((fake_tanh + 1.0) / 2.0).clamp(0.0, 1.0).cpu()

    # 3. Hitung Skor FID (Perbandingan kemiripan distribusi gambar asli vs buatan)
    fid_score = compute_fid(real_imgs, fake_imgs, device=device)

    # 4. Ukur Latensi untuk 1 batch (64 citra)
    latency_ms = measure_latency(
        lambda: generator(torch.randn(64, latent_dim, device=device)),
        device=device,
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

    # Pemilihan arsitektur berdasarkan flag --model
    if args.model == "dcgan":
        name = "dcgan"
        display_name = "Deep Convolutional GAN (DCGAN)"
        generator = DCGANGenerator(latent_dim=args.latent_dim).to(device)
        discriminator = DCGANDiscriminator().to(device)
        # Pasang inisialisasi bobot khusus DCGAN
        generator.apply(weights_init_dcgan)
        discriminator.apply(weights_init_dcgan)
    else:
        name = "vanilla_gan"
        display_name = "Generative Adversarial Network (GAN)"
        generator = VanillaGenerator(latent_dim=args.latent_dim).to(device)
        discriminator = VanillaDiscriminator().to(device)

    # Optimizer Adam standar GAN: lr=0.0002, betas=(0.5, 0.999)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    history = {"d_losses": [], "g_losses": [], "epoch_times": []}

    print(f"=== Mulai Training {display_name} ({args.epochs} Epochs) ===")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        d_loss, g_loss = train_one_epoch(
            generator, discriminator, train_loader, opt_g, opt_d, criterion, device, args.latent_dim
        )
        epoch_time = time.perf_counter() - t0

        history["d_losses"].append(d_loss)
        history["g_losses"].append(g_loss)
        history["epoch_times"].append(epoch_time)

        print(f"Epoch {epoch:02d}/{args.epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f} | {epoch_time:.1f}s")

    # Evaluasi metrik 
    metrics = evaluate_test_metrics(generator, test_loader, device, args.latent_dim, args.fid_samples)

    # save Checkpoint Model
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{name}.pt")
    torch.save({
        "epoch": args.epochs,
        "generator_state": generator.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "opt_g_state": opt_g.state_dict(),
        "opt_d_state": opt_d.state_dict(),
    }, ckpt_path)

    # save hasil Metrik JSON
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, f"{name}.json")
    results = {
        "model": display_name,
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "time_per_epoch_s": sum(history["epoch_times"]) / len(history["epoch_times"]),
        **metrics,
        "d_losses": history["d_losses"],
        "g_losses": history["g_losses"],
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Selesai! Hasil disimpan ke {results_path} dan {ckpt_path}")


if __name__ == "__main__":
    main()