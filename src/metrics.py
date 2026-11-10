import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
import time
import torch
from src.data import get_datasets

def compute_recon_metrics(originals, recons): 
    mse = F.mse_loss(recons, originals).item()

    psnrs_metrics = PeakSignalNoiseRatio(data_range=1.0).to(originals.device)
    psnr = psnrs_metrics(recons, originals).item()

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(originals.device)
    ssim = ssim_metric(recons, originals).item()

    return mse, psnr, ssim

def compute_fid(real, fake, device): 

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device=device)
    # gambar grayscale , butuh 3 channel buat fid

    for i in range(0, len(real), 256): 
        batch = real[i:i+256].repeat(1,3,1,1).to(device)
        fid.update(batch, real=True)

    for i in range(0, len(fake), 256): 
        batch = fake[i:i+256].repeat(1,3,1,1).to(device)
        fid.update(batch, real=False)
    
    score = fid.compute().item()
    return score

def measure_latency(generate_fn, device, n_runs=20): 
    with torch.no_grad(): 
        for _ in range(3):
            generate_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(n_runs): 
            generate_fn()
        if device.type == "cuda": 
            torch.cuda.synchronize()
        end = time.perf_counter()
    return (end - start) / n_runs * 1000 


if __name__ == "__main__": 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, _ = get_datasets()
    x, _ = next(iter(test_loader))   # 1 batch = 128 gambar asli

    mse, psnr, ssim = compute_recon_metrics(x,x)
    print("A | MSE:", mse, "PSNR:", psnr, "SSIM:", ssim)
    assert mse == 0, "tes A gagal: MSE harus 0"
    assert abs(ssim - 1.0) < 1e-4, "tes A gagal, SSIM harus 1"

    # ---- Tes B: gambar asli + noise ----
    noisy = (x + 0.1 * torch.randn_like(x)).clamp(0, 1)
    mse, psnr, ssim = compute_recon_metrics(x, noisy)
    print("B | MSE:", mse, "PSNR:", psnr, "SSIM:", ssim)
    assert mse > 0, "tes B gagal: MSE harus > 0"
    assert ssim < 1, "tes B gagal: SSIM harus < 1"
    
    imgs = []
    total = 0
    for batch, _ in test_loader:
        imgs.append(batch)
        total += len(batch)
        if total >= 1000:
            break
    imgs = torch.cat(imgs)[:1000]
    # ---- Tes C: asli vs asli lain, skor harus kecil ----
    fid_real = compute_fid(imgs[:500], imgs[500:1000], device)
    print("C | FID asli vs asli:", fid_real)

    # ---- Tes D: asli vs noise, skor harus jauh lebih besar ----
    fid_noise = compute_fid(imgs[:500], torch.rand(500, 1, 28, 28), device)
    print("D | FID asli vs noise:", fid_noise)
    assert fid_noise > fid_real, "tes C/D gagal: noise harus skornya lebih besar"

    # ---- Tes E: latency ----
    ms = measure_latency(lambda: torch.randn(64, 1, 28, 28, device=device), device)
    print("E | Latency (ms):", ms)

    print("semua tes lolos")