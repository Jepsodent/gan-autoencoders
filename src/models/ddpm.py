import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    #mengubah langkah (t=x) menjadi vektor dimensi 64. 
    #tujuannya kasih tau U-net buat tau sbrp banyak/tebal noise pada langkah t 

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class Block(nn.Module): 
    def __init__(self, in_c: int, out_c: int, time_emb_dim = 64) -> None:
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(self.conv1(x)))
        # Suntikkan informasi waktu t ke dalam fitur gambar
        time_feat = self.act(self.time_mlp(t_emb))[(..., ) + (None, ) * 2]
        h = h + time_feat
        return self.act(self.norm(self.conv2(h)))
    
class UNet(nn.Module):
    def __init__(self, time_emb_dim: int = 64) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        # 1. Jalur Turun (Encoder)
        self.b1 = Block(1, 32, time_emb_dim)       # 28x28
        self.down1 = nn.Conv2d(32, 32, 4, 2, 1)    # Turun ke 14x14
        self.b2 = Block(32, 64, time_emb_dim)      # 14x14
        self.down2 = nn.Conv2d(64, 64, 4, 2, 1)    # Turun ke 7x7
        # 2.(Bottleneck)
        self.mid = Block(64, 64, time_emb_dim)     # 7x7
        # 3. Jalur Naik (Decoder dengan Skip Connection)
        self.up1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # Naik ke 14x14
        self.b3 = Block(96, 32, time_emb_dim)           # 32 (up) + 64 (skip) = 96 kanal
        self.up2 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # Naik ke 28x28
        self.b4 = Block(64, 32, time_emb_dim)           # 32 (up) + 32 (skip) = 64 kanal
        # Layer Akhir: Prediksi derau noise (1 kanal, 28x28)
        self.out = nn.Conv2d(32, 1, kernel_size=1)
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        # Turun
        h1 = self.b1(x, t_emb)
        h2 = self.b2(self.down1(h1), t_emb)
        # Tengah
        m = self.mid(self.down2(h2), t_emb)
        # Naik (pasang skip connection menggunakan torch.cat)
        u1 = self.b3(torch.cat([self.up1(m), h2], dim=1), t_emb)
        u2 = self.b4(torch.cat([self.up2(u1), h1], dim=1), t_emb)
        return self.out(u2)


class DDPM(nn.Module):
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    def __init__(self, timesteps: int = 300, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.unet = UNet()

        # Jadwal penaburan debu linear (Linear Beta Schedule)
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Simpan konstanta matematika sebagai buffer 
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        #Proses Maju: Campur gambar asli x_0 dengan noise murni pada langkah t
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        x_t = (sqrt_alpha * x_0) + (sqrt_one_minus_alpha * noise)
        return x_t, noise

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        #Proses Mundur: Dari derau acak x_T dibersihkan bertahap sampai jadi x_0
        self.eval()
        # 1. Mulai dari debu murni
        x = torch.randn(num_samples, 1, 28, 28, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            pred_noise = self.unet(x, t)

            beta_t = self.betas[i] 
            sqrt_recip_alpha_t = self.sqrt_recip_alphas[i]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[i]

            # Rumus reverse mean
            mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * pred_noise)
            if i > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

        return x.clamp(-1.0, 1.0)
    @torch.no_grad()
    def sample_trajectory(self, num_samples: int, device: torch.device, steps_to_save: list[int] | None = None) -> list[tuple[int, torch.Tensor]]:
        #simpen foto proses bersihin debu di langkah" tertentu, buat laporan
        self.eval()
        if steps_to_save is None:
            # Rekam di 6 titik waktu: debu murni -> makin bersih -> baju utuh
            steps_to_save = [299, 200, 150, 100, 50, 0]

        x = torch.randn(num_samples, 1, 28, 28, device=device)
        trajectory = []

        for i in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            pred_noise = self.unet(x, t)

            beta_t = self.betas[i]
            sqrt_recip_alpha_t = self.sqrt_recip_alphas[i]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[i]

            mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_cumprod_t) * pred_noise)
            if i > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

            if i in steps_to_save:
                trajectory.append((i, x.clamp(-1.0, 1.0).cpu()))

        return trajectory