from typing import Any
import torch
import torch.nn as nn

class DCGANGenerator(nn.Module): 
    def __init__(self, latent_dim = 100) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # Proyeksikan noise 100 ke grid 7x7 dengan 128 kanal fitur
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
        )
        # Perbesar ukuran bertahap: (7x7) -> (14x14) -> (28x28)
        self.deconv = nn.Sequential(
            # Dari 7x7 naik jadi 14x14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Dari 14x14 naik jadi 28x28 (citra utuh Fashion-MNIST)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(), 
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, 128, 7, 7)  # Ubah jadi matriks 3D (Kanal, Tinggi, Lebar)
        return self.deconv(h)
    

class DCGANDiscriminator(nn.Module): 
    def __init__(self) -> None:
        super().__init__()
        # Input: 28x28 -> Menyusut bertahap: (28x28) -> (14x14) -> (7x7)
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2: Dari 14x14 turun jadi 7x7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    