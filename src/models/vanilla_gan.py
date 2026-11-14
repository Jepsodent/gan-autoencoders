from typing import Any
import torch
import torch.nn as nn

class VanillaGenerator(nn.Module): 
    def __init__(self, latent_dim = 100) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # Input: Noise z (vektor 64) -> Diperbesar bertahap -> Output: Citra 28x28 (784 piksel)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh(),  # Skala output [-1, 1]
            nn.Unflatten(1, (1, 28, 28)),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
    
class VanillaDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Input: 28x28 (784 piksel) -> Menyusut bertahap -> Output: 1 angka
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Mencegah Discriminator terlalu pintar dari awal
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),  # Output skor logit mentah (bisa positif/negatif)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)