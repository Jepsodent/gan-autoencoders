import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module): 
    def __init__(self, latent_dim = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        #1. Jalur utama encoder (untuk ekstraksi fitur )
        self.encoder_backbone = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(784, 256), 
            nn.ReLU()
        )

        #2 cabang output laten *yg bkin beda vanila ae
        self.fc_mu = nn.Linear(256, latent_dim)  # Cabang 1: Prediksi Titik Pusat (μ)
        self.fc_logvar= nn.Linear(256, latent_dim) # Cabang 2: Prediksi Range nya (log σ²)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256,784), 
            nn.Sigmoid(),
            nn.Unflatten(1,(1,28,28))
        )

    def encode(self, x: torch.Tensor): 
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor: 
        #Rumus: z = μ + (σ * ε) , ε diambil dari distribusi normal acak N(0, 1)
        if self.training:  
            #transform log(σ²) menjadi σ
            std = torch.exp(0.5 *logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        # Saat evaluasi/testing rekonstruksi, pakai titik tengahnya (μ) saja
        # Biar gambarnya stabil, nggak goyang-goyang kena noise acak
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor: 
        return self.decoder(z)

    def forward(self, x: torch.Tensor): 
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
    
    def sample(self, num_samples:int, device: torch.device) -> torch.Tensor: 
        #ini generatifnya, gabutuh gambar asli,cukup lempar noise acak N(0, I)
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    