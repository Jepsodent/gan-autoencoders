from typing import Any
import torch
import torch.nn as nn


class VanillaAE(nn.Module): 
    def __init__(self, latent_dim=64) -> None:
        super().__init__()
        self.encoder =nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,256), 
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(), 
            nn.Linear(256,784), 
            nn.Sigmoid(),
            nn.Unflatten(1, (1,28,28))
        )
    
    def encode(self, x): 
        return self.encoder(x)

    def decode(self, z): 
        return self.decoder(z)
    
    def forward(self, x): 
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

if __name__ == "__main__": 
    model  = VanillaAE()
    x = torch.rand(8,1,28,28)
    z = model.encode(x)
    x_hat = model(x) #auto jalanin forward , dari parent class

    print("z shape:", z.shape)
    print("x_hat shape:", x_hat.shape)
    print("x_hat min/max:", x_hat.min().item(), x_hat.max().item())
