import torch
from src.models.vanilla_ae import VanillaAE


class SparseAE(VanillaAE): 
    def forward_with_hidden(self, x): 
        h = self.encoder[:3](x)  # Flatten → Linear(784, 256) → ReLU
        z = self.encoder[3](h) # Linear(256, 64)
        x_hat = self.decode(z)
        return x_hat, h
    

if __name__ == "__main__":
    model = SparseAE()
    x = torch.rand(8, 1, 28, 28)
    x_hat, h = model.forward_with_hidden(x)

    print("x_hat shape:", x_hat.shape)
    print("h shape:", h.shape)
    print("persen h == 0:", (h == 0).float().mean().item() * 100)
    print("sama dengan model(x)?", torch.allclose(x_hat, model(x)))
