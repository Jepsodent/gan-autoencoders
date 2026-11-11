import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, latent_dim: int = 64, num_classes: int = 10) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x))

    def weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fc1, fc2 = self.encoder[1], self.encoder[3]
        assert isinstance(fc1, nn.Linear) and isinstance(fc2, nn.Linear)
        return fc1.weight, fc2.weight, self.classifier.weight


class LatentDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


if __name__ == "__main__":
    model = MLPClassifier()
    decoder = LatentDecoder()
    x = torch.rand(8, 1, 28, 28)

    z = model.encode(x)
    logits = model(x)
    x_hat = decoder(z)
    W1, W2, W3 = model.weights()

    print("z shape:", z.shape)
    print("logits shape:", logits.shape)
    print("x_hat shape:", x_hat.shape)
    print("W1, W2, W3:", W1.shape, W2.shape, W3.shape)
    print("z_MLP minimum:", z.min().item())
