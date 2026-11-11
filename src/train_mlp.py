import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import get_datasets
from src.models.mlp_classifier import MLPClassifier
from src.utils import save_checkpoint, set_seed

TARGET_ACCURACY = 88.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total * 100


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total * 100


def main():
    args = parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader, test_loader, val_loader = get_datasets(
        batch_size=128, num_workers=args.num_workers
    )

    model = MLPClassifier(latent_dim=64, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    ckpt_path = os.path.join(args.out_dir, "mlp_classifier.pt")

    history = {
        "train_losses": [], "val_losses": [],
        "train_accs": [], "val_accs": [],
        "epoch_times": [],
    }
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        epoch_time = time.perf_counter() - t0
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accs"].append(train_acc)
        history["val_accs"].append(val_acc)
        history["epoch_times"].append(epoch_time)
        print(f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.2f}% | "
              f"val loss {val_loss:.4f} acc {val_acc:.2f}% | {epoch_time:.1f}s")

        save_checkpoint(model, optimizer, epoch, ckpt_path)

    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    status = "TERPENUHI" if test_acc >= TARGET_ACCURACY else "BELUM TERPENUHI"
    print(f"Test loss {test_loss:.4f} | test acc {test_acc:.2f}% "
          f"(syarat >= {TARGET_ACCURACY}%: {status})")

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    results = {
        "model": "MLP Classifier",
        "architecture": "784-256-64-10 (ReLU)",
        "latent_dim": 64,
        "epochs": args.epochs,
        "gpu": gpu_name,
        "time_per_epoch_s": sum(history["epoch_times"]) / len(history["epoch_times"]),
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "target_accuracy": TARGET_ACCURACY,
        **history,
    }
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, "mlp_classifier.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("hasil disimpan ke", results_path)


if __name__ == "__main__":
    main()
