from torchvision import transforms
from torchvision import datasets
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch

def get_datasets(data_dir="data", batch_size=128, num_workers=0): 
    # Ubah jadi tensor pytorch , shape diubah jadi tambah grayscale , dan scale feature nya diubah jadi range 0 - 1 (bukan 0-255)
    transform = transforms.ToTensor()
    train_part = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_part = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    #train parameter itu split bawaan nya dia , true = ambil split train bawaan, train=false, ambil test split bawaan

    full_data = ConcatDataset([train_part, test_part])

    generator = torch.Generator().manual_seed(42)
    #pake manual seed lagi biar random seed nya selalu pake yg ini (bukan global, tktnya beda)

    train_set, val_set, test_set = random_split(full_data, [0.8, 0.1, 0.1], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, val_loader


if __name__ == "__main__": 
    train_loader,test_loader, val_loader  = get_datasets()
    print("Train:", len(train_loader.dataset)) #type: ignore
    print("Val:  ", len(val_loader.dataset)) #type: ignore
    print("Test: ", len(test_loader.dataset)) #type: ignore
    images, labels = next(iter(train_loader))
    print("Batch shape:", images.shape)
    print("Min/Max:", images.min().item(), images.max().item())