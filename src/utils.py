import os
import random
import numpy as np
import torch

#ngunci seed ajah  (ada 4 seed berbeda)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed_all(seed) #gpu

def save_checkpoint(model, optimizer, epoch, path):
    folder = os.path.dirname(path)
    if folder != "": 
        os.makedirs(folder, exist_ok=True)
    checkpoint = {
        "epoch": epoch, 
        "model_state": model.state_dict(), #state_dict = angka penting model , formatnya dictionary {nama_layer: tensor array} -> weight: tensor([[...]]) 
        "optimizer_state": optimizer.state_dict()
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path, device): 
    res = torch.load(path, map_location=device)
    model.load_state_dict(res["model_state"])
    optimizer.load_state_dict(res["optimizer_state"])
    return res["epoch"]
