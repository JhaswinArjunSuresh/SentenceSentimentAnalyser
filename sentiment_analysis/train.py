import torch
import torch.nn.functional as F
from tqdm import tqdm

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in tqdm(dataloader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)