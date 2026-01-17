# train.py（Day 3：CNN 训练与评估）

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from datasets.mnist_dataset import MNISTDataset
from datasets.transforms import get_transforms
from models.cnn import SimpleCNN
import yaml

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return correct / total


def main():

    cfg = load_config("configs/config.yaml")

    if cfg["runtime"]["device"] == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg["runtime"]["device"]
    print("Using device:", device)

    # 1. 数据
    train_dataset = MNISTDataset(
        root="./data",
        train=True,
        transform=get_transforms(train=True)
    )
    test_dataset = MNISTDataset(
        root="./data",
        train=False,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True
    )
    test_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False
    )

    # 2. 模型
    model = SimpleCNN().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["train"]["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss()

    # 3. 训练 3–5 epoch
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc*100:.2f}%")

    print("✅ Day 3 finished")


if __name__ == "__main__":
    main()
