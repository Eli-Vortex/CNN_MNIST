# train.py（Day 3：CNN 训练与评估）

import torch
import os
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.logger import get_logger
from datasets.mnist_dataset import MNISTDataset
from datasets.transforms import get_transforms
from models.cnn import SimpleCNN
import yaml

# ------------------------
# Config
# ------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ------------------------
# Train / Eval
# ------------------------
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


# ------------------------
# Checkpoint
# ------------------------
def save_checkpoint(path, model, optimizer, epoch, best_acc):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc
    }, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["best_acc"]


# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="path to config file"
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="path to checkpoint (optional)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # device
    if cfg["runtime"]["device"] == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg["runtime"]["device"]

    # logger
    logger = get_logger(cfg["log"]["log_dir"])
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {args.config}")

    # data
    train_dataset = MNISTDataset(
        root=cfg["data"]["root"],
        train=True,
        transform=get_transforms(train=True)
    )
    test_dataset = MNISTDataset(
        root=cfg["data"]["root"],
        train=False,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False
    )

    # model
    model = SimpleCNN(
        num_classes=cfg["model"]["num_classes"]
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["train"]["learning_rate"]
    )
    criterion = nn.CrossEntropyLoss()

    # resume
    start_epoch = 1
    best_acc = 0.0
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_acc = load_checkpoint(
            args.resume, model, optimizer, device
        )
        start_epoch += 1

    os.makedirs(cfg["checkpoint"]["dir"], exist_ok=True)

    # train loop
    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        acc = evaluate(model, test_loader, device)

        logger.info(
            f"Epoch [{epoch}/{cfg['train']['epochs']}], "
            f"Loss={loss:.4f}, Acc={acc*100:.2f}%"
        )

        # save best
        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(
                cfg["checkpoint"]["dir"], "best.pth"
            )
            save_checkpoint(
                save_path, model, optimizer, epoch, best_acc
            )
            logger.info(f"Best model saved: {save_path}")

    logger.info("✅ Training finished")


if __name__ == "__main__":
    main()