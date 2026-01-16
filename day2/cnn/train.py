# train.py
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datasets.mnist_dataset import MNISTDataset
from datasets.mnist_dataset import MNISTDataset, get_transforms
from datasets.mnist_dataset import MNISTDataset
from utils.visualize import save_augmentation_comparison
from datasets.mnist_dataset import MNISTDataset, get_transforms




# 1. 数据
transform = transforms.ToTensor()
# 原始数据（无增强）
raw_dataset = MNISTDataset(
    root="./data",
    train=True,
    transform=get_transforms(train=False)
)

# 增强数据
aug_dataset = MNISTDataset(
    root="./data",
    train=True,
    transform=get_transforms(train=True)
)

# 生成 5 组对比样本
save_augmentation_comparison(
    raw_dataset,
    aug_dataset,
    indices=[0, 1, 2, 3, 4,5,6,7,8,9]
)
train_dataset = aug_dataset
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 2. 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 13 * 13, 10)
        )

    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN().to(device)

# 3. 训练组件
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 4. 训练 1 个 epoch
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % 100 == 0:
        print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

print("✅ 1 epoch finished")
