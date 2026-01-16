from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

class MNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        """
        root: 数据存放路径
        train: 是否使用训练集
        transform: 数据预处理 / 增强
        """
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label

# datasets/transforms.py
from torchvision import transforms


def get_transforms(train=True):
    """
    数据增强策略定义

    train=True  : 训练集，启用完整数据增强
    train=False : 测试集，仅做基础预处理
    """
    if train:
        return transforms.Compose([
            # 1️⃣ 随机旋转（模拟手写倾斜）
            transforms.RandomRotation(degrees=10),

            # 2️⃣ 随机裁剪（模拟位置偏移）
            # 先 padding 再裁剪，保持尺寸不变
            transforms.RandomCrop(28, padding=4),

            # 3️⃣ 随机仿射变换（核心：平移 + 缩放 + 剪切）
            transforms.RandomAffine(
                degrees=0,                 # 旋转已单独处理
                translate=(0.1, 0.1),      # 平移
                scale=(0.9, 1.1),          # 缩放
                shear=10                   # ⭐ 剪切（关键）
            ),

            # 4️⃣ 转为 Tensor
            transforms.ToTensor(),

            # 5️⃣ 归一化
            transforms.Normalize(
                mean=(0.1307,),
                std=(0.3081,)
            )
        ])
    else:
        # 测试集：禁止任何几何增强
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.1307,),
                std=(0.3081,)
            )
        ])

