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
            transforms.RandomRotation(degrees=10),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=10
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
