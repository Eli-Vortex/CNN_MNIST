from torch.utils.data import Dataset
from torchvision import datasets


class MNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
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
