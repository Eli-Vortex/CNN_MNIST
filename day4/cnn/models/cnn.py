# models/cnn.py
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),  # 28→26
            nn.ReLU(),
            nn.MaxPool2d(2),         # 26→13

            nn.Conv2d(16, 32, 3, 1), # 13→11
            nn.ReLU(),
            nn.MaxPool2d(2)          # 11→5
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
