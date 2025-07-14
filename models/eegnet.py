
# === File: models/eegnet.py ===
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, n_channels=23, n_times=512, n_classes=2):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding='same', bias=False),
            nn.BatchNorm2d(8)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(0.25)
        )
        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(0.25)
        )
        self.classify = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * (n_times // 32), n_classes)
        )
    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return self.classify(x)