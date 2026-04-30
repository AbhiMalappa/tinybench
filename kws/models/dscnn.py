import torch.nn as nn


class DSBlock(nn.Module):
    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DSCNN(nn.Module):
    """DS-CNN for keyword spotting. Zhang et al., 2017 — Hello Edge.
    Input shape: (batch, 1, n_frames=49, n_mfcc=10).
    """

    def __init__(self, n_classes=35, n_channels=64, n_ds_blocks=4, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=(10, 4), padding=(5, 2)),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.ds_blocks = nn.Sequential(
            *[DSBlock(n_channels, dropout=dropout) for _ in range(n_ds_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.ds_blocks(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)
