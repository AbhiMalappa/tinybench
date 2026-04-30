import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=9):
        super().__init__()
        pad = kernel_size // 2
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.main(x) + self.skip(x))


class TCResNet8(nn.Module):
    """TC-ResNet8 for keyword spotting. Choi et al., 2019.

    Input shape: (batch, 1, n_frames=49, n_mfcc=10) — same as DS-CNN from dataset.
    Treats n_mfcc coefficients as channels, n_frames as the 1D temporal axis.
    """

    def __init__(self, n_classes=35, n_mfcc=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_mfcc, 16, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResBlock(16, 24, stride=2),
            ResBlock(24, 32, stride=2),
            ResBlock(32, 48, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(48, n_classes)

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)  # (batch, n_mfcc, n_frames)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
