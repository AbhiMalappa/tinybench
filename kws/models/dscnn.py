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

    stem_stride downsamples the feature map at the stem (canonical Hello-Edge DS-CNN uses a
    strided first conv). This is REQUIRED for the STM32N6 Neural-ART NPU: its HW pooling unit only
    handles small windows, so the final global-average-pool maps to hardware only when the pre-pool
    map is small (≤ ~300 elements). With the original stride (1,1) the pre-pool map is 50×11 (=550),
    which forces a software-fallback float epoch and yields wrong NPU output (see
    firmware/stm32n6/FINDINGS.md). Verified all-HW (0 software epochs) pre-pool sizes:
        stem_stride=(2, 2) -> 25×6  (=150, ~3.5M MACC)   <- default, fastest, balanced
        stem_stride=(1, 2) -> 50×6  (=300, ~7.1M MACC)   <- keeps full time resolution (accuracy hedge)
        stem_stride=(2, 1) -> 25×11 (=275, ~6.5M MACC)
    All three also keep the CPU/Arduino/ESP32 cells valid (smaller maps => fewer MACCs => faster).
    """

    def __init__(self, n_classes=35, n_channels=64, n_ds_blocks=4, dropout=0.2,
                 stem_stride=(2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=(10, 4), stride=stem_stride, padding=(5, 2)),
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
