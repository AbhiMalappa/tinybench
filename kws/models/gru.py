import torch.nn as nn


class GRU48(nn.Module):
    """GRU with hidden size 48 for keyword spotting.

    Input shape: (batch, 1, n_frames=49, n_mfcc=10) — same as DS-CNN from dataset.
    Recurrent architecture included specifically to test whether the STM32N6 NPU
    accelerates non-CNN compute patterns (hypothesis: less speedup than CNNs).
    """

    def __init__(self, n_classes=35, n_mfcc=10, hidden_size=48):
        super().__init__()
        self.gru = nn.GRU(input_size=n_mfcc, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = x.squeeze(1)           # (batch, n_frames, n_mfcc)
        _, h = self.gru(x)         # h: (1, batch, hidden_size)
        return self.classifier(h.squeeze(0))
