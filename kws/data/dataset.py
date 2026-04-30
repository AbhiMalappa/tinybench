import json
import os
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset


# All 35 classes in Speech Commands v2, sorted alphabetically for determinism.
LABELS = sorted([
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
    'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
    'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila',
    'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero',
])
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}


class SpeechCommandsDataset(Dataset):
    """Wraps torchaudio SPEECHCOMMANDS and returns MFCC features.

    Returns tensors of shape (1, n_frames, n_mfcc) — time as height, MFCC as
    width, matching the DS-CNN stem kernel convention (10, 4).
    """

    def __init__(self, root, subset, config, stats=None):
        self.data = torchaudio.datasets.SPEECHCOMMANDS(
            root, url='speech_commands_v0.02', download=True, subset=subset
        )
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=config['sample_rate'],
            n_mfcc=config['n_mfcc'],
            melkwargs={
                'n_fft': config['n_fft'],
                'hop_length': config['hop_length'],
                'win_length': config['win_length'],
                'n_mels': config['n_mels'],
            },
        )
        self.target_samples = config['sample_rate']  # 16000
        self.n_frames = config['n_frames']           # 49
        self.stats = stats  # {'mean': (n_mfcc, 1), 'std': (n_mfcc, 1)}

    def _extract(self, waveform):
        """Compute MFCC and return (n_mfcc, n_frames) — raw, unnormalized."""
        if waveform.shape[-1] < self.target_samples:
            waveform = F.pad(waveform, (0, self.target_samples - waveform.shape[-1]))
        else:
            waveform = waveform[..., :self.target_samples]

        feat = self.mfcc_transform(waveform).squeeze(0)  # (n_mfcc, T)
        if feat.shape[1] > self.n_frames:
            feat = feat[:, :self.n_frames]
        elif feat.shape[1] < self.n_frames:
            feat = F.pad(feat, (0, self.n_frames - feat.shape[1]))
        return feat  # (n_mfcc, n_frames)

    def __getitem__(self, idx):
        waveform, _, label, *_ = self.data[idx]
        feat = self._extract(waveform)  # (n_mfcc, n_frames)

        if self.stats is not None:
            feat = (feat - self.stats['mean']) / (self.stats['std'] + 1e-8)

        # DS-CNN expects (1, n_frames, n_mfcc): transpose so time is the spatial height
        feat = feat.T.unsqueeze(0)  # (1, n_frames, n_mfcc)
        return feat, LABEL_TO_IDX[label]

    def __len__(self):
        return len(self.data)


def compute_normalization_stats(data_root, config):
    """Compute per-MFCC mean and std over the training set using an online pass.

    Returns {'mean': (n_mfcc, 1), 'std': (n_mfcc, 1)} — shaped for broadcasting
    against (n_mfcc, n_frames) before the final transpose in __getitem__.
    """
    dataset = SpeechCommandsDataset(data_root, 'training', config, stats=None)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

    n_mfcc = config['n_mfcc']
    mean_sum = torch.zeros(n_mfcc)
    sq_sum = torch.zeros(n_mfcc)
    n_total = 0

    print("Computing normalization stats on training set...")
    for feats, _ in loader:
        # feats: (batch, 1, n_frames, n_mfcc)
        mfcc = feats.squeeze(1)                  # (batch, n_frames, n_mfcc)
        mean_sum += mfcc.sum(dim=(0, 1))         # (n_mfcc,)
        sq_sum += (mfcc ** 2).sum(dim=(0, 1))
        n_total += mfcc.shape[0] * mfcc.shape[1]

    mean = mean_sum / n_total
    std = (sq_sum / n_total - mean ** 2).sqrt().clamp(min=1e-8)

    # Unsqueeze to (n_mfcc, 1) so they broadcast with (n_mfcc, n_frames)
    return {'mean': mean.unsqueeze(1), 'std': std.unsqueeze(1)}


def get_dataloaders(data_root, config_path, batch_size=64, num_workers=4, stats_path=None):
    with open(config_path) as f:
        config = json.load(f)

    if stats_path and os.path.exists(stats_path):
        stats = torch.load(stats_path, weights_only=True)
        print(f"Loaded normalization stats from {stats_path}")
    else:
        stats = compute_normalization_stats(data_root, config)
        if stats_path:
            torch.save(stats, stats_path)
            print(f"Saved normalization stats to {stats_path}")

    train_ds = SpeechCommandsDataset(data_root, 'training', config, stats=stats)
    val_ds = SpeechCommandsDataset(data_root, 'validation', config, stats=stats)
    test_ds = SpeechCommandsDataset(data_root, 'testing', config, stats=stats)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader, stats
