import json
import os
import random
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

# ±100ms at 16kHz = ±1600 samples = ±5 MFCC frames (hop_length=320)
_MAX_FRAME_SHIFT = 5
# SpecAugment: mask up to 2 MFCC coefficient rows at a time
_FREQ_MASK_MAX = 2


class SpeechCommandsDataset(Dataset):
    """Wraps torchaudio SPEECHCOMMANDS and returns MFCC features.

    Returns tensors of shape (1, n_frames, n_mfcc).

    cache_dir behaviour:
      - Not set  → compute MFCC from WAV on every __getitem__ (slow, no disk needed)
      - Set, cache exists  → load pre-computed tensors instantly
      - Set, no cache yet  → build cache once, save to disk, then use it
    All three splits share the same cache_dir; each gets its own file.
    """

    def __init__(self, root, subset, config, stats=None, augment=False, cache_dir=None):
        self.target_samples = config['sample_rate']
        self.n_frames = config['n_frames']
        self.stats = stats
        self.augment = augment
        self.cache = None
        self.raw_data = None

        # Check if cache already exists — if so, skip the 2.3GB download entirely
        cache_path = os.path.join(cache_dir, f'mfcc_{subset}.pt') if cache_dir else None
        cache_ready = cache_path and os.path.exists(cache_path)

        if not cache_ready:
            # Need raw WAV files to build the cache (or to run without cache)
            os.makedirs(root, exist_ok=True)
            self.raw_data = torchaudio.datasets.SPEECHCOMMANDS(
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

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            if cache_ready:
                print(f"Loading MFCC cache: {cache_path}")
                self.cache = torch.load(cache_path, weights_only=True)
            else:
                self.cache = self._build_and_save_cache(cache_path, subset)

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f'mfcc_{subset}.pt')
            if os.path.exists(cache_path):
                print(f"Loading MFCC cache: {cache_path}")
                self.cache = torch.load(cache_path, weights_only=True)
            else:
                self.cache = self._build_and_save_cache(cache_path, subset)

    def _extract_mfcc(self, waveform):
        """Pad/trim to 1s, compute MFCC, return (n_mfcc, n_frames) unnormalized."""
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

    def _build_and_save_cache(self, cache_path, subset):
        """Compute MFCC for every sample once and save to disk."""
        n = len(self.raw_data)
        print(f"Building MFCC cache for '{subset}' ({n} samples) — runs once...")
        all_feats, all_labels = [], []
        for waveform, _, label, *_ in self.raw_data:
            all_feats.append(self._extract_mfcc(waveform))
            all_labels.append(LABEL_TO_IDX[label])
        cache = {
            'features': torch.stack(all_feats),   # (N, n_mfcc, n_frames)
            'labels': torch.tensor(all_labels),    # (N,)
        }
        torch.save(cache, cache_path)
        size_mb = os.path.getsize(cache_path) / 1024 / 1024
        print(f"Cache saved: {cache_path}  ({size_mb:.0f} MB)")
        return cache

    def _time_shift(self, feat):
        """Shift MFCC frames by ±5 frames (≈ ±100ms), zero-padding the gap."""
        shift = random.randint(-_MAX_FRAME_SHIFT, _MAX_FRAME_SHIFT)
        if shift == 0:
            return feat
        feat = torch.roll(feat, shift, dims=1)
        if shift > 0:
            feat[:, :shift] = 0
        else:
            feat[:, shift:] = 0
        return feat

    def _freq_mask(self, feat):
        """SpecAugment: zero out up to 2 consecutive MFCC coefficient rows."""
        n_mfcc = feat.shape[0]
        width = random.randint(1, _FREQ_MASK_MAX)
        start = random.randint(0, n_mfcc - width)
        feat[start:start + width, :] = 0
        return feat

    def __getitem__(self, idx):
        if self.cache is not None:
            feat = self.cache['features'][idx].clone()  # (n_mfcc, n_frames)
            label = self.cache['labels'][idx].item()
        else:
            waveform, _, label, *_ = self.raw_data[idx]
            feat = self._extract_mfcc(waveform)
            label = LABEL_TO_IDX[label]

        if self.augment:
            feat = self._time_shift(feat)
            feat = self._freq_mask(feat)

        if self.stats is not None:
            feat = (feat - self.stats['mean']) / (self.stats['std'] + 1e-8)

        feat = feat.T.unsqueeze(0)  # (1, n_frames, n_mfcc)
        return feat, label

    def __len__(self):
        if self.cache is not None:
            return len(self.cache['labels'])
        return len(self.raw_data)


def compute_normalization_stats(data_root, config, cache_dir=None):
    """Compute per-MFCC mean and std over the training set."""
    dataset = SpeechCommandsDataset(
        data_root, 'training', config, stats=None, augment=False, cache_dir=cache_dir
    )
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

    n_mfcc = config['n_mfcc']
    mean_sum = torch.zeros(n_mfcc)
    sq_sum = torch.zeros(n_mfcc)
    n_total = 0

    print("Computing normalization stats on training set...")
    for feats, _ in loader:
        mfcc = feats.squeeze(1)               # (batch, n_frames, n_mfcc)
        mean_sum += mfcc.sum(dim=(0, 1))
        sq_sum += (mfcc ** 2).sum(dim=(0, 1))
        n_total += mfcc.shape[0] * mfcc.shape[1]

    mean = mean_sum / n_total
    std = (sq_sum / n_total - mean ** 2).sqrt().clamp(min=1e-8)
    return {'mean': mean.unsqueeze(1), 'std': std.unsqueeze(1)}


def get_dataloaders(data_root, config_path, batch_size=64, num_workers=4,
                    stats_path=None, augment=False, cache_dir=None):
    with open(config_path) as f:
        config = json.load(f)

    if stats_path and os.path.exists(stats_path):
        stats = torch.load(stats_path, weights_only=True)
        print(f"Loaded normalization stats from {stats_path}")
    else:
        stats = compute_normalization_stats(data_root, config, cache_dir=cache_dir)
        if stats_path:
            torch.save(stats, stats_path)
            print(f"Saved normalization stats to {stats_path}")

    train_ds = SpeechCommandsDataset(data_root, 'training',   config, stats=stats, augment=augment,  cache_dir=cache_dir)
    val_ds   = SpeechCommandsDataset(data_root, 'validation', config, stats=stats, augment=False,    cache_dir=cache_dir)
    test_ds  = SpeechCommandsDataset(data_root, 'testing',    config, stats=stats, augment=False,    cache_dir=cache_dir)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader, stats
