"""Train a KWS model on Google Speech Commands v2.

Usage:
    python train.py --model dscnn
    python train.py --model tcresnet
    python train.py --model gru
    python train.py --model dscnn --epochs 50 --lr 1e-3
"""

import argparse
import json
import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kws.data.dataset import get_dataloaders
from kws.models.dscnn import DSCNN
from kws.models.tcresnet import TCResNet8
from kws.models.gru import GRU48


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='dscnn', choices=['dscnn', 'tcresnet', 'gru'])
    parser.add_argument('--data-root', default='./data/speechcommands')
    parser.add_argument('--config', default='./kws/mfcc_config.json')
    parser.add_argument('--epochs', type=int, default=150, help='Maximum epochs (early stopping usually kicks in before this)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--checkpoints-dir', default='./kws/checkpoints')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--augment', action='store_true', help='Enable time-shift augmentation during training')
    parser.add_argument('--cache-dir', default=None,
                        help='Directory for pre-computed MFCC cache. Built on first run, reused after.')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping: stop if val acc does not improve for this many epochs')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    stats_path = os.path.join(args.checkpoints_dir, 'mfcc_stats.pt')

    train_loader, val_loader, test_loader, stats = get_dataloaders(
        args.data_root, args.config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stats_path=stats_path,
        augment=args.augment,
        cache_dir=args.cache_dir,
    )

    with open(args.config) as f:
        config = json.load(f)

    if args.model == 'dscnn':
        model = DSCNN(n_classes=config['n_classes']).to(device)
    elif args.model == 'tcresnet':
        model = TCResNet8(n_classes=config['n_classes'], n_mfcc=config['n_mfcc']).to(device)
    elif args.model == 'gru':
        model = GRU48(n_classes=config['n_classes'], n_mfcc=config['n_mfcc']).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    epochs_no_improve = 0
    best_ckpt = os.path.join(args.checkpoints_dir, f'{args.model}_best.pt')

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
            f"val loss {val_loss:.4f} acc {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model': args.model,
                'model_state': model.state_dict(),
                'val_acc': val_acc,
                'config': config,
                'stats': stats,
            }, best_ckpt)
            print(f"  -> Best model saved (val acc {val_acc*100:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    _, test_acc = evaluate(model, test_loader, criterion, device)

    target = 0.93
    print(f"\nTest accuracy: {test_acc*100:.2f}%  |  Target: >{target*100:.0f}%  |  {'PASS' if test_acc >= target else 'FAIL'}")


if __name__ == '__main__':
    main()
