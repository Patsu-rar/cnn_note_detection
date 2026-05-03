"""
Training script for second-stage offset refinement model.

Trains on note-level mel regions extracted using ground truth annotations.
Requires first-stage mel cache to exist (run train.py --experiment 3 first).

Usage:
    python train_offset.py                     # Train with defaults (exp4)
    python train_offset.py --experiment 6      # Train 7-layer offset model for exp6
    python train_offset.py --epochs 30         # Custom epoch count
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from config import Config
from offset_model import OffsetRefineModel
from offset_dataset import NoteRegionDataset


def make_config():
    """Config for offset model training — uses exp3's mel cache."""
    cfg = Config(experiment=4)
    cfg.checkpoint_dir = "./checkpoints/exp4"
    cfg.cache_dir = "./cache"
    cfg.soft_label_spread = True  # match exp3 cache tag
    cfg.deeper_backbone = True
    return cfg


def make_config_exp5():
    """Config for augmented offset model training (exp5)."""
    cfg = Config(experiment=5)
    cfg.checkpoint_dir = "./checkpoints/exp5"
    cfg.cache_dir = "./cache"
    cfg.soft_label_spread = True
    cfg.deeper_backbone = True
    return cfg


def make_config_exp6():
    """Config for 7-layer offset model training (exp6)."""
    cfg = Config(experiment=6)
    cfg.checkpoint_dir = "./checkpoints/exp6"
    cfg.cache_dir = "./cache"
    cfg.soft_label_spread = True
    cfg.deeper_backbone = True
    return cfg


def train_offset(cfg, epochs=20, batch_size=64, lr=1e-3, augment=False,
                 n_temporal_layers=5, onset_jitter=2, offset_sigma=1.0,
                 peak_loss_weight=0.0, resume_from=""):
    device = torch.device(
        cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    )
    print(f"Device: {device}")
    print(f"Augmentation: {augment}")
    print(f"Temporal layers: {n_temporal_layers}, Onset jitter: ±{onset_jitter}, "
          f"Offset σ: {offset_sigma}, Peak loss weight: {peak_loss_weight}")

    # Data
    train_set = NoteRegionDataset(cfg, split="train", augment=augment,
                                  onset_jitter=onset_jitter,
                                  offset_sigma=offset_sigma)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    print(f"Note regions: {len(train_set):,}, Batches: {len(train_loader):,}")

    # Model
    model = OffsetRefineModel(
        n_mels=cfg.n_mels, in_channels=cfg.in_channels,
        n_temporal_layers=n_temporal_layers,
    ).to(device)
    print(f"Offset model parameters: {model.count_parameters():,}")

    # Resume from existing checkpoint (fine-tuning)
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed from {resume_from} (epoch {ckpt.get('epoch', '?')})")

    # Optimiser + scheduler
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=1e-6
    )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    history = {"loss": [], "peak_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_peak_correct = 0
        total_notes = 0
        t0 = time.time()

        for mel, target, mask in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}",
                                      leave=False):
            mel = mel.to(device)
            target = target.to(device)
            mask = mask.to(device)

            logits = model(mel, mask)  # (B, L)

            # Masked BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
            bce_loss = (bce_loss * mask).sum() / mask.sum()

            # Peak position loss: soft-argmax distance
            if peak_loss_weight > 0:
                probs_for_peak = torch.sigmoid(logits)
                frame_idx = torch.arange(logits.size(1), device=device).float().unsqueeze(0)
                # Weighted center of mass (soft peak position)
                weighted = probs_for_peak * mask
                soft_pred = (weighted * frame_idx).sum(dim=1) / (weighted.sum(dim=1) + 1e-8)
                soft_true = (target * mask * frame_idx).sum(dim=1) / ((target * mask).sum(dim=1) + 1e-8)
                peak_loss = F.smooth_l1_loss(soft_pred, soft_true)
                loss = bce_loss + peak_loss_weight * peak_loss
            else:
                loss = bce_loss

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            total_loss += loss.item()

            # Peak accuracy: does argmax match target argmax within ±2 frames?
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                for b in range(mel.size(0)):
                    valid_len = int(mask[b].sum().item())
                    if valid_len == 0:
                        continue
                    pred_peak = probs[b, :valid_len].argmax().item()
                    true_peak = target[b, :valid_len].argmax().item()
                    if abs(pred_peak - true_peak) <= 2:
                        total_peak_correct += 1
                    total_notes += 1

        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)
        peak_acc = total_peak_correct / max(total_notes, 1)
        dt = time.time() - t0

        history["loss"].append(avg_loss)
        history["peak_acc"].append(peak_acc)

        print(f"Epoch {epoch:2d}/{epochs} | loss={avg_loss:.4f} | "
              f"peak_acc={peak_acc:.3f} (±2f) | "
              f"lr={optimiser.param_groups[0]['lr']:.2e} | {dt:.0f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": avg_loss,
            }, Path(cfg.checkpoint_dir) / "offset_model.pt")
            print(f"  → Saved best (loss={avg_loss:.4f})")

    # Final save
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "loss": avg_loss,
    }, Path(cfg.checkpoint_dir) / "offset_model_final.pt")
    print(f"\nDone. Models saved in {cfg.checkpoint_dir}/")

    # Plot training curves
    plot_offset_curves(history, cfg)


def plot_offset_curves(history, cfg):
    """Save offset model training curves."""
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(epochs, history["loss"], marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Offset Refinement Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["peak_acc"], marker="o", markersize=3, color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Peak Accuracy (±2 frames)")
    ax.set_title("Offset Peak Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Experiment {cfg.experiment} — Offset Refinement Training", fontsize=14)
    fig.tight_layout()
    plot_path = Path(cfg.checkpoint_dir) / "offset_training_curves.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train second-stage offset refinement model."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--experiment", type=int, default=4, choices=[4, 5, 6],
                        help="4=original, 5=augmented, 6=7-layer improved")
    args = parser.parse_args()

    if args.experiment == 6:
        cfg = make_config_exp6()
        augment = True
        default_epochs = 50
        n_temporal_layers = 7
        onset_jitter = 4
        offset_sigma = 2.0
        peak_loss_weight = 0.0
        resume_from = ""
    elif args.experiment == 5:
        cfg = make_config_exp5()
        augment = True
        default_epochs = 30
        n_temporal_layers = 5
        onset_jitter = 2
        offset_sigma = 1.0
        peak_loss_weight = 0.0
        resume_from = ""
    else:
        cfg = make_config()
        augment = False
        default_epochs = 20
        n_temporal_layers = 5
        onset_jitter = 2
        offset_sigma = 1.0
        peak_loss_weight = 0.0
        resume_from = ""

    epochs = args.epochs if args.epochs != 20 else default_epochs

    print(f"\n{'='*60}")
    print(f"  Training Offset Refinement Model (Experiment {args.experiment})")
    print(f"  Checkpoint dir: {cfg.checkpoint_dir}")
    print(f"  Augmentation: {augment}")
    print(f"  Epochs: {epochs}")
    print(f"  Temporal layers: {n_temporal_layers}, Jitter: ±{onset_jitter}, σ: {offset_sigma}")
    print(f"  Peak loss weight: {peak_loss_weight}")
    if resume_from:
        print(f"  Resume from: {resume_from}")
    print(f"{'='*60}\n")

    train_offset(cfg, epochs=epochs, batch_size=args.batch_size,
                 lr=args.lr, augment=augment,
                 n_temporal_layers=n_temporal_layers,
                 onset_jitter=onset_jitter,
                 offset_sigma=offset_sigma,
                 peak_loss_weight=peak_loss_weight,
                 resume_from=resume_from)
