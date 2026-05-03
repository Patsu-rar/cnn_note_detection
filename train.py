"""
Training script for per-frame onset/offset + pitch detection on MIR-ST500.

Uses balanced sampling with context window CNN, extended with factored
pitch classification heads (octave × pitch class).

Experiments:
  1 – Baseline: single mel (n_fft=2048), no augmentation.
  2 – Multi-resolution: 3 stacked mels (1024/2048/4096).
  3 – Enhanced training: soft labels, stochastic augmentation, deeper
      backbone, pitch voting, probability smoothing.
  5 – Onset-focused retraining: fine-tune from exp3 with onset loss
      weight 1.5×, augmentations, 50 epochs.
  6 – Pitch-focused fine-tuning: freeze onset head, pitch loss 3.0×,
      3 epochs from exp5.

Usage:
    python train.py --experiment 1     # Baseline
    python train.py --experiment 2     # Multi-resolution
    python train.py --experiment 3     # Enhanced training
    python train.py --experiment 5     # Onset-focused
    python train.py --experiment 6     # Pitch-focused (final)
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from config import Config
from dataset import MIRST500Dataset
from model import OnsetOffsetPitchModel


def make_config(experiment: int) -> Config:
    """Build Config for the given experiment."""
    cfg = Config(experiment=experiment)

    tag = f"exp{experiment}"
    cfg.checkpoint_dir = f"./checkpoints/{tag}"
    cfg.cache_dir = "./cache"

    if experiment == 3:
        # Enhanced training: soft labels, stochastic aug, deeper backbone
        cfg.aug_noise = True
        cfg.aug_pitch_shift = True
        cfg.aug_probability = 0.5
        cfg.context_frames = 21
        cfg.epochs = 30
        cfg.use_cosine_lr = True
        cfg.threshold_search = True
        cfg.soft_label_spread = True
        cfg.soft_label_sigma = 1.0
        cfg.deeper_backbone = True
        cfg.pitch_voting = True
        cfg.smooth_probs = True
        cfg.smooth_kernel = 5
    elif experiment == 5:
        # Onset-focused retraining from exp3
        cfg.aug_noise = True
        cfg.aug_pitch_shift = True
        cfg.aug_gain = True
        cfg.aug_probability = 0.5
        cfg.context_frames = 21
        cfg.epochs = 50
        cfg.learning_rate = 5e-5
        cfg.use_cosine_lr = True
        cfg.threshold_search = True
        cfg.soft_label_spread = True
        cfg.soft_label_sigma = 1.5
        cfg.deeper_backbone = True
        cfg.pitch_voting = True
        cfg.smooth_probs = True
        cfg.smooth_kernel = 5
        cfg.onset_loss_weight = 1.5
        cfg.resume_from = "./checkpoints/exp3/best_model.pt"
    elif experiment == 6:
        # Pitch-focused fine-tuning from exp5
        cfg.context_frames = 21
        cfg.epochs = 3
        cfg.learning_rate = 2e-5
        cfg.use_cosine_lr = True
        cfg.threshold_search = True
        cfg.soft_label_spread = True
        cfg.soft_label_sigma = 1.5
        cfg.deeper_backbone = True
        cfg.pitch_voting = True
        cfg.smooth_probs = True
        cfg.smooth_kernel = 5
        cfg.onset_loss_weight = 1.0
        cfg.pitch_loss_weight = 3.0
        cfg.freeze_onset_head = True
        cfg.resume_from = "./checkpoints/exp5/best_model.pt"

    return cfg


def train(cfg: Config):
    device = torch.device(
        cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    )
    print(f"Device: {device}")

    # Data
    train_set = MIRST500Dataset(cfg, split="train")
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Batches per epoch: {len(train_loader):,}")

    # Model
    model = OnsetOffsetPitchModel(
        n_mels=cfg.n_mels,
        context_frames=cfg.context_frames,
        n_octaves=cfg.n_octaves,
        n_pitch_classes=cfg.n_pitch_classes,
        in_channels=cfg.in_channels,
        deeper_backbone=cfg.deeper_backbone,
    ).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # Load pretrained weights (fine-tuning)
    if cfg.resume_from:
        ckpt = torch.load(cfg.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded weights from {cfg.resume_from} (epoch {ckpt.get('epoch', '?')})")

    # Freeze onset head if requested (pitch-focused fine-tuning)
    if cfg.freeze_onset_head:
        for param in model.onset_head.parameters():
            param.requires_grad = False
        print("Onset head frozen")

    # Loss
    onset_criterion = nn.BCEWithLogitsLoss()
    offset_criterion = nn.BCEWithLogitsLoss()
    pitch_oct_criterion = nn.CrossEntropyLoss(ignore_index=cfg.pitch_ignore_index)
    pitch_cls_criterion = nn.CrossEntropyLoss(ignore_index=cfg.pitch_ignore_index)
    if cfg.offset_loss_weight != 1.0:
        print(f"Offset loss weight: {cfg.offset_loss_weight}x")
    if cfg.onset_loss_weight != 1.0:
        print(f"Onset loss weight: {cfg.onset_loss_weight}x")
    if cfg.pitch_loss_weight != 1.0:
        print(f"Pitch loss weight: {cfg.pitch_loss_weight}x")

    # Optimiser
    optimiser = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # LR scheduler
    scheduler = None
    if cfg.use_cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=cfg.epochs, eta_min=1e-6
        )
        print(f"Using CosineAnnealingLR (T_max={cfg.epochs}, eta_min=1e-6)")

    # Train
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    # History for plotting
    history = {
        "onset_loss": [], "offset_loss": [], "octave_loss": [], "class_loss": [],
        "total_loss": [], "onset_f1": [], "offset_f1": [], "pitch_acc": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        # Resample negatives each epoch for variety
        if hasattr(train_set, '_resample_negatives'):
            train_set._resample_negatives()

        model.train()
        total_on_loss = total_off_loss = total_oct_loss = total_cls_loss = 0.0
        n_batches = 0
        # Onset/offset diagnostics
        tp_on, fp_on, fn_on = 0, 0, 0
        tp_off, fp_off, fn_off = 0, 0, 0
        # Pitch diagnostics
        pitch_correct = pitch_total = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{cfg.epochs}", leave=True)
        for batch in pbar:
            mel = batch["mel"].to(device)
            on_target = batch["onset"].to(device)
            off_target = batch["offset"].to(device)
            p_oct_target = batch["pitch_octave"].to(device)
            p_cls_target = batch["pitch_class"].to(device)

            on_logits, off_logits, oct_logits, cls_logits = model(mel)

            loss_on = onset_criterion(on_logits, on_target)
            loss_off = offset_criterion(off_logits, off_target)
            loss_oct = pitch_oct_criterion(oct_logits, p_oct_target)
            loss_cls = pitch_cls_criterion(cls_logits, p_cls_target)

            loss = cfg.onset_loss_weight * loss_on + cfg.offset_loss_weight * loss_off + cfg.pitch_loss_weight * (loss_oct + loss_cls)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimiser.step()

            total_on_loss += loss_on.item()
            total_off_loss += loss_off.item()
            total_oct_loss += loss_oct.item()
            total_cls_loss += loss_cls.item()
            n_batches += 1

            with torch.no_grad():
                on_pred = (torch.sigmoid(on_logits) > 0.5).cpu()
                on_gt = (on_target > 0.5).cpu()
                tp_on += (on_pred & on_gt).sum().item()
                fp_on += (on_pred & ~on_gt).sum().item()
                fn_on += (~on_pred & on_gt).sum().item()

                off_pred = (torch.sigmoid(off_logits) > 0.5).cpu()
                off_gt = (off_target > 0.5).cpu()
                tp_off += (off_pred & off_gt).sum().item()
                fp_off += (off_pred & ~off_gt).sum().item()
                fn_off += (~off_pred & off_gt).sum().item()

                # Pitch accuracy (only on voiced frames)
                voiced_mask = p_oct_target != cfg.pitch_ignore_index
                if voiced_mask.any():
                    oct_correct = (oct_logits[voiced_mask].argmax(-1) ==
                                   p_oct_target[voiced_mask]).sum().item()
                    cls_correct = (cls_logits[voiced_mask].argmax(-1) ==
                                   p_cls_target[voiced_mask]).sum().item()
                    n_voiced = voiced_mask.sum().item()
                    pitch_correct += oct_correct + cls_correct
                    pitch_total += 2 * n_voiced

            if n_batches % 200 == 0:
                pbar.set_postfix(
                    on=f"{total_on_loss/n_batches:.4f}",
                    off=f"{total_off_loss/n_batches:.4f}",
                    oct=f"{total_oct_loss/n_batches:.4f}",
                    cls=f"{total_cls_loss/n_batches:.4f}",
                )

        avg_on = total_on_loss / n_batches
        avg_off = total_off_loss / n_batches
        avg_oct = total_oct_loss / n_batches
        avg_cls = total_cls_loss / n_batches
        avg_total = avg_on + avg_off + avg_oct + avg_cls

        on_p = tp_on / (tp_on + fp_on) if (tp_on + fp_on) > 0 else 0
        on_r = tp_on / (tp_on + fn_on) if (tp_on + fn_on) > 0 else 0
        on_f1 = 2 * on_p * on_r / (on_p + on_r) if (on_p + on_r) > 0 else 0

        off_p = tp_off / (tp_off + fp_off) if (tp_off + fp_off) > 0 else 0
        off_r = tp_off / (tp_off + fn_off) if (tp_off + fn_off) > 0 else 0
        off_f1 = 2 * off_p * off_r / (off_p + off_r) if (off_p + off_r) > 0 else 0

        pitch_acc = pitch_correct / max(pitch_total, 1)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:2d}/{cfg.epochs} | "
            f"loss: on={avg_on:.4f} off={avg_off:.4f} "
            f"oct={avg_oct:.4f} cls={avg_cls:.4f} total={avg_total:.4f} | "
            f"lr={optimiser.param_groups[0]['lr']:.2e} | {dt:.0f}s"
        )
        print(
            f"  Onset  → P={on_p:.3f} R={on_r:.3f} F1={on_f1:.3f} "
            f"(TP={tp_on:,} FP={fp_on:,} FN={fn_on:,})"
        )
        print(
            f"  Offset → P={off_p:.3f} R={off_r:.3f} F1={off_f1:.3f} "
            f"(TP={tp_off:,} FP={fp_off:,} FN={fn_off:,})"
        )
        print(f"  Pitch  → Acc={pitch_acc:.3f}")

        # Step LR scheduler
        if scheduler is not None:
            scheduler.step()

        # Record history
        history["onset_loss"].append(avg_on)
        history["offset_loss"].append(avg_off)
        history["octave_loss"].append(avg_oct)
        history["class_loss"].append(avg_cls)
        history["total_loss"].append(avg_total)
        history["onset_f1"].append(on_f1)
        history["offset_f1"].append(off_f1)
        history["pitch_acc"].append(pitch_acc)

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": avg_total,
            }, Path(cfg.checkpoint_dir) / "best_model.pt")
            print(f"  → Saved best (loss={avg_total:.4f})")

    # Final save
    torch.save({
        "epoch": cfg.epochs,
        "model_state_dict": model.state_dict(),
        "loss": avg_total,
    }, Path(cfg.checkpoint_dir) / "final_model.pt")
    print(f"\nDone. Models saved in {cfg.checkpoint_dir}/")

    # ── Plot training curves ──────────────────────────────────────────
    plot_training_curves(history, cfg)


def plot_training_curves(history: dict, cfg: Config):
    """Save training loss and metric curves as PNG."""
    epochs = range(1, len(history["total_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (1) Loss curves
    ax = axes[0]
    ax.plot(epochs, history["onset_loss"], label="Onset", marker="o", markersize=3)
    ax.plot(epochs, history["offset_loss"], label="Offset", marker="s", markersize=3)
    ax.plot(epochs, history["octave_loss"], label="Octave", marker="^", markersize=3)
    ax.plot(epochs, history["class_loss"], label="Class", marker="v", markersize=3)
    ax.plot(epochs, history["total_loss"], label="Total", linewidth=2, color="black")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) F1 curves
    ax = axes[1]
    ax.plot(epochs, history["onset_f1"], label="Onset F1", marker="o", markersize=3)
    ax.plot(epochs, history["offset_f1"], label="Offset F1", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title("Onset / Offset F1")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (3) Pitch accuracy
    ax = axes[2]
    ax.plot(epochs, history["pitch_acc"], label="Pitch Accuracy", marker="o",
            markersize=3, color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pitch Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Experiment {cfg.experiment} — Training Curves", fontsize=14)
    fig.tight_layout()
    plot_path = Path(cfg.checkpoint_dir) / "training_curves.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train singing transcription model.")
    parser.add_argument(
        "--experiment", type=int, default=1, choices=[1, 2, 3, 5, 6],
        help="Experiment number: 1=baseline, 2=multi-res, 3=enhanced, 5=onset-focused, 6=pitch-focused",
    )
    args = parser.parse_args()

    cfg = make_config(args.experiment)

    print(f"\n{'='*60}")
    print(f"  Experiment {cfg.experiment}")
    print(f"  Multi-resolution: {cfg.use_multi_res} (channels={cfg.in_channels})")
    print(f"  Augmentation: {cfg.augmentation_enabled}")
    if cfg.augmentation_enabled:
        aug_names = ["time_stretch", "pitch_shift", "spec_augment", "noise", "gain"]
        enabled = [n for n in aug_names if getattr(cfg, f'aug_{n}')]
        print(f"    Enabled: {', '.join(enabled)}")
    print(f"  Checkpoint dir: {cfg.checkpoint_dir}")
    print(f"{'='*60}\n")

    train(cfg)

