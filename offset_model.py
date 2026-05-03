"""
Second-stage offset refinement model.

Takes a mel spectrogram region starting from a detected note onset and
predicts per-frame offset probability. Uses dilated 1D convolutions for
a ~1.5s temporal receptive field — much wider than the first-stage CNN's
~480ms context window.

The key insight: offsets require seeing the full note's energy envelope
to determine where the note ends. Per-frame classification with narrow
context fundamentally cannot do this. This model sees the whole note.
"""

import torch
import torch.nn as nn


class DilatedResBlock(nn.Module):
    """1D dilated conv with residual connection."""
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3,
                              padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.bn(self.conv(x)))


class OffsetRefineModel(nn.Module):
    """
    Offset refinement via note-region analysis.

    Architecture:
      1. Frequency encoder (2D CNN): compresses mel bins to per-frame features
      2. Temporal stack (dilated 1D CNN): models note-level temporal patterns
      3. Per-frame classifier: outputs offset probability for each frame

    n_temporal_layers controls receptive field:
      5 layers → RF=63 frames ≈ 1.5s (exp10, exp13)
      7 layers → RF=255 frames ≈ 5.9s (exp15)

    Input: (B, C, n_mels, L) mel region from onset to search boundary
    Output: (B, L) per-frame offset logits
    """
    def __init__(self, n_mels=128, in_channels=3, n_temporal_layers=5):
        super().__init__()
        # Frequency encoder: compress n_mels → per-frame feature vector
        # Conv2d operates on (B, C, n_mels, L): pools along n_mels, preserves L
        self.freq_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4, 1)),           # n_mels: 128 → 32
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4, 1)),           # 32 → 8
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),  # 8 → 1, time preserved
        )
        # Output: (B, 128, 1, L)

        # Temporal model: dilated 1D conv stack for wide receptive field
        dilations = [2 ** i for i in range(n_temporal_layers)]
        temporal_layers = [DilatedResBlock(128, d) for d in dilations]
        temporal_layers.append(nn.Conv1d(128, 1, 1))
        self.temporal = nn.Sequential(*temporal_layers)

    def forward(self, mel, mask=None):
        """
        mel: (B, C, n_mels, L)
        mask: (B, L) binary mask for valid (non-padded) frames
        Returns: (B, L) offset logits
        """
        h = self.freq_encoder(mel)            # (B, 128, 1, L)
        h = h.squeeze(2)                      # (B, 128, L)
        logits = self.temporal(h).squeeze(1)   # (B, L)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = OffsetRefineModel(n_mels=128, in_channels=3)
    print(f"Parameters: {model.count_parameters():,}")
    x = torch.randn(4, 3, 128, 200)
    mask = torch.ones(4, 200)
    out = model(x, mask)
    print(f"Output shape: {out.shape}")  # (4, 200)
