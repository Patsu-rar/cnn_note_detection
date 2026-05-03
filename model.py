"""
CNN model for per-frame onset/offset + pitch classification.

Based on the proven step_by_step CNN architecture (which achieved very good
onset/offset results), extended with pitch heads.

Input (B, C, context_frames, n_mels), each frame classified independently.
Output: onset (B,), offset (B,), pitch_octave (B, n_oct), pitch_class (B, n_cls)
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class OnsetOffsetPitchModel(nn.Module):
    """
    Per-frame CNN with 4 heads: onset, offset, pitch_octave, pitch_class.
    """

    def __init__(self, n_mels=128, context_frames=11,
                 n_octaves=4, n_pitch_classes=12, in_channels=1,
                 deeper_backbone=False, dedicated_heads=False):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),    # (B,32, ctx//2, n_mels//2)
            ConvBlock(32, 64),             # (B,64, ctx//4, n_mels//4)
            ConvBlock(64, 128),            # (B,128, ctx//8, n_mels//8)
        )

        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, context_frames, n_mels)
            feat = self.features(dummy)
            flat_size = feat.view(1, -1).shape[1]

        if deeper_backbone:
            self.backbone = nn.Sequential(
                nn.Linear(flat_size, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
            )
        else:
            self.backbone = nn.Sequential(
                nn.Linear(flat_size, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )

        head_dim = 256

        # Onset/offset heads
        if dedicated_heads:
            self.onset_head = nn.Sequential(
                nn.Linear(head_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )
            self.offset_head = nn.Sequential(
                nn.Linear(head_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )
        else:
            self.onset_head = nn.Linear(head_dim, 1)
            self.offset_head = nn.Linear(head_dim, 1)

        # Pitch heads
        self.pitch_octave_head = nn.Linear(head_dim, n_octaves)
        self.pitch_class_head = nn.Linear(head_dim, n_pitch_classes)

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: onset (B,), offset (B,), pitch_oct (B, n_oct), pitch_cls (B, n_cls)
        """
        # CNN feature extraction
        h = self.features(x)
        h = h.view(h.size(0), -1)
        h = self.backbone(h)  # (B, 256)

        onset = self.onset_head(h).squeeze(-1)
        offset = self.offset_head(h).squeeze(-1)
        pitch_oct = self.pitch_octave_head(h)
        pitch_cls = self.pitch_class_head(h)

        return onset, offset, pitch_oct, pitch_cls

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test baseline (1 channel)
    model = OnsetOffsetPitchModel()
    print(f"Baseline parameters: {model.count_parameters():,}")
    x = torch.randn(4, 1, 11, 128)
    on, off, p_oct, p_cls = model(x)
    print(f"onset={on.shape}, offset={off.shape}, "
          f"pitch_oct={p_oct.shape}, pitch_cls={p_cls.shape}")

    # Test multi-resolution (3 channels)
    model3 = OnsetOffsetPitchModel(in_channels=3)
    print(f"\nMulti-res parameters: {model3.count_parameters():,}")
    x3 = torch.randn(4, 3, 11, 128)
    on, off, p_oct, p_cls = model3(x3)
    print(f"onset={on.shape}, offset={off.shape}, "
          f"pitch_oct={p_oct.shape}, pitch_cls={p_cls.shape}")
