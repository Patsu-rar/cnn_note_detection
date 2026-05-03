"""
Note-region dataset for second-stage offset refinement training.

Each sample is a mel spectrogram region from a note onset to the search
boundary (next onset or max duration), with a Gaussian target label at
the ground truth offset position.

Reuses mel spectrograms from the first-stage cache.
"""

import os
import json
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from config import Config
from dataset import find_audio


class NoteRegionDataset(Dataset):
    """
    Dataset of note-level mel regions for offset refinement.

    For each ground truth note: extract mel from onset_frame to
    min(onset + max_note_frames, next_onset, song_end).
    Target: Gaussian peak at the true offset position (relative).

    Includes onset jitter augmentation (±2 frames) during training
    to handle imprecise onset detection at test time.

    When augment=True, applies additional augmentations:
    - Additive Gaussian noise
    - Random gain scaling
    - Frequency masking (SpecAugment-style)
    """
    def __init__(self, cfg: Config, split="train", augment=False,
                 onset_jitter=2, offset_sigma=1.0):
        self.cfg = cfg
        self.split = split
        self.max_frames = cfg.max_note_frames
        self.n_mels = cfg.n_mels
        self.in_channels = cfg.in_channels
        self.augment = augment and (split == "train")
        self.onset_jitter = onset_jitter
        self.offset_sigma = offset_sigma

        data_dir = cfg.train_dir if split == "train" else cfg.test_dir

        # Find mel cache directory — try multiple cache tags since
        # the mel data is identical across soft-label configurations
        use_multi_res = len(cfg.multi_res_n_ffts) > 0 and cfg.experiment >= 2
        base_tag = "multires_v1" if use_multi_res else "pitch_v1"
        possible_tags = [
            base_tag + "_soft",      # exp6/exp9 cache
            base_tag,                # exp1-5 cache
        ]

        cache_dir = None
        for tag in possible_tags:
            candidate = os.path.join(cfg.cache_dir, f"{split}_{tag}")
            if os.path.isdir(candidate):
                cache_dir = candidate
                break

        if cache_dir is None:
            raise RuntimeError(
                f"No mel cache found for {split}. "
                f"Run train.py --experiment 6 first to generate mel cache."
            )

        print(f"[{split} offset] Using mel cache: {cache_dir}")

        with open(cfg.annotation_file, "r") as f:
            all_annotations = json.load(f)

        song_dirs = sorted(
            [d for d in os.listdir(data_dir)
             if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()],
            key=lambda x: int(x),
        )

        self.song_mels = []
        self.song_means = []
        self.song_stds = []
        self.notes = []  # (song_idx, onset_frame, region_end, rel_offset)
        fd = cfg.frame_duration
        skipped = 0

        print(f"[{split} offset] Loading {len(song_dirs)} songs...")
        for sid in tqdm(song_dirs, desc=f"[{split} offset]"):
            song_dir = os.path.join(data_dir, sid)
            audio_path = find_audio(song_dir, cfg.audio_source)
            if audio_path is None:
                skipped += 1
                continue

            song_notes = all_annotations.get(sid)
            if song_notes is None:
                skipped += 1
                continue

            # Build cache key (same as first-stage dataset)
            if use_multi_res:
                cache_extra = "|".join(str(n) for n in cfg.multi_res_n_ffts)
            else:
                cache_extra = str(cfg.n_fft)
            cache_key = hashlib.md5(
                f"{audio_path}|{cfg.sample_rate}|{cache_extra}|{cfg.hop_length}|"
                f"{cfg.n_mels}|{cfg.fmin}|{cfg.fmax}|{cfg.midi_lowest}|"
                f"{cfg.n_octaves}".encode()
            ).hexdigest()
            cache_file = os.path.join(cache_dir, f"{sid}_{cache_key}.npz")

            if not os.path.exists(cache_file):
                skipped += 1
                continue

            data = np.load(cache_file)
            if "mel" not in data.files or data["mel"].ndim != 3:
                skipped += 1
                continue

            mel_stack = data["mel"]  # (C, n_mels, T)
            s_idx = len(self.song_mels)
            self.song_mels.append(mel_stack)
            self.song_means.append(mel_stack.mean(axis=(1, 2)))     # (C,)
            self.song_stds.append(mel_stack.std(axis=(1, 2)) + 1e-8)  # (C,)

            T = mel_stack.shape[2]

            # Sort notes by onset time and extract regions
            sorted_notes = sorted(song_notes, key=lambda n: float(n[0]))
            for i, note in enumerate(sorted_notes):
                onset_sec = float(note[0])
                offset_sec = float(note[1])
                on_f = int(round(onset_sec / fd))
                off_f = int(round(offset_sec / fd))

                # Region end: next onset or max range or song end
                if i + 1 < len(sorted_notes):
                    next_on_f = int(round(float(sorted_notes[i + 1][0]) / fd))
                else:
                    next_on_f = T

                region_end = min(on_f + self.max_frames, next_on_f, T)
                rel_offset = off_f - on_f

                # Skip invalid notes
                if on_f >= T or region_end <= on_f:
                    continue
                if rel_offset < 1 or rel_offset >= region_end - on_f:
                    continue

                self.notes.append((s_idx, on_f, region_end, rel_offset))

        print(f"[{split} offset] {len(self.song_mels)} songs, "
              f"{len(self.notes):,} note regions, skipped {skipped}")

    def __len__(self):
        return len(self.notes)

    def __getitem__(self, idx):
        s_idx, onset_f, region_end, rel_offset = self.notes[idx]

        mel = self.song_mels[s_idx]
        means = self.song_means[s_idx]
        stds = self.song_stds[s_idx]
        C = mel.shape[0]
        T = mel.shape[2]

        # Onset jitter augmentation (train only)
        jitter = 0
        if self.split == "train":
            jitter = np.random.randint(-self.onset_jitter, self.onset_jitter + 1)
            onset_j = max(0, onset_f + jitter)
            # Don't shift past the offset
            if onset_j >= onset_f + rel_offset:
                onset_j = onset_f
                jitter = 0
        else:
            onset_j = onset_f

        actual_end = min(region_end, T)
        L = min(actual_end - onset_j, self.max_frames)
        if L <= 0:
            L = 1
            onset_j = max(0, actual_end - 1)

        # Extract and normalize mel region
        region = mel[:, :, onset_j:onset_j + L].copy()
        for c in range(C):
            region[c] = (region[c] - means[c]) / stds[c]

        # Data augmentation (train only)
        if self.augment:
            # Additive Gaussian noise
            if np.random.random() < 0.5:
                region = region + np.random.randn(*region.shape).astype(np.float32) * 0.1
            # Random gain per channel
            if np.random.random() < 0.5:
                gains = np.random.uniform(0.8, 1.2, size=(C, 1, 1)).astype(np.float32)
                region = region * gains
            # Frequency masking (SpecAugment-style)
            if np.random.random() < 0.5:
                f_width = np.random.randint(1, 16)
                f_start = np.random.randint(0, max(self.n_mels - f_width, 1))
                region[:, f_start:f_start + f_width, :] = 0.0

        # Target: Gaussian at adjusted relative offset position
        adj_rel_offset = rel_offset - jitter
        adj_rel_offset = max(0, min(adj_rel_offset, L - 1))

        target = np.zeros(L, dtype=np.float32)
        sigma = self.offset_sigma
        spread = int(np.ceil(3 * sigma))
        for delta in range(-spread, spread + 1):
            f = adj_rel_offset + delta
            if 0 <= f < L:
                target[f] = max(target[f], np.exp(-0.5 * (delta / sigma) ** 2))

        # Pad to max_frames
        padded_mel = np.zeros((C, self.n_mels, self.max_frames), dtype=np.float32)
        padded_mel[:, :, :L] = region
        padded_target = np.zeros(self.max_frames, dtype=np.float32)
        padded_target[:L] = target
        mask = np.zeros(self.max_frames, dtype=np.float32)
        mask[:L] = 1.0

        return (
            torch.from_numpy(padded_mel),
            torch.from_numpy(padded_target),
            torch.from_numpy(mask),
        )
