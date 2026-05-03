"""
MIR-ST500 Per-Frame Dataset with Pitch Labels.

Based on the proven step_by_step approach:
  - Per-frame classification with context window
  - Balanced sampling (equal positives and negatives per epoch)

Extended with pitch labels (octave + pitch class) for voiced frames.
Silent frames get pitch_ignore_index (ignored in CrossEntropyLoss).

Supports:
  - Multi-resolution mel input (Experiments 2–4): stacked mel spectrograms
    from window sizes 1024, 2048, 4096 → 3 input channels.
  - Data augmentation (Experiments 3–4): time-stretch, pitch-shift,
    SpecAugment, noise injection, random gain.
"""

import json
import os
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from tqdm import tqdm

from config import Config


# Audio file discovery

AUDIO_FILE_PATTERNS = {
    "vocal": [
        "Mixture_(Vocals)_htdemucs_ft.wav",
        "Vocals.wav", "Vocal.wav", "vocal.wav", "vocals.wav",
    ],
    "mixture": [
        "Mixture.mp3", "Mixture.wav", "mixture.mp3", "mixture.wav",
    ],
}


def find_audio(song_dir: str, source: str) -> str | None:
    patterns = AUDIO_FILE_PATTERNS.get(source, [])
    for name in patterns:
        p = os.path.join(song_dir, name)
        if os.path.exists(p):
            return p
    for ext in [".wav", ".mp3", ".flac"]:
        for f in os.listdir(song_dir):
            if f.lower().endswith(ext):
                return os.path.join(song_dir, f)
    return None


# Label generation

def make_labels(notes, n_frames, frame_dur, cfg: Config):
    """
    Per-frame labels: onset/offset + pitch.

    If cfg.soft_label_spread is True, onset/offset labels are spread with a
    Gaussian of width cfg.soft_label_sigma around each event frame.
    Otherwise, hard binary single-frame labels (proven approach).
    Pitch: octave + pitch_class for voiced frames, ignore_index for silence.
    """
    onset = np.zeros(n_frames, dtype=np.float32)
    offset = np.zeros(n_frames, dtype=np.float32)
    pitch_octave = np.full(n_frames, cfg.pitch_ignore_index, dtype=np.int64)
    pitch_class = np.full(n_frames, cfg.pitch_ignore_index, dtype=np.int64)

    for note in notes:
        note_on_sec = float(note[0])
        note_off_sec = float(note[1])
        midi_pitch = int(round(float(note[2])))

        on_frame = int(round(note_on_sec / frame_dur))
        off_frame = int(round(note_off_sec / frame_dur))

        if cfg.soft_label_spread:
            # Gaussian-spread labels
            on_sigma = cfg.soft_label_sigma
            off_sigma = cfg.soft_label_offset_sigma
            on_spread = int(np.ceil(3 * on_sigma))
            off_spread = int(np.ceil(3 * off_sigma))
            for delta in range(-max(on_spread, off_spread), max(on_spread, off_spread) + 1):
                # Onset
                if abs(delta) <= on_spread:
                    weight_on = np.exp(-0.5 * (delta / on_sigma) ** 2)
                    f_on = on_frame + delta
                    if 0 <= f_on < n_frames:
                        onset[f_on] = max(onset[f_on], weight_on)
                # Offset
                if abs(delta) <= off_spread:
                    weight_off = np.exp(-0.5 * (delta / off_sigma) ** 2)
                    f_off = off_frame + delta
                    if 0 <= f_off < n_frames:
                        offset[f_off] = max(offset[f_off], weight_off)
        else:
            # Single-frame onset/offset (no spread)
            if 0 <= on_frame < n_frames:
                onset[on_frame] = 1.0
            if 0 <= off_frame < n_frames:
                offset[off_frame] = 1.0

        # Pitch for voiced frames (onset to offset inclusive)
        clamped = max(cfg.midi_lowest,
                      min(midi_pitch, cfg.midi_lowest + cfg.n_octaves * 12 - 1))
        oct_label = (clamped - cfg.midi_lowest) // cfg.n_pitch_classes
        cls_label = clamped % cfg.n_pitch_classes

        f_start = max(0, on_frame)
        f_end = min(n_frames, off_frame + 1)
        pitch_octave[f_start:f_end] = oct_label
        pitch_class[f_start:f_end] = cls_label

    return onset, offset, pitch_octave, pitch_class


# Multi-resolution mel spectrogram

def compute_multi_res_mel(y: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Compute stacked mel spectrograms at multiple FFT window sizes.

    All spectrograms use the same hop_length so they have the same number
    of frames. Returns shape (C, n_mels, T) where C = len(multi_res_n_ffts).
    """
    mels = []
    for n_fft in cfg.multi_res_n_ffts:
        mel = librosa.feature.melspectrogram(
            y=y, sr=cfg.sample_rate, n_fft=n_fft,
            hop_length=cfg.hop_length, n_mels=cfg.n_mels,
            fmin=cfg.fmin, fmax=cfg.fmax,
        )
        mels.append(np.log(mel + 1e-8).astype(np.float32))

    # Align to shortest length (slight differences possible due to n_fft)
    min_len = min(m.shape[1] for m in mels)
    mels = [m[:, :min_len] for m in mels]

    return np.stack(mels, axis=0)  # (C, n_mels, T)


# Data augmentation

def apply_augmentations(context: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Apply enabled augmentations to a context window.

    context: (C, ctx_frames, n_mels) — already normalised spectrogram patch.
    Augmentations are applied in-place on copies and only during training.
    Each augmentation is applied with probability cfg.aug_probability.
    """
    p = cfg.aug_probability

    if cfg.aug_spec_augment and np.random.random() < p:
        context = _spec_augment(context, cfg)

    if cfg.aug_noise and np.random.random() < p:
        context = _add_noise(context, cfg)

    if cfg.aug_gain and np.random.random() < p:
        context = _random_gain(context, cfg)

    if cfg.aug_time_stretch and np.random.random() < p:
        context = _time_stretch_spec(context, cfg)

    if cfg.aug_pitch_shift and np.random.random() < p:
        context = _pitch_shift_spec(context, cfg)

    return context


def _spec_augment(context: np.ndarray, cfg: Config) -> np.ndarray:
    """SpecAugment: random frequency and time masking on spectrogram patches."""
    C, T, F = context.shape
    context = context.copy()

    # Frequency masks
    for _ in range(cfg.spec_freq_masks):
        f_width = np.random.randint(1, cfg.spec_freq_width + 1)
        f_start = np.random.randint(0, max(F - f_width, 1))
        context[:, :, f_start:f_start + f_width] = 0.0

    # Time masks
    for _ in range(cfg.spec_time_masks):
        t_width = np.random.randint(1, cfg.spec_time_width + 1)
        t_start = np.random.randint(0, max(T - t_width, 1))
        context[:, t_start:t_start + t_width, :] = 0.0

    return context


def _add_noise(context: np.ndarray, cfg: Config) -> np.ndarray:
    """Additive Gaussian noise."""
    noise = np.random.randn(*context.shape).astype(np.float32) * cfg.noise_std
    return context + noise


def _random_gain(context: np.ndarray, cfg: Config) -> np.ndarray:
    """Random gain (volume scaling) per channel."""
    lo, hi = cfg.gain_range
    gains = np.random.uniform(lo, hi, size=(context.shape[0], 1, 1)).astype(np.float32)
    return context * gains


def _time_stretch_spec(context: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Simulate time stretching by resampling the time axis of the
    spectrogram context window.
    """
    lo, hi = cfg.time_stretch_range
    rate = np.random.uniform(lo, hi)
    C, T, F = context.shape
    new_T = max(1, int(round(T * rate)))

    # Resample each channel along time axis
    stretched = np.zeros_like(context)
    indices = np.linspace(0, new_T - 1, T).astype(np.float32)
    src_indices = np.linspace(0, T - 1, new_T).astype(np.float32)

    for ch in range(C):
        # Simple linear interpolation resample
        from numpy import interp
        for f in range(F):
            original = context[ch, :, f]
            # Stretch then crop/pad back to original length
            stretched_col = np.interp(
                np.linspace(0, len(original) - 1, new_T),
                np.arange(len(original)),
                original,
            )
            # Resample back to original T
            stretched[ch, :, f] = np.interp(
                np.arange(T),
                np.linspace(0, T - 1, new_T),
                stretched_col,
            )

    return stretched


def _pitch_shift_spec(context: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Simulate pitch shifting by rolling the frequency axis of the
    spectrogram context window.
    """
    lo, hi = cfg.pitch_shift_range
    # Number of mel bins to shift (~1 semitone ≈ 1 mel bin at typical resolution)
    shift = np.random.randint(lo, hi + 1)
    if shift == 0:
        return context

    return np.roll(context, shift, axis=2)


# ── Dataset class ──────────────────────────────────────────────────────

class MIRST500Dataset(Dataset):
    """
    Per-frame dataset with balanced sampling.

    Each __getitem__ returns:
        mel:           (C, context_frames, n_mels) — context window
                       C=1 for baseline, C=3 for multi-resolution
        onset:         scalar 0 or 1
        offset:        scalar 0 or 1
        pitch_octave:  scalar int (0..n_octaves-1 or ignore_index)
        pitch_class:   scalar int (0..11 or ignore_index)
    """

    def __init__(self, cfg: Config, split="train"):
        self.cfg = cfg
        self.split = split
        self.context = cfg.context_frames
        self.half_ctx = self.context // 2
        self.use_multi_res = cfg.use_multi_res
        self.in_channels = cfg.in_channels

        data_dir = cfg.train_dir if split == "train" else cfg.test_dir
        cache_tag = "multires_v1" if self.use_multi_res else "pitch_v1"
        if cfg.soft_label_spread:
            cache_tag += "_soft"
            if cfg.soft_label_offset_sigma != cfg.soft_label_sigma:
                cache_tag += f"_offsig{cfg.soft_label_offset_sigma}"
        cache_dir = os.path.join(cfg.cache_dir, f"{split}_{cache_tag}")
        os.makedirs(cache_dir, exist_ok=True)

        with open(cfg.annotation_file, "r") as f:
            all_annotations = json.load(f)

        song_dirs = sorted(
            [d for d in os.listdir(data_dir)
             if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()],
            key=lambda x: int(x),
        )

        # Load all songs
        self.song_mels = []       # list of (C, n_mels, T) arrays
        self.song_means = []      # list of (C,) arrays
        self.song_stds = []       # list of (C,) arrays
        self.frame_indices = []   # (song_idx, frame_idx, onset, offset, p_oct, p_cls)
        skipped = 0

        print(f"[{split}] Loading {len(song_dirs)} songs "
              f"(multi-res={self.use_multi_res}, channels={self.in_channels})...")
        for sid in tqdm(song_dirs, desc=f"[{split}]"):
            song_dir = os.path.join(data_dir, sid)
            audio_path = find_audio(song_dir, cfg.audio_source)
            if audio_path is None:
                skipped += 1
                continue

            notes = all_annotations.get(sid)
            if notes is None:
                skipped += 1
                continue

            # Cache key includes multi-res config
            if self.use_multi_res:
                cache_extra = "|".join(str(n) for n in cfg.multi_res_n_ffts)
            else:
                cache_extra = str(cfg.n_fft)
            cache_key = hashlib.md5(
                f"{audio_path}|{cfg.sample_rate}|{cache_extra}|{cfg.hop_length}|"
                f"{cfg.n_mels}|{cfg.fmin}|{cfg.fmax}|{cfg.midi_lowest}|"
                f"{cfg.n_octaves}".encode()
            ).hexdigest()
            cache_file = os.path.join(cache_dir, f"{sid}_{cache_key}.npz")

            if os.path.exists(cache_file):
                data = np.load(cache_file)
                expected_keys = {"mel", "onsets", "offsets", "pitch_octave", "pitch_class"}
                if expected_keys.issubset(data.files) and data["mel"].ndim == 3:
                    mel_stack = data["mel"]       # (C, n_mels, T)
                    on_labels = data["onsets"]
                    off_labels = data["offsets"]
                    p_oct = data["pitch_octave"]
                    p_cls = data["pitch_class"]
                else:
                    os.remove(cache_file)

            if not os.path.exists(cache_file):
                try:
                    y, _ = librosa.load(audio_path, sr=cfg.sample_rate, mono=True)
                except Exception as e:
                    print(f"  Song {sid}: {e}")
                    skipped += 1
                    continue

                if self.use_multi_res:
                    mel_stack = compute_multi_res_mel(y, cfg)
                else:
                    mel_single = librosa.feature.melspectrogram(
                        y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft,
                        hop_length=cfg.hop_length, n_mels=cfg.n_mels,
                        fmin=cfg.fmin, fmax=cfg.fmax,
                    )
                    mel_stack = np.log(mel_single + 1e-8).astype(np.float32)[np.newaxis]  # (1, n_mels, T)

                n_frames = mel_stack.shape[2]
                on_labels, off_labels, p_oct, p_cls = make_labels(
                    notes, n_frames, cfg.frame_duration, cfg
                )
                np.savez(cache_file, mel=mel_stack, onsets=on_labels, offsets=off_labels,
                         pitch_octave=p_oct, pitch_class=p_cls)

            s_idx = len(self.song_mels)
            self.song_mels.append(mel_stack)  # (C, n_mels, T)
            # Per-channel mean/std
            self.song_means.append(mel_stack.mean(axis=(1, 2)))   # (C,)
            self.song_stds.append(mel_stack.std(axis=(1, 2)) + 1e-8)  # (C,)

            n_frames = mel_stack.shape[2]
            for f_idx in range(n_frames):
                self.frame_indices.append((
                    s_idx, f_idx,
                    on_labels[f_idx], off_labels[f_idx],
                    p_oct[f_idx], p_cls[f_idx],
                ))

        total_onset = sum(1 for x in self.frame_indices if x[2] > 0.5)
        print(f"[{split}] Loaded {len(self.song_mels)} songs, skipped {skipped}")
        print(f"[{split}] Total frames: {len(self.frame_indices):,}, "
              f"onset positives: {total_onset:,} "
              f"({100*total_onset/max(len(self.frame_indices),1):.2f}%)")

        # Balanced sampling (train only)
        if split == "train":
            self.pos_indices = []
            self.neg_indices = []
            for i, (s_idx, f_idx, on, off, p_oct, p_cls) in enumerate(self.frame_indices):
                if on > 0.5 or off > 0.5:
                    self.pos_indices.append(i)
                else:
                    self.neg_indices.append(i)
            self.pos_indices = np.array(self.pos_indices)
            self.neg_indices = np.array(self.neg_indices)
            self._resample_negatives()
            print(f"[{split}] Balanced: {len(self.pos_indices):,} pos + "
                  f"{len(self.pos_indices):,} neg = {len(self.epoch_indices):,} per epoch")

    def _resample_negatives(self):
        """Resample negatives for a new epoch."""
        n_pos = len(self.pos_indices)
        neg_sample = np.random.choice(self.neg_indices, size=n_pos, replace=False)
        self.epoch_indices = np.concatenate([self.pos_indices, neg_sample])
        np.random.shuffle(self.epoch_indices)

    def __len__(self):
        if self.split == "train":
            return len(self.epoch_indices)
        return len(self.frame_indices)

    def __getitem__(self, idx):
        if self.split == "train":
            real_idx = self.epoch_indices[idx]
        else:
            real_idx = idx

        s_idx, f_idx, onset_label, offset_label, p_oct, p_cls = self.frame_indices[real_idx]

        mel_stack = self.song_mels[s_idx]  # (C, n_mels, T)
        means = self.song_means[s_idx]     # (C,)
        stds = self.song_stds[s_idx]       # (C,)
        n_frames = mel_stack.shape[2]
        n_mels = mel_stack.shape[1]
        C = mel_stack.shape[0]

        # Extract context window with zero padding — shape (C, ctx, n_mels)
        context_channels = []
        for ch in range(C):
            cols = []
            for ci in range(f_idx - self.half_ctx, f_idx + self.half_ctx + 1):
                if 0 <= ci < n_frames:
                    col = (mel_stack[ch, :, ci] - means[ch]) / stds[ch]
                else:
                    col = np.zeros(n_mels, dtype=np.float32)
                cols.append(col)
            context_channels.append(np.stack(cols, axis=0))  # (ctx, n_mels)

        context = np.stack(context_channels, axis=0).astype(np.float32)  # (C, ctx, n_mels)

        # Apply augmentation (training only, experiments 3–4)
        if self.split == "train" and self.cfg.augmentation_enabled:
            context = apply_augmentations(context, self.cfg)

        return {
            "mel": torch.from_numpy(context),  # (C, ctx, n_mels)
            "onset": torch.tensor(onset_label, dtype=torch.float32),
            "offset": torch.tensor(offset_label, dtype=torch.float32),
            "pitch_octave": torch.tensor(int(p_oct), dtype=torch.long),
            "pitch_class": torch.tensor(int(p_cls), dtype=torch.long),
        }
