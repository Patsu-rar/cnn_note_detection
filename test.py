"""
Evaluation on MIR-ST500 test split.

Per-frame inference: for each frame, extract context window and run model.
Then post-process to extract notes (onset, offset, pitch) and evaluate
with COn / COnP / COnPOff metrics via mir_eval.transcription.

Experiments:
  1 – Baseline: single mel (n_fft=2048), no augmentation.
  2 – Multi-resolution: 3 stacked mels (1024/2048/4096).
  3 – Enhanced training: soft labels, deeper backbone, pitch voting.
  4 – Two-stage: exp3 CNN + offset refinement model.
  5 – Onset-focused retraining from exp3.
  6 – Pitch-focused fine-tuning from exp5 + 7-layer offset model.

Usage:
    python test.py --experiment 1
    python test.py --experiment 6
    python test.py --experiment 2 --checkpoint checkpoints/exp2/best_model.pt
"""

import argparse
import json
import os
import numpy as np
import torch
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from model import OnsetOffsetPitchModel
from dataset import compute_multi_res_mel

try:
    import mir_eval
    HAS_MIR_EVAL = True
except ImportError:
    HAS_MIR_EVAL = False
    print("Warning: mir_eval not found. Using basic metrics.")


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


# Post-processing

def median_smooth(arr, kernel_size):
    """Apply 1D median filter to a numpy array."""
    if kernel_size <= 1:
        return arr
    pad = kernel_size // 2
    padded = np.pad(arr, pad, mode='edge')
    out = np.empty_like(arr)
    for i in range(len(arr)):
        out[i] = np.median(padded[i:i + kernel_size])
    return out

def peak_pick(activation, threshold, min_distance):
    peaks = []
    last_peak = -min_distance
    for i in range(len(activation)):
        if activation[i] < threshold:
            continue
        lo = max(0, i - min_distance)
        hi = min(len(activation), i + min_distance + 1)
        if activation[i] >= activation[lo:hi].max():
            if (i - last_peak) >= min_distance:
                peaks.append(i)
                last_peak = i
    return peaks


def decode_pitch(oct_logits, cls_logits, cfg):
    """Decode MIDI pitch from per-frame octave/class logits."""
    oct_pred = oct_logits.argmax()
    cls_pred = cls_logits.argmax()
    if oct_pred >= cfg.n_octaves or cls_pred >= cfg.n_pitch_classes:
        return -1
    return cfg.midi_lowest + int(oct_pred) * cfg.n_pitch_classes + int(cls_pred)


def decode_pitch_voting(oct_logits_range, cls_logits_range, cfg):
    """
    Majority-vote MIDI pitch across a range of frames.
    oct_logits_range: (N, n_octaves), cls_logits_range: (N, n_pitch_classes)
    """
    if len(oct_logits_range) == 0:
        return -1
    oct_preds = oct_logits_range.argmax(axis=1)
    cls_preds = cls_logits_range.argmax(axis=1)
    midi_preds = cfg.midi_lowest + oct_preds * cfg.n_pitch_classes + cls_preds
    values, counts = np.unique(midi_preds, return_counts=True)
    best = values[counts.argmax()]
    return int(best)


def decode_pitch_soft(oct_logits_range, cls_logits_range, cfg):
    """
    Soft pitch aggregation: average full-resolution pitch distributions
    across frames, then argmax. Avoids octave/pitch-class boundary errors.

    For each frame, computes softmax(octave) × softmax(pitch_class) to get
    a full (n_octaves × n_pitch_classes)-dimensional distribution, averages
    these distributions, then takes argmax.
    """
    if len(oct_logits_range) == 0:
        return -1

    N = len(oct_logits_range)
    n_oct = cfg.n_octaves
    n_cls = cfg.n_pitch_classes

    def _softmax(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    oct_probs = _softmax(oct_logits_range)   # (N, n_oct)
    cls_probs = _softmax(cls_logits_range)   # (N, n_cls)

    # Outer product per frame → full pitch distribution (N, n_oct * n_cls)
    full_probs = (oct_probs[:, :, None] * cls_probs[:, None, :]).reshape(N, n_oct * n_cls)
    avg_probs = full_probs.mean(axis=0)

    best_idx = int(avg_probs.argmax())
    return cfg.midi_lowest + best_idx


def extract_predicted_notes(onset_probs, offset_probs, oct_logits, cls_logits,
                            cfg, fd):
    """Extract note list from per-frame model outputs with improved offset pairing."""
    if cfg.smooth_probs:
        onset_probs = median_smooth(onset_probs, cfg.smooth_kernel)
        offset_probs = median_smooth(offset_probs, cfg.smooth_kernel)

    onset_frames = peak_pick(onset_probs, cfg.onset_threshold, cfg.peak_min_distance)
    offset_frames = peak_pick(offset_probs, cfg.offset_threshold, cfg.peak_min_distance)

    notes = []
    for i, on_f in enumerate(onset_frames):
        next_onset = onset_frames[i + 1] if i + 1 < len(onset_frames) else len(onset_probs)
        max_off = min(on_f + cfg.max_note_frames, next_onset, len(onset_probs) - 1)

        # Find nearest offset peak after onset but before boundary
        off_f = None
        for of in offset_frames:
            if of > on_f and of <= max_off:
                off_f = of
                break

        # Fallback: use point of lowest onset prob in the valid range
        if off_f is None:
            search_start = on_f + cfg.min_note_frames
            search_end = max_off
            if search_start < search_end:
                region = onset_probs[search_start:search_end]
                off_f = search_start + int(np.argmin(region))
            else:
                off_f = max_off

        if off_f - on_f < cfg.min_note_frames:
            off_f = min(on_f + cfg.min_note_frames, len(onset_probs) - 1)

        # Pitch: soft aggregation, majority vote, or single-frame
        if cfg.pitch_soft_voting and off_f > on_f:
            midi = decode_pitch_soft(
                oct_logits[on_f:off_f + 1], cls_logits[on_f:off_f + 1], cfg
            )
        elif cfg.pitch_voting and off_f > on_f:
            midi = decode_pitch_voting(
                oct_logits[on_f:off_f + 1], cls_logits[on_f:off_f + 1], cfg
            )
        else:
            midi = decode_pitch(oct_logits[on_f], cls_logits[on_f], cfg)
        if midi < 0:
            continue

        notes.append([on_f * fd, off_f * fd, float(midi)])
    return notes


# Second-stage offset refinement

def extract_notes_refined(onset_probs, offset_probs, oct_logits, cls_logits,
                          mel_stack, means, stds, offset_model, cfg, fd, device):
    """
    Extract notes using first-stage onsets + second-stage offset refinement.

    For each detected onset, extracts the mel region and runs the offset
    refinement model to find a precise offset position. This gives the
    offset model access to the full note's energy envelope (~4.6s max),
    vastly more context than the first-stage CNN's ~480ms window.
    """
    if cfg.smooth_probs:
        onset_probs = median_smooth(onset_probs, cfg.smooth_kernel)

    onset_frames = peak_pick(onset_probs, cfg.onset_threshold, cfg.peak_min_distance)

    T = len(onset_probs)
    C, n_mels = mel_stack.shape[0], mel_stack.shape[1]

    if len(onset_frames) == 0:
        return []

    # Prepare batched inputs for offset model
    batch_mels = []
    batch_masks = []
    batch_lengths = []
    valid_indices = []

    for i, on_f in enumerate(onset_frames):
        next_onset = onset_frames[i + 1] if i + 1 < len(onset_frames) else T
        max_off = min(on_f + cfg.max_note_frames, next_onset, T)
        L = max_off - on_f

        if L < cfg.min_note_frames:
            continue

        # Extract and normalize mel region
        region = mel_stack[:, :, on_f:max_off].copy()
        for c in range(C):
            region[c] = (region[c] - means[c]) / stds[c]

        # Pad to max_note_frames
        padded = np.zeros((C, n_mels, cfg.max_note_frames), dtype=np.float32)
        padded[:, :, :L] = region
        mask = np.zeros(cfg.max_note_frames, dtype=np.float32)
        mask[:L] = 1.0

        batch_mels.append(padded)
        batch_masks.append(mask)
        batch_lengths.append(L)
        valid_indices.append(i)

    if not batch_mels:
        return []

    # Batched inference through offset model
    mel_batch = torch.from_numpy(np.stack(batch_mels)).to(device)
    mask_batch = torch.from_numpy(np.stack(batch_masks)).to(device)

    all_probs = []
    with torch.no_grad():
        sub_batch = 64
        for start in range(0, len(batch_mels), sub_batch):
            end = min(start + sub_batch, len(batch_mels))
            logits = offset_model(mel_batch[start:end], mask_batch[start:end])
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0)

    # Build notes from onset + refined offset
    notes = []
    for j, oi in enumerate(valid_indices):
        on_f = onset_frames[oi]
        L = batch_lengths[j]
        probs = all_probs[j, :L].copy()

        # Find offset: argmax after min_note_frames
        search_start = cfg.min_note_frames
        if search_start < L:
            off_rel = search_start + int(np.argmax(probs[search_start:]))
        else:
            off_rel = L - 1
        off_f = on_f + off_rel

        if off_f - on_f < cfg.min_note_frames:
            off_f = min(on_f + cfg.min_note_frames, T - 1)

        # Pitch
        if cfg.pitch_soft_voting and off_f > on_f:
            midi = decode_pitch_soft(
                oct_logits[on_f:off_f + 1], cls_logits[on_f:off_f + 1], cfg
            )
        elif cfg.pitch_voting and off_f > on_f:
            midi = decode_pitch_voting(
                oct_logits[on_f:off_f + 1], cls_logits[on_f:off_f + 1], cfg
            )
        else:
            midi = decode_pitch(oct_logits[on_f], cls_logits[on_f], cfg)
        if midi < 0:
            continue

        notes.append([on_f * fd, off_f * fd + cfg.offset_bias_sec, float(midi)])

    return notes


# Per-frame inference on full song

def infer_full_song(mel_stack, means, stds, model, cfg, device):
    """
    Run per-frame inference on a full song's mel spectrogram(s).

    Args:
        mel_stack: (C, n_mels, T) log-mel spectrogram(s) (unnormalised)
        means: (C,) per-channel means
        stds: (C,) per-channel stds
        model: loaded OnsetOffsetPitchModel
        cfg: Config
        device: torch device

    Returns:
        onset_probs, offset_probs: (T,) numpy arrays
        oct_logits, cls_logits: (T, n_oct), (T, n_cls) numpy arrays
    """
    C = mel_stack.shape[0]
    n_mels = mel_stack.shape[1]
    n_frames = mel_stack.shape[2]
    half_ctx = cfg.context_frames // 2

    onset_probs = np.zeros(n_frames, dtype=np.float32)
    offset_probs = np.zeros(n_frames, dtype=np.float32)
    oct_logits_all = np.zeros((n_frames, cfg.n_octaves), dtype=np.float32)
    cls_logits_all = np.zeros((n_frames, cfg.n_pitch_classes), dtype=np.float32)

    def _build_context(f_idx):
        channels = []
        for ch in range(C):
            cols = []
            for ci in range(f_idx - half_ctx, f_idx + half_ctx + 1):
                if 0 <= ci < n_frames:
                    col = (mel_stack[ch, :, ci] - means[ch]) / stds[ch]
                else:
                    col = np.zeros(n_mels, dtype=np.float32)
                cols.append(col)
            channels.append(np.stack(cols, axis=0))
        return np.stack(channels, axis=0)  # (C, ctx, n_mels)

    with torch.no_grad():
        batch_size = cfg.batch_size
        for start in range(0, n_frames, batch_size):
                end = min(start + batch_size, n_frames)
                contexts = []
                for f_idx in range(start, end):
                    contexts.append(_build_context(f_idx))

                batch = torch.from_numpy(
                    np.array(contexts, dtype=np.float32)
                ).to(device)  # (B, C, ctx, n_mels)

                on_logits, off_logits, oct_logits, cls_logits = model(batch)
                onset_probs[start:end] = torch.sigmoid(on_logits).cpu().numpy()
                offset_probs[start:end] = torch.sigmoid(off_logits).cpu().numpy()
                oct_logits_all[start:end] = oct_logits.cpu().numpy()
                cls_logits_all[start:end] = cls_logits.cpu().numpy()

    return onset_probs, offset_probs, oct_logits_all, cls_logits_all


# Threshold grid search

def threshold_grid_search(all_onset_probs, all_offset_probs, all_oct_logits,
                          all_cls_logits, all_ref, cfg, fd,
                          offset_model=None, all_mel_stacks=None,
                          all_mel_means=None, all_mel_stds=None, device=None):
    """
    Grid-search onset/offset thresholds to maximise COnPOff F1.
    Returns (best_onset_thresh, best_offset_thresh, best_f1).

    When offset refinement is active, only onset_threshold is searched
    (offset comes from the refinement model, not thresholding).

    Phase 1:  Coarse onset search (0.10–0.75, step 0.025)
    Phase 1b: Fine onset search (±0.025 around best, step 0.005)
    Phase 2:  Joint bias / smooth_kernel / peak_min_distance / min_note_frames
    """
    if not HAS_MIR_EVAL:
        print("Warning: mir_eval required for threshold search.")
        return cfg.onset_threshold, cfg.offset_threshold, 0.0

    use_refine = (cfg.use_offset_refine and offset_model is not None)

    def _eval_config(cfg_tmp, n_songs):
        """Evaluate a config across all songs, return average COnPOff F1."""
        total_f1 = 0.0
        for s_idx in range(n_songs):
            if use_refine:
                est_notes = extract_notes_refined(
                    all_onset_probs[s_idx], all_offset_probs[s_idx],
                    all_oct_logits[s_idx], all_cls_logits[s_idx],
                    all_mel_stacks[s_idx], all_mel_means[s_idx],
                    all_mel_stds[s_idx], offset_model, cfg_tmp, fd, device,
                )
            else:
                est_notes = extract_predicted_notes(
                    all_onset_probs[s_idx], all_offset_probs[s_idx],
                    all_oct_logits[s_idx], all_cls_logits[s_idx],
                    cfg_tmp, fd,
                )
            ref = all_ref[s_idx]
            ref_intervals = np.array([[n[0], n[1]] for n in ref]) if ref else np.zeros((0, 2))
            ref_pitches = np.array([n[2] for n in ref]) if ref else np.zeros(0)
            est_intervals = np.array([[n[0], n[1]] for n in est_notes]) if est_notes else np.zeros((0, 2))
            est_pitches = np.array([n[2] for n in est_notes]) if est_notes else np.zeros(0)

            if len(est_intervals) == 0 or len(ref_intervals) == 0:
                continue

            ref_hz = mir_eval.util.midi_to_hz(ref_pitches)
            est_hz = mir_eval.util.midi_to_hz(est_pitches)
            raw = mir_eval.transcription.evaluate(
                ref_intervals, ref_hz, est_intervals, est_hz,
                onset_tolerance=cfg.onset_tolerance,
                pitch_tolerance=cfg.pitch_tolerance,
            )
            total_f1 += raw['F-measure']
        return total_f1 / max(n_songs, 1)

    def _make_tmp_cfg(**overrides):
        """Create a temporary config with specified overrides."""
        cfg_tmp = Config(experiment=cfg.experiment)
        cfg_tmp.onset_threshold = overrides.get('onset_threshold', best_on_t)
        cfg_tmp.offset_threshold = overrides.get('offset_threshold', best_off_t)
        cfg_tmp.peak_min_distance = overrides.get('peak_min_distance', cfg.peak_min_distance)
        cfg_tmp.onset_tolerance = cfg.onset_tolerance
        cfg_tmp.pitch_tolerance = cfg.pitch_tolerance
        cfg_tmp.n_octaves = cfg.n_octaves
        cfg_tmp.n_pitch_classes = cfg.n_pitch_classes
        cfg_tmp.midi_lowest = cfg.midi_lowest
        cfg_tmp.max_note_frames = cfg.max_note_frames
        cfg_tmp.min_note_frames = overrides.get('min_note_frames', cfg.min_note_frames)
        cfg_tmp.pitch_voting = cfg.pitch_voting
        cfg_tmp.pitch_soft_voting = cfg.pitch_soft_voting
        cfg_tmp.smooth_probs = cfg.smooth_probs
        cfg_tmp.smooth_kernel = overrides.get('smooth_kernel', cfg.smooth_kernel)
        cfg_tmp.use_offset_refine = cfg.use_offset_refine
        cfg_tmp.offset_bias_sec = overrides.get('offset_bias_sec', cfg.offset_bias_sec)
        return cfg_tmp

    n_songs = len(all_ref)
    best_f1 = -1.0
    best_on_t = cfg.onset_threshold
    best_off_t = cfg.offset_threshold

    # Phase 1: Coarse onset search
    onset_range = np.arange(0.10, 0.76, 0.025)
    offset_range = [cfg.offset_threshold] if use_refine else np.arange(0.3, 0.71, 0.05)

    print(f"Phase 1: Coarse onset search [{onset_range[0]:.2f}..{onset_range[-1]:.2f}]")

    for on_t in onset_range:
        for off_t in offset_range:
            cfg_tmp = _make_tmp_cfg(onset_threshold=float(on_t), offset_threshold=float(off_t))
            avg_f1 = _eval_config(cfg_tmp, n_songs)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_on_t = float(on_t)
                best_off_t = float(off_t)

    print(f"Best (coarse): onset={best_on_t:.3f}, offset={best_off_t:.2f} "
          f"→ COnPOff F1={best_f1:.4f}")

    # Phase 1b: Fine-grained onset search around coarse best
    fine_onset_range = np.arange(
        max(0.05, best_on_t - 0.025), best_on_t + 0.03, 0.005
    )
    print(f"Phase 1b: Fine onset search [{fine_onset_range[0]:.3f}..{fine_onset_range[-1]:.3f}]")

    for on_t in fine_onset_range:
        for off_t in offset_range:
            cfg_tmp = _make_tmp_cfg(onset_threshold=float(on_t), offset_threshold=float(off_t))
            avg_f1 = _eval_config(cfg_tmp, n_songs)
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_on_t = float(on_t)
                best_off_t = float(off_t)

    print(f"Best (fine): onset={best_on_t:.3f}, offset={best_off_t:.2f} "
          f"→ COnPOff F1={best_f1:.4f}")

    # Phase 2: Joint bias / smooth_kernel / peak_min_distance / min_note_frames
    if use_refine:
        print(f"\nPhase 2: Joint bias / smooth_kernel / pmd / min_note_frames search...")
        bias_range = np.arange(-0.005, 0.016, 0.002)
        kernel_range = [3, 5]
        pmd_range = [4, 5, 6, 7]
        mnf_range = [2, 3, 4, 5]

        best_bias = cfg.offset_bias_sec
        best_kernel = cfg.smooth_kernel
        best_pmd = cfg.peak_min_distance
        best_mnf = cfg.min_note_frames

        total_combos = len(bias_range) * len(kernel_range) * len(pmd_range) * len(mnf_range)
        print(f"  {total_combos} combinations to test...")
        combo_i = 0

        for bias in bias_range:
            for kern in kernel_range:
                for pmd in pmd_range:
                    for mnf in mnf_range:
                        combo_i += 1
                        cfg_tmp = _make_tmp_cfg(
                            offset_bias_sec=float(bias),
                            smooth_kernel=kern,
                            peak_min_distance=pmd,
                            min_note_frames=mnf,
                        )
                        avg_f1 = _eval_config(cfg_tmp, n_songs)
                        if avg_f1 > best_f1:
                            best_f1 = avg_f1
                            best_bias = float(bias)
                            best_kernel = kern
                            best_pmd = pmd
                            best_mnf = mnf

                        if combo_i % 50 == 0:
                            print(f"  [{combo_i}/{total_combos}] best so far: F1={best_f1:.4f}")

        print(f"Best Phase 2: bias={best_bias:.3f}, kernel={best_kernel}, "
              f"pmd={best_pmd}, mnf={best_mnf} → COnPOff F1={best_f1:.4f}")
        cfg.offset_bias_sec = best_bias
        cfg.smooth_kernel = best_kernel
        cfg.peak_min_distance = best_pmd
        cfg.min_note_frames = best_mnf

    return best_on_t, best_off_t, best_f1


# Main evaluation

def evaluate(cfg: Config, checkpoint_path: str = None):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")

    device = torch.device(
        cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    )

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = OnsetOffsetPitchModel(
        n_mels=cfg.n_mels,
        context_frames=cfg.context_frames,
        n_octaves=cfg.n_octaves,
        n_pitch_classes=cfg.n_pitch_classes,
        in_channels=cfg.in_channels,
        deeper_backbone=cfg.deeper_backbone,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Load offset refinement model
    offset_model = None
    if cfg.use_offset_refine and cfg.offset_refine_checkpoint:
        from offset_model import OffsetRefineModel
        offset_ckpt = torch.load(
            cfg.offset_refine_checkpoint, map_location=device, weights_only=False
        )
        offset_model = OffsetRefineModel(
            n_mels=cfg.n_mels, in_channels=cfg.in_channels,
            n_temporal_layers=cfg.offset_n_temporal_layers,
        ).to(device)
        offset_model.load_state_dict(offset_ckpt["model_state_dict"])
        offset_model.eval()
        print(f"Loaded offset refinement model from {cfg.offset_refine_checkpoint}")

    # Load annotations
    with open(cfg.annotation_file, "r") as f:
        all_annotations = json.load(f)

    # Test songs
    test_dir = cfg.test_dir
    song_dirs = sorted(
        [d for d in os.listdir(test_dir)
         if os.path.isdir(os.path.join(test_dir, d)) and d.isdigit()],
        key=lambda x: int(x),
    )

    fd = cfg.frame_duration
    all_ref = []
    all_est = []
    song_ids = []
    all_onset_probs = []
    all_offset_probs = []
    all_oct_logits = []
    all_cls_logits = []
    all_mel_stacks = []
    all_mel_means = []
    all_mel_stds = []

    print(f"\nEvaluating {len(song_dirs)} test songs...\n")

    for sid in song_dirs:
        song_dir = os.path.join(test_dir, sid)
        audio_path = find_audio(song_dir, cfg.audio_source)
        if audio_path is None:
            continue

        ref_notes_raw = all_annotations.get(sid)
        if ref_notes_raw is None:
            continue

        try:
            y, _ = librosa.load(audio_path, sr=cfg.sample_rate, mono=True)
        except Exception as e:
            print(f"  Song {sid}: {e}")
            continue

        mel = librosa.feature.melspectrogram(
            y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft,
            hop_length=cfg.hop_length, n_mels=cfg.n_mels,
            fmin=cfg.fmin, fmax=cfg.fmax,
        )
        log_mel = np.log(mel + 1e-8).astype(np.float32)

        if cfg.use_multi_res:
            mel_stack = compute_multi_res_mel(y, cfg)
        else:
            mel_stack = log_mel[np.newaxis]

        means = mel_stack.mean(axis=(1, 2))
        stds = mel_stack.std(axis=(1, 2)) + 1e-8

        onset_probs, offset_probs, oct_log, cls_log = infer_full_song(
            mel_stack, means, stds, model, cfg, device
        )

        all_onset_probs.append(onset_probs)
        all_offset_probs.append(offset_probs)
        all_oct_logits.append(oct_log)
        all_cls_logits.append(cls_log)

        if cfg.use_offset_refine:
            all_mel_stacks.append(mel_stack)
            all_mel_means.append(means)
            all_mel_stds.append(stds)

        ref_notes = []
        for note in ref_notes_raw:
            on_sec, off_sec, midi = float(note[0]), float(note[1]), float(note[2])
            if off_sec > on_sec:
                ref_notes.append([on_sec, off_sec, midi])

        all_ref.append(ref_notes)
        song_ids.append(sid)

    # Threshold grid search
    if cfg.threshold_search and HAS_MIR_EVAL:
        best_on_t, best_off_t, _ = threshold_grid_search(
            all_onset_probs, all_offset_probs, all_oct_logits,
            all_cls_logits, all_ref, cfg, fd,
            offset_model=offset_model,
            all_mel_stacks=all_mel_stacks,
            all_mel_means=all_mel_means,
            all_mel_stds=all_mel_stds,
            device=device,
        )
        cfg.onset_threshold = best_on_t
        cfg.offset_threshold = best_off_t

    # Extract notes with (possibly optimised) thresholds
    all_est = []
    for s_idx in range(len(all_ref)):
        if cfg.use_offset_refine and offset_model is not None:
            est_notes = extract_notes_refined(
                all_onset_probs[s_idx], all_offset_probs[s_idx],
                all_oct_logits[s_idx], all_cls_logits[s_idx],
                all_mel_stacks[s_idx], all_mel_means[s_idx],
                all_mel_stds[s_idx], offset_model, cfg, fd, device,
            )
        else:
            est_notes = extract_predicted_notes(
                all_onset_probs[s_idx], all_offset_probs[s_idx],
                all_oct_logits[s_idx], all_cls_logits[s_idx], cfg, fd,
            )
        all_est.append(est_notes)

    # Compute metrics
    print(f"\nComputing metrics for {len(song_ids)} songs...\n")

    if HAS_MIR_EVAL:
        avg = np.zeros(14)
        for i, (ref, est) in enumerate(zip(all_ref, all_est)):
            ref_intervals = np.array([[n[0], n[1]] for n in ref]) if ref else np.zeros((0, 2))
            ref_pitches = np.array([n[2] for n in ref]) if ref else np.zeros(0)
            est_intervals = np.array([[n[0], n[1]] for n in est]) if est else np.zeros((0, 2))
            est_pitches = np.array([n[2] for n in est]) if est else np.zeros(0)

            ref_pitches_hz = mir_eval.util.midi_to_hz(ref_pitches) if len(ref_pitches) > 0 else ref_pitches
            est_pitches_hz = mir_eval.util.midi_to_hz(est_pitches) if len(est_pitches) > 0 else est_pitches

            if len(est_intervals) == 0:
                ret = np.zeros(14)
                ret[9] = len(ref_pitches)
            else:
                raw = mir_eval.transcription.evaluate(
                    ref_intervals, ref_pitches_hz,
                    est_intervals, est_pitches_hz,
                    onset_tolerance=cfg.onset_tolerance,
                    pitch_tolerance=cfg.pitch_tolerance,
                )
                ret = np.zeros(14)
                ret[0] = raw['Precision']
                ret[1] = raw['Recall']
                ret[2] = raw['F-measure']
                ret[3] = raw['Precision_no_offset']
                ret[4] = raw['Recall_no_offset']
                ret[5] = raw['F-measure_no_offset']
                ret[6] = raw['Onset_Precision']
                ret[7] = raw['Onset_Recall']
                ret[8] = raw['Onset_F-measure']
                ret[9] = len(ref_pitches)
                ret[10] = len(est_pitches)

            print(f"  Song {song_ids[i]:>4s} | "
                  f"COnPOff={ret[2]:.3f} COnP={ret[5]:.3f} COn={ret[8]:.3f} | "
                  f"ref={int(ret[9])} est={int(ret[10])}")

            avg += ret

        n = len(all_ref)
        for i in range(9):
            avg[i] /= max(n, 1)

        print("\n" + "=" * 60)
        print("         Precision  Recall  F1-score")
        print(f"COnPOff  {avg[0]:.4f}     {avg[1]:.4f}  {avg[2]:.4f}")
        print(f"COnP     {avg[3]:.4f}     {avg[4]:.4f}  {avg[5]:.4f}")
        print(f"COn      {avg[6]:.4f}     {avg[7]:.4f}  {avg[8]:.4f}")
        print(f"Total ref notes: {int(avg[9])}, Total est notes: {int(avg[10])}")
        print(f"Songs evaluated: {n}")
        print("=" * 60)

        plot_test_results(song_ids, all_ref, all_est, avg, cfg, checkpoint_path)

    else:
        # Fallback without mir_eval
        all_con = all_conp = all_conpoff = 0.0
        for i, (ref, est) in enumerate(zip(all_ref, all_est)):
            if not ref and not est:
                m = {"COn_F1": 1.0, "COnP_F1": 1.0, "COnPOff_F1": 1.0}
            elif not ref or not est:
                m = {"COn_F1": 0.0, "COnP_F1": 0.0, "COnPOff_F1": 0.0}
            else:
                matched = set()
                con = conp = conpoff = 0
                for e_on, e_off, e_pitch in est:
                    best_ri = -1
                    best_d = float("inf")
                    for ri, (r_on, r_off, r_pitch) in enumerate(ref):
                        if ri in matched:
                            continue
                        d = abs(e_on - r_on)
                        if d <= cfg.onset_tolerance and d < best_d:
                            best_d = d
                            best_ri = ri
                    if best_ri >= 0:
                        con += 1
                        matched.add(best_ri)
                        r_on, r_off, r_pitch = ref[best_ri]
                        if int(e_pitch) == int(r_pitch):
                            conp += 1
                            if abs(e_off - r_off) <= cfg.offset_tolerance:
                                conpoff += 1
                def _f1(tp, n_p, n_r):
                    p = tp / max(n_p, 1)
                    r = tp / max(n_r, 1)
                    return 2 * p * r / max(p + r, 1e-8)
                m = {
                    "COn_F1": _f1(con, len(est), len(ref)),
                    "COnP_F1": _f1(conp, len(est), len(ref)),
                    "COnPOff_F1": _f1(conpoff, len(est), len(ref)),
                }
            all_con += m["COn_F1"]
            all_conp += m["COnP_F1"]
            all_conpoff += m["COnPOff_F1"]
            print(f"  Song {song_ids[i]:>4s} | "
                  f"COnPOff={m['COnPOff_F1']:.3f} COnP={m['COnP_F1']:.3f} "
                  f"COn={m['COn_F1']:.3f}")

        n = len(all_ref)
        print("\n" + "=" * 60)
        print(f"  Mean COn     F1: {all_con / max(n, 1):.4f}")
        print(f"  Mean COnP    F1: {all_conp / max(n, 1):.4f}")
        print(f"  Mean COnPOff F1: {all_conpoff / max(n, 1):.4f}")
        print("=" * 60)


def plot_test_results(song_ids, all_ref, all_est, avg, cfg, checkpoint_path):
    """Save per-song and summary test result plots as PNG."""
    song_con = []
    song_conp = []
    song_conpoff = []

    for ref, est in zip(all_ref, all_est):
        ref_intervals = np.array([[n[0], n[1]] for n in ref]) if ref else np.zeros((0, 2))
        ref_pitches = np.array([n[2] for n in ref]) if ref else np.zeros(0)
        est_intervals = np.array([[n[0], n[1]] for n in est]) if est else np.zeros((0, 2))
        est_pitches = np.array([n[2] for n in est]) if est else np.zeros(0)

        if len(est_intervals) == 0 or len(ref_intervals) == 0:
            song_con.append(0.0)
            song_conp.append(0.0)
            song_conpoff.append(0.0)
        else:
            ref_hz = mir_eval.util.midi_to_hz(ref_pitches)
            est_hz = mir_eval.util.midi_to_hz(est_pitches)
            raw = mir_eval.transcription.evaluate(
                ref_intervals, ref_hz, est_intervals, est_hz,
                onset_tolerance=cfg.onset_tolerance,
                pitch_tolerance=cfg.pitch_tolerance,
            )
            song_con.append(raw['Onset_F-measure'])
            song_conp.append(raw['F-measure_no_offset'])
            song_conpoff.append(raw['F-measure'])

    save_dir = os.path.dirname(checkpoint_path) if checkpoint_path else cfg.checkpoint_dir
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Per-song F1 bar chart (sorted by COnPOff)
    ax = axes[0]
    order = np.argsort(song_conpoff)[::-1]
    x = np.arange(len(song_ids))
    ax.bar(x, [song_conpoff[i] for i in order], alpha=0.7, label="COnPOff", width=0.8)
    ax.set_xlabel("Song (sorted by COnPOff F1)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Song COnPOff F1")
    ax.set_ylim(0, 1)
    ax.axhline(y=avg[2], color="red", linestyle="--", label=f"Mean={avg[2]:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    if len(song_ids) <= 50:
        ax.set_xticks(x)
        ax.set_xticklabels([song_ids[i] for i in order], rotation=90, fontsize=6)
    else:
        ax.set_xticks([])

    # Summary bar chart
    ax = axes[1]
    metrics = ["COn", "COnP", "COnPOff"]
    precision = [avg[6], avg[3], avg[0]]
    recall = [avg[7], avg[4], avg[1]]
    f1 = [avg[8], avg[5], avg[2]]
    x = np.arange(len(metrics))
    w = 0.25
    ax.bar(x - w, precision, w, label="Precision", color="#2196F3")
    ax.bar(x, recall, w, label="Recall", color="#FF9800")
    ax.bar(x + w, f1, w, label="F1", color="#4CAF50")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Overall Metrics")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for i in range(len(metrics)):
        ax.text(x[i] + w, f1[i] + 0.02, f"{f1[i]:.3f}", ha="center", fontsize=9)

    fig.suptitle(f"Experiment {cfg.experiment} — Test Results", fontsize=14)
    fig.tight_layout()
    plot_path = os.path.join(save_dir, "test_results.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nTest results plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--experiment", type=int, default=1, choices=[1, 2, 3, 4, 5, 6],
        help="Experiment number",
    )
    args = parser.parse_args()

    cfg = Config(experiment=args.experiment)

    # Apply experiment-specific config
    if args.experiment == 3:
        cfg.context_frames = 21
        cfg.threshold_search = True
        cfg.deeper_backbone = True
        cfg.pitch_voting = True
        cfg.smooth_probs = True
        cfg.smooth_kernel = 5
    elif args.experiment == 4:
        # Two-stage: exp3 first-stage + offset refinement model
        cfg.context_frames = 21
        cfg.threshold_search = True
        cfg.deeper_backbone = True
        cfg.pitch_voting = True
        cfg.smooth_probs = True
        cfg.smooth_kernel = 5
        cfg.use_offset_refine = True
        cfg.offset_refine_checkpoint = "./checkpoints/exp4/offset_model.pt"
    elif args.experiment == 5:
        # Onset-focused retrained CNN + offset refinement
        cfg.context_frames = 21
        cfg.threshold_search = True
        cfg.deeper_backbone = True
        cfg.pitch_soft_voting = True
        cfg.smooth_probs = True
        cfg.smooth_kernel = 3
        cfg.peak_min_distance = 3
        cfg.use_offset_refine = True
        cfg.offset_refine_checkpoint = "./checkpoints/exp5/offset_model.pt"
    elif args.experiment == 6:
        # Pitch-focused fine-tuning from exp5 + 7-layer offset model
        cfg.context_frames = 21
        cfg.threshold_search = True
        cfg.deeper_backbone = True
        cfg.pitch_soft_voting = True
        cfg.smooth_probs = True
        cfg.smooth_kernel = 3
        cfg.peak_min_distance = 3
        cfg.use_offset_refine = True
        cfg.offset_refine_checkpoint = "./checkpoints/exp6/offset_model.pt"
        cfg.offset_n_temporal_layers = 7
        cfg.offset_bias_sec = 0.011

    # Default checkpoint path based on experiment
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        if args.experiment == 4:
            tag = "exp3"   # exp4 uses exp3's first-stage model
        else:
            tag = f"exp{args.experiment}"
        checkpoint_path = f"./checkpoints/{tag}/best_model.pt"

    print(f"\n{'='*60}")
    print(f"  Evaluating Experiment {cfg.experiment}")
    print(f"  Multi-resolution: {cfg.use_multi_res} (channels={cfg.in_channels})")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

    evaluate(cfg, checkpoint_path=checkpoint_path)
