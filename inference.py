"""
Inference script - predict notes (onset, offset, pitch) from any audio file.

Uses per-frame CNN with context window (same as training/test).
Supports experiments 1, 2, 3, 4, 5, 6.

Usage:
    python inference.py path/to/song.mp3
    python inference.py path/to/song.wav --experiment 6
    python inference.py path/to/song.wav --checkpoint checkpoints/exp6/best_model.pt --experiment 6
"""

import argparse
import os
import numpy as np
import torch
import librosa
import mido

from config import Config
from model import OnsetOffsetPitchModel
from dataset import compute_multi_res_mel


# Post-processing

def peak_pick(activation: np.ndarray, threshold: float, min_distance: int) -> np.ndarray:
    candidates = activation > threshold
    peaks = []
    last_peak = -min_distance

    for i in range(len(candidates)):
        if not candidates[i]:
            continue
        win_start = max(0, i - min_distance)
        win_end = min(len(activation), i + min_distance + 1)
        if activation[i] == activation[win_start:win_end].max():
            if (i - last_peak) >= min_distance:
                peaks.append(i)
                last_peak = i

    return np.array(peaks, dtype=int)


def decode_pitch_at_frame(oct_logits, cls_logits, cfg: Config) -> int:
    oct_pred = oct_logits.argmax()
    cls_pred = cls_logits.argmax()
    if oct_pred >= cfg.n_octaves or cls_pred >= cfg.n_pitch_classes:
        return -1
    return cfg.midi_lowest + int(oct_pred) * cfg.n_pitch_classes + int(cls_pred)


# Core inference function

def predict_notes(
    audio_path: str,
    checkpoint_path: str = "checkpoints/best_model.pt",
    cfg: Config = None,
    onset_threshold: float = None,
    offset_threshold: float = None,
) -> dict:
    if cfg is None:
        cfg = Config()
    if onset_threshold is None:
        onset_threshold = cfg.onset_threshold
    if offset_threshold is None:
        offset_threshold = cfg.offset_threshold

    device = torch.device(
        cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    )

    # Load model
    model = OnsetOffsetPitchModel(
        n_mels=cfg.n_mels,
        context_frames=cfg.context_frames,
        n_octaves=cfg.n_octaves,
        n_pitch_classes=cfg.n_pitch_classes,
        in_channels=cfg.in_channels,
        deeper_backbone=cfg.deeper_backbone,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load audio
    y, sr = librosa.load(audio_path, sr=cfg.sample_rate, mono=True)
    duration = len(y) / sr

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, n_mels=cfg.n_mels,
        fmin=cfg.fmin, fmax=cfg.fmax,
    )
    log_mel = np.log(mel + 1e-8).astype(np.float32)

    if cfg.use_multi_res:
        mel_stack = compute_multi_res_mel(y, cfg)
    else:
        mel_stack = log_mel[np.newaxis]  # (1, n_mels, T)

    C = mel_stack.shape[0]
    n_mels = mel_stack.shape[1]
    n_frames = mel_stack.shape[2]
    means = mel_stack.mean(axis=(1, 2))   # (C,)
    stds = mel_stack.std(axis=(1, 2)) + 1e-8  # (C,)

    # Inference
    half_ctx = cfg.context_frames // 2
    batch_size = cfg.batch_size

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
        return np.stack(channels, axis=0)

    with torch.no_grad():
        # Per-frame CNN inference (features + head predictions)
        for start in range(0, n_frames, batch_size):
                end = min(start + batch_size, n_frames)
                contexts = []
                for f_idx in range(start, end):
                    contexts.append(_build_context(f_idx))

                batch = torch.from_numpy(
                    np.array(contexts, dtype=np.float32)
                ).to(device)

                on_logits, off_logits, oct_logits, cls_logits = model(batch)
                onset_probs[start:end] = torch.sigmoid(on_logits).cpu().numpy()
                offset_probs[start:end] = torch.sigmoid(off_logits).cpu().numpy()
                oct_logits_all[start:end] = oct_logits.cpu().numpy()
                cls_logits_all[start:end] = cls_logits.cpu().numpy()

    # Note extraction
    fd = cfg.frame_duration
    if cfg.use_offset_refine and cfg.offset_refine_checkpoint:
        from offset_model import OffsetRefineModel
        from test import extract_notes_refined
        offset_ckpt = torch.load(
            cfg.offset_refine_checkpoint, map_location=device, weights_only=False
        )
        offset_mdl = OffsetRefineModel(
            n_mels=cfg.n_mels, in_channels=cfg.in_channels,
            n_temporal_layers=getattr(cfg, 'offset_n_temporal_layers', 5),
        ).to(device)
        offset_mdl.load_state_dict(offset_ckpt["model_state_dict"])
        offset_mdl.eval()
        notes = extract_notes_refined(
            onset_probs, offset_probs, oct_logits_all, cls_logits_all,
            mel_stack, means, stds, offset_mdl, cfg, fd, device,
        )
    else:
        onset_frames = peak_pick(onset_probs, onset_threshold, cfg.peak_min_distance)
        offset_frames = peak_pick(offset_probs, offset_threshold, cfg.peak_min_distance)

        notes = []
        for on_f in onset_frames:
            off_f = None
            for of in offset_frames:
                if of > on_f:
                    off_f = of
                    break
            if off_f is None:
                off_f = min(on_f + 10, n_frames - 1)

            midi_pitch = decode_pitch_at_frame(oct_logits_all[on_f], cls_logits_all[on_f], cfg)
            if midi_pitch < 0:
                continue

            notes.append((on_f * fd, off_f * fd, midi_pitch))

    return {
        "notes": notes,
        "onset_probs": onset_probs,
        "offset_probs": offset_probs,
        "frame_duration": fd,
        "audio_duration": duration,
    }


# MIDI export

def notes_to_midi(notes, output_path, tempo=500000, velocity=80):
    """Save detected notes as a MIDI file.
    
    Args:
        notes: list of (onset_sec, offset_sec, midi_pitch)
        output_path: path for the .mid file
        tempo: microseconds per beat (default 500000 = 120 BPM)
        velocity: note velocity 1-127
    """
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    ticks_per_beat = mid.ticks_per_beat  # default 480
    sec_per_tick = tempo / (1_000_000 * ticks_per_beat)

    # Build note_on / note_off events sorted by time
    events = []
    for on_sec, off_sec, pitch in notes:
        pitch = int(pitch)
        events.append((on_sec, 'note_on', pitch, velocity))
        events.append((off_sec, 'note_off', pitch, 0))
    events.sort(key=lambda e: e[0])

    prev_tick = 0
    for time_sec, msg_type, pitch, vel in events:
        abs_tick = int(round(time_sec / sec_per_tick))
        delta = max(0, abs_tick - prev_tick)
        track.append(mido.Message(msg_type, note=pitch, velocity=vel, time=delta))
        prev_tick = abs_tick

    mid.save(output_path)
    return output_path


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Predict vocal notes (onset, offset, pitch) from audio."
    )
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--experiment", type=int, default=1, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--onset_thresh", type=float, default=None)
    parser.add_argument("--offset_thresh", type=float, default=None)
    parser.add_argument("--no_midi", action="store_true", help="Skip MIDI file output")
    parser.add_argument("--midi_out", default=None, help="Output MIDI path (default: <audio>.mid)")
    args = parser.parse_args()

    cfg = Config(experiment=args.experiment)
    if args.experiment >= 3:
        cfg.context_frames = 21
        cfg.deeper_backbone = True
    if args.experiment == 3:
        cfg.pitch_voting = True
        cfg.smooth_probs = True
        cfg.smooth_kernel = 5
    if args.experiment == 4:
        cfg.pitch_voting = True
        cfg.smooth_probs = True
        cfg.smooth_kernel = 5
        cfg.use_offset_refine = True
        cfg.offset_refine_checkpoint = "checkpoints/exp4/offset_model.pt"
    if args.experiment == 5:
        cfg.pitch_soft_voting = True
        cfg.smooth_probs = True
        cfg.smooth_kernel = 3
        cfg.peak_min_distance = 3
        cfg.use_offset_refine = True
        cfg.offset_refine_checkpoint = "checkpoints/exp5/offset_model.pt"
    if args.experiment == 6:
        cfg.pitch_soft_voting = True
        cfg.smooth_probs = True
        cfg.smooth_kernel = 3
        cfg.peak_min_distance = 5
        cfg.onset_threshold = 0.600
        cfg.use_offset_refine = True
        cfg.offset_refine_checkpoint = "checkpoints/exp6/offset_model.pt"
        cfg.offset_n_temporal_layers = 7
        cfg.offset_bias_sec = 0.005
    checkpoint = args.checkpoint
    if checkpoint is None:
        if args.experiment == 4:
            checkpoint = "checkpoints/exp3/best_model.pt"
        else:
            checkpoint = f"checkpoints/exp{args.experiment}/best_model.pt"

    result = predict_notes(
        args.audio,
        checkpoint_path=checkpoint,
        cfg=cfg,
        onset_threshold=args.onset_thresh,
        offset_threshold=args.offset_thresh,
    )

    notes = result["notes"]
    print(f"\nDetected {len(notes)} notes "
          f"(duration: {result['audio_duration']:.1f}s)\n")

    if notes:
        print(f"{'Onset':>8s}  {'Offset':>8s}  {'MIDI':>4s}  {'Note':>6s}")
        print("-" * 35)
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                       'F#', 'G', 'G#', 'A', 'A#', 'B']
        for on_sec, off_sec, midi in notes:
            midi = int(midi)
            name = note_names[midi % 12] + str(midi // 12 - 1)
            print(f"{on_sec:8.3f}  {off_sec:8.3f}  {midi:4d}  {name:>6s}")

    # Save MIDI file
    if notes and not args.no_midi:
        midi_path = args.midi_out
        if midi_path is None:
            base = os.path.splitext(args.audio)[0]
            midi_path = base + ".mid"
        notes_to_midi(notes, midi_path)
        print(f"\nMIDI saved to: {midi_path}")


if __name__ == "__main__":
    main()
