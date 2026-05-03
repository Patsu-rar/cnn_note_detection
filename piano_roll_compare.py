"""
Piano roll comparison: ground truth vs predicted notes.

Generates a figure showing both reference and estimated notes as horizontal
bars on a time × pitch grid — useful for visual comparison in a journal.

Usage:
    python piano_roll_compare.py --song 1
    python piano_roll_compare.py --song 42 --start 10 --end 25
    python piano_roll_compare.py --song 42 --experiment 6 --start 10 --end 25 --out figure.png
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import Config
from inference import predict_notes
from test import find_audio


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]


def midi_to_label(midi):
    return f"{NOTE_NAMES[int(midi) % 12]}{int(midi) // 12 - 1}"


def plot_piano_roll(ref_notes, est_notes, start_sec, end_sec, output_path,
                    title="Piano Roll Comparison"):
    """
    Plot ground truth and predicted notes as overlapping horizontal bars.

    ref_notes / est_notes: list of [onset, offset, midi_pitch]
    """
    # Filter notes within the time window
    def in_window(notes):
        return [n for n in notes
                if n[1] > start_sec and n[0] < end_sec]

    ref = in_window(ref_notes)
    est = in_window(est_notes)

    if not ref and not est:
        print(f"No notes in [{start_sec:.1f}, {end_sec:.1f}]s window.")
        return

    # Determine pitch range from both sets
    all_pitches = [n[2] for n in ref + est]
    pitch_lo = int(min(all_pitches)) - 1
    pitch_hi = int(max(all_pitches)) + 2

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, sharey=True)

    for ax, notes, label, color in [
        (axes[0], ref, "Ground Truth", "#2196F3"),
        (axes[1], est, "Predicted", "#F44336"),
    ]:
        for on, off, midi in notes:
            on_c = max(on, start_sec)
            off_c = min(off, end_sec)
            ax.barh(
                y=int(midi), width=off_c - on_c, left=on_c,
                height=0.7, color=color, alpha=0.85, edgecolor="black",
                linewidth=0.4,
            )
        ax.set_ylabel("MIDI Pitch")
        ax.set_ylim(pitch_lo, pitch_hi)
        ax.set_xlim(start_sec, end_sec)
        ax.set_title(label, fontsize=12, fontweight="bold", loc="left")
        ax.grid(True, axis="x", alpha=0.3)
        ax.grid(True, axis="y", alpha=0.15)

        # Y-axis note labels
        yticks = range(pitch_lo + 1, pitch_hi)
        ax.set_yticks(list(yticks))
        ax.set_yticklabels([midi_to_label(m) for m in yticks], fontsize=7)

    axes[1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_piano_roll_overlay(ref_notes, est_notes, start_sec, end_sec,
                            output_path, title="Piano Roll Comparison"):
    """
    Single-axis overlay: reference in blue, predicted in red, overlap in purple.
    """
    def in_window(notes):
        return [n for n in notes if n[1] > start_sec and n[0] < end_sec]

    ref = in_window(ref_notes)
    est = in_window(est_notes)

    if not ref and not est:
        print(f"No notes in [{start_sec:.1f}, {end_sec:.1f}]s window.")
        return

    all_pitches = [n[2] for n in ref + est]
    pitch_lo = int(min(all_pitches)) - 1
    pitch_hi = int(max(all_pitches)) + 2

    fig, ax = plt.subplots(figsize=(14, 5))

    # Draw reference first (blue, behind)
    for on, off, midi in ref:
        on_c = max(on, start_sec)
        off_c = min(off, end_sec)
        ax.barh(
            y=int(midi), width=off_c - on_c, left=on_c,
            height=0.7, color="#2196F3", alpha=0.5, edgecolor="#1565C0",
            linewidth=0.5,
        )

    # Draw predicted on top (red, semi-transparent)
    for on, off, midi in est:
        on_c = max(on, start_sec)
        off_c = min(off, end_sec)
        ax.barh(
            y=int(midi), width=off_c - on_c, left=on_c,
            height=0.7, color="#F44336", alpha=0.5, edgecolor="#B71C1C",
            linewidth=0.5,
        )

    ax.set_xlim(start_sec, end_sec)
    ax.set_ylim(pitch_lo, pitch_hi)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MIDI Pitch")
    ax.grid(True, axis="x", alpha=0.3)
    ax.grid(True, axis="y", alpha=0.15)

    yticks = range(pitch_lo + 1, pitch_hi)
    ax.set_yticks(list(yticks))
    ax.set_yticklabels([midi_to_label(m) for m in yticks], fontsize=7)

    ref_patch = mpatches.Patch(color="#2196F3", alpha=0.5, label="Ground Truth")
    est_patch = mpatches.Patch(color="#F44336", alpha=0.5, label="Predicted")
    ax.legend(handles=[ref_patch, est_patch], loc="upper right", fontsize=10)

    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Piano roll comparison: ground truth vs predicted."
    )
    parser.add_argument("--song", type=str, required=True,
                        help="Test song ID (e.g. 1, 42)")
    parser.add_argument("--experiment", type=int, default=6,
                        choices=[1, 2, 3, 4, 5, 6],
                        help="1=baseline, 2=multi-res, 3=enhanced, "
                             "4=two-stage offset, 5=onset-focused, 6=pitch-focused")
    parser.add_argument("--start", type=float, default=None,
                        help="Start time in seconds (default: auto)")
    parser.add_argument("--end", type=float, default=None,
                        help="End time in seconds (default: start+15)")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Window duration in seconds (default: 15)")
    parser.add_argument("--overlay", action="store_true",
                        help="Single-axis overlay instead of stacked panels")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path (default: piano_roll_<song>.png)")
    args = parser.parse_args()

    # Build config matching experiment
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

    # Checkpoint
    if args.experiment == 4:
        checkpoint = "checkpoints/exp3/best_model.pt"
    else:
        checkpoint = f"checkpoints/exp{args.experiment}/best_model.pt"

    # Load ground truth
    with open(cfg.annotation_file, "r") as f:
        annotations = json.load(f)

    sid = args.song
    if sid not in annotations:
        print(f"Song '{sid}' not found in annotations. "
              f"Available: {sorted(annotations.keys(), key=lambda x: int(x))[:10]}...")
        sys.exit(1)

    ref_notes = []
    for note in annotations[sid]:
        on, off, midi = float(note[0]), float(note[1]), float(note[2])
        if off > on:
            ref_notes.append([on, off, midi])

    # Find audio
    song_dir = os.path.join(cfg.test_dir, sid)
    audio_path = find_audio(song_dir, cfg.audio_source)
    if audio_path is None:
        print(f"No audio found in {song_dir}")
        sys.exit(1)

    print(f"Song {sid}: {len(ref_notes)} ground truth notes")
    print(f"Audio: {audio_path}")
    print(f"Experiment: {args.experiment}, checkpoint: {checkpoint}")

    # Predict
    result = predict_notes(audio_path, checkpoint_path=checkpoint, cfg=cfg)
    est_notes = [[n[0], n[1], n[2]] for n in result["notes"]]
    print(f"Predicted: {len(est_notes)} notes")

    # Time window
    if args.start is not None:
        start = args.start
    else:
        # Pick a window with decent note density
        if ref_notes:
            mid = np.median([n[0] for n in ref_notes])
            start = max(0, mid - args.duration / 2)
        else:
            start = 0
    end = args.end if args.end is not None else start + args.duration

    print(f"Window: [{start:.1f}, {end:.1f}]s")

    # Output path
    out = args.out or f"piano_roll_song{sid}_exp{args.experiment}.png"
    title = f"Song {sid} — Experiment {args.experiment}"

    if args.overlay:
        plot_piano_roll_overlay(ref_notes, est_notes, start, end, out, title)
    else:
        plot_piano_roll(ref_notes, est_notes, start, end, out, title)


if __name__ == "__main__":
    main()
