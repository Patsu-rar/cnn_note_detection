"""
Diagnostic: break down COnPOff errors by note duration, pitch, and song.

Runs exp6 config (exp5 CNN + exp6 offset model), then analyzes per-note
matching to find where offset errors cluster.

Usage:
    python diagnose.py                          # runs grid search first
    python diagnose.py --onset_threshold 0.325  # skip grid search, use known value
"""

import argparse
import json
import os
import numpy as np
import torch
import librosa

from config import Config
from model import OnsetOffsetPitchModel
from offset_model import OffsetRefineModel
from dataset import compute_multi_res_mel
from test import (
    find_audio, infer_full_song, extract_notes_refined,
    median_smooth, peak_pick, threshold_grid_search,
)

import mir_eval


def match_notes(ref_notes, est_notes, onset_tol=0.05, pitch_tol=50.0):
    """
    Match estimated notes to reference notes using mir_eval criteria.

    Returns list of dicts with match info for each ref note:
      - matched: bool
      - onset_matched: bool
      - pitch_matched: bool
      - offset_matched: bool
      - ref_onset, ref_offset, ref_pitch, ref_duration
      - est_onset, est_offset, est_pitch (if matched)
      - offset_error (seconds, if onset+pitch matched)
    """
    ref_intervals = np.array([[n[0], n[1]] for n in ref_notes]) if ref_notes else np.zeros((0, 2))
    ref_pitches = np.array([n[2] for n in ref_notes]) if ref_notes else np.zeros(0)
    est_intervals = np.array([[n[0], n[1]] for n in est_notes]) if est_notes else np.zeros((0, 2))
    est_pitches = np.array([n[2] for n in est_notes]) if est_notes else np.zeros(0)

    results = []

    if len(ref_intervals) == 0:
        return results

    # Use mir_eval matching for COn (onset only)
    ref_hz = mir_eval.util.midi_to_hz(ref_pitches) if len(ref_pitches) > 0 else ref_pitches
    est_hz = mir_eval.util.midi_to_hz(est_pitches) if len(est_pitches) > 0 else est_pitches

    # Manual matching to get per-note details
    est_used = set()

    for ri in range(len(ref_notes)):
        r_on, r_off, r_pitch = ref_notes[ri]
        r_dur = r_off - r_on
        r_hz = mir_eval.util.midi_to_hz(np.array([r_pitch]))[0]

        info = {
            "ref_onset": r_on, "ref_offset": r_off, "ref_pitch": r_pitch,
            "ref_duration": r_dur,
            "onset_matched": False, "pitch_matched": False,
            "offset_matched": False, "matched": False,
            "est_onset": None, "est_offset": None, "est_pitch": None,
            "offset_error": None, "offset_tolerance": None,
        }

        # Find best onset match
        best_ei = -1
        best_d = float("inf")
        for ei in range(len(est_notes)):
            if ei in est_used:
                continue
            e_on = est_notes[ei][0]
            d = abs(e_on - r_on)
            if d <= onset_tol and d < best_d:
                best_d = d
                best_ei = ei

        if best_ei >= 0:
            info["onset_matched"] = True
            e_on, e_off, e_pitch = est_notes[best_ei]
            info["est_onset"] = e_on
            info["est_offset"] = e_off
            info["est_pitch"] = e_pitch
            est_used.add(best_ei)

            # Pitch check (50 cents = half semitone)
            e_hz = mir_eval.util.midi_to_hz(np.array([e_pitch]))[0]
            cents = abs(1200 * np.log2(e_hz / r_hz)) if r_hz > 0 and e_hz > 0 else 999
            if cents <= pitch_tol:
                info["pitch_matched"] = True

                # Offset check: max(50ms, 20% of ref duration)
                off_tol = max(0.05, 0.2 * r_dur)
                info["offset_tolerance"] = off_tol
                info["offset_error"] = e_off - r_off
                if abs(e_off - r_off) <= off_tol:
                    info["offset_matched"] = True
                    info["matched"] = True

        results.append(info)

    return results


def run_diagnostic(onset_threshold=None):
    # Config: exp6 settings
    cfg = Config(experiment=6)
    cfg.context_frames = 21
    cfg.deeper_backbone = True
    cfg.pitch_soft_voting = True
    cfg.smooth_probs = True
    cfg.smooth_kernel = 3
    cfg.peak_min_distance = 3
    cfg.use_offset_refine = True
    cfg.offset_refine_checkpoint = "./checkpoints/exp6/offset_model.pt"
    cfg.offset_n_temporal_layers = 7

    if onset_threshold is not None:
        cfg.onset_threshold = onset_threshold
        cfg.threshold_search = False
        print(f"Using fixed onset threshold: {onset_threshold}")
    else:
        cfg.threshold_search = True
        print("Will run grid search to find best onset threshold...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    checkpoint_path = "./checkpoints/exp5/best_model.pt"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = OnsetOffsetPitchModel(
        n_mels=cfg.n_mels, context_frames=cfg.context_frames,
        n_octaves=cfg.n_octaves, n_pitch_classes=cfg.n_pitch_classes,
        in_channels=cfg.in_channels, deeper_backbone=cfg.deeper_backbone,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    offset_ckpt = torch.load(cfg.offset_refine_checkpoint, map_location=device, weights_only=False)
    offset_model = OffsetRefineModel(
        n_mels=cfg.n_mels, in_channels=cfg.in_channels,
        n_temporal_layers=cfg.offset_n_temporal_layers,
    ).to(device)
    offset_model.load_state_dict(offset_ckpt["model_state_dict"])
    offset_model.eval()

    with open(cfg.annotation_file, "r") as f:
        all_annotations = json.load(f)

    song_dirs = sorted(
        [d for d in os.listdir(cfg.test_dir)
         if os.path.isdir(os.path.join(cfg.test_dir, d)) and d.isdigit()],
        key=lambda x: int(x),
    )

    fd = cfg.frame_duration
    all_ref = []
    all_onset_probs = []
    all_offset_probs = []
    all_oct_logits = []
    all_cls_logits = []
    all_mel_stacks = []
    all_mel_means = []
    all_mel_stds = []
    song_ids = []

    print(f"Running diagnostic on {len(song_dirs)} test songs...\n")
    print("Phase 1: Inference...")

    for si, sid in enumerate(song_dirs):
        song_dir = os.path.join(cfg.test_dir, sid)
        audio_path = find_audio(song_dir, cfg.audio_source)
        if audio_path is None:
            continue

        ref_notes_raw = all_annotations.get(sid)
        if ref_notes_raw is None:
            continue

        y, _ = librosa.load(audio_path, sr=cfg.sample_rate, mono=True)
        if cfg.use_multi_res:
            mel_stack = compute_multi_res_mel(y, cfg)
        else:
            mel = librosa.feature.melspectrogram(
                y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft,
                hop_length=cfg.hop_length, n_mels=cfg.n_mels,
                fmin=cfg.fmin, fmax=cfg.fmax,
            )
            mel_stack = np.log(mel + 1e-8).astype(np.float32)[np.newaxis]

        means = mel_stack.mean(axis=(1, 2))
        stds = mel_stack.std(axis=(1, 2)) + 1e-8

        onset_probs, offset_probs, oct_log, cls_log = infer_full_song(
            mel_stack, means, stds, model, cfg, device
        )

        all_onset_probs.append(onset_probs)
        all_offset_probs.append(offset_probs)
        all_oct_logits.append(oct_log)
        all_cls_logits.append(cls_log)
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

        if (si + 1) % 20 == 0:
            print(f"  Processed {si + 1}/{len(song_dirs)} songs...")

    # Grid search if needed
    if cfg.threshold_search:
        print("\nPhase 2: Grid search for best onset threshold...")
        best_on_t, best_off_t, best_f1 = threshold_grid_search(
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

    print(f"\nUsing onset_threshold={cfg.onset_threshold:.3f}")

    # Phase 3: Extract notes and do per-note matching
    print("\nPhase 3: Per-note analysis...")
    all_results = []
    song_stats = []

    for s_idx in range(len(all_ref)):
        est_notes = extract_notes_refined(
            all_onset_probs[s_idx], all_offset_probs[s_idx],
            all_oct_logits[s_idx], all_cls_logits[s_idx],
            all_mel_stacks[s_idx], all_mel_means[s_idx],
            all_mel_stds[s_idx], offset_model, cfg, fd, device,
        )

        matches = match_notes(all_ref[s_idx], est_notes)
        sid = song_ids[s_idx]
        for m in matches:
            m["song_id"] = sid
        all_results.extend(matches)

        n_ref = len(matches)
        n_onset = sum(1 for m in matches if m["onset_matched"])
        n_pitch = sum(1 for m in matches if m["pitch_matched"])
        n_offset = sum(1 for m in matches if m["offset_matched"])
        song_stats.append((sid, n_ref, n_onset, n_pitch, n_offset))

    # ========== ANALYSIS ==========
    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC RESULTS — {len(all_results)} reference notes")
    print(f"{'='*70}\n")

    total = len(all_results)
    onset_ok = [r for r in all_results if r["onset_matched"]]
    pitch_ok = [r for r in all_results if r["pitch_matched"]]
    offset_ok = [r for r in all_results if r["offset_matched"]]

    print(f"  Onset matched:  {len(onset_ok):>6d} / {total}  ({100*len(onset_ok)/total:.1f}%)")
    print(f"  +Pitch matched: {len(pitch_ok):>6d} / {total}  ({100*len(pitch_ok)/total:.1f}%)")
    print(f"  +Offset matched:{len(offset_ok):>6d} / {total}  ({100*len(offset_ok)/total:.1f}%)")
    print(f"  Offset failures: {len(pitch_ok) - len(offset_ok)} notes (pitch OK but offset wrong)")

    # ---- Duration breakdown ----
    print(f"\n{'─'*70}")
    print("  OFFSET ERRORS BY NOTE DURATION")
    print(f"{'─'*70}")

    duration_bins = [
        ("< 0.1s", 0, 0.1),
        ("0.1–0.2s", 0.1, 0.2),
        ("0.2–0.5s", 0.2, 0.5),
        ("0.5–1.0s", 0.5, 1.0),
        ("1.0–2.0s", 1.0, 2.0),
        ("> 2.0s", 2.0, 999),
    ]

    for label, lo, hi in duration_bins:
        in_bin = [r for r in all_results if lo <= r["ref_duration"] < hi]
        if not in_bin:
            continue
        n_bin = len(in_bin)
        n_onset = sum(1 for r in in_bin if r["onset_matched"])
        n_pitch = sum(1 for r in in_bin if r["pitch_matched"])
        n_off = sum(1 for r in in_bin if r["offset_matched"])
        pitch_fail_off = sum(1 for r in in_bin if r["pitch_matched"] and not r["offset_matched"])

        # Offset error stats for pitch-matched notes
        off_errors = [abs(r["offset_error"]) for r in in_bin
                      if r["pitch_matched"] and r["offset_error"] is not None]
        off_tols = [r["offset_tolerance"] for r in in_bin
                    if r["pitch_matched"] and r["offset_tolerance"] is not None]

        mean_err = np.mean(off_errors) if off_errors else 0
        median_err = np.median(off_errors) if off_errors else 0
        mean_tol = np.mean(off_tols) if off_tols else 0

        print(f"\n  {label:>10s}:  {n_bin:>5d} notes  "
              f"(onset={n_onset}, pitch={n_pitch}, offset={n_off})")
        print(f"             offset failures: {pitch_fail_off}  "
              f"| mean |error|={mean_err:.3f}s  median={median_err:.3f}s  "
              f"| mean tolerance={mean_tol:.3f}s")

    # ---- Offset error distribution for pitch-matched notes ----
    print(f"\n{'─'*70}")
    print("  SIGNED OFFSET ERROR DISTRIBUTION (pitch-matched notes)")
    print(f"{'─'*70}")

    signed_errors = [r["offset_error"] for r in all_results
                     if r["pitch_matched"] and r["offset_error"] is not None]
    if signed_errors:
        se = np.array(signed_errors)
        print(f"  N = {len(se)}")
        print(f"  Mean = {se.mean():.4f}s  (positive = predicted too late)")
        print(f"  Median = {np.median(se):.4f}s")
        print(f"  Std = {se.std():.4f}s")
        print(f"  |error| percentiles:  50%={np.percentile(np.abs(se),50):.3f}s  "
              f"75%={np.percentile(np.abs(se),75):.3f}s  "
              f"90%={np.percentile(np.abs(se),90):.3f}s  "
              f"95%={np.percentile(np.abs(se),95):.3f}s")

        # How many are within various tolerances
        for tol in [0.023, 0.046, 0.05, 0.07, 0.10, 0.15, 0.20]:
            within = np.sum(np.abs(se) <= tol)
            print(f"  Within ±{tol:.3f}s: {within} ({100*within/len(se):.1f}%)")

    # ---- Worst songs ----
    print(f"\n{'─'*70}")
    print("  WORST 15 SONGS BY OFFSET FAILURE COUNT")
    print(f"{'─'*70}")

    song_failures = []
    for sid, n_ref, n_onset, n_pitch, n_offset in song_stats:
        fail = n_pitch - n_offset
        rate = fail / max(n_ref, 1)
        song_failures.append((sid, n_ref, n_onset, n_pitch, n_offset, fail, rate))

    song_failures.sort(key=lambda x: -x[5])
    print(f"  {'Song':>6s}  {'Ref':>4s}  {'On':>4s}  {'Pitch':>5s}  {'Off':>4s}  {'Fail':>4s}  {'Rate':>5s}")
    for sid, n_ref, n_on, n_p, n_off, fail, rate in song_failures[:15]:
        print(f"  {sid:>6s}  {n_ref:>4d}  {n_on:>4d}  {n_p:>5d}  {n_off:>4d}  {fail:>4d}  {rate:>5.1%}")

    # ---- Pitch breakdown ----
    print(f"\n{'─'*70}")
    print("  OFFSET FAILURES BY PITCH RANGE")
    print(f"{'─'*70}")

    pitch_bins = [
        ("C2-B2 (36-47)", 36, 48),
        ("C3-B3 (48-59)", 48, 60),
        ("C4-B4 (60-71)", 60, 72),
        ("C5-B5 (72-83)", 72, 84),
    ]

    for label, lo, hi in pitch_bins:
        in_bin = [r for r in all_results if lo <= r["ref_pitch"] < hi]
        if not in_bin:
            continue
        n_bin = len(in_bin)
        n_pitch = sum(1 for r in in_bin if r["pitch_matched"])
        n_off = sum(1 for r in in_bin if r["offset_matched"])
        fail = n_pitch - n_off
        print(f"  {label}: {n_bin} notes, {n_pitch} pitch-matched, "
              f"{n_off} offset-matched, {fail} failures")

    # ---- PITCH ERROR ANALYSIS ----
    print(f"\n{'─'*70}")
    print("  PITCH ERRORS (onset matched but pitch wrong)")
    print(f"{'─'*70}")

    pitch_errors = [r for r in all_results if r["onset_matched"] and not r["pitch_matched"]]
    print(f"  Total pitch errors: {len(pitch_errors)} / {len(onset_ok)} onset-matched notes "
          f"({100*len(pitch_errors)/max(len(onset_ok),1):.1f}%)")

    if pitch_errors:
        # Semitone error distribution
        semitone_errors = []
        octave_errors = 0
        for r in pitch_errors:
            diff = r["est_pitch"] - r["ref_pitch"]
            semitone_errors.append(diff)
            if abs(diff) == 12 or abs(diff) == 24:
                octave_errors += 1

        se = np.array(semitone_errors)
        print(f"\n  Semitone error distribution:")
        print(f"    Mean = {se.mean():.2f}, Median = {np.median(se):.1f}, Std = {se.std():.2f}")
        print(f"    Octave errors (±12 or ±24 semitones): {octave_errors} ({100*octave_errors/len(pitch_errors):.1f}%)")

        # Histogram of absolute semitone errors
        abs_se = np.abs(se)
        print(f"\n  |Semitone error| histogram:")
        for d in range(0, 25):
            count = np.sum(abs_se == d)
            if count > 0:
                bar = "█" * min(count, 60)
                print(f"    {d:>2d} semitones: {count:>4d}  {bar}")

        # Duration of pitch-error notes
        durations = [r["ref_duration"] for r in pitch_errors]
        print(f"\n  Duration of pitch-error notes:")
        print(f"    Mean = {np.mean(durations):.3f}s, Median = {np.median(durations):.3f}s")
        short_count = sum(1 for d in durations if d < 0.1)
        print(f"    < 0.1s (very short): {short_count} ({100*short_count/len(pitch_errors):.1f}%)")

        # Pitch errors by ref pitch range
        print(f"\n  Pitch errors by ref pitch range:")
        for label, lo, hi in pitch_bins:
            in_bin = [r for r in pitch_errors if lo <= r["ref_pitch"] < hi]
            if not in_bin:
                continue
            oct_err = sum(1 for r in in_bin if abs(r["est_pitch"] - r["ref_pitch"]) in (12, 24))
            print(f"    {label}: {len(in_bin)} errors ({oct_err} octave)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onset_threshold", type=float, default=None,
                        help="Fixed onset threshold (skip grid search)")
    args = parser.parse_args()
    run_diagnostic(onset_threshold=args.onset_threshold)
