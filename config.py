"""
Configuration for CNN Singing Transcription with Pitch Detection.

Per-frame classification using balanced sampling and context window CNN,
extended with factored pitch heads (octave × pitch class).

Evaluated with COn / COnP / COnPOff metrics on MIR-ST500.

Experiments:
  1 – Baseline: single mel spectrogram (n_fft=2048), no augmentation.
  2 – Multi-resolution input: 3 stacked mel spectrograms (1024/2048/4096).
  3 – Enhanced training: multi-res + soft labels, stochastic augmentation,
      deeper backbone (512→256), pitch voting, probability smoothing.
  4 – Two-stage: exp3 first-stage CNN + second-stage offset refinement
      model with dilated temporal convolutions for note-region offset.
  5 – Onset-focused retraining: fine-tune from exp3 with onset loss
      weight 1.5×, augmentations (noise/pitch_shift/gain), 50 epochs.
  6 – Pitch-focused fine-tuning: freeze onset head, pitch loss weight
      3.0×, 3 epochs from exp5 — fixes ±1-2 semitone pitch errors.
      + 7-layer offset model (5.9s RF) with grid-searched
      hyperparameters (bias=0.005, kernel=3, pmd=5).
"""

from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class Config:
    # Experiment selection
    # 1 = baseline, 2 = multi-res, 3 = enhanced, 4 = two-stage,
    # 5 = onset-focused, 6 = pitch-focused (final)
    experiment: int = 1

    # Paths
    dataset_root: str = os.path.expanduser(
        "~/AI/dissertation/singing_transcription_ICASSP2021"
    )
    annotation_file: str = os.path.expanduser(
        "~/AI/dissertation/singing_transcription_ICASSP2021/"
        "MIR-ST500_20210206/MIR-ST500_corrected.json"
    )
    train_dir: str = os.path.expanduser(
        "~/AI/dissertation/singing_transcription_ICASSP2021/train"
    )
    test_dir: str = os.path.expanduser(
        "~/AI/dissertation/singing_transcription_ICASSP2021/test"
    )
    checkpoint_dir: str = "./checkpoints"
    cache_dir: str = "./cache"

    # Which audio file to use: "vocal", "mixture"
    audio_source: str = "vocal"

    # Audio / Mel Spectrogram
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512           # ~23ms per frame at 22050Hz
    n_mels: int = 128
    fmin: float = 80.0
    fmax: float = 8000.0

    # Multi-resolution (Experiments 2+)
    multi_res_n_ffts: List[int] = field(default_factory=lambda: [1024, 2048, 4096])

    # Augmentation
    aug_time_stretch: bool = False      # random time-stretch  (0.9–1.1)
    aug_pitch_shift: bool = False       # random pitch shift   (±2 semitones)
    aug_spec_augment: bool = False      # SpecAugment (freq + time masks)
    aug_noise: bool = False             # additive Gaussian noise
    aug_gain: bool = False              # random gain / volume change

    # Augmentation hyper-parameters
    time_stretch_range: tuple = (0.9, 1.1)
    pitch_shift_range: tuple = (-2, 2)      # semitones
    spec_freq_masks: int = 2
    spec_freq_width: int = 15
    spec_time_masks: int = 2
    spec_time_width: int = 10
    noise_std: float = 0.01
    gain_range: tuple = (0.8, 1.2)
    aug_probability: float = 1.0    # per-sample probability of applying each aug

    # Label
    soft_label_spread: bool = False  # Gaussian-spread onset/offset labels (exp3+)
    soft_label_sigma: float = 1.0   # σ in frames for soft label Gaussian (onsets)
    soft_label_offset_sigma: float = 1.0  # σ for offset labels (can differ from onset)

    # Dataset
    context_frames: int = 11        # context window centered on target frame (21 for exp3+)

    # Pitch
    midi_lowest: int = 36           # C2
    n_octaves: int = 4              # C2..B5 → octaves 0,1,2,3
    n_pitch_classes: int = 12       # chromatic pitch classes
    pitch_ignore_index: int = 100   # ignore_index for silent frames

    # Training
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_cosine_lr: bool = False      # cosine annealing LR scheduler (exp3+)
    deeper_backbone: bool = False    # 512→256 backbone instead of flat→256 (exp3+)
    offset_loss_weight: float = 1.0  # weight multiplier for offset loss
    onset_loss_weight: float = 1.0   # weight multiplier for onset loss (exp5)
    pitch_loss_weight: float = 1.0   # weight multiplier for pitch (octave+class) loss
    freeze_onset_head: bool = False  # freeze onset head during training (exp6)

    # Fine-tuning
    resume_from: str = ""            # checkpoint to load weights from before training

    # Second-stage offset refinement (exp4+)
    use_offset_refine: bool = False  # use second-stage offset model
    offset_refine_checkpoint: str = ""  # path to offset refinement model
    offset_n_temporal_layers: int = 5   # temporal layers in offset model (5=exp4, 7=exp6)
    offset_bias_sec: float = 0.0       # systematic offset correction in seconds (positive = later)

    # Inference / Post-processing
    onset_threshold: float = 0.5
    offset_threshold: float = 0.5
    onset_tolerance: float = 0.05   # seconds
    offset_tolerance: float = 0.05  # seconds
    pitch_tolerance: float = 50.0   # cents (standard: 50 = quarter tone)
    peak_min_distance: int = 5      # frames
    max_note_frames: int = 200      # max note duration in frames (~4.6s)
    min_note_frames: int = 3        # min note duration in frames (~70ms)
    threshold_search: bool = False  # grid-search onset/offset thresholds (exp3+)
    pitch_voting: bool = False       # majority-vote pitch across note frames (exp3)
    pitch_soft_voting: bool = False  # soft probability aggregation for pitch (exp5+)
    smooth_probs: bool = False       # median-filter onset/offset probs (exp3+)
    smooth_kernel: int = 5           # kernel size for median smoothing

    # Device
    device: str = "cuda"

    @property
    def use_multi_res(self) -> bool:
        """Multi-resolution input is used in experiments 2+."""
        return self.experiment >= 2

    @property
    def in_channels(self) -> int:
        """Number of input channels: 1 for baseline, 3 for multi-res."""
        return len(self.multi_res_n_ffts) if self.use_multi_res else 1

    @property
    def frame_duration(self) -> float:
        return self.hop_length / self.sample_rate

    @property
    def augmentation_enabled(self) -> bool:
        return any([
            self.aug_time_stretch, self.aug_pitch_shift,
            self.aug_spec_augment, self.aug_noise, self.aug_gain,
        ])
