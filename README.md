# Final Approach

This project contains a singing transcription training and inference workflow, including:

- dataset preparation
- model training
- offset model training
- inference and diagnostics
- experiment checkpoints and result plots

## Project Structure

- `config.py`: central configuration values and paths.
- `dataset.py`: dataset loading and preprocessing utilities.
- `model.py`: main model definition.
- `train.py`: main model training script.
- `offset_model.py`: offset model definition.
- `train_offset.py`: offset model training script.
- `inference.py`: inference pipeline.
- `diagnose.py`: debugging and diagnosis helpers.
- `prepare_audio.py`: audio preparation utilities.
- `checkpoints/`: saved models, metrics, and plots.

## Ignored Local Folders

The following folders are excluded from git tracking:

- `cache/`
- `songs/`
- `__pycache__/` (including nested `__pycache__` folders)

These are configured in `.gitignore`.

## Quick Start

1. Install dependencies required by your environment.
2. Adjust paths and settings in `config.py`.
3. Run training:

```bash
python train.py
```

4. Run inference:

```bash
python inference.py
```
