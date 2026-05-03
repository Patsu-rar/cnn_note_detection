"""
Download audio from a YouTube / YouTube Music URL and extract vocals using HTDemucs.

Usage:
    python prepare_audio.py "https://www.youtube.com/watch?v=..."
    python prepare_audio.py "https://music.youtube.com/watch?v=..."
    python prepare_audio.py "https://youtu.be/..." --output_dir my_songs
    python prepare_audio.py "https://youtu.be/..." --skip_separation   # download only

Then run inference on the extracted vocals:
    python inference.py <output_dir>/<title>/<title>_(Vocals)_htdemucs_ft.wav --experiment 3
"""

import argparse
import os
import re
import subprocess
import sys

import yt_dlp


def sanitize_filename(name: str) -> str:
    """Remove characters that are problematic in file paths."""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.strip('. ')
    return name[:120]  # keep reasonable length


def download_audio(url: str, output_dir: str) -> str:
    """Download audio from YouTube URL, return path to the downloaded mp3."""
    os.makedirs(output_dir, exist_ok=True)

    # First, extract the title to build the output path
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        title = sanitize_filename(info.get("title", "audio"))

    song_dir = os.path.join(output_dir, title)
    os.makedirs(song_dir, exist_ok=True)

    outtmpl = os.path.join(song_dir, "Mixture")
    mp3_path = outtmpl + ".mp3"

    if os.path.isfile(mp3_path):
        print(f"Already downloaded: {mp3_path}")
        return mp3_path

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "320",
        }],
        "quiet": False,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print(f"Downloaded: {mp3_path}")
    return mp3_path


def separate_vocals(mp3_path: str) -> str:
    """Run audio-separator with htdemucs_ft to extract vocals, return vocals path."""
    song_dir = os.path.dirname(mp3_path)

    # audio-separator names the output: <filename>_(Vocals)_htdemucs_ft.wav
    base = os.path.splitext(os.path.basename(mp3_path))[0]
    vocals_path = os.path.join(song_dir, f"{base}_(Vocals)_htdemucs_ft.wav")

    if os.path.isfile(vocals_path):
        print(f"Vocals already exist: {vocals_path}")
        return vocals_path

    cmd = [
        "audio-separator", mp3_path,
        "--model_filename", "htdemucs_ft.yaml",
        "--output_format=wav",
        "--single_stem=Vocals",
        "--output_dir", song_dir,
    ]
    print(f"Running vocal separation...")
    subprocess.run(cmd, check=True)
    print(f"Vocals saved: {vocals_path}")
    return vocals_path


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube audio and extract vocals with HTDemucs."
    )
    parser.add_argument("url", help="YouTube or YouTube Music URL")
    parser.add_argument("--output_dir", default="songs",
                        help="Base output directory (default: songs/)")
    parser.add_argument("--skip_separation", action="store_true",
                        help="Only download, skip vocal separation")
    args = parser.parse_args()

    mp3_path = download_audio(args.url, args.output_dir)

    if args.skip_separation:
        print(f"\nDone. To separate vocals later:\n"
              f"  audio-separator \"{mp3_path}\" --model_filename htdemucs_ft.yaml "
              f"--output_format=wav --single_stem=Vocals --output_dir \"{os.path.dirname(mp3_path)}\"")
        return

    vocals_path = separate_vocals(mp3_path)

    print(f"\n--- Ready for inference ---")
    print(f"  python inference.py \"{vocals_path}\" --experiment 3")


if __name__ == "__main__":
    main()
