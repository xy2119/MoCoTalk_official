#!/usr/bin/env python
# coding=utf-8

"""
Step 1: run this script in an environment that has audio-separator installed.
It separates the vocals of every evaluation video in a directory and writes <stem>-vocals.wav.

Example:

    python prepare_vocals_offline.py \
        --input_dir /path/to/eval_videos \
        --output_dir /path/to/vocals_offline \
        --vocal_separator_model /path/to/Kim_Vocal_2.onnx

The output looks roughly like:
    /path/to/vocals_offline/
        video1-vocals.wav
        video2-vocals.wav
        ...
"""

import argparse
import os
from pathlib import Path

from audio_separator.separator import Separator


def is_video_file(path: Path):
    return path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser("Offline vocal separation for evaluation videos")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory of evaluation driving videos (with audio).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store the separated vocals.wav files.",
    )
    parser.add_argument(
        "--vocal_separator_model",
        type=str,
        default="checkpoints/misc/vocal_separator/Kim_Vocal_2.onnx",
        help="Path to Kim_Vocal_2.onnx (the ONNX model used by audio-separator).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.vocal_separator_model)
    if not model_path.exists():
        raise FileNotFoundError(f"vocal_separator_model not found: {model_path}")

    model_dir = model_path.parent
    model_name = model_path.name

    print(f"[INIT] audio_separator model_dir = {model_dir}")
    print(f"[INIT] audio_separator model_name = {model_name}")

    # Only output the "vocals" stem
    separator = Separator(
        output_dir=output_dir,
        output_single_stem="vocals",
        model_file_dir=str(model_dir),
    )
    separator.load_model(model_name)

    # Collect videos
    video_files = [
        p for p in sorted(input_dir.iterdir())
        if p.is_file() and is_video_file(p)
    ]
    if not video_files:
        print(f"[WARN] No video files found in {input_dir}")
        return

    print(f"[INFO] Found {len(video_files)} videos in {input_dir}")

    for vid_path in video_files:
        stem = vid_path.stem
        target_vocals = output_dir / f"{stem}-vocals.wav"

        # skip videos that are already separated
        if target_vocals.exists():
            print(f"[SKIP] {target_vocals} already exists.")
            continue

        print(f"\n[PROC] {vid_path}")
        print(f"[INFO] separating vocals -> {target_vocals.name}")

        # audio_separator.Separator.separate may return:
        # - a single path (str / Path)
        # - a list of paths
        outputs = separator.separate(vid_path.as_posix())

        # normalise to a list
        if isinstance(outputs, (str, Path)):
            output_paths = [outputs]
        elif isinstance(outputs, (list, tuple)):
            output_paths = outputs
        else:
            raise TypeError(f"Unexpected type from Separator.separate: {type(outputs)}")

        if len(output_paths) == 0:
            print(f"[WARN] Separator returned empty outputs for {vid_path}")
            continue

        first_out = Path(output_paths[0])
        # audio_separator may return relative paths; join them with output_dir
        if not first_out.is_absolute():
            first_out = output_dir / first_out

        if not first_out.exists():
            print(f"[WARN] output file not found: {first_out}")
            continue

        # rename to <stem>-vocals.wav so the later steps can find it by convention
        if first_out.resolve() != target_vocals.resolve():
            if target_vocals.exists():
                target_vocals.unlink()
            first_out.rename(target_vocals)

        print(f"[OK] saved vocals: {target_vocals}")

    print("\n[DONE] All videos processed.")


if __name__ == "__main__":
    main()