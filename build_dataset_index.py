# rebuild_data_json_with_audio.py
import json
import argparse
from pathlib import Path

JSON_FILENAME_TMPL = "{dataset}-data-consistent.json"


def _build_stem2path(dir_path: Path):
    """
    Map every file in a directory to:
        stem -> full path
    For example:
        TFHP_TH_00054_000.mp4    -> stem = TFHP_TH_00054_000
        TFHP_TH_00054_000.png    -> stem = TFHP_TH_00054_000
    If one stem has several extensions, only the first one found is kept (this should not normally happen).
    """
    stem2path = {}
    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        stem = p.stem  # file name without extension
        # If several files share a stem, keep only the first; add an assert here for a stricter check
        if stem not in stem2path:
            stem2path[stem] = p
    return stem2path


def rebuild_json_from_dirs(save_path, dataset):
    """
    Scan the five sub-directories under save_path/dataset:
        {dataset}-video
        {dataset}-kpmap
        {dataset}-mask
        {dataset}-mesh
        {dataset}-audioemb

    Intersect them on the file name without extension (stem), keep the stems that exist
    in all 5 directories, and regenerate data-consistent.json.

    Output fields:
        {
            "video_path":      ...,
            "kps_path":        ...,
            "light_path":      ...,
            "mask_path":       ...,
            "audio_emb_path":  ...
        }
    """

    ds_dir = Path(save_path) / dataset

    video_dir = ds_dir / f"{dataset}-video"
    kp_dir    = ds_dir / f"{dataset}-kpmap"
    mask_dir  = ds_dir / f"{dataset}-mask"
    mesh_dir  = ds_dir / f"{dataset}-mesh"
    audio_dir = ds_dir / f"{dataset}-audioemb"

    for d in [video_dir, kp_dir, mask_dir, mesh_dir, audio_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Directory does not exist: {d}")

    # Build a stem -> path mapping for each directory
    video_map = _build_stem2path(video_dir)
    kp_map    = _build_stem2path(kp_dir)
    mask_map  = _build_stem2path(mask_dir)
    mesh_map  = _build_stem2path(mesh_dir)
    audio_map = _build_stem2path(audio_dir)

    # Keep only stems that exist in all 5 directories
    common_stems = (
        set(video_map.keys())
        & set(kp_map.keys())
        & set(mask_map.keys())
        & set(mesh_map.keys())
        & set(audio_map.keys())
    )
    common_stems = sorted(common_stems)

    print(f"[INFO] {len(video_map)} video stems, "
          f"{len(kp_map)} kpmap stems, "
          f"{len(mask_map)} mask stems, "
          f"{len(mesh_map)} mesh stems, "
          f"{len(audio_map)} audioemb stems")
    print(f"[INFO] {len(common_stems)} stems exist in all five directories and will be written to the json")

    data = []
    for stem in common_stems:
        video_path = video_map[stem].as_posix()
        kp_path    = kp_map[stem].as_posix()
        mask_path  = mask_map[stem].as_posix()
        mesh_path  = mesh_map[stem].as_posix()
        audio_path = audio_map[stem].as_posix()

        # Note: the shading mesh is used as light_path
        data.append(
            {
                "video_path":      video_path,
                "kps_path":        kp_path,
                "light_path":      mesh_path,
                "mask_path":       mask_path,
                "audio_emb_path":  audio_path,
            }
        )

    json_out = ds_dir / JSON_FILENAME_TMPL.format(dataset=dataset)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"[OK] Rebuilt and overwrote: {json_out}")
    print(f"[OK] Entries written: {len(data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./TalkingHeadVideo",
        help="Root directory of the preprocessing output (same as --save_path in preprocess_dataset.py)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="VFHQ",
        help="Dataset name (same as --dataset in preprocess_dataset.py)",
    )
    args = parser.parse_args()

    rebuild_json_from_dirs(args.save_path, args.dataset)
