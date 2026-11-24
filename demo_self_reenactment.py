#!/usr/bin/env python
# coding=utf-8
"""
Self-reenactment inference for RelightablePA Audio+Router.

Input: a talking-head video with audio (vocal separation already done offline)
Output: the self-reenacted mp4, with the original audio muxed back in

Usage:
    python eval_self_reenact.py \
        --input_video /path/to/video.mp4 \
        --output_dir results/self_reenact \
        --offline_vocals_root /path/to/vocals_offline \
        --pretrained_model_name_or_path pretrained_weights/stable-video-diffusion-img2vid-xt \
        --checkpoint_path pretrained_weights/mocotalk

A directory can also be processed in batch:
    python eval_self_reenact.py \
        --input_dir /path/to/HDTF \
        --output_dir results/self_reenact \
        --offline_vocals_root /path/to/HDTF/vocals_offline
"""

import os
import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, EulerDiscreteScheduler

from prepare_conditions import AudioProcessor, PreprocessSingleWithAudio

from mocotalk.models.head_net import HeadNet
from mocotalk.models.light_net import LightNet
from mocotalk.models.ref_net import RefNet
from mocotalk.models.audio_adapter import AudioProjModel
from mocotalk.models.unet_router import UNetSpatioTemporalConditionModel

from mocotalk.pipelines.pipeline_mocotalk import (
    MoCoTalkPipeline,
)


# ============================================================
#                      Utility functions
# ============================================================
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def is_video_file(path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def extract_first_frame(video_path: str, frame_save_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read first frame from video: {video_path}")
    os.makedirs(os.path.dirname(frame_save_path), exist_ok=True)
    cv2.imwrite(frame_save_path, frame)


def extract_frames_ffmpeg(video_path: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = f'ffmpeg -y -loglevel error -i "{video_path}" "{(out_dir / "%05d.png").as_posix()}"'
    code = os.system(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg extract frames failed, cmd: {cmd}")


def load_shading_tiles(frames_dir: Path):
    """Parse the five 512x512 tiles [ref | ref_mask | tar | head | light] from a frame extracted from shading.mp4"""
    files = sorted(frames_dir.glob("*.png"))
    if len(files) == 0:
        raise RuntimeError(f"No frames found in {frames_dir}")

    first = np.array(Image.open(files[0]).convert("RGB"), dtype=np.uint8)
    if first.shape[1] < 512 * 5:
        raise RuntimeError(
            f"Frame width too small, expect >= {512*5}, got {first.shape}"
        )

    ref_rgb = first[:, 0:512]
    ref_mask_rgb = first[:, 512:1024]

    tar_list, head_list, light_list = [], [], []
    for p in files:
        img = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        tar_list.append(img[:, 1024:1536])
        head_list.append(img[:, 1536:2048])
        light_list.append(img[:, 2048:2560])

    return ref_rgb, ref_mask_rgb, tar_list, head_list, light_list


def make_audio_windows(audio_tlc: torch.Tensor, num_frames: int, win_size: int = 5):
    """(T,L,C) -> (F,W,L,C)"""
    if audio_tlc.dim() != 3:
        raise ValueError(f"audio_emb expected (T,L,C), got {tuple(audio_tlc.shape)}")

    T = audio_tlc.shape[0]
    F = num_frames
    if T < F:
        pad = audio_tlc[-1:].repeat(F - T, 1, 1)
        audio_tlc = torch.cat([audio_tlc, pad], dim=0)
    elif T > F:
        audio_tlc = audio_tlc[:F]

    margin = win_size // 2
    left = audio_tlc[0:1].repeat(margin, 1, 1)
    right = audio_tlc[-1:].repeat(margin, 1, 1)
    padded = torch.cat([left, audio_tlc, right], dim=0)

    chunks = [padded[t : t + win_size] for t in range(F)]
    return torch.stack(chunks, dim=0)


def export_video_ffmpeg(frames_dir: Path, save_path: str, fps: int = 20):
    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        f'ffmpeg -y -loglevel error -r {fps} -i "{(frames_dir / "%05d.png").as_posix()}" '
        f'-pix_fmt yuv420p -c:v libx264 -crf 18 "{save_path}"'
    )
    code = os.system(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg export failed, cmd: {cmd}")


def to_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if frame.ndim == 2:
        return np.stack([frame] * 3, axis=-1)
    if frame.ndim != 3:
        raise ValueError(f"Unexpected frame ndim={frame.ndim}, shape={frame.shape}")
    if frame.shape[0] in (1, 3, 4) and frame.shape[1] > 8 and frame.shape[2] > 8:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] >= 4:
        frame = frame[..., :3]
    if frame.shape[-1] != 3:
        raise ValueError(f"Cannot convert to RGB, shape={frame.shape}")
    return frame


def get_video_fps(video_path: str, default_fps: int = 25) -> float:
    """Read the fps of the source video, used as the reference output fps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return float(default_fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 0 or np.isnan(fps):
        return float(default_fps)
    return float(fps)


def has_audio_stream(video_path: str) -> bool:
    """Use ffprobe to check whether a video has an audio stream."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, check=False,
        )
        return "audio" in out.stdout.strip().lower()
    except FileNotFoundError:
        return False


def mux_audio_into_video(
    silent_video: str,
    audio_source: str,
    out_path: str,
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
):
    """
    Mux the audio track of audio_source onto silent_video and write out_path.
    - If audio_source is longer than silent_video, it is truncated with -shortest;
    - the video stream is copied, not re-encoded;
    - audio_source can be the original video or a vocals.wav file.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", silent_video,
        "-i", audio_source,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", audio_codec,
        "-b:a", audio_bitrate,
        "-shortest",
        out_path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"ffmpeg mux failed.\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )


# ============================================================
#                      Router Pipeline
# ============================================================
def build_router_pipeline(
    pretrained_model_name_or_path: str,
    checkpoint_path: str,
    device: torch.device,
    weight_dtype: torch.dtype,
):
    scheduler = EulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path, subfolder="scheduler"
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        pretrained_model_name_or_path, subfolder="feature_extractor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name_or_path, subfolder="image_encoder", variant="fp16"
    )
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="sd-vae-ft-mse"
    )
    unet = UNetSpatioTemporalConditionModel.from_config(
        pretrained_model_name_or_path, subfolder="unet"
    )

    head_embedder = HeadNet(noise_latent_channels=320)
    light_embedder = LightNet(noise_latent_channels=320)
    ref_embedder = RefNet(noise_latent_channels=320)
    audio_proj = AudioProjModel(
        seq_len=5, blocks=12, channels=768,
        intermediate_dim=512, noise_latent_channels=320,
    )

    ckpt = Path(checkpoint_path)
    unet.load_state_dict(torch.load(ckpt / "unet.pth", map_location="cpu"))
    head_embedder.load_state_dict(torch.load(ckpt / "head_embedder.pth", map_location="cpu"))
    light_embedder.load_state_dict(torch.load(ckpt / "light_embedder.pth", map_location="cpu"))
    ref_embedder.load_state_dict(torch.load(ckpt / "ref_embedder.pth", map_location="cpu"))
    audio_proj.load_state_dict(torch.load(ckpt / "audio_proj.pth", map_location="cpu"))

    for m in [vae, image_encoder, unet, head_embedder, light_embedder, ref_embedder, audio_proj]:
        m.requires_grad_(False)
        m.eval()
        m.to(device, dtype=weight_dtype)

    pipe = MoCoTalkPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=scheduler,
        feature_extractor=feature_extractor,
        head_embedder=head_embedder,
        light_embedder=light_embedder,
        ref_embedder=ref_embedder,
        audio_proj=audio_proj,
    ).to(device)

    pipe.set_progress_bar_config(disable=False)
    return pipe


def run_router_inference_one_video(
    pipe,
    shading_video_path: Path,
    audio_emb_path,
    out_video_path: Path,
    device: torch.device,
    weight_dtype: torch.dtype,
    width: int,
    height: int,
    fps: int,
    guidance: float,
    steps: int,
    tile_size: int,
    tile_overlap: int,
    motion_bucket_id: int,
    noise_aug_strength: float,
    audio_win_size: int,
):
    tmp_root = shading_video_path.parent
    tmp_frames = tmp_root / "router_tmp_shading_frames"
    tmp_out = tmp_root / "router_tmp_out_frames"
    shutil.rmtree(tmp_frames, ignore_errors=True)
    shutil.rmtree(tmp_out, ignore_errors=True)
    tmp_frames.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)

    extract_frames_ffmpeg(str(shading_video_path), tmp_frames)
    ref_rgb, ref_mask_rgb, tar_list, head_list, light_list = load_shading_tiles(tmp_frames)
    num_frames = len(tar_list)

    reference_pil = [Image.fromarray(ref_rgb)]

    pixel_head = (
        torch.tensor(np.stack(head_list, 0)[None], dtype=torch.float32, device=device)
        .permute(0, 1, 4, 2, 3) / 255.0
    )
    pixel_light = (
        torch.tensor(np.stack(light_list, 0)[None], dtype=torch.float32, device=device)
        .permute(0, 1, 4, 2, 3) / 255.0
    )
    ref_rep = (
        torch.tensor(ref_rgb[None, None], dtype=torch.float32, device=device)
        .repeat(1, num_frames, 1, 1, 1).permute(0, 1, 4, 2, 3)
    )
    ref_rep = ref_rep / 127.5 - 1.0

    mask64 = cv2.resize(ref_mask_rgb, (64, 64), interpolation=cv2.INTER_AREA)
    mask64 = (
        torch.tensor(mask64[None, None], dtype=torch.float32, device=device)
        .repeat(1, num_frames, 1, 1, 1).permute(0, 1, 4, 2, 3)
    )
    mask64 = mask64[:, :, 0:1] / 255.0

    audio_in = None
    if audio_emb_path is not None and Path(audio_emb_path).exists():
        audio_tlc = torch.load(audio_emb_path, map_location="cpu")
        if isinstance(audio_tlc, np.ndarray):
            audio_tlc = torch.from_numpy(audio_tlc)
        audio_tlc = audio_tlc.float()
        audio_fwlc = make_audio_windows(audio_tlc, num_frames=num_frames, win_size=audio_win_size)
        audio_in = audio_fwlc.unsqueeze(0).to(device=device, dtype=weight_dtype)

    pixel_head = pixel_head.to(dtype=weight_dtype)
    pixel_light = pixel_light.to(dtype=weight_dtype)
    ref_rep = ref_rep.to(dtype=weight_dtype)
    mask64 = mask64.to(dtype=weight_dtype)

    use_autocast = device.type == "cuda" and weight_dtype == torch.float16
    with torch.autocast(device_type="cuda", enabled=use_autocast):
        out = pipe(
            image=reference_pil,
            image_head=pixel_head,
            image_light=pixel_light,
            image_mask=mask64,
            image_ref=ref_rep,
            audio_emb=audio_in,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            num_inference_steps=steps,
            min_guidance_scale=guidance,
            max_guidance_scale=guidance,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            output_type="pt",
            device=device,
        )

    frames = out.frames
    if isinstance(frames, list):
        frames_np = np.stack([np.array(im) for im in frames[0]], 0)
    else:
        frames_np = (frames[0].clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()

    num_out = frames_np.shape[0]
    for i in range(num_out):
        frm = to_hwc_uint8(frames_np[i])
        Image.fromarray(frm, mode="RGB").save(tmp_out / f"{i:05d}.png")

    export_video_ffmpeg(tmp_out, str(out_video_path), fps=fps)

    shutil.rmtree(tmp_frames, ignore_errors=True)
    shutil.rmtree(tmp_out, ignore_errors=True)


# ============================================================
#                  Single-video pipeline (self-reenactment)
# ============================================================
def process_one_video(
    vid_path: Path,
    output_dir: Path,
    tmp_dir: Path,
    pre: PreprocessSingleWithAudio,
    pipe,
    args,
    device: torch.device,
    weight_dtype: torch.dtype,
):
    name = vid_path.stem
    per_video_tmp = tmp_dir / name
    per_video_tmp.mkdir(parents=True, exist_ok=True)

    first_frame_path = per_video_tmp / "ref.png"
    shading_video_path = per_video_tmp / "shading.mp4"
    audio_emb_path = per_video_tmp / "audio_emb.pt"

    silent_out_path = per_video_tmp / f"{name}_silent.mp4"        # video only
    final_out_path = output_dir / f"{name}_result.mp4"            # with audio

    if final_out_path.exists() and not args.overwrite:
        print(f"[Skip] {final_out_path} already exists.")
        return

    print(f"\n=== Processing (self-reenact): {vid_path} ===")

    # 1) first frame = ref = light
    print(f"[Step 1] Extracting first frame -> {first_frame_path}")
    extract_first_frame(str(vid_path), str(first_frame_path))

    # 2) Preprocessing: source and light are the same first frame; driving is the whole video
    print("[Step 2] DECA/SPECTRE + KP + Audio preprocess ...")
    save_shading_path, save_audio_emb_path = pre.run(
        source_path=str(first_frame_path),
        light_path=str(first_frame_path),
        video_path=str(vid_path),  # self-reenactment: driving == source video
        save_shading_path=str(shading_video_path),
        motion_align=args.motion_align,
        audio_emb_path=str(audio_emb_path),
        spectre_head_path=None,
        spectre_feature_path=None,
        spectre_light_path=None,
    )

    # 3) Run the router + audio SVD (writes a silent video first)
    print("[Step 3] RelightablePA Audio+Router inference ...")
    out_fps = args.fps if args.fps > 0 else int(round(get_video_fps(str(vid_path))))
    run_router_inference_one_video(
        pipe=pipe,
        shading_video_path=Path(save_shading_path),
        audio_emb_path=Path(save_audio_emb_path) if save_audio_emb_path else None,
        out_video_path=silent_out_path,
        device=device,
        weight_dtype=weight_dtype,
        width=args.width,
        height=args.height,
        fps=out_fps,
        guidance=args.guidance,
        steps=args.inference_steps,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        motion_bucket_id=args.motion_bucket_id,
        noise_aug_strength=args.noise_aug_strength,
        audio_win_size=args.audio_win_size,
    )

    # 4) Mux the audio into the result video
    print("[Step 4] Muxing audio ...")
    audio_source = None
    if args.audio_source == "vocals":
        # use the offline-separated vocals
        vocals_path = Path(args.offline_vocals_root) / f"{name}-vocals.wav"
        if vocals_path.exists():
            audio_source = str(vocals_path)
        else:
            print(f"[Warn] vocals not found at {vocals_path}, fallback to original audio.")
    if audio_source is None:
        # use the source video's own audio track
        if has_audio_stream(str(vid_path)):
            audio_source = str(vid_path)

    if audio_source is None:
        print("[Warn] No audio available, copying silent video as final result.")
        shutil.copy2(silent_out_path, final_out_path)
    else:
        try:
            mux_audio_into_video(
                silent_video=str(silent_out_path),
                audio_source=audio_source,
                out_path=str(final_out_path),
            )
        except Exception as e:
            print(f"[Warn] mux failed ({e}), saving silent result.")
            shutil.copy2(silent_out_path, final_out_path)

    # Clean up intermediate files
    if not args.keep_tmp:
        shutil.rmtree(per_video_tmp, ignore_errors=True)

    print(f"[Done] -> {final_out_path}")


# ============================================================
#                          main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-reenactment inference for RelightablePA Audio+Router (with audio)."
    )

    # Input: either a single video or a directory
    parser.add_argument("--input_video", type=str, default=None,
                        help="Path to a single driving video (mutually exclusive with --input_dir)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory to evaluate in batch (mutually exclusive with --input_video)")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for the result videos")
    parser.add_argument("--tmp_dir", type=str, default="tmp_self_reenact",
                        help="Directory for intermediate files")
    parser.add_argument("--keep_tmp", action="store_true",
                        help="Keep intermediate files (deleted after inference by default)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite results that already exist")

    # SVD / Pipeline
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="pretrained_weights/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--checkpoint_path", type=str,
                        default="pretrained_weights/mocotalk")
    parser.add_argument("--guidance", type=float, default=2.5)
    parser.add_argument("--inference_steps", type=int, default=25)
    parser.add_argument("--fps", type=int, default=0,
                        help="Output fps; if <= 0, use the fps of the source video")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--tile_size", type=int, default=16)
    parser.add_argument("--tile_overlap", type=int, default=6)
    parser.add_argument("--motion_bucket_id", type=int, default=127)
    parser.add_argument("--noise_aug_strength", type=float, default=0.02)
    parser.add_argument("--audio_win_size", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="fp16",
                        choices=["fp16", "bf16", "fp32"])

    # DECA / SPECTRE
    parser.add_argument("--motion_align", type=str, default="relative",
                        choices=["relative", "absolute"])
    parser.add_argument("--use_spectre", type=int, default=1)
    parser.add_argument("--spectre_exp_scale", type=float, default=1.0)

    # Audio
    parser.add_argument("--use_audio", type=int, default=1)
    parser.add_argument("--wav2vec_model", type=str,
                        default="facebook/wav2vec2-base-960h")
    parser.add_argument("--offline_vocals_root", type=str, required=True,
                        help="Directory of offline-separated vocals, containing <stem>-vocals.wav")
    parser.add_argument("--audio_sample_rate", type=int, default=16000)

    # Source of the output audio track
    parser.add_argument("--audio_source", type=str, default="vocals",
                        choices=["vocals", "original"],
                        help="Audio track muxed into the result video: vocals=separated vocals only, original=original audio of the source video")

    return parser.parse_args()


def main():
    args = parse_args()

    if (args.input_video is None) == (args.input_dir is None):
        raise ValueError("Specify exactly one of --input_video or --input_dir.")

    output_dir = Path(args.output_dir)
    tmp_dir = Path(args.tmp_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # device & dtype
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weight_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]

    # Collect videos
    if args.input_video is not None:
        vp = Path(args.input_video)
        if not vp.exists() or not is_video_file(vp):
            raise FileNotFoundError(f"Invalid input video: {vp}")
        video_files = [vp]
    else:
        in_dir = Path(args.input_dir)
        video_files = [
            p for p in sorted(in_dir.iterdir())
            if p.is_file() and is_video_file(p)
        ]
        if not video_files:
            print(f"[Warn] No videos in {in_dir}")
            return
    print(f"Found {len(video_files)} video(s) to process.")

    # AudioProcessor
    audio_processor = None
    if args.use_audio:
        audio_processor = AudioProcessor(
            wav2vec_model_name=args.wav2vec_model,
            sample_rate=args.audio_sample_rate,
            device=device,
            offline_vocals_root=args.offline_vocals_root,
        )

    pre = PreprocessSingleWithAudio(
        use_spectre=bool(args.use_spectre),
        spectre_exp_scale=args.spectre_exp_scale,
        device=str(device),
        use_audio=bool(args.use_audio),
        audio_processor=audio_processor,
    )

    # Pipeline
    print("Initialising RelightablePA Audio+Router (SVD pipeline)...")
    pipe = build_router_pipeline(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
        weight_dtype=weight_dtype,
    )

    for vp in video_files:
        try:
            process_one_video(
                vid_path=vp,
                output_dir=output_dir,
                tmp_dir=tmp_dir,
                pre=pre,
                pipe=pipe,
                args=args,
                device=device,
                weight_dtype=weight_dtype,
            )
        except Exception as e:
            print(f"[Error] processing {vp} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nAll done.")


if __name__ == "__main__":
    main()