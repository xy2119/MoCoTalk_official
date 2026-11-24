#!/usr/bin/env python
# coding=utf-8
import os
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, EulerDiscreteScheduler

from mocotalk.models.head_net import HeadNet
from mocotalk.models.light_net import LightNet
from mocotalk.models.ref_net import RefNet
from mocotalk.models.audio_adapter import AudioProjModel
from mocotalk.models.unet_router import UNetSpatioTemporalConditionModel

from mocotalk.pipelines.pipeline_mocotalk import MoCoTalkPipeline


def extract_frames_ffmpeg(video_path: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Only -y is forced; other ffmpeg behaviour is left at its defaults
    cmd = f'ffmpeg -y -i "{video_path}" "{(out_dir / "%05d.png").as_posix()}"'
    code = os.system(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg extract frames failed, cmd: {cmd}")


def load_shading_tiles(frames_dir: Path):
    """
    Each frame of shading.mp4 is a horizontal concatenation of:
      [ref | ref_mask | tar | head | light]  each tile 512x512

    Returns:
      ref_rgb: (512,512,3) uint8
      ref_mask_rgb: (512,512,3) uint8
      tar_list/head_list/light_list: list of uint8 (512,512,3)
    """
    files = sorted(frames_dir.glob("*.png"))
    if len(files) == 0:
        raise RuntimeError(f"No frames found in {frames_dir}")

    # Read the first frame to get ref + mask
    first = np.array(Image.open(files[0]).convert("RGB"), dtype=np.uint8)
    if first.shape[1] < 512 * 5:
        raise RuntimeError(f"Frame width too small, expect >= {512*5}, got {first.shape}")

    ref_rgb = first[:, 0:512]
    ref_mask_rgb = first[:, 512:1024]

    tar_list, head_list, light_list = [], [], []
    for p in files:
        img = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        tar = img[:, 1024:1536]
        head = img[:, 1536:2048]
        light = img[:, 2048:2560]
        tar_list.append(tar)
        head_list.append(head)
        light_list.append(light)

    return ref_rgb, ref_mask_rgb, tar_list, head_list, light_list


def make_audio_windows(audio_tlc: torch.Tensor, num_frames: int, win_size: int = 5):
    """
    Input:
      audio_tlc: must be (T, L, C); other layouts such as (T, C) are not supported
    Output:
      audio_fwlc: (F,W,L,C)
    Alignment:
      - if T != F: truncate or pad
      - W=win_size, margin=win_size//2
    """
    if audio_tlc.dim() != 3:
        raise ValueError(f"audio_emb expected (T,L,C), got {tuple(audio_tlc.shape)}")

    T, L, C = audio_tlc.shape
    F = num_frames
    if T < F:
        pad = audio_tlc[-1:].repeat(F - T, 1, 1)
        audio_tlc = torch.cat([audio_tlc, pad], dim=0)
    elif T > F:
        audio_tlc = audio_tlc[:F]
    # now T==F
    margin = win_size // 2
    left = audio_tlc[0:1].repeat(margin, 1, 1)
    right = audio_tlc[-1:].repeat(margin, 1, 1)
    padded = torch.cat([left, audio_tlc, right], dim=0)  # (F+2m, L, C)

    chunks = []
    for t in range(F):
        chunks.append(padded[t : t + win_size])  # (W,L,C)
    audio_fwlc = torch.stack(chunks, dim=0)  # (F,W,L,C)
    return audio_fwlc


def export_video_ffmpeg(frames_dir: Path, save_path: str, fps: int = 20):
    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = f'ffmpeg -y -r {fps} -i "{(frames_dir / "%05d.png").as_posix()}" -pix_fmt yuv420p -c:v libx264 "{save_path}"'
    code = os.system(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg export failed, cmd: {cmd}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    ap.add_argument("--checkpoint_path", type=str, required=True, help="dir containing unet.pth/head_embedder.pth/light_embedder.pth/ref_embedder.pth/audio_proj.pth")
    ap.add_argument("--shading_video", type=str, required=True, help="resources/shading.mp4")
    ap.add_argument("--audio_emb", type=str, default=None, help="audio embedding .pt. If it's (T,L,C) it will be windowed to (F,W,L,C). If None -> no audio cond.")
    ap.add_argument("--save_path", type=str, default="result.mp4")

    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--fps", type=int, default=20)

    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)

    ap.add_argument("--tile_size", type=int, default=16)
    ap.add_argument("--tile_overlap", type=int, default=6)

    ap.add_argument("--motion_bucket_id", type=int, default=127)
    ap.add_argument("--noise_aug_strength", type=float, default=0.02)

    ap.add_argument("--audio_win_size", type=int, default=5, help="W=2*margin+1, should match training AudioProjModel(seq_len)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.dtype == "fp16":
        weight_dtype = torch.float16
    elif args.dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # -------------------------
    # 1) load base svd components
    # -------------------------
    scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder", variant="fp16")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="sd-vae-ft-mse")

    # IMPORTANT: use your router UNet
    unet = UNetSpatioTemporalConditionModel.from_config(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        # low_cpu_mem_usage=True,
    )

    head_embedder = HeadNet(noise_latent_channels=320)
    light_embedder = LightNet(noise_latent_channels=320)
    ref_embedder = RefNet(noise_latent_channels=320)

    # AudioProjModel must match training config
    audio_proj = AudioProjModel(
        seq_len=args.audio_win_size,   # W
        blocks=12,
        channels=768,
        intermediate_dim=512,
        noise_latent_channels=320,
    )

    # -------------------------
    # 2) load finetuned weights
    # -------------------------
    ckpt = Path(args.checkpoint_path)
    unet.load_state_dict(torch.load(ckpt / "unet.pth", map_location="cpu"))
    head_embedder.load_state_dict(torch.load(ckpt / "head_embedder.pth", map_location="cpu"))
    light_embedder.load_state_dict(torch.load(ckpt / "light_embedder.pth", map_location="cpu"))
    ref_embedder.load_state_dict(torch.load(ckpt / "ref_embedder.pth", map_location="cpu"))
    audio_proj.load_state_dict(torch.load(ckpt / "audio_proj.pth", map_location="cpu"))

    # freeze
    for m in [vae, image_encoder, unet, head_embedder, light_embedder, ref_embedder, audio_proj]:
        m.requires_grad_(False)
        m.eval()

    # dtype + device
    image_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    head_embedder.to(device, dtype=weight_dtype)
    light_embedder.to(device, dtype=weight_dtype)
    ref_embedder.to(device, dtype=weight_dtype)
    audio_proj.to(device, dtype=weight_dtype)

    # pipeline
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

    # -------------------------
    # 3) read shading video -> tensors
    # -------------------------
    base_dir = Path(args.save_path).resolve().parent
    tmp_frames = base_dir / "tmp_shading_frames"
    tmp_out = base_dir / "tmp_out_frames"
    shutil.rmtree(tmp_frames, ignore_errors=True)
    shutil.rmtree(tmp_out, ignore_errors=True)
    tmp_frames.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)

    extract_frames_ffmpeg(args.shading_video, tmp_frames)

    ref_rgb, ref_mask_rgb, tar_list, head_list, light_list = load_shading_tiles(tmp_frames)
    num_frames = len(tar_list)

    # ref image: pipeline expects PIL list for CLIP embedding input
    reference_pil = [Image.fromarray(ref_rgb)]

    # build tensors
    # tar in [-1,1]
    pixel_tar = torch.tensor(np.stack(tar_list, 0)[None], dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3)
    pixel_tar = pixel_tar / 127.5 - 1.0  # (1,F,3,512,512)

    # head/light in [0,1]
    pixel_head = torch.tensor(np.stack(head_list, 0)[None], dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3) / 255.0
    pixel_light = torch.tensor(np.stack(light_list, 0)[None], dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3) / 255.0

    # ref in [-1,1] and repeat to (1,F,3,512,512)
    ref_rep = torch.tensor(ref_rgb[None, None], dtype=torch.float32, device=device).repeat(1, num_frames, 1, 1, 1).permute(0, 1, 4, 2, 3)
    ref_rep = ref_rep / 127.5 - 1.0

    # mask -> (1,F,1,64,64) in [0,1]
    mask64 = cv2.resize(ref_mask_rgb, (64, 64), interpolation=cv2.INTER_AREA)
    mask64 = torch.tensor(mask64[None, None], dtype=torch.float32, device=device).repeat(1, num_frames, 1, 1, 1).permute(0, 1, 4, 2, 3)
    mask64 = mask64[:, :, 0:1] / 255.0

    # audio emb: load and window
    audio_in = None
    if args.audio_emb is not None and os.path.exists(args.audio_emb):
        audio_tlc = torch.load(args.audio_emb, map_location="cpu")
        if isinstance(audio_tlc, np.ndarray):
            audio_tlc = torch.from_numpy(audio_tlc)
        audio_tlc = audio_tlc.float()

        # support saved as (T,L,C) only
        audio_fwlc = make_audio_windows(audio_tlc, num_frames=num_frames, win_size=args.audio_win_size)  # (F,W,L,C)
        audio_in = audio_fwlc.unsqueeze(0).to(device=device, dtype=weight_dtype)  # (1,F,W,L,C)

    # cast visual tensors to dtype
    pixel_head = pixel_head.to(dtype=weight_dtype)
    pixel_light = pixel_light.to(dtype=weight_dtype)
    ref_rep = ref_rep.to(dtype=weight_dtype)
    mask64 = mask64.to(dtype=weight_dtype)

    # -------------------------
    # 4) run pipeline
    # -------------------------
    with torch.autocast(device_type="cuda", enabled=(device.type == "cuda" and weight_dtype == torch.float16)):
        out = pipe(
            image=reference_pil,
            image_head=pixel_head,
            image_light=pixel_light,
            image_mask=mask64,
            image_ref=ref_rep,
            audio_emb=audio_in,
            num_frames=num_frames,
            height=args.height,
            width=args.width,
            fps=7,  # keep consistent with your training/inference default
            motion_bucket_id=args.motion_bucket_id,
            noise_aug_strength=args.noise_aug_strength,
            num_inference_steps=args.steps,
            min_guidance_scale=args.guidance,
            max_guidance_scale=args.guidance,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            output_type="pt",
            device=device,
        )

    frames = out.frames  # (B,F,H,W,3) in [0,1] for pt
    if isinstance(frames, list):
        # pil list
        frames_np = np.stack([np.array(im) for im in frames[0]], 0)
    else:
        # pt: (B,F,H,W,3)
        frames_np = (frames[0].clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()

    # save frames
    # for i in range(frames_np.shape[0]):
    #     Image.fromarray(frames_np[i]).save(tmp_out / f"{i:05d}.png")

    # ---- robust frame saving ----
    def to_hwc_uint8(frame: np.ndarray) -> np.ndarray:
        """
        Accept frame in possible shapes:
        - (H,W,3) uint8
        - (3,H,W) uint8
        - (H,W) uint8
        - (1,H,W) or (H,W,1)
        Return (H,W,3) uint8.
        """
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        if frame.ndim == 2:
            # (H,W) -> (H,W,3)
            frame = np.stack([frame]*3, axis=-1)
            return frame

        if frame.ndim != 3:
            raise ValueError(f"Unexpected frame ndim={frame.ndim}, shape={frame.shape}")

        # channels-first (C,H,W)
        if frame.shape[0] in (1, 3, 4) and frame.shape[1] > 8 and frame.shape[2] > 8:
            frame = np.transpose(frame, (1, 2, 0))  # (H,W,C)

        # now channels-last
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        elif frame.shape[-1] >= 4:
            frame = frame[..., :3]  # drop alpha / extra

        if frame.shape[-1] != 3:
            raise ValueError(f"Cannot convert to RGB, shape={frame.shape}")

        return frame

    # Save frames
    num_out = frames_np.shape[0]
    for i in range(num_out):
        frm = to_hwc_uint8(frames_np[i])
        Image.fromarray(frm, mode="RGB").save(tmp_out / f"{i:05d}.png")

    export_video_ffmpeg(tmp_out, args.save_path, fps=args.fps)

    print(f"[OK] saved: {args.save_path}")

    shutil.rmtree(tmp_frames, ignore_errors=True)
    shutil.rmtree(tmp_out, ignore_errors=True)


if __name__ == "__main__":
    main()