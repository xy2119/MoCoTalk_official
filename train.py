#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Video Diffusion."""
import argparse
import random
import logging
import math
import os
import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.utils.checkpoint
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import diffusers

from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available

from mocotalk.data.dataset import TalkingHeadVideoDataset
from mocotalk.models.head_net import HeadNet
from mocotalk.models.light_net import LightNet
from mocotalk.models.ref_net import RefNet
from mocotalk.models.unet_router import UNetSpatioTemporalConditionModel
from mocotalk.pipelines.pipeline_mocotalk_train import MoCoTalkTrainPipeline

from mocotalk.models.audio_adapter import AudioProjModel

# Lip (mouth-region) loss
from configparser import ConfigParser

from spectre.external.Visual_Speech_Recognition_for_Multiple_Languages.lipreading.model import Lipreading
from spectre.external.Visual_Speech_Recognition_for_Multiple_Languages.dataloader.transform import (
    Compose, Normalize, CenterCrop, Identity
)

from spectre.external.Visual_Speech_Recognition_for_Multiple_Languages.tracker.face_tracker import FaceTracker
from spectre.external.Visual_Speech_Recognition_for_Multiple_Languages.tracker.utils import get_landmarks
from spectre.datasets.data_utils import landmarks_interpolate

import torchvision.transforms.functional as F_v
import collections
from torchvision.utils import save_image

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class Net(nn.Module):
    def __init__(
        self,
        denoising_unet: UNetSpatioTemporalConditionModel,
        head_embedder: HeadNet,
        light_embedder: LightNet,
        ref_embedder: RefNet,
        audio_proj: AudioProjModel,        # audio -> latent branch
    ):
        super().__init__()
        self.denoising_unet = denoising_unet
        self.head_embedder = head_embedder
        self.light_embedder = light_embedder
        self.ref_embedder = ref_embedder
        self.audio_proj = audio_proj

    def forward(
        self,
        inp_noisy_latents,      # (B, F, C_in, H, W)
        inp_head,
        inp_light,
        inp_ref,
        inp_audio,              # (B, F, W, L, C)
        timesteps,
        encoder_hidden_states,  # (B, D) or (B, 1, D)
        added_time_ids,
    ):
        b, f = inp_noisy_latents.shape[:2]
        h_lat, w_lat = inp_noisy_latents.shape[-2:]

        # 1) head / light / ref branches
        inp_head_latents = self.head_embedder(inp_head)     # usually (B*F, C, H, W)
        inp_light_latents = self.light_embedder(inp_light)
        inp_ref_latents = self.ref_embedder(inp_ref)

        # Head condition probility.
        head_prob = np.random.uniform(0, 1, [b])
        head_mask = np.zeros_like(head_prob)
        head_mask[head_prob > 0.5] = 1
        head_mask = torch.from_numpy(head_mask).to(
            dtype=inp_head.dtype, device=inp_head.device
        ).reshape(b, 1, 1, 1, 1)

        # Light condition probility.
        light_prob = np.random.uniform(0, 1, [b])
        light_mask = np.zeros_like(light_prob)
        light_mask[light_prob > 0.5] = 1
        light_mask = torch.from_numpy(light_mask).to(
            dtype=inp_light.dtype, device=inp_light.device
        ).reshape(b, 1, 1, 1, 1)

        # Reference condition probility.
        ref_prob = np.random.uniform(0, 1, [b])
        ref_mask = np.zeros_like(ref_prob)
        ref_mask[ref_prob > 0.5] = 1
        ref_mask = torch.from_numpy(ref_mask).to(
            dtype=inp_light.dtype, device=inp_light.device
        ).reshape(b, 1, 1, 1, 1)

        # Full condition probility
        full_prob = np.random.uniform(0, 1, [b])
        full_contain_mask = np.zeros_like(full_prob)
        full_contain_mask[full_prob > 0.9] = 1
        full_drop_mask = np.zeros_like(full_prob)
        full_drop_mask[full_prob < 0.1] = 1
        full_contain_mask = torch.from_numpy(full_contain_mask).to(
            dtype=inp_light.dtype, device=inp_light.device
        ).reshape(b, 1, 1, 1, 1)
        full_drop_mask = torch.from_numpy(full_drop_mask).to(
            dtype=inp_light.dtype, device=inp_light.device
        ).reshape(b, 1, 1, 1, 1)

        # Probility mask fusion (head / light / ref)
        head_mask = (1 - full_drop_mask) * head_mask
        light_mask = (1 - full_drop_mask) * light_mask
        ref_mask = (1 - full_drop_mask) * ref_mask
        head_mask = torch.clamp(full_contain_mask + head_mask, 0, 1).detach()
        light_mask = torch.clamp(full_contain_mask + light_mask, 0, 1).detach()
        ref_mask = torch.clamp(full_contain_mask + ref_mask, 0, 1).detach()

        # Accept both output layouts: (B*F, C, H, W) or (B, F, C, H, W)
        def _apply_mask(latents, mask):
            if latents is None:
                return None
            if latents.dim() == 5:
                # (B, F, C, H, W)
                return (mask * latents)      # mask: (B,1,1,1,1)
            elif latents.dim() == 4:
                # (B*F, C, H, W)
                latents = latents.view(b, f, *latents.shape[1:])
                latents = mask * latents
                return latents
            else:
                raise ValueError(f"latents should be 4D or 5D, got {latents.shape}")

        inp_head_latents = _apply_mask(inp_head_latents, head_mask)
        inp_light_latents = _apply_mask(inp_light_latents, light_mask)
        inp_ref_latents = _apply_mask(inp_ref_latents, ref_mask)


        # 2) audio -> latent feature map (fourth branch)
        if inp_audio is not None:
            # (B, F, C_lat, H, W)
            audio_latents = self.audio_proj(inp_audio, height=h_lat, width=w_lat)

            audio_prob = np.random.uniform(0, 1, [b])
            audio_mask = np.zeros_like(audio_prob)
            audio_mask[audio_prob > 0.5] = 1
            audio_mask = torch.from_numpy(audio_mask).to(
                dtype=audio_latents.dtype, device=audio_latents.device
            ).reshape(b, 1, 1, 1, 1)

            # apply full_drop / full_contain like the head/light/ref branches
            audio_mask = (1 - full_drop_mask) * audio_mask
            audio_mask = torch.clamp(full_contain_mask + audio_mask, 0, 1).detach()

            audio_latents = audio_latents * audio_mask
        else:
            audio_latents = None


        # reshape every condition to (B*F, C, H, W)
        def _flatten_cond(latents):
            if latents is None:
                return None
            if latents.dim() == 5:
                return latents.flatten(0, 1)
            elif latents.dim() == 4:
                return latents
            else:
                raise ValueError(f"Unexpected condition latent shape: {latents.shape}")

        inp_head_latents = _flatten_cond(inp_head_latents)
        inp_light_latents = _flatten_cond(inp_light_latents)
        inp_ref_latents = _flatten_cond(inp_ref_latents)
        audio_latents = _flatten_cond(audio_latents)

        # 3) make sure encoder_hidden_states is (B, 1, D)
        if encoder_hidden_states.dim() == 2:
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
        elif encoder_hidden_states.dim() == 3:
            pass
        else:
            raise ValueError(
                f"encoder_hidden_states should be 2D or 3D, got {encoder_hidden_states.shape}"
            )

        # 4) call the UNet with audio_latents (not audio tokens)
        model_pred = self.denoising_unet(
            inp_noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_time_ids=added_time_ids,
            head_latents=inp_head_latents,
            light_latents=inp_light_latents,
            ref_latents=inp_ref_latents,
            audio_latents=audio_latents,
        ).sample

        return model_pred

# copy from https://github.com/crowsonkb/k-diffusion.git
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(
        shape, group=0, groups=1, dtype=dtype, device=device
    )
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data

def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()

# min_value = 0.002
# max_value = 700
# image_d = 64
# noise_d_low = 32
# noise_d_high = 64
# sigma_data = 0.5

# resizing utils
# TODO: clean up later
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames,
                       save_all=True,
                       duration=500,
                       loop=0)


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="pretrained_weights/stable-video-diffusion-img2vid-xt",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is sampled during training for inference.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=666, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=300000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        type=bool,
        default=True,
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--data_meta_path",
        type=str,
        default="./TalkingHeadVideo/",
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help=(
            "Training image width."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help=(
            "Training image height"
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        type=bool,
        default=True,
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=128,
        help=("The dimension of the LoRA update matrices."),
    )

    parser.add_argument(
        "--lip_loss_weight",
        type=float,
        default=0,
        help="Weight for lip movement loss (1 - cosine similarity between Lipreading features). Set 0 to disable.",
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def download_image(url):
    original_image = (
        lambda image_url_or_path: load_image(image_url_or_path)
        if urlparse(image_url_or_path).scheme
        else PIL.Image.open(image_url_or_path).convert("RGB")
    )(url)
    return original_image


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

    generator = torch.Generator(
        device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision, variant="fp16"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="sd-vae-ft-mse", revision=args.revision)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=False,
        device_map=None, 
    )
    head_embedder = HeadNet(noise_latent_channels=320)
    light_embedder = LightNet(noise_latent_channels=320)
    ref_embedder = RefNet(noise_latent_channels=320)


    audio_proj = AudioProjModel(
        seq_len=5,          # W = 2*audio_margin+1
        blocks=12,
        channels=768,
        intermediate_dim=512,
        noise_latent_channels=320,   # matches the shallowest UNet layer
    )

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(True)
    head_embedder.requires_grad_(True)
    light_embedder.requires_grad_(True)
    ref_embedder.requires_grad_(True)
    audio_proj.requires_grad_(True) 

    net = Net(denoising_unet=unet, 
              head_embedder=head_embedder,
              light_embedder=light_embedder,
              ref_embedder=ref_embedder,
              audio_proj=audio_proj,             
        )
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    
    optimizer = optimizer_cls(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # check parameters
    if accelerator.is_main_process:
        rec_txt1 = open('rec_para.txt', 'w')
        rec_txt2 = open('rec_para_train.txt', 'w')
        for name, para in net.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        rec_txt1.close()
        rec_txt2.close()

    # --------------------------------------------------
    # Lip-reading network & mouth-crop tools (only used when lip_loss_weight > 0)
    # --------------------------------------------------
    lip_reader = None
    mouth_transform = None
    face_tracker = None

    if args.lip_loss_weight > 0.0:
        # 1) Lip-reading network (same as SPECTRE)
        config = ConfigParser()
        # config.read("spectre/external/Visual_Speech_Recognition_for_Multiple_Languages/configs/LRS3_V_WER32.3.ini")
        config.read("spectre/configs/lipread_config.ini")
        lip_reader = Lipreading(config, device=accelerator.device)
        lip_reader.model.eval()
        lip_reader.eval()
        for p in lip_reader.parameters():
            p.requires_grad_(False)

        # 2) Mouth image preprocessing (same as SPECTRE)
        crop_size = (88, 88)
        mean, std = 0.421, 0.165
        mouth_transform = Compose(
            [
                Normalize(0.0, 1.0),
                CenterCrop(crop_size),
                Normalize(mean, std),
                Identity(),
            ]
        )

        # 3) Landmark detector (the SPECTRE FaceTracker, as used in preprocess_dataset.py)
        face_tracker = FaceTracker()

    # --------------------------------------------------
    # Helper: detect landmarks on a video sequence
    #   frames: (F, 3, 224, 224) in [0, 1]
    #   returns: (F, 68, 2) pixel coordinates (224 scale)
    # --------------------------------------------------
    def detect_landmarks_video(frames_224):
        """
        frames_224: torch.Tensor, shape (F, 3, 224, 224), value in [0,1]
        """
        F_ = frames_224.shape[0]
        H, W = frames_224.shape[-2:]
        face_info = collections.defaultdict(list)

        for t in range(F_):
            img = frames_224[t].detach().cpu()      # (3,224,224)
            img = img.permute(1, 2, 0).numpy()
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            detected_faces = face_tracker.face_detector(img_bgr, rgb=False)
            lms, scores = face_tracker.landmark_detector(img_bgr, detected_faces, rgb=False)
            face_info["bbox"].append(detected_faces)
            face_info["landmarks"].append(lms)
            face_info["landmarks_scores"].append(scores)

        landmarks = get_landmarks(face_info)
        landmarks = landmarks_interpolate(landmarks)
        if landmarks is None:
            return None

        landmarks = np.array(landmarks, dtype=np.float32)  # (F,68,2)

        # If the coordinates look normalised (0~1), map them to pixel coordinates
        if landmarks.max() <= 2.0:  # 2.0 is far below 224, so the values are almost certainly normalised
            landmarks[..., 0] *= W
            landmarks[..., 1] *= H

        return torch.from_numpy(landmarks).to(frames_224.device)

    # --------------------------------------------------
    # Helper: crop the mouth sequence (based on SPECTRE.Trainer.cut_mouth,
    #  but assuming landmarks are already pixel coordinates in 0..224)
    #   frames_224: (F, 3, 224, 224), [0,1]
    #   landmarks:  (F, 68, 2), pixel coordinates
    #   returns: mouth_seq: (F, Hm, Wm), single channel
    # --------------------------------------------------
    _crop_min_size = 32          # minimum side length of the crop patch (tunable)
    _mouth_expand_w = 1.8        # expansion factor over the mouth width (tunable)
    _mouth_expand_h = 1.2        # expansion factor over the mouth height (tunable)
    _start_idx = 48
    _stop_idx = 68

    def cut_mouth_sequence(frames_224, landmarks_224, convert_grayscale=True):
        """
        frames_224: (F, 3, 224, 224), [0,1]
        landmarks_224: (F, 68, 2), pixel coordinates
        returns mouth_sequence: (F, Hm, Wm), single channel
        """
        T_, _, H, W = frames_224.shape

        # 1) Compute one mouth bbox over the whole sequence from all frames, for temporal stability
        mouth_lms = landmarks_224[:, _start_idx:_stop_idx]    # (T, 20, 2)

        x_min = mouth_lms[..., 0].min()
        x_max = mouth_lms[..., 0].max()
        y_min = mouth_lms[..., 1].min()
        y_max = mouth_lms[..., 1].max()

        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        mouth_w = (x_max - x_min)
        mouth_h = (y_max - y_min)

        # Shift the box down slightly, towards the chin and away from the eyes
        # 0.15 means "move down by 0.15 * mouth height"
        cy = cy + 0.10 * mouth_h

        # 2) Add a margin so mouth-shape changes stay inside the box
        half_w = max(_crop_min_size / 2.0, mouth_w * _mouth_expand_w * 0.5)
        half_h = max(_crop_min_size / 2.0, mouth_h * _mouth_expand_h * 0.5)

        # 3) Clamp to the image bounds
        x1 = int(max(cx - half_w, 0))
        x2 = int(min(cx + half_w, W))
        y1 = int(max(cy - half_h, 0))
        y2 = int(min(cy + half_h, H))

        mouth_sequence = []
        for frame_idx in range(T_):
            if convert_grayscale:
                img = F_v.rgb_to_grayscale(frames_224[frame_idx]).squeeze(0)  # (H,W)
            else:
                img = frames_224[frame_idx]  # (3,H,W) also works; the last two dims are H,W when cropping the mouth

            mouth = img[..., y1:y2, x1:x2]    # same bbox for every frame
            mouth_sequence.append(mouth)

        mouth_sequence = torch.stack(mouth_sequence, dim=0)   # (F, Hm, Wm)
        return mouth_sequence


    # --------------------------------------------------
    # Core: compute the lip movement loss for one batch
    #   pixel_values: (B,F,3,H,W), ground-truth video frames in [-1,1]
    #   denoised_latents: (B,F,4,H/8,W/8) —— denoised_latents
    #   visualize: whether to also return mouth crops for saving
    #   returns: lip_loss, vis_gt, vis_pred
    #         where vis_* have shape (F,1,Hc,Wc) or are None
    # --------------------------------------------------
    def compute_lip_loss(pixel_values, denoised_latents, visualize: bool = False, max_frames_for_lip: int = 2):
        if args.lip_loss_weight <= 0.0 or lip_reader is None:
            # return a scalar 0 so it can still take part in backward
            zero = pixel_values.new_zeros([])
            return zero, None, None

        B, F_, _, H_, W_ = pixel_values.shape

        # ---- Compute the lip loss on a subset of frames only, to save memory ----
        if F_ > max_frames_for_lip:
            # uniformly sample max_frames_for_lip frames
            idx = torch.linspace(0, F_ - 1, max_frames_for_lip, device=pixel_values.device).long()
            pixel_values = pixel_values[:, idx]         # (B, maxF, 3, H, W)
            denoised_latents = denoised_latents[:, idx] # (B, maxF, 4, H/8, W/8)
            F_ = max_frames_for_lip
            
        # 1) Decode the predicted frames with the VAE ([-1,1])
        video_length = F_
        latents_flat = denoised_latents.reshape(B * F_, *denoised_latents.shape[2:])
        latents_flat = latents_flat / vae.config.scaling_factor

        vae_dtype = next(vae.parameters()).dtype
        latents_flat = latents_flat.to(dtype=vae_dtype)

        pred_flat = vae.decode(latents_flat).sample  # (B*F,3,H_dec,W_dec)
        pred_frames = pred_flat.reshape(B, F_, *pred_flat.shape[1:])  # (B,F,3,H_dec,W_dec)

        # 2) Convert GT & Pred to [0,1] and resize to 224x224
        gt = (pixel_values + 1.0) / 2.0
        gt = gt.clamp(0.0, 1.0)
        pred = (pred_frames + 1.0) / 2.0
        pred = pred.clamp(0.0, 1.0)

        gt_224 = F.interpolate(
            gt.view(B * F_, 3, H_, W_),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).view(B, F_, 3, 224, 224)

        H_dec, W_dec = pred_frames.shape[-2], pred_frames.shape[-1]
        pred_224 = F.interpolate(
            pred.view(B * F_, 3, H_dec, W_dec),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).view(B, F_, 3, 224, 224)

        per_sample_losses = []

        # These two hold the mouth crops of the first sample with a detected face, for visualisation
        vis_gt = None   # (F,1,Hc,Wc)
        vis_pred = None # (F,1,Hc,Wc)

        for b in range(B):
            frames_gt = gt_224[b]    # (F,3,224,224)
            frames_pred = pred_224[b]

            # 3.1 Detect landmarks on GT first
            landmarks = detect_landmarks_video(frames_gt)
            if landmarks is None:
                # skip this sample if detection fails
                continue

            # 3.2 Crop the mouth sequence: (F,Hm,Wm), single channel
            mouths_gt_raw = cut_mouth_sequence(frames_gt, landmarks)       # (F,Hm,Wm)
            mouths_pred_raw = cut_mouth_sequence(frames_pred, landmarks)   # (F,Hm,Wm)

            # keep only one sample for visualisation
            if visualize and vis_gt is None:
                # add a channel dim (F,1,Hm,Wm) for save_image
                vis_gt = mouths_gt_raw.detach().clone().unsqueeze(1)
                vis_pred = mouths_pred_raw.detach().clone().unsqueeze(1)

            # 3.3 mouth_transform (Normalize + CenterCrop, etc.)
            mouths_gt = mouth_transform(mouths_gt_raw)     # (F,Hc,Wc)
            mouths_pred = mouth_transform(mouths_pred_raw) # (F,Hc,Wc)

            # 3.4 Feed the lipreader encoder with shape (1,F,Hc,Wc)

            device_type = "cuda"  # training runs on GPU

            # cast to float32 to avoid Half/Float mismatches
            mouths_gt_in = mouths_gt.unsqueeze(0).to(accelerator.device, dtype=torch.float32)
            mouths_pred_in = mouths_pred.unsqueeze(0).to(accelerator.device, dtype=torch.float32)

            # disable autocast so the lip reader runs fully in fp32
            with torch.cuda.amp.autocast(enabled=False):
                with torch.no_grad():
                    # GT needs no gradient
                    lip_features_gt = lip_reader.model.encoder(
                        mouths_gt_in, None, extract_resnet_feats=True
                    )

            with torch.cuda.amp.autocast(enabled=False):
                # Pred needs gradients, which flow back to denoised_latents / UNet
                lip_features_pred = lip_reader.model.encoder(
                    mouths_pred_in, None, extract_resnet_feats=True
                )


            # lip_features_*: (1,F,feat_dim) -> flatten the time dimension
            lip_features_gt = lip_features_gt.reshape(-1, lip_features_gt.shape[-1])
            lip_features_pred = lip_features_pred.reshape(-1, lip_features_pred.shape[-1])

            # 3.5 Per-frame cosine similarity, then average
            cos_sim = F.cosine_similarity(lip_features_gt, lip_features_pred, dim=1)  # (F,)
            lip_loss_b = 1.0 - cos_sim.mean()
            per_sample_losses.append(lip_loss_b)

        if len(per_sample_losses) == 0:
            zero = pixel_values.new_zeros([])
            return zero, None, None

        lip_loss = torch.stack(per_sample_losses).mean()
        return lip_loss, vis_gt, vis_pred


    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    train_dataset = TalkingHeadVideoDataset(sample_rate=args.sample_rate, n_sample_frames=args.num_frames, width=args.width, height=args.height, data_meta_path=args.data_meta_path)
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    net, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        net, optimizer, lr_scheduler, train_dataloader
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("SVDFaceDriving", config=vars(args))

    # Train!
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    def encode_image(pixel_values):
        # pixel: [-1, 1]
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        # We unnormalize it after resizing.
        pixel_values = (pixel_values + 1.0) / 2.0

        # Normalize the image with for CLIP input
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(
            device=accelerator.device, dtype=weight_dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        return image_embeddings

    def _get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = 256 * \
            len(add_time_ids)
        expected_add_embed_dim = 768

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        net.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # # Skip steps until we reach the resumed step
            # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     if step % args.gradient_accumulation_steps == 0:
            #         progress_bar.update(1)
            #     continue

            with accelerator.accumulate(net):
                # first, convert images to latent space.
                pixel_values = batch["pixel_values_vid"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                pixel_head = batch["pixel_values_head"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                pixel_ref_mask = batch["pixel_values_ref_mask"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                pixel_ref_mask = pixel_ref_mask.repeat(1, pixel_head.shape[1], 1, 1, 1)

                pixel_values_light = batch["pixel_values_light"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                pixel_values_mask = batch["pixel_values_mask"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                pixel_ref_values = batch["pixel_values_ref_img"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )

                # audio_emb from the dataset
                audio_emb = batch["audio_embeds"].to(weight_dtype).to(
                    accelerator.device, non_blocking=True
                )
                # expected shape: (B, F, W, L, C)

                conditional_pixel_values = pixel_ref_values
                latents = tensor_to_vae_latent(pixel_values, vae)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                cond_sigmas = rand_log_normal(shape=[bsz,], loc=-3.0, scale=0.5).to(latents)
                noise_aug_strength = cond_sigmas[0] # TODO: support batch > 1
                cond_sigmas = cond_sigmas[:, None, None, None, None]
                conditional_pixel_values = \
                    torch.randn_like(conditional_pixel_values) * cond_sigmas + conditional_pixel_values
                conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae)[:, 0, :, :, :]
                conditional_latents = conditional_latents / vae.config.scaling_factor

                # Sample a random timestep for each image
                # P_mean=0.7 P_std=1.6
                sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(latents.device)
                # sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(latents)
 
                
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas = sigmas[:, None, None, None, None]
                noisy_latents = latents + noise * sigmas
                timesteps = torch.Tensor(
                    [0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)

                inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

                # Get the text embedding for conditioning.
                encoder_hidden_states = encode_image(
                    pixel_ref_values[:, 0, :, :, :].float()) # (B, 1, D)

                # Here I input a fixed numerical value for 'motion_bucket_id', which is not reasonable.
                # However, I am unable to fully align with the calculation method of the motion score,
                # so I adopted this approach. The same applies to the 'fps' (frames per second).
                added_time_ids = _get_add_time_ids(
                    6, # fixed
                    127, # motion_bucket_id = 127, fixed
                    noise_aug_strength, # noise_aug_strength == cond_sigmas
                    encoder_hidden_states.dtype,
                    bsz,
                )
                added_time_ids = added_time_ids.to(latents.device)

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(
                        bsz, device=latents.device, generator=generator)

                    # ---------------------------
                    # text / image condition dropout
                    # ---------------------------
                    # make sure encoder_hidden_states is (B, 1, D)
                    if encoder_hidden_states.dim() == 2:
                        encoder_hidden_states_3d = encoder_hidden_states.unsqueeze(1)  # (B, 1, D)
                    elif encoder_hidden_states.dim() == 3:
                        encoder_hidden_states_3d = encoder_hidden_states
                    else:
                        raise ValueError(f"encoder_hidden_states should be 2D or 3D, got {encoder_hidden_states.shape}")



                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    # null_conditioning = torch.zeros_like(encoder_hidden_states)
                    # encoder_hidden_states = torch.where(
                    #     prompt_mask, null_conditioning.unsqueeze(1), encoder_hidden_states.unsqueeze(1))   

                    null_conditioning = torch.zeros_like(encoder_hidden_states_3d)  # (B,1,D)

                    # final text condition: (B,1,D)
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states_3d)
                
                    
                    # Sample masks for the original images.
                    image_mask_dtype = conditional_latents.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(
                            image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    conditional_latents = image_mask * conditional_latents

                # Concatenate the `conditional_latents` with the `noisy_latents`.
                conditional_latents = conditional_latents.unsqueeze(
                    1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
                
                conditional_latents = conditional_latents * (1 - pixel_ref_mask)
                inp_noisy_latents = torch.cat(
                    [inp_noisy_latents, conditional_latents], dim=2)

                # check https://arxiv.org/abs/2206.00364(the EDM-framework) for more details.
                pixel_ref_values = pixel_ref_values.repeat(1, noisy_latents.shape[1], 1, 1, 1)
                target = latents
                model_pred = net(inp_noisy_latents, 
                                pixel_head, 
                                pixel_values_light, 
                                pixel_ref_values, 
                                audio_emb,
                                timesteps, 
                                encoder_hidden_states, 
                                added_time_ids)

                # Denoise the latents
                c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas ** 2) * (sigmas**-2.0)

                # MSE loss
                loss = torch.mean(
                    (weighing.float() * (denoised_latents.float() -
                     target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1,
                )
                app_loss = torch.sum(
                    (weighing.float() * pixel_values_mask * (denoised_latents.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1) / (torch.sum(pixel_values_mask.reshape(target.shape[0], -1), dim=1) * 4 + 1e-8)

                # loss = loss.mean() + app_loss.mean()

                base_loss = loss.mean() + app_loss.mean()

                # -----------------------------
                # Lip movement loss + visualisation
                # -----------------------------
                # the global_step about to run (it is incremented below)
                current_step = global_step + 1

                # visualise only on the main process, every 10 steps
                visualize_mouth = (
                    args.lip_loss_weight > 0.0
                    and accelerator.is_main_process
                    and (current_step % 1000 == 0)
                )
                # print('current_step: ',current_step,' global_step: ',global_step, ' visualize_mouth: ', visualize_mouth)
                lip_loss_interval = 1  # could be exposed as a command-line argument

                if (
                    args.lip_loss_weight > 0.0 
                    # and accelerator.sync_gradients
                    and (current_step % lip_loss_interval == 0)
                ):
                    lip_loss, vis_gt, vis_pred = compute_lip_loss(
                        pixel_values, denoised_latents, visualize=visualize_mouth
                    )
                    loss = base_loss + args.lip_loss_weight * lip_loss
                else:
                    lip_loss = pixel_values.new_zeros([])
                    vis_gt, vis_pred = None, None
                    loss = base_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss,
                                "lip_loss": lip_loss.item() if args.lip_loss_weight > 0.0 else 0.0,
                        }, step=global_step)
                train_loss = 0.0

                # ---------------------------------------
                # save the mouth crops (GT vs Pred) every 10 steps
                # ---------------------------------------
                if (
                    accelerator.is_main_process
                    and args.lip_loss_weight > 0.0
                    and (global_step % 1000 == 0)
                    and (vis_gt is not None)
                    and (vis_pred is not None)
                ):
                    mouth_vis_dir = os.path.join(args.output_dir, "mouth_vis")
                    os.makedirs(mouth_vis_dir, exist_ok=True)

                    # vis_gt / vis_pred: (F,1,H,W)
                    num_frames_vis = vis_gt.shape[0]

                    # stack them: the top F images are GT, the bottom F are Pred
                    grid = torch.cat([vis_gt, vis_pred], dim=0)  # (2F,1,H,W)
                    save_path = os.path.join(
                        mouth_vis_dir, f"step_{global_step:06d}_mouth.png"
                    )

                    # save as a single image, laid out along time
                    save_image(grid.cpu(), save_path, nrow=num_frames_vis)
                    logger.info(f"Saved mouth visualization to: {save_path}")


                if accelerator.is_main_process:
                    # save checkpoints!
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        torch.save(accelerator.unwrap_model(net).denoising_unet.state_dict(), os.path.join(save_path, "unet.pth"))
                        torch.save(accelerator.unwrap_model(net).head_embedder.state_dict(), os.path.join(save_path, "head_embedder.pth"))
                        torch.save(accelerator.unwrap_model(net).light_embedder.state_dict(), os.path.join(save_path, "light_embedder.pth"))
                        torch.save(accelerator.unwrap_model(net).ref_embedder.state_dict(), os.path.join(save_path, "ref_embedder.pth"))
                        torch.save(accelerator.unwrap_model(net).audio_proj.state_dict(), os.path.join(save_path, "audio_proj.pth"))

                        logger.info(f"Saved state to {save_path}")
                    # sample images!
                    if (
                        (global_step % args.validation_steps == 0)
                        or (global_step == 1)
                    ):
                        logger.info(
                            f"Running validation... \n Generating videos."
                        )
                        # The models need unwrapping because for compatibility in distributed training mode.
                        pipeline = MoCoTalkTrainPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(net).denoising_unet,
                            image_encoder=accelerator.unwrap_model(
                                image_encoder),
                            vae=accelerator.unwrap_model(vae),
                            head_embedder=accelerator.unwrap_model(net).head_embedder,
                            light_embedder=accelerator.unwrap_model(net).light_embedder,
                            ref_embedder=accelerator.unwrap_model(net).ref_embedder,
                            audio_proj=accelerator.unwrap_model(net).audio_proj,
                            revision=args.revision,
                            torch_dtype=weight_dtype,
                        )
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)

                        # run inference
                        val_save_dir = os.path.join(
                            args.output_dir, "validation_images")

                        if not os.path.exists(val_save_dir):
                            os.makedirs(val_save_dir)

                        with torch.autocast(
                            str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
                        ):
                            pixel_pil = [Image.fromarray(np.uint8((pixel_values.permute(0, 1, 3, 4, 2).cpu().numpy()[0, i] + 1) * 127.5)) for i in range(args.num_frames)]
                            heads_pil = [Image.fromarray(np.uint8((pixel_head.permute(0, 1, 3, 4, 2).cpu().numpy()[0, i]) * 255)) for i in range(args.num_frames)]
                            lights_pil = [Image.fromarray(np.uint8((pixel_values_light.permute(0, 1, 3, 4, 2).cpu().numpy()[0, i]) * 255)) for i in range(args.num_frames)]
                            reference_pil = [Image.fromarray(np.uint8((pixel_ref_values.permute(0, 1, 3, 4, 2).cpu().numpy()[0, 0] + 1) * 127.5))]
                            mask_pil = [Image.fromarray(np.uint8(pixel_values_mask.permute(0, 1, 3, 4, 2).repeat(1, 1, 1, 1, 3).cpu().numpy()[0, i] * 255)).resize([args.width, args.height]) for i in range(args.num_frames)]
                            frames = pipeline(
                                reference_pil, image_head=pixel_head, image_ref=pixel_ref_values,
                                image_light=pixel_values_light, image_mask=pixel_ref_mask,
                                audio_emb=audio_emb,
                                num_frames=pixel_head.size(1),
                                tile_size=args.num_frames, tile_overlap=6,
                                height=args.height, width=args.width, fps=7,
                                noise_aug_strength=0., num_inference_steps=25,
                                generator=None, min_guidance_scale=2.5, 
                                max_guidance_scale=2.5, decode_chunk_size=8, output_type="pt", device="cuda"
                            ).frames.cpu()
                            video_frames = (frames.permute(0, 1, 3, 4, 2) * 255.0).to(torch.uint8).numpy()[0]

                            out_file = os.path.join(
                                val_save_dir,
                                f"step_{global_step}_val_img.mp4",
                            )
                            final = []
                            for i in range(pixel_head.size(1)):
                                img = video_frames[i]
                                head = np.array(heads_pil[i])
                                light = np.array(lights_pil[i])
                                tar = np.array(pixel_pil[i])
                                ref = np.array(reference_pil[0])
                                ref_mask = np.array(mask_pil[i])
                                final.append(np.concatenate([ref, ref_mask, head, light, img, tar], axis=1))
                            export_to_gif(final, out_file, 16)

                        del pipeline
                        torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = MoCoTalkTrainPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(net).denoising_unet,
            image_encoder=accelerator.unwrap_model(
                image_encoder),
            vae=accelerator.unwrap_model(vae),
            head_embedder=accelerator.unwrap_model(net).head_embedder,
            light_embedder=accelerator.unwrap_model(net).light_embedder,
            ref_embedder=accelerator.unwrap_model(net).ref_embedder,
            audio_proj=accelerator.unwrap_model(net).audio_proj,
            revision=args.revision,
            torch_dtype=weight_dtype,
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
