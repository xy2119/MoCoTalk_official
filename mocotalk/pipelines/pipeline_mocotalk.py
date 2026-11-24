# -*- coding: utf-8 -*-
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union, Tuple

import PIL.Image
import einops
import numpy as np
import torch

from diffusers.image_processor import VaeImageProcessor, PipelineImageInput
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing, _append_dims
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from mocotalk.models.head_net import HeadNet
from mocotalk.models.light_net import LightNet
from mocotalk.models.ref_net import RefNet
from mocotalk.models.audio_adapter import AudioProjModel

logger = logging.get_logger(__name__)


# Copied from diffusers.pipelines.animatediff.pipeline_animatediff.tensor2vid
def tensor2vid(video: torch.Tensor, processor: "VaeImageProcessor", output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)
        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)
    elif output_type == "pt":
        outputs = torch.stack(outputs)
    elif output_type != "pil":
        raise ValueError(f"{output_type} does not exist. Choose one of ['np', 'pt', 'pil']")
    return outputs


@dataclass
class MoCoTalkPipelineOutput(BaseOutput):
    frames: Union[List[List[PIL.Image.Image]], np.ndarray, torch.Tensor]


class MoCoTalkPipeline(DiffusionPipeline):
    """
    SVD-based pipeline with:
      - head/light/ref latent branches
      - audio latent branch (AudioProjModel -> (B,F,320,64,64))
      - router-UNet (injects multi-conds by channel-wise gating)
      - standard CFG (cond vs uncond by duplicating batch and using zeros in uncond branch)
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModelWithProjection,
        unet,  # mocotalk.models.unet_router.UNetSpatioTemporalConditionModel
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
        head_embedder: HeadNet,
        light_embedder: LightNet,
        ref_embedder: RefNet,
        audio_proj: AudioProjModel,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            head_embedder=head_embedder,
            light_embedder=light_embedder,
            ref_embedder=ref_embedder,
            audio_proj=audio_proj,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        # internal
        self._guidance_scale = 1.0
        self._num_timesteps = 0

    # -------------------------
    # CFG control
    # -------------------------
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        gs = self.guidance_scale
        if isinstance(gs, (float, int)):
            return gs > 1.0
        return bool(gs.max() > 1.0)

    @property
    def num_timesteps(self):
        return self._num_timesteps

    # -------------------------
    # encoders
    # -------------------------
    def _encode_image(
        self,
        image: PipelineImageInput,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_cfg: bool,
    ):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

            image = self.feature_extractor(
                images=image,
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds  # (B,D)
        image_embeddings = image_embeddings.unsqueeze(1)           # (B,1,D)

        bs, seq_len, dim = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs * num_videos_per_prompt, seq_len, dim)

        if do_cfg:
            negative = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([negative, image_embeddings], dim=0)  # (2B,1,D)

        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_cfg: bool,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()  # (B,4,64,64)

        if do_cfg:
            negative = torch.zeros_like(image_latents)
            image_latents = torch.cat([negative, image_latents], dim=0)  # (2B,4,64,64)

        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)  # (2B*prompt,4,64,64) or (B*prompt,...)
        return image_latents

    def _encode_head_image(self, head_image: torch.Tensor, do_cfg: bool):
        head_latents = self.head_embedder(head_image)  # typically (B*F,320,64,64)
        if do_cfg:
            head_latents = torch.cat([torch.zeros_like(head_latents), head_latents], dim=0)
        return head_latents

    def _encode_light_image(self, image_light: torch.Tensor, do_cfg: bool):
        light_latents = self.light_embedder(image_light)
        if do_cfg:
            light_latents = torch.cat([torch.zeros_like(light_latents), light_latents], dim=0)
        return light_latents

    def _encode_ref_image(self, image_ref: torch.Tensor, do_cfg: bool):
        ref_latents = self.ref_embedder(image_ref)
        if do_cfg:
            ref_latents = torch.cat([torch.zeros_like(ref_latents), ref_latents], dim=0)
        return ref_latents

    def _encode_audio(
        self,
        audio_embeds: torch.Tensor,  # (B,F,W,L,C)
        height_lat: int,
        width_lat: int,
        num_videos_per_prompt: int,
        do_cfg: bool,
    ):
        if audio_embeds is None:
            return None

        device = audio_embeds.device
        dtype = next(self.audio_proj.parameters()).dtype
        audio_embeds = audio_embeds.to(device=device, dtype=dtype)

        audio_latents = self.audio_proj(audio_embeds, height=height_lat, width=width_lat)  # (B,F,320,64,64)

        # repeat prompt
        audio_latents = audio_latents.repeat_interleave(num_videos_per_prompt, dim=0)

        if do_cfg:
            audio_latents = torch.cat([torch.zeros_like(audio_latents), audio_latents], dim=0)  # (2B,F,...)

        return audio_latents

    def _get_add_time_ids(
        self,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int,
        do_cfg: bool,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"added_time_ids dim mismatch: expected {expected_add_embed_dim}, got {passed_add_embed_dim}. "
                f"Check unet.config.addition_time_embed_dim and add_embedding."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)  # (B,3)
        if do_cfg:
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)  # (2B,3)
        return add_time_ids

    # -------------------------
    # decoding
    # -------------------------
    def decode_latents_no_temporal(self, latents: torch.Tensor):
        num_frames = latents.shape[1]
        latents = latents.flatten(0, 1)  # (B*F,4,64,64)

        latents = latents / self.vae.config.scaling_factor
        frames = []
        for i in range(latents.shape[0]):
            frame = self.vae.decode(latents[i : i + 1]).sample
            frames.append(frame.cpu())
        frames = torch.cat(frames, dim=0)
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        return frames.float()

    def check_inputs(self, image, height, width):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError("`image` must be torch.Tensor or PIL.Image or list of PIL.Image.")
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 8, got {height}, {width}")

    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: Optional[torch.Generator],
        latents: Optional[torch.Tensor] = None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # -------------------------
    # main call
    # -------------------------
    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        image_head: torch.FloatTensor,      # (B,F,3,H,W) in [0,1] typically
        image_light: torch.FloatTensor,     # (B,F,3,H,W) in [0,1]
        image_mask: torch.FloatTensor,      # (B,F,1,64,64) in [0,1]
        image_ref: torch.FloatTensor,       # (B,F,3,H,W) in [-1,1] or [0,1] depending on your ref_embedder training; keep consistent with your inference script
        audio_emb: Optional[torch.FloatTensor] = None,  # (B,F,W,L,C)
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        tile_size: int = 16,
        tile_overlap: int = 4,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        image_only_indicator: bool = False,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: str = "pt",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        self.check_inputs(image, height, width)

        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]

        device = device if device is not None else self._execution_device

        # set guidance schedule (frame-wise)
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device)
        self._guidance_scale = guidance_scale  # store tensor (1,F)

        do_cfg = self.do_classifier_free_guidance

        # 1) encode CLIP image embedding
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, do_cfg)

        # NOTE: SVD uses fps-1 conditioning
        fps = fps - 1

        # 2) preprocess init image + noise aug
        image_in = self.image_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image_in.shape, generator=generator, device=device, dtype=image_in.dtype)
        image_in = image_in + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and getattr(self.vae.config, "force_upcast", False)
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        # 3) encode VAE latents for init image
        image_latents = self._encode_vae_image(image_in, device, num_videos_per_prompt, do_cfg)
        image_latents = image_latents.to(image_embeddings.dtype)
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)  # (B_eff,F,4,64,64)
        image_latents = image_latents * (1 - image_mask.to(image_latents.dtype))

        h_lat, w_lat = image_latents.shape[-2], image_latents.shape[-1]

        # 4) encode head/light/ref/audio
        head_latents = self._encode_head_image(image_head, do_cfg=do_cfg)
        light_latents = self._encode_light_image(image_light, do_cfg=do_cfg)
        ref_latents = self._encode_ref_image(image_ref, do_cfg=do_cfg)

        # reshape to (B_eff,F,320,64,64)
        head_latents = einops.rearrange(head_latents, "(b f) c h w -> b f c h w", f=num_frames)
        light_latents = einops.rearrange(light_latents, "(b f) c h w -> b f c h w", f=num_frames)
        ref_latents = einops.rearrange(ref_latents, "(b f) c h w -> b f c h w", f=num_frames)

        audio_latents = None
        if audio_emb is not None:
            audio_latents = self._encode_audio(
                audio_embeds=audio_emb,
                height_lat=h_lat,
                width_lat=w_lat,
                num_videos_per_prompt=num_videos_per_prompt,
                do_cfg=do_cfg,
            )

        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # 5) added time ids
        added_time_ids = self._get_add_time_ids(
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            dtype=image_embeddings.dtype,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            do_cfg=do_cfg,
        ).to(device)

        # 6) timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None)

        # 7) init noise latents
        # IMPORTANT: base batch for latents is (B * num_videos_per_prompt), CFG doubling is done by cat later
        latents = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            num_frames=tile_size,
            num_channels_latents=self.unet.config.in_channels,
            height=height,
            width=width,
            dtype=image_embeddings.dtype,
            device=device,
            generator=generator if isinstance(generator, torch.Generator) else None,
            latents=latents,
        )
        latents = latents.repeat(1, num_frames // tile_size + 1, 1, 1, 1)[:, :num_frames]

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, 0.0)

        # guidance per-frame -> (B, F, 1, 1, 1)
        gs = self.guidance_scale.to(device, latents.dtype).repeat(batch_size * num_videos_per_prompt, 1)
        gs = _append_dims(gs, latents.ndim)  # (B, F, 1, 1, 1)

        # 8) tiling indices
        self._num_timesteps = len(timesteps)
        indices = [[0, *range(i + 1, min(i + tile_size, num_frames))] for i in range(0, num_frames - tile_size + 1, tile_size - tile_overlap)]
        if indices[-1][-1] < num_frames - 1:
            indices.append([0, *range(num_frames - tile_size + 1, num_frames)])

        # 9) denoise
        with self.progress_bar(total=len(timesteps) * len(indices)) as progress_bar:
            for step_idx, t in enumerate(timesteps):
                # CFG duplicates batch here
                latent_model_input = torch.cat([latents] * 2, dim=0) if do_cfg else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # concat SVD image latents along channel dim: (B_eff,F,4,H,W) -> input channels 8
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                # accumulate tiled predictions
                noise_pred = torch.zeros_like(image_latents)
                noise_pred_cnt = image_latents.new_zeros((num_frames,))

                weight = (torch.arange(tile_size, device=device) + 0.5) * 2.0 / tile_size
                weight = torch.minimum(weight, 2 - weight)

                for idx in indices:
                    _noise_pred = self.unet(
                        latent_model_input[:, idx],  # (B_eff, tile, 8, 64, 64)
                        t,
                        encoder_hidden_states=image_embeddings,
                        added_time_ids=added_time_ids,
                        head_latents=head_latents[:, idx].flatten(0, 1),
                        light_latents=light_latents[:, idx].flatten(0, 1),
                        ref_latents=ref_latents[:, idx].flatten(0, 1),
                        audio_latents=None if audio_latents is None else audio_latents[:, idx].flatten(0, 1),
                        image_only_indicator=image_only_indicator,
                        return_dict=False,
                    )[0]  # (B_eff, tile, 4, 64, 64)

                    noise_pred[:, idx] += _noise_pred * weight[:, None, None, None]
                    noise_pred_cnt[idx] += weight
                    progress_bar.update()

                noise_pred.div_(noise_pred_cnt[:, None, None, None])

                if do_cfg:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
                    noise_pred = noise_pred_uncond + gs * (noise_pred_cond - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, step_idx, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

        # 10) decode
        if output_type != "latent":
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents_no_temporal(latents)  # (B,3,F,H,W)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames
        return MoCoTalkPipelineOutput(frames=frames)