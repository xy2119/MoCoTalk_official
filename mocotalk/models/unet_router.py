# mocotalk/models/unet_router.py

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.models.attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging

from diffusers.models.unets.unet_3d_blocks import get_down_block, get_up_block, UNetMidBlockSpatioTemporal

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# -------------------------
# NEW: lightweight Condition Router (channel-wise gating)
# -------------------------
class ConditionRouter(nn.Module):
    """
    Channel-wise router for multi-condition additive injection.

    Given:
        h: (BF, C, H, W)    backbone feature after conv_in
        emb: (BF, T)        timestep embedding (after time + added_time_ids fusion), repeated over frames already
        c_k: (BF, C, H, W)  condition latents (head/light/ref/audio), each optional

    Produces:
        fused = h + sum_k g_k * c_k
    where g_k is channel-wise:
        g_k: (BF, C, 1, 1)
    computed via:
        u = GAP(h)      -> (BF, C)
        t = MLP(emb)    -> (BF, C)
        s_k = GAP(c_k)  -> (BF, C)
        logits = MLP([u, t, s_h, s_l, s_r, s_a]) -> (BF, 4, C)
        g = softmax(logits, dim=1)
    """
    def __init__(
        self,
        channels: int,
        emb_dim: int,
        hidden_mult: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.channels = channels
        self.emb_dim = emb_dim
        self.eps = eps

        # project emb -> C so it can compete with u/summaries in same space
        self.emb_proj = nn.Sequential(
            nn.Linear(emb_dim, channels),
            nn.SiLU(),
        )

        in_dim = channels * 6  # [u, t, s_h, s_l, s_r, s_a]
        hidden_dim = channels * hidden_mult

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, channels * 4),  # 4 experts/conditions
        )

    @staticmethod
    def _gap(x: torch.Tensor) -> torch.Tensor:
        # (BF, C, H, W) -> (BF, C)
        return x.mean(dim=(-2, -1))

    def forward(
        self,
        h: torch.Tensor,                         # (BF, C, H, W)
        emb: torch.Tensor,                       # (BF, emb_dim)
        head_latents: Optional[torch.Tensor] = None,
        light_latents: Optional[torch.Tensor] = None,
        ref_latents: Optional[torch.Tensor] = None,
        audio_latents: Optional[torch.Tensor] = None,
        return_gates: bool = False,
    ):
        BF, C, H, W = h.shape
        if C != self.channels:
            raise ValueError(f"[ConditionRouter] channels mismatch: got {C}, expected {self.channels}")

        u = self._gap(h)                          # (BF, C)
        t = self.emb_proj(emb)                    # (BF, C)

        # If a condition is None, use zeros for summary and mask its logits to -inf so softmax -> 0.
        device = h.device
        dtype = u.dtype

        def _summary_and_mask(cond: Optional[torch.Tensor]):
            if cond is None:
                return torch.zeros((BF, C), device=device, dtype=dtype), torch.zeros((BF, 1, 1), device=device, dtype=torch.bool)
            if cond.dim() != 4:
                raise ValueError(f"[ConditionRouter] cond must be 4D (BF,C,H,W), got {cond.shape}")
            if cond.shape[0] != BF or cond.shape[1] != C:
                raise ValueError(f"[ConditionRouter] cond shape mismatch: {cond.shape} vs BF={BF}, C={C}")
            return self._gap(cond), torch.ones((BF, 1, 1), device=device, dtype=torch.bool)

        s_h, m_h = _summary_and_mask(head_latents)
        s_l, m_l = _summary_and_mask(light_latents)
        s_r, m_r = _summary_and_mask(ref_latents)
        s_a, m_a = _summary_and_mask(audio_latents)

        # concat: (BF, 6C)
        feats = torch.cat([u, t, s_h, s_l, s_r, s_a], dim=1)

        logits = self.mlp(feats)                  # (BF, 4C)
        logits = logits.view(BF, 4, C)            # (BF, 4, C)

        # availability mask per condition
        avail = torch.cat([m_h, m_l, m_r, m_a], dim=1)  # (BF, 4, 1)
        # set logits to very negative where not available
        # Use -1e4 (safe for fp16) instead of -inf to avoid NaNs in some kernels.
        logits = torch.where(avail, logits, torch.full_like(logits, -1e4))

        gates = torch.softmax(logits, dim=1)      # (BF, 4, C)
        # channel-wise gate: (BF, C, 1, 1)
        g_h = gates[:, 0].view(BF, C, 1, 1)
        g_l = gates[:, 1].view(BF, C, 1, 1)
        g_r = gates[:, 2].view(BF, C, 1, 1)
        g_a = gates[:, 3].view(BF, C, 1, 1)

        fused = h
        if head_latents is not None:
            fused = fused + g_h * head_latents
        if light_latents is not None:
            fused = fused + g_l * light_latents
        if ref_latents is not None:
            fused = fused + g_r * ref_latents
        if audio_latents is not None:
            fused = fused + g_a * audio_latents

        if return_gates:
            # return (BF, 4, C) for logging/analysis
            return fused, gates
        return fused


@dataclass
class UNetSpatioTemporalConditionOutput(BaseOutput):
    """
    The output of [`UNetSpatioTemporalConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """
    sample: torch.FloatTensor = None


class UNetSpatioTemporalConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 8,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        up_block_types: Tuple[str] = (
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 768,
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = (5, 10, 10, 20),
        num_frames: int = 25,
    ):
        super().__init__()
        self.sample_size = sample_size

        # checks
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. "
                f"`down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. "
                f"`block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )
        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. "
                f"`num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )
        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. "
                f"`cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )
        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. "
                f"`layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # time embedding
        time_embed_dim = block_out_channels[0] * 4  # 1280 when base is 320
        self.time_proj = Timesteps(block_out_channels[0], True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)

        self.add_time_proj = Timesteps(addition_time_embed_dim, True, downscale_freq_shift=0)
        self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        # NEW: router (operates right after conv_in)
        self.condition_router = ConditionRouter(
            channels=block_out_channels[0],
            emb_dim=time_embed_dim,
            hidden_mult=2,
        )

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)
        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)
        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)
        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                resnet_act_fn="silu",
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlockSpatioTemporal(
            block_out_channels[-1],
            temb_channels=blocks_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            cross_attention_dim=cross_attention_dim[-1],
            num_attention_heads=num_attention_heads[-1],
        )

        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=1e-5,
                resolution_idx=i,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                resnet_act_fn="silu",
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-5)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)
        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        count = len(self.attn_processors.keys())
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the "
                f"number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` "
                f"when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )
        self.set_attn_processor(processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)
            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def forward(
        self,
        sample: torch.FloatTensor,                      # (B, F, C, H, W)
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,            # (B, 1, D)
        added_time_ids: torch.Tensor,                   # (B, 3) typically
        head_latents: torch.Tensor = None,              # (BF, C, H, W) or (B,F,C,H,W)
        light_latents: torch.Tensor = None,
        ref_latents: torch.Tensor = None,
        audio_latents: Optional[torch.Tensor] = None,
        image_only_indicator: bool = False,
        return_dict: bool = True,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:

        # ---- time ----
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)  # (B, time_embed_dim)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1)).to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb  # (B, time_embed_dim)

        # ---- flatten conditions to (BF, C, H, W) ----
        def _maybe_flatten_cond(cond: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if cond is None:
                return None
            if cond.dim() == 5:
                b, f = cond.shape[:2]
                if b != batch_size or f != num_frames:
                    raise ValueError(f"cond shape {cond.shape} mismatches sample {sample.shape}")
                return cond.flatten(0, 1)
            if cond.dim() == 4:
                if cond.shape[0] != batch_size * num_frames:
                    raise ValueError(f"cond BF dim={cond.shape[0]} expected {batch_size * num_frames}")
                return cond
            raise ValueError(f"cond must be 4D/5D, got {cond.shape}")

        head_latents = _maybe_flatten_cond(head_latents)
        light_latents = _maybe_flatten_cond(light_latents)
        ref_latents = _maybe_flatten_cond(ref_latents)
        audio_latents = _maybe_flatten_cond(audio_latents)

        # ---- flatten sample and repeat embeddings over frames ----
        sample = sample.flatten(0, 1)  # (BF, C_in, H, W)

        # (B, T) -> (BF, T)
        emb = emb.repeat_interleave(num_frames, dim=0)

        # (B, 1, D) -> (BF, 1, D)
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # ---- pre-process + NEW gated injection ----
        sample = self.conv_in(sample)  # (BF, 320, H, W)

        # router does: sample = sample + sum g_k * c_k
        sample = self.condition_router(
            sample,
            emb,
            head_latents=head_latents,
            light_latents=light_latents,
            ref_latents=ref_latents,
            audio_latents=audio_latents,
            return_gates=False,
        )

        image_only_indicator = (
            torch.ones(batch_size, num_frames, dtype=sample.dtype, device=sample.device)
            if image_only_indicator
            else torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)
        )

        # ---- down ----
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )
            down_block_res_samples += res_samples

        # ---- mid ----
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # ---- up ----
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )

        # ---- out ----
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # reshape back: (BF, C, H, W) -> (B, F, C, H, W)
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if not return_dict:
            return (sample,)
        return UNetSpatioTemporalConditionOutput(sample=sample)