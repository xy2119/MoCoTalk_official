# mocotalk/models/audio_adapter.py

import torch
from diffusers import ConfigMixin, ModelMixin
from einops import rearrange
from torch import nn


class AudioProjModel(ModelMixin, ConfigMixin):
    """
    Map the wav2vec audio_emb (B, F, W, L, C) to a condition feature map with the same
    resolution as the UNet noise latent. This is the fourth condition branch; like HeadNet/LightNet/RefNet,
    it is fused into the UNet by addition.

    Expected input:
        audio_embeds: (B, F, W, L, C)
            B: batch
            F: number of sampled video frames (n_sample_frames)
            W: temporal window length (2 * audio_margin + 1)
            L: number of wav2vec blocks used
            C: wav2vec hidden size

    Output:
        audio_latents: (B, F, C_lat, H_lat, W_lat)
            C_lat: matches the channel count of the shallowest UNet layer (usually 320)
            H_lat, W_lat: match the spatial size of the VAE latent / UNet input (e.g. 64x64)
    """

    def __init__(
        self,
        seq_len: int = 5,
        blocks: int = 12,
        channels: int = 768,
        intermediate_dim: int = 512,
        noise_latent_channels: int = 320,
    ):
        super().__init__()

        # Store the config so save_pretrained / from_pretrained work
        self.register_to_config(
            seq_len=seq_len,
            blocks=blocks,
            channels=channels,
            intermediate_dim=intermediate_dim,
            noise_latent_channels=noise_latent_channels,
        )

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.noise_latent_channels = noise_latent_channels

        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, noise_latent_channels)

        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(noise_latent_channels)

    def forward(
        self,
        audio_embeds: torch.Tensor,   # (B, F, W, L, C)
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Returns audio_latents: (B, F, C_lat, H_lat, W_lat)
        The audio condition is spatially uniform: a global vector is computed first, then broadcast to HxW.
        """

        if audio_embeds.ndim != 5:
            raise ValueError(
                f"audio_embeds should have shape (B, F, W, L, C), but got {audio_embeds.shape}"
            )

        B, F, W, L, C = audio_embeds.shape

        # (B, F, W, L, C) -> (B*F, W*L*C)
        audio_flat = rearrange(audio_embeds, "b f w l c -> (b f) (w l c)")
        if audio_flat.shape[1] != self.input_dim:
            raise ValueError(
                f"AudioProjModel is configured with input_dim={self.input_dim}, "
                f"but the flattened input has {audio_flat.shape[1]}; check that seq_len/blocks/channels match the data."
            )

        x = self.activation(self.proj1(audio_flat))
        x = self.activation(self.proj2(x))
        x = self.proj3(x)                     # (B*F, C_lat)
        x = self.norm(x)

        # (B*F, C_lat) -> (B, F, C_lat, H, W): expand to 1x1 first, then broadcast
        x = x.view(B * F, self.noise_latent_channels, 1, 1)
        x = x.expand(B * F, self.noise_latent_channels, height, width)
        audio_latents = rearrange(x, "(b f) c h w -> b f c h w", b=B, f=F)

        return audio_latents
