import json
import random
import os
from typing import List

import torch
import torch.nn.functional as F
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np



class TalkingHeadVideoDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        data_meta_path="data.json",
        audio_margin=2,          # window length W = 2 * audio_margin + 1
        audio_drop_rate=0.05,    # same role as VideoDataset.drop_rate
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio
        self.audio_margin = audio_margin
        self.audio_drop_rate = audio_drop_rate

        # expected fixed seq_len (= W)
        self.seq_len = 2 * self.audio_margin + 1  # e.g. audio_margin=2 -> seq_len=5

        # -------------------------------------------------
        # 1) Load meta
        # -------------------------------------------------
        with open(data_meta_path, "r") as f:
            raw_meta = json.load(f)

        # Use audio only if the meta entries contain audio_emb_path
        self.use_audio = len(raw_meta) > 0 and "audio_emb_path" in raw_meta[0]
        print(f"[TalkingHeadVideoDataset] use_audio = {self.use_audio}")

        self.vid_meta = []
        self.bad_indices = set()  # samples marked as bad at runtime

        # -------------------------------------------------
        # 2) Filter out invalid samples as far as possible at construction time
        # -------------------------------------------------
        for m in raw_meta:
            video_path = m.get("video_path", None)
            if video_path is None:
                continue

            if not os.path.isfile(video_path):
                print(f"[TalkingHeadVideoDataset] Skip (file not found): {video_path}")
                continue

            # Check that decord can open the video
            try:
                vr = VideoReader(video_path)
                video_length = len(vr)
            except Exception as e:
                print(f"[TalkingHeadVideoDataset] Skip (VideoReader error) {video_path}: {e}")
                continue

            if video_length == 0:
                print(f"[TalkingHeadVideoDataset] Skip (empty video): {video_path}")
                continue

            # If audio is used, run a basic check on it as well
            if self.use_audio:
                audio_path = m.get("audio_emb_path", None)
                if audio_path is None:
                    print(f"[TalkingHeadVideoDataset] Skip (no audio_emb_path) for {video_path}")
                    continue

                if not os.path.isfile(audio_path):
                    print(f"[TalkingHeadVideoDataset] Skip (audio file not found): {audio_path}")
                    continue

                # It must be long enough to hold one fixed-length window
                if video_length < self.seq_len:
                    print(
                        f"[TalkingHeadVideoDataset] Skip (video too short for audio window): "
                        f"{video_path}, length={video_length} < {self.seq_len}"
                    )
                    continue

            # Reaching this point means the sample is basically OK
            self.vid_meta.append(m)

        if len(self.vid_meta) == 0:
            raise RuntimeError(
                "[TalkingHeadVideoDataset] No valid samples after filtering. "
                "Please check your data_meta_path and files."
            )

        # -------------------------------------------------
        # 3) Image augmentation transforms
        # -------------------------------------------------
        self.pixel_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, list):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (F, C, H, W)
        else:
            ret_tensor = transform(images)  # (C, H, W)
        return ret_tensor

    def __getitem__(self, index):
        """
        Robust version:
        - retries in a while loop instead of recursing;
        - marks bad samples and avoids them afterwards;
        - only raises in the extreme case where every sample is bad.
        """
        if len(self.vid_meta) == 0:
            raise RuntimeError("[TalkingHeadVideoDataset] Empty dataset.")

        # The dataloader index is always in [0, len-1]; clamp it again to be safe
        index = int(index) % len(self.vid_meta)

        max_tries = min(50, len(self.vid_meta))  # maximum samples tried per __getitem__ call
        tries = 0

        # Indices already tried in this call, to avoid retrying the same one forever
        tried_indices = set()

        while True:
            # If the current index is already in the global bad_indices, pick another
            if index in self.bad_indices or index in tried_indices:
                # pick a random index that is not marked bad
                candidate_indices = [
                    i for i in range(len(self.vid_meta))
                    if i not in self.bad_indices and i not in tried_indices
                ]
                if not candidate_indices:
                    # Every sample is either bad or was just tried: the data is unusable
                    raise RuntimeError(
                        "[TalkingHeadVideoDataset] All samples are corrupted or failed to read. "
                        "Please check your dataset."
                    )
                index = random.choice(candidate_indices)

            video_meta = self.vid_meta[index]
            video_path = video_meta["video_path"]
            kps_path = video_meta["kps_path"]
            light_path = video_meta["light_path"]
            mask_path = video_meta["mask_path"]

            tries += 1
            tried_indices.add(index)

            try:
                # ------------------------------------------------------
                # 1. Read the 4 video streams (RGB / kps / light / mask)
                # ------------------------------------------------------
                video_reader = VideoReader(video_path)
                kps_reader = VideoReader(kps_path)
                light_reader = VideoReader(light_path)
                mask_reader = VideoReader(mask_path)

                len_v = len(video_reader)
                len_k = len(kps_reader)
                len_l = len(light_reader)
                len_m = len(mask_reader)

                lengths = [len_v, len_k, len_l, len_m]
                video_length = min(lengths)

                if video_length == 0:
                    raise RuntimeError(f"Empty video or cond video in {video_path}")

                if len(set(lengths)) != 1:
                    print(
                        f"[TalkingHeadVideoDataset] length mismatch in {video_path}: "
                        f"video={len_v}, kps={len_k}, light={len_l}, mask={len_m}. "
                        f"Truncating to min_length={video_length}."
                    )

                # ------------------------------------------------------
                # 2. Read / align the audio embedding (if any)
                # ------------------------------------------------------
                full_audio_emb = None
                if self.use_audio and "audio_emb_path" in video_meta:
                    audio_emb_path = video_meta["audio_emb_path"]
                    full_audio_emb = torch.load(audio_emb_path, map_location="cpu")  # (T_audio, L, C)

                    T_audio = full_audio_emb.shape[0]
                    # Common length of all modalities: RGB/kps/light/mask/audio
                    T_common = min(video_length, T_audio)

                    if T_common < self.seq_len:
                        raise RuntimeError(
                            f"[TalkingHeadVideoDataset] too short after alignment: "
                            f"T_common={T_common} < seq_len={self.seq_len} in {video_path}"
                        )

                    if (T_common != video_length) or (T_common != T_audio):
                        print(
                            f"[TalkingHeadVideoDataset] audio length mismatch in {video_path}: "
                            f"video_len={video_length}, audio_len={T_audio}. "
                            f"Truncating to T_common={T_common}."
                        )

                    video_length = T_common
                    full_audio_emb = full_audio_emb[:video_length]  # align to the common length

                # ------------------------------------------------------
                # 3. Sample the centre frames batch_index from video_length
                # ------------------------------------------------------
                if self.use_audio and full_audio_emb is not None:
                    effective_margin = self.audio_margin
                else:
                    effective_margin = 0

                min_required_length = 2 * effective_margin + 1
                if video_length < min_required_length:
                    raise RuntimeError(
                        f"[TalkingHeadVideoDataset] video too short even after alignment: "
                        f"video_length={video_length}, required={min_required_length}, "
                        f"path={video_path}"
                    )

                min_center = effective_margin
                max_center = video_length - 1 - effective_margin
                if max_center < min_center:
                    raise RuntimeError(
                        f"[TalkingHeadVideoDataset] invalid center range: "
                        f"min_center={min_center}, max_center={max_center}, "
                        f"video_length={video_length}, margin={effective_margin}"
                    )

                center_span = max_center - min_center

                if self.n_sample_frames == 1:
                    batch_index = [random.randint(min_center, max_center)]
                else:
                    desired_total_span = self.sample_rate * (self.n_sample_frames - 1)
                    if center_span >= desired_total_span:
                        start_min = min_center
                        start_max = max_center - desired_total_span
                        start_idx = random.randint(start_min, start_max)
                        batch_index = [
                            start_idx + i * self.sample_rate
                            for i in range(self.n_sample_frames)
                        ]
                    else:
                        # Not enough centre positions for sample_rate; fall back to evenly spaced sampling
                        batch_index = np.linspace(
                            min_center,
                            max_center,
                            self.n_sample_frames,
                            dtype=int,
                        ).tolist()

                # ------------------------------------------------------
                # 4. Reference frame (can still be any frame in [0, video_length-1])
                # ------------------------------------------------------
                ref_img_idx = random.randint(0, video_length - 1)
                ref_img = Image.fromarray(video_reader[ref_img_idx].asnumpy())
                ref_pose = Image.fromarray(kps_reader[ref_img_idx].asnumpy())
                ref_light = Image.fromarray(light_reader[ref_img_idx].asnumpy())
                ref_mask = Image.fromarray(mask_reader[ref_img_idx].asnumpy())

                # ------------------------------------------------------
                # 5. Gather frames by batch_index
                # ------------------------------------------------------
                vid_pil_image_list = [ref_img]
                pose_pil_image_list = [ref_pose]
                light_pil_image_list = [ref_light]
                mask_pil_image_list = [ref_mask]

                for idx_frame in batch_index[1:]:
                    img = video_reader[idx_frame]
                    vid_pil_image_list.append(Image.fromarray(img.asnumpy()))

                    img = kps_reader[idx_frame]
                    pose_pil_image_list.append(Image.fromarray(img.asnumpy()))

                    img = light_reader[idx_frame]
                    light_pil_image_list.append(Image.fromarray(img.asnumpy()))

                    img = mask_reader[idx_frame]
                    mask_pil_image_list.append(Image.fromarray(img.asnumpy()))

                # ------------------------------------------------------
                # 6. Image transforms
                # ------------------------------------------------------
                state = torch.get_rng_state()

                pixel_values_vid = self.augmentation(
                    vid_pil_image_list, self.pixel_transform, state
                )  # (F, 3, H, W)

                pixel_values_pose = self.augmentation(
                    pose_pil_image_list, self.cond_transform, state
                )  # (F, 3, H, W)

                pixel_values_light = self.augmentation(
                    light_pil_image_list, self.cond_transform, state
                )  # (F, 3, H, W)

                pixel_values_mask = self.augmentation(
                    mask_pil_image_list, self.cond_transform, state
                )  # (F, 3, H, W)

                pixel_values_mask[pixel_values_mask > 0.5] = 1.0
                pixel_values_mask[pixel_values_mask <= 0.5] = 0.0
                pixel_values_mask = F.interpolate(
                    pixel_values_mask, [self.height // 8, self.width // 8]
                )[:, 0:1]  # (F, 1, H/8, W/8)

                pixel_values_ref_img = self.augmentation(
                    ref_img, self.pixel_transform, state
                ).unsqueeze(0)  # (1, 3, H, W)

                pixel_values_ref_mask = self.augmentation(
                    ref_mask, self.cond_transform, state
                )
                pixel_values_ref_mask[pixel_values_ref_mask > 0.5] = 1.0
                pixel_values_ref_mask[pixel_values_ref_mask <= 0.5] = 0.0
                pixel_values_ref_mask = pixel_values_ref_mask.unsqueeze(0)
                pixel_values_ref_mask = F.interpolate(
                    pixel_values_ref_mask, [self.height // 8, self.width // 8]
                )[:, 0:1]  # (1, 1, H/8, W/8)

                # ------------------------------------------------------
                # 7. Audio: build (F, W, L, C)
                # ------------------------------------------------------
                audio_embeds = None
                if self.use_audio and (full_audio_emb is not None):
                    # full_audio_emb: (T, L, C), where T = video_length here
                    assert (
                        full_audio_emb.shape[0] == video_length
                    ), f"Audio embedding shape {full_audio_emb.shape[0]} != video length {video_length}"

                    if effective_margin > 0:
                        W = 2 * effective_margin + 1
                        indices = torch.arange(
                            -effective_margin, effective_margin + 1
                        )  # (W,)

                        centers = torch.tensor(batch_index, dtype=torch.long).unsqueeze(1)  # (F,1)
                        center_indices = centers + indices.unsqueeze(0)  # (F, W)

                        assert center_indices.min() >= 0 and center_indices.max() < video_length, (
                            f"center_indices out of range: "
                            f"[{int(center_indices.min())}, {int(center_indices.max())}] vs T={video_length}, "
                            f"margin={effective_margin}, batch_index={batch_index}"
                        )

                        audio_embeds = full_audio_emb[center_indices]  # (F, W, L, C)
                    else:
                        # Without a margin this reduces to (F, 1, L, C)
                        centers = torch.tensor(batch_index, dtype=torch.long)
                        audio_embeds = full_audio_emb[centers]  # (F, L, C)
                        audio_embeds = audio_embeds.unsqueeze(1)  # (F, 1, L, C)

                    # Original drop logic
                    if torch.rand(1).item() < self.audio_drop_rate and effective_margin > 0:
                        for i in range(effective_margin):
                            audio_embeds[i, : effective_margin - i] = audio_embeds[
                                i, effective_margin - i
                            ]

                # ------------------------------------------------------
                # 8. Assemble and return the sample
                # ------------------------------------------------------
                sample = dict(
                    video_dir=video_path,
                    pixel_values_vid=pixel_values_vid,           # (F, 3, H, W)
                    pixel_values_head=pixel_values_pose,         # (F, 3, H, W)
                    pixel_values_light=pixel_values_light,       # (F, 3, H, W)
                    pixel_values_mask=pixel_values_mask,         # (F, 1, H/8, W/8)
                    pixel_values_ref_mask=pixel_values_ref_mask, # (1, 1, H/8, W/8)
                    pixel_values_ref_img=pixel_values_ref_img,   # (1, 3, H, W)
                )

                if audio_embeds is not None:
                    sample["audio_embeds"] = audio_embeds       # (F, W, L, C)

                # Reaching this point means everything is OK: this index is not a bad sample
                return sample

            except Exception as e:
                # Log the error, mark the sample as bad, then try another index
                print(
                    f"[TalkingHeadVideoDataset] error processing {video_path} "
                    f"(index={index}, try={tries}): {e}"
                )
                self.bad_indices.add(index)

                if tries >= max_tries:
                    # Too many attempts: usually too many bad samples, or the data has a wider problem
                    raise RuntimeError(
                        "[TalkingHeadVideoDataset] Too many failures in __getitem__. "
                        "Please check your dataset and meta."
                    )

                # Try again with a new random index
                index = random.randint(0, len(self.vid_meta) - 1)

    def __len__(self):
        return len(self.vid_meta)