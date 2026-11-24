#!/usr/bin/env python
# coding=utf-8

import os
import cv2
import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput

# ============== Baseline / existing modules ==============
from mocotalk.third_party.facepose.mp_utils import LMKExtractor
from mocotalk.third_party.facepose.draw_utils import FaceMeshVisualizer
from mocotalk.third_party.facepose.motion_utils import motion_sync
from mocotalk.third_party.facematting.u2net_matting import U2NET
from mocotalk.third_party.decalib.utils import util
from mocotalk.third_party.decalib.utils.tensor_cropper import transform_points
from mocotalk.third_party.decalib.deca import DECA
from mocotalk.third_party.decalib.utils.config import cfg as deca_cfg

# ============== SPECTRE ==============
import collections
from skimage.transform import estimate_transform, warp

from spectre.src.spectre import SPECTRE
from spectre.config import cfg as spectre_cfg
from spectre.datasets.data_utils import landmarks_interpolate
from spectre.external.Visual_Speech_Recognition_for_Multiple_Languages.tracker.face_tracker import FaceTracker
from spectre.external.Visual_Speech_Recognition_for_Multiple_Languages.tracker.utils import get_landmarks

# ============== Audio ==============
from einops import rearrange
import librosa


# ============================================================
#                         Utils
# ============================================================
def run_ffmpeg_extract(video_path: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        str(out_dir / "%05d.png"),
    ]
    subprocess.run(cmd, check=True)


def run_ffmpeg_encode(frames_dir: Path, save_path: str, fps: int = 20):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-r",
        str(fps),
        "-i",
        str(frames_dir / "%05d.png"),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(save_path),
    ]
    subprocess.run(cmd, check=True)


def center_square_resize(frame_rgb, target_size=512):
    h, w = frame_rgb.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    crop = frame_rgb[top: top + side, left: left + side]
    crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return crop


# ============================================================
#                    Baseline: U2NET matting
# ============================================================
class FaceMatting:
    def __init__(self, device="cuda") -> None:
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.net = U2NET(3, 1).to(self.device)
        self.net.load_state_dict(
            torch.load("./mocotalk/third_party/facematting/u2net_human_seg.pth", map_location=self.device)
        )
        self.net.eval()

    def portrait_matting(self, rgb_image):
        rgb_image = cv2.resize(rgb_image, (320, 320))[None] / 255.0
        rgb_image[:, :, 0] = (rgb_image[:, :, 0] - 0.485) / 0.229
        rgb_image[:, :, 1] = (rgb_image[:, :, 1] - 0.456) / 0.224
        rgb_image[:, :, 2] = (rgb_image[:, :, 2] - 0.406) / 0.225

        rgb_image_th = torch.tensor(rgb_image, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)

        with torch.no_grad():
            d1, d2, d3, d4, d5, d6, d7 = self.net(rgb_image_th)
            pred = d1[:, 0, :, :]
            ma = torch.max(pred)
            mi = torch.min(pred)
            alpha = (pred - mi) / (ma - mi + 1e-8)

        alpha = alpha.detach().cpu().numpy()[0]
        alpha[alpha > 0.5] = 255
        alpha[alpha <= 0.5] = 0
        alpha = np.dstack([alpha, alpha, alpha])
        alpha = cv2.resize(alpha, (512, 512))
        alpha = cv2.dilate(alpha, np.ones([7, 7], np.uint8))
        return alpha.astype(np.uint8)


# ============================================================
#                    DECA renderer
# ============================================================
class FaceImageRender:
    def __init__(self) -> None:
        self.deca = DECA(config=deca_cfg)

        f_mask = np.load(
            "./mocotalk/third_party/decalib/data/FLAME_masks_face-id.pkl",
            allow_pickle=True,
            encoding="latin1"
        )
        v_mask = np.load(
            "./mocotalk/third_party/decalib/data/FLAME_masks.pkl",
            allow_pickle=True,
            encoding="latin1"
        )
        self.mask = {
            "v_mask": v_mask["face"].tolist(),
            "f_mask": f_mask["face"].tolist()
        }

    def image_to_3dcoeff(self, rgb_image):
        with torch.no_grad():
            codedict, detected_flag = self.deca.img_to_3dcoeff(rgb_image)
        return codedict

    def render_shape(self, shape, exp, pose, cam, light, tform, h, w):
        with torch.no_grad():
            verts, landmarks2d, landmarks3d = self.deca.flame(
                shape_params=shape,
                expression_params=exp,
                pose_params=pose
            )

            trans_verts = util.batch_orth_proj(verts, cam)
            trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

            points_scale = [self.deca.image_size, self.deca.image_size]
            trans_verts = transform_points(trans_verts, tform, points_scale, [h, w])

            shape_images, _, grid, alpha_images, albedo_images = self.deca.render.render_shape(
                verts,
                trans_verts,
                h=h,
                w=w,
                lights=light,
                images=None,
                return_grid=True,
                mask=self.mask,
            )

            shape_images = (
                shape_images.permute(0, 2, 3, 1)
                .clamp(0, 1)
                .detach()
                .cpu()
                .numpy()[0]
                * 255
            )
            albedo_images = (
                albedo_images.permute(0, 2, 3, 1)
                .clamp(0, 1)
                .detach()
                .cpu()
                .numpy()[0]
                * 255
            )

        return shape_images.astype(np.uint8), albedo_images.astype(np.uint8)

    def render_shape_with_light(self, codedict, target_light=None):
        if target_light is None:
            target_light = codedict["light"]
        shape, exp, pose = codedict["shape"], codedict["exp"], codedict["pose"]
        cam, tform, h, w = codedict["cam"], codedict["tform"], codedict["height"], codedict["width"]
        shape_image, albedo_image = self.render_shape(shape, exp, pose, cam, target_light, tform, h, w)
        return shape_image

    def render_motion_single_with_light(self, image, target_light_image):
        codedict = self.image_to_3dcoeff(image)
        target_light = self.image_to_3dcoeff(target_light_image)["light"]
        shading = self.render_shape_with_light(codedict, target_light=target_light)
        return shading

    def render_motion_sync_relative(self, ref_image, driver_frames, target_light_image):
        """
        cross-reenact shading:
        Use the shape/identity/light of ref,
        driven by the relative pose/exp of the driver.
        """
        ref_codedict = self.image_to_3dcoeff(ref_image)
        target_light = self.image_to_3dcoeff(target_light_image)["light"]

        drv_codedict_list = []
        shading_frames = []

        for drv_frm in tqdm(driver_frames, desc="DECA encode driver"):
            drv_codedict = self.image_to_3dcoeff(drv_frm)
            drv_codedict_list.append(drv_codedict)

        if len(drv_codedict_list) == 0:
            return []

        best_pose = drv_codedict_list[0]["pose"]
        best_exp = drv_codedict_list[0]["exp"]

        for drv_codedict in tqdm(drv_codedict_list, desc="DECA render relative"):
            relative_pose = drv_codedict["pose"] - best_pose + ref_codedict["pose"]
            relative_exp = drv_codedict["exp"] - best_exp + ref_codedict["exp"]

            shape = ref_codedict["shape"]
            exp = relative_exp
            pose = relative_pose
            cam = ref_codedict["cam"]
            tform = ref_codedict["tform"]
            h = ref_codedict["height"]
            w = ref_codedict["width"]

            shape_image, albedo_image = self.render_shape(
                shape, exp, pose, cam, target_light, tform, h, w
            )
            shading_frames.append(shape_image)

        return shading_frames


# ============================================================
#                    Face KP Detector
# ============================================================
class FaceKPDetector:
    def __init__(self) -> None:
        self.vis = FaceMeshVisualizer(
            draw_iris=False,
            draw_mouse=True,
            draw_eye=True,
            draw_nose=True,
            draw_eyebrow=True,
            draw_pupil=True,
        )
        self.lmk_extractor = LMKExtractor()

    def motion_sync(self, ref_image, driver_frames):
        """
        Start from the ref face as the initial state
        and align the driver motion to ref.
        """
        try:
            ref_image = cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR)
            ref_frame = cv2.resize(ref_image, (512, 512))
            ref_det = self.lmk_extractor(ref_frame)
            if ref_det is None:
                return None

            sequence_driver_det = []
            for frame in tqdm(driver_frames, desc="KP detect + motion sync"):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (512, 512))
                result = self.lmk_extractor(frame)
                if result is None:
                    return None
                sequence_driver_det.append(result)

            sequence_det_ms = motion_sync(sequence_driver_det, ref_det)
            pose_frames = [
                self.vis.draw_landmarks((512, 512), i, normed=False)
                for i in sequence_det_ms
            ]
            return pose_frames
        except Exception:
            return None

    def single_kp(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (512, 512))
        frame_det = self.lmk_extractor(image)
        if frame_det is None or "lmks" not in frame_det:
            return None
        kpmap = self.vis.draw_landmarks((512, 512), frame_det["lmks"], normed=True)
        return kpmap


# ============================================================
#                    SPECTRE helpers
# ============================================================
def crop_face_spectre(frame, landmarks, scale=1.6, image_size=224):
    h, w = frame.shape[:2]

    lms = np.asarray(landmarks).copy()
    if lms.ndim == 3:
        lms = lms[0]

    valid_mask = np.isfinite(lms).all(axis=1)
    lms = lms[valid_mask]

    if lms.shape[0] == 0:
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        src_pts = np.array([
            [left, top],
            [left, top + side - 1],
            [left + side - 1, top],
        ])
    else:
        xs = np.clip(lms[:, 0], 0, w - 1)
        ys = np.clip(lms[:, 1], 0, h - 1)

        left = xs.min()
        right = xs.max()
        top = ys.min()
        bottom = ys.max()

        old_size = (right - left + bottom - top) / 2.0
        old_size = max(old_size, 1.0)

        center = np.array([
            (left + right) / 2.0,
            (top + bottom) / 2.0,
        ])

        size = float(old_size * scale)
        max_side = max(h, w)
        size = np.clip(size, 32.0, max_side * 1.2)
        half = size / 2.0

        cx = np.clip(center[0], half, w - half)
        cy = np.clip(center[1], half, h - half)

        src_pts = np.array([
            [cx - half, cy - half],
            [cx - half, cy + half - 1],
            [cx + half - 1, cy - half],
        ])

    dst_pts = np.array([
        [0, 0],
        [0, image_size - 1],
        [image_size - 1, 0],
    ])

    tform = estimate_transform("similarity", src_pts, dst_pts)
    return tform


def extract_frames_with_landmarks(video_path, face_tracker, detect_landmarks=True):
    videofolder = os.path.splitext(video_path)[0] + "_frames_spectre_cross"
    os.makedirs(videofolder, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    imagepath_list = []
    face_info = collections.defaultdict(list)

    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=frame_count, desc=f"[SPECTRE] Extract {os.path.basename(video_path)}") as pbar:
        idx = 0
        while True:
            success, image = vidcap.read()
            if not success:
                break

            if detect_landmarks:
                detected_faces = face_tracker.face_detector(image, rgb=False)
                lms, scores = face_tracker.landmark_detector(image, detected_faces, rgb=False)
                face_info["bbox"].append(detected_faces)
                face_info["landmarks"].append(lms)
                face_info["landmarks_scores"].append(scores)

            imagepath = os.path.join(videofolder, f"{idx:06d}.jpg")
            cv2.imwrite(imagepath, image)
            imagepath_list.append(imagepath)

            idx += 1
            pbar.update(1)

    landmarks = None
    if detect_landmarks:
        landmarks = get_landmarks(face_info)
        landmarks = landmarks_interpolate(landmarks)
        if landmarks is None:
            print(f"[WARN] No faces detected in {video_path}")

    return imagepath_list, landmarks, videofolder


def extract_spectre_motion(video_path, spectre, device, face_tracker):
    image_paths, landmarks, videofolder = extract_frames_with_landmarks(
        video_path, face_tracker, detect_landmarks=True
    )
    if landmarks is None:
        raise RuntimeError(f"No landmarks detected in {video_path}")

    original_len = len(image_paths)

    image_paths.insert(0, image_paths[0])
    image_paths.insert(0, image_paths[0])
    image_paths.append(image_paths[-1])
    image_paths.append(image_paths[-1])

    landmarks.insert(0, landmarks[0])
    landmarks.insert(0, landmarks[0])
    landmarks.append(landmarks[-1])
    landmarks.append(landmarks[-1])

    image_paths = np.array(image_paths)
    landmarks = np.array(landmarks)

    L = 50
    indices = list(range(len(image_paths)))
    overlapping_indices = [indices[i: i + L] for i in range(0, len(indices), L - 4)]
    if len(overlapping_indices[-1]) < 5:
        overlapping_indices[-2] = overlapping_indices[-2] + overlapping_indices[-1]
        overlapping_indices[-2] = np.unique(overlapping_indices[-2]).tolist()
        overlapping_indices = overlapping_indices[:-1]

    all_exps = []
    all_poses = []

    with torch.no_grad():
        for chunk_id, idx_chunk in enumerate(overlapping_indices):
            print(f"[SPECTRE] {os.path.basename(video_path)} frames {idx_chunk[0]}-{idx_chunk[-1]}")
            image_paths_chunk = image_paths[idx_chunk]
            landmarks_chunk = landmarks[idx_chunk]

            images_list = []
            for j, img_path in enumerate(image_paths_chunk):
                frame_bgr = cv2.imread(img_path)
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                kpt = landmarks_chunk[j]

                tform = crop_face_spectre(frame, kpt, scale=1.6)
                cropped = warp(frame, tform.inverse, output_shape=(224, 224))
                images_list.append(cropped.transpose(2, 0, 1))

            images_array = torch.from_numpy(np.array(images_list)).float().to(device)

            codedict, deca_exp, deca_jaw = spectre.encode(images_array)
            exp_final = codedict["exp"] + deca_exp
            pose_final = codedict["pose"].clone()
            pose_final[..., 3:] = pose_final[..., 3:] + deca_jaw

            def trim(x):
                if chunk_id == 0 and chunk_id == len(overlapping_indices) - 1:
                    return x
                elif chunk_id == 0:
                    return x[:-2]
                elif chunk_id == len(overlapping_indices) - 1:
                    return x[2:]
                else:
                    return x[2:-2]

            exp_final = trim(exp_final)
            pose_final = trim(pose_final)

            all_exps.append(exp_final.cpu())
            all_poses.append(pose_final.cpu())

    exp_seq = torch.cat(all_exps, dim=0)
    pose_seq = torch.cat(all_poses, dim=0)

    exp_seq = exp_seq[2:-2]
    pose_seq = pose_seq[2:-2]

    assert exp_seq.shape[0] == original_len, f"Length mismatch, got {exp_seq.shape[0]} vs {original_len}"

    return exp_seq, pose_seq


class SpectreDecaCrossRetargetor:
    """
    cross-reenact shading:
    - identity (shape/cam/tform/light): from the ref image
    - motion / pose / expression: from the current driving video
    """
    def __init__(self, deca_renderer, device="cuda", exp_scale=1.0):
        self.deca_renderer = deca_renderer
        self.device = device
        self.exp_scale = exp_scale

        spectre_cfg.pretrained_modelpath = "spectre/pretrained/spectre_model.tar"
        spectre_cfg.model.use_tex = False
        self.spectre = SPECTRE(spectre_cfg, device)
        self.spectre.eval()

        self.face_tracker = FaceTracker()

    def fuse_cross(self, feature_path, driving_video_path, light_path):
        device = self.device
        fir = self.deca_renderer

        feature_bgr = cv2.imread(feature_path)
        if feature_bgr is None:
            raise RuntimeError(f"Cannot read feature image: {feature_path}")
        feature_rgb = cv2.cvtColor(feature_bgr, cv2.COLOR_BGR2RGB)
        feature_rgb = cv2.resize(feature_rgb, (512, 512))

        ref_codedict = fir.image_to_3dcoeff(feature_rgb)
        shape_ref = ref_codedict["shape"]
        cam_ref = ref_codedict["cam"]
        tform_ref = ref_codedict["tform"]
        h_ref = ref_codedict["height"]
        w_ref = ref_codedict["width"]

        light_bgr = cv2.imread(light_path)
        if light_bgr is None:
            raise RuntimeError(f"Cannot read lighting image: {light_path}")
        light_rgb = cv2.cvtColor(light_bgr, cv2.COLOR_BGR2RGB)
        light_rgb = cv2.resize(light_rgb, (512, 512))
        light_codedict = fir.image_to_3dcoeff(light_rgb)
        light_target = light_codedict["light"]

        drv_exp_seq, drv_pose_seq = extract_spectre_motion(
            driving_video_path, self.spectre, device, self.face_tracker
        )

        if self.exp_scale != 1.0:
            drv_exp_seq = drv_exp_seq * self.exp_scale

        if drv_exp_seq.shape[0] == 0:
            return []

        best_pose = drv_pose_seq[0:1].to(device)
        best_exp = drv_exp_seq[0:1].to(device)
        ref_pose = ref_codedict["pose"].to(device)
        ref_exp = ref_codedict["exp"].to(device)

        shading_frames = []
        shape_ref = shape_ref.to(device)
        cam_ref = cam_ref.to(device)
        light_target = light_target.to(device)

        print("[SPECTRE+DECA] rendering cross shading frames ...")
        for t in tqdm(range(drv_exp_seq.shape[0]), desc="Render shading (cross SPECTRE+DECA)"):
            exp_t = drv_exp_seq[t:t + 1].to(device)
            pose_t = drv_pose_seq[t:t + 1].to(device)

            relative_pose = pose_t - best_pose + ref_pose
            relative_exp = exp_t - best_exp + ref_exp

            shape_img, _ = fir.render_shape(
                shape=shape_ref,
                exp=relative_exp,
                pose=relative_pose,
                cam=cam_ref,
                light=light_target,
                tform=tform_ref,
                h=h_ref,
                w=w_ref
            )
            shape_img = shape_img.astype(np.uint8)
            gray = cv2.cvtColor(shape_img, cv2.COLOR_RGB2GRAY)
            gray3 = np.stack([gray, gray, gray], axis=-1)
            shading_frames.append(gray3)

        return shading_frames


# ============================================================
#                    Audio model
# ============================================================
class Wav2VecModel(Wav2Vec2Model):
    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if output_attentions is None:
            output_attentions = False
        if output_hidden_states is None:
            output_hidden_states = True
        if return_dict is None:
            return_dict = True

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states,
            mask_time_indices=mask_time_indices,
            attention_mask=attention_mask,
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def linear_interpolation(features, seq_len):
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode="linear")
    return output_features.transpose(1, 2)


class AudioProcessor:
    """
    Read the audio of the current driving video.
    Prefer offline_vocals_root/<stem>-vocals.wav,
    otherwise fall back to the video's original audio track.
    Returns (T, L, C)
    """
    def __init__(
        self,
        wav2vec_model_name="facebook/wav2vec2-base-960h",
        sample_rate=16000,
        device="cuda",
        offline_vocals_root=None,
    ):
        self.device = device
        self.sample_rate = sample_rate

        self.audio_encoder = Wav2VecModel.from_pretrained(
            wav2vec_model_name,
            local_files_only=True
        ).to(device)
        self.audio_encoder.feature_extractor._freeze_parameters()

        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_model_name,
            local_files_only=True
        )

        if offline_vocals_root is not None:
            self.offline_vocals_root = Path(offline_vocals_root)
            self.offline_vocals_root.mkdir(parents=True, exist_ok=True)
            print(f"[AUDIO] offline_vocals_root = {self.offline_vocals_root}")
        else:
            self.offline_vocals_root = None

    def _extract_wave(self, video_path: str):
        video_path = str(video_path)
        stem = Path(video_path).stem

        if self.offline_vocals_root is not None:
            cand = self.offline_vocals_root / f"{stem}-vocals.wav"
            if cand.exists():
                print(f"[AUDIO] use pre-separated vocals: {cand}")
                y, sr = librosa.load(cand.as_posix(), sr=None)
                speech_array = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
                return speech_array
            else:
                print(f"[WARN] pre-separated vocals not found: {cand}, fallback to raw audio from video.")

        speech_array, _ = librosa.load(video_path, sr=self.sample_rate)
        return speech_array

    def get_audio_embedding_from_video(self, video_path: str, video_length: int):
        speech_array = self._extract_wave(video_path)

        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(
                speech_array, sampling_rate=self.sample_rate
            ).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)
        audio_feature = audio_feature.unsqueeze(0)

        with torch.no_grad():
            outputs = self.audio_encoder(
                audio_feature,
                seq_len=video_length,
                output_hidden_states=True
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) <= 1:
            raise RuntimeError("Fail to get hidden_states from wav2vec")

        audio_emb = torch.stack(hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "l t d -> t l d")

        return audio_emb.cpu().detach()


# ============================================================
#              Cross preprocess with audio
# ============================================================
class PreprocessCrossWithAudio:
    """
    Cross-reenactment preprocessing:
    - source/light: first frame of the next video
    - video_path: the current driving video
    - kp: motion_sync(ref, driver_frames)
    - shading:
        * use_spectre=0 -> DECA relative
        * use_spectre=1 -> SPECTRE+DECA cross relative
    - audio: the audio of the current video_path
    """
    def __init__(
        self,
        use_spectre=False,
        spectre_exp_scale=1.0,
        device="cuda",
        use_audio=True,
        audio_processor: AudioProcessor = None,
        fps: int = 20,
    ):
        self.fm = FaceMatting(device=device)
        self.fir = FaceImageRender()
        self.fkpd = FaceKPDetector()

        self.use_spectre = bool(use_spectre)
        self.use_audio = bool(use_audio)
        self.audio_processor = audio_processor
        self.fps = fps

        if self.use_spectre:
            print("[INIT] Use SPECTRE+DECA for cross shading")
            self.spectre_retargetor = SpectreDecaCrossRetargetor(
                deca_renderer=self.fir,
                device=device,
                exp_scale=spectre_exp_scale
            )
        else:
            self.spectre_retargetor = None

        if self.use_audio and self.audio_processor is None:
            raise ValueError("use_audio=True but audio_processor is None")

    def run(
        self,
        source_path: str,
        light_path: str,
        video_path: str,
        save_shading_path: str,
        motion_align: str = "relative",
        audio_emb_path: str = None,
        spectre_feature_path: str = None,
        spectre_light_path: str = None,
        work_dir: str = None,
    ):
        if motion_align != "relative":
            raise ValueError("Cross-reenact only supports motion_align='relative'.")

        video_path = str(video_path)
        source_path = str(source_path)
        light_path = str(light_path) if light_path is not None else source_path

        if work_dir is None:
            work_dir = str(Path(save_shading_path).parent / "_tmp_cross_audio_preprocess")

        work_dir = Path(work_dir)
        raw_frames_dir = work_dir / "raw_frames"
        concat_tmp_dir = work_dir / "concat_frames"

        shutil.rmtree(work_dir, ignore_errors=True)
        raw_frames_dir.mkdir(parents=True, exist_ok=True)
        concat_tmp_dir.mkdir(parents=True, exist_ok=True)

        source_image = np.array(Image.open(source_path).convert("RGB").resize((512, 512)), dtype=np.uint8)
        target_lighting = np.array(Image.open(light_path).convert("RGB").resize((512, 512)), dtype=np.uint8)

        run_ffmpeg_extract(video_path, raw_frames_dir)

        frame_files = sorted(raw_frames_dir.glob("*.png"))
        if len(frame_files) == 0:
            raise RuntimeError("ffmpeg frame extraction failed or produced no frames")

        driver_frames = []
        for p in frame_files:
            bgr = cv2.imread(str(p))
            if bgr is None:
                raise RuntimeError(f"Cannot read frame: {p}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = center_square_resize(rgb, target_size=512)
            driver_frames.append(rgb)

        aligned_kpmaps = self.fkpd.motion_sync(source_image, driver_frames)
        if aligned_kpmaps is None or len(aligned_kpmaps) != len(driver_frames):
            raise RuntimeError("face detection failed in cross kp motion_sync")

        alpha = self.fm.portrait_matting(source_image)

        if self.use_spectre:
            feature_path = spectre_feature_path if spectre_feature_path is not None else source_path
            light_img_path = spectre_light_path if spectre_light_path is not None else light_path

            shadings = self.spectre_retargetor.fuse_cross(
                feature_path=feature_path,
                driving_video_path=video_path,
                light_path=light_img_path,
            )
            if shadings is None or len(shadings) == 0:
                raise RuntimeError("SPECTRE+DECA cross shading failed")

            T = min(len(driver_frames), len(aligned_kpmaps), len(shadings))
            driver_frames = driver_frames[:T]
            aligned_kpmaps = aligned_kpmaps[:T]
            shadings = shadings[:T]
        else:
            shadings = self.fir.render_motion_sync_relative(
                source_image,
                driver_frames,
                target_lighting
            )
            if shadings is None or len(shadings) != len(driver_frames):
                raise RuntimeError("DECA cross shading failed or length mismatch")

        num_frames = len(driver_frames)

        if self.use_audio:
            if audio_emb_path is None:
                save_root = Path(save_shading_path).parent
                stem = Path(save_shading_path).stem
                audio_emb_path = (save_root / f"{stem}.pt").as_posix()

            audio_emb = self.audio_processor.get_audio_embedding_from_video(
                video_path, video_length=num_frames
            )
            torch.save(audio_emb, audio_emb_path)
            print(f"[AUDIO] saved audio embedding to {audio_emb_path}")
        else:
            audio_emb_path = None

        for idx, (drv, kp, shading) in enumerate(zip(driver_frames, aligned_kpmaps, shadings)):
            tile_ref = np.uint8(source_image)
            tile_mask = np.uint8(alpha)
            tile_tar = np.uint8(drv)
            tile_head = np.uint8(kp)
            tile_light = np.uint8(shading)

            img_concat = np.concatenate(
                [tile_ref, tile_mask, tile_tar, tile_head, tile_light],
                axis=1
            )
            Image.fromarray(img_concat).save(concat_tmp_dir / f"{idx + 1:05d}.png")

        source_kp = self.fkpd.single_kp(source_image)
        if source_kp is None:
            raise RuntimeError(f"Cannot extract keypoint from ref image: {source_path}")

        source_shading = self.fir.render_motion_single_with_light(source_image, source_image)
        img0 = np.concatenate(
            [source_image, alpha, source_image, source_kp, source_shading],
            axis=1
        )
        Image.fromarray(np.uint8(img0)).save(concat_tmp_dir / "00000.png")

        run_ffmpeg_encode(concat_tmp_dir, save_shading_path, fps=self.fps)

        print(f"[OK] cross shading video saved to {save_shading_path}")
        if audio_emb_path is not None:
            print(f"[OK] audio embedding path: {audio_emb_path}")

        shutil.rmtree(work_dir, ignore_errors=True)

        return save_shading_path, audio_emb_path


# ============================================================
#                         main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", type=str, required=True, help="current driving video path")
    parser.add_argument("--source_path", type=str, required=True, help="next video's first frame as reference")
    parser.add_argument("--light_path", type=str, default=None, help="next video's first frame as lighting")
    parser.add_argument("--save_shading_path", type=str, required=True, help="output shading video path")
    parser.add_argument("--motion_align", type=str, default="relative", choices=["relative"], help="must be relative")

    parser.add_argument("--use_spectre", type=int, default=1, help="1=use SPECTRE+DECA; 0=use DECA relative")
    parser.add_argument("--spectre_feature_path", type=str, default=None, help="identity feature image path, default source_path")
    parser.add_argument("--spectre_light_path", type=str, default=None, help="lighting image path, default light_path/source_path")
    parser.add_argument("--spectre_exp_scale", type=float, default=1.0, help="expression scaling for SPECTRE")

    parser.add_argument("--use_audio", type=int, default=1, help="1=extract audio embedding; 0=skip audio")
    parser.add_argument("--audio_emb_path", type=str, default=None, help="where to save audio embedding .pt")
    parser.add_argument("--wav2vec_model", type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument("--audio_sample_rate", type=int, default=16000)
    parser.add_argument("--offline_vocals_root", type=str, default="eval_HDTF_50/vocals_offline")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--work_dir", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    if args.use_audio:
        audio_processor = AudioProcessor(
            wav2vec_model_name=args.wav2vec_model,
            sample_rate=args.audio_sample_rate,
            device=device,
            offline_vocals_root=args.offline_vocals_root,
        )
    else:
        audio_processor = None

    pre = PreprocessCrossWithAudio(
        use_spectre=bool(args.use_spectre),
        spectre_exp_scale=args.spectre_exp_scale,
        device=device,
        use_audio=bool(args.use_audio),
        audio_processor=audio_processor,
        fps=args.fps,
    )

    save_shading_path, audio_emb_path = pre.run(
        source_path=args.source_path,
        light_path=args.light_path,
        video_path=args.video_path,
        save_shading_path=args.save_shading_path,
        motion_align=args.motion_align,
        audio_emb_path=args.audio_emb_path,
        spectre_feature_path=args.spectre_feature_path,
        spectre_light_path=args.spectre_light_path,
        work_dir=args.work_dir,
    )

    print("\n=== DONE ===")
    print(f"cross_shading.mp4 : {save_shading_path}")
    if audio_emb_path is not None:
        print(f"audio_emb.pt     : {audio_emb_path}")