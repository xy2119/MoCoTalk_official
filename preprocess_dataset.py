# preprocess_dataset.py
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

from mocotalk.preprocessing.face_video import InferVideo  # DECA preprocessing class

# -------- SPECTRE + DECA imports --------
import collections
import cv2
import torch
from skimage.transform import estimate_transform, warp

from spectre.src.spectre import SPECTRE
from spectre.config import cfg as spectre_cfg
from spectre.datasets.data_utils import landmarks_interpolate
from spectre.external.Visual_Speech_Recognition_for_Multiple_Languages.tracker.face_tracker import FaceTracker
from spectre.external.Visual_Speech_Recognition_for_Multiple_Languages.tracker.utils import get_landmarks

# -------- Audio imports (following MEMO) --------
import subprocess
import librosa
from decord import VideoReader
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput
from audio_separator.separator import Separator

STATUS_FILENAME_TMPL = "{dataset}-status.txt"
JSON_FILENAME_TMPL   = "{dataset}-data-consistent.json"


# ============================================================
#                   Status file helpers
# ============================================================
def load_status_map(status_file):
    """Read an existing status file -> {video_name: (status, payload_json)}"""
    done = {}
    if not os.path.exists(status_file):
        return done
    with open(status_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                name, status, payload = line.split("\t", 2)
                done[name] = (status, json.loads(payload))
            except Exception:
                # tolerate malformed lines
                continue
    return done


def append_status(status_file, video_name, status, payload_dict):
    """Append one status entry to the file"""
    line = f"{video_name}\t{status}\t{json.dumps(payload_dict, ensure_ascii=False)}\n"
    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    with open(status_file, "a", encoding="utf-8") as f:
        f.write(line)


def build_json_from_status(status_file, json_out):
    """
    Collect every Done entry from status.txt and write the final data json.
    Works with and without audio_emb_path.
    """
    data = []
    if os.path.exists(status_file):
        with open(status_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    name, status, payload = line.strip().split("\t", 2)
                    if status == "Done":
                        info = json.loads(payload)
                        item = {
                            "video_path": info["video_path"],
                            "kps_path": info["kpmap_path"],
                            "light_path": info["mesh_path"],
                            "mask_path": info["mask_path"],
                        }
                        if "audio_emb_path" in info and info["audio_emb_path"] is not None:
                            item["audio_emb_path"] = info["audio_emb_path"]
                        data.append(item)
                except Exception:
                    continue
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ============================================================
#                   SPECTRE face crop / motion
# ============================================================
def crop_face_spectre(frame, landmarks, scale=1.6, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    old_size = (right - left + bottom - top) / 2.0
    center = np.array([
        right - (right - left) / 2.0,
        bottom - (bottom - top) / 2.0
    ])

    size = int(old_size * scale)

    src_pts = np.array([
        [center[0] - size / 2, center[1] - size / 2],
        [center[0] - size / 2, center[1] + size / 2],
        [center[0] + size / 2, center[1] - size / 2]
    ])
    dst_pts = np.array([
        [0, 0],
        [0, image_size - 1],
        [image_size - 1, 0]
    ])

    tform = estimate_transform('similarity', src_pts, dst_pts)
    return tform


def extract_frames_with_landmarks(video_path, face_tracker, detect_landmarks=True):
    videofolder = os.path.splitext(video_path)[0] + "_frames_spectre"
    os.makedirs(videofolder, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    imagepath_list = []
    face_info = collections.defaultdict(list)

    fps = vidcap.get(cv2.CAP_PROP_FPS)
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

    return imagepath_list, landmarks, videofolder, fps


def extract_spectre_motion(video_path, spectre, device, face_tracker):
    image_paths, landmarks, videofolder, fps = extract_frames_with_landmarks(
        video_path, face_tracker, detect_landmarks=True
    )
    if landmarks is None:
        raise RuntimeError(f"No landmarks detected in {video_path}")

    original_len = len(image_paths)

    # pad 2 frames
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

    # chunking (same as the official SPECTRE inference)
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
                images_list.append(cropped.transpose(2, 0, 1))  # (3,224,224)

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

    # remove the 2 padding frames
    exp_seq = exp_seq[2:-2]
    pose_seq = pose_seq[2:-2]

    assert exp_seq.shape[0] == original_len, f"Length mismatch, got {exp_seq.shape[0]} vs {original_len}"

    return exp_seq, pose_seq, fps, videofolder


# ============================================================
#          SPECTRE + DECA four-source fusion -> shading
# ============================================================
class SpectreDecaRetargetor:
    def __init__(self, deca_renderer, device="cuda", exp_scale=1.0):
        """
        deca_renderer: the self.fir (FaceImageRender) already initialised in InferVideo
        """
        self.deca_renderer = deca_renderer
        self.device = device
        self.exp_scale = exp_scale

        # Initialise SPECTRE
        spectre_cfg.pretrained_modelpath = "spectre/pretrained/spectre_model.tar"
        spectre_cfg.model.use_tex = False
        self.spectre = SPECTRE(spectre_cfg, device)
        self.spectre.eval()

        # Initialise FaceTracker
        self.face_tracker = FaceTracker()

    def fuse(self, feature_path, head_video_path, speech_video_path, light_path):
        device = self.device
        fir = self.deca_renderer

        # A) feature → identity
        print(f"[SPECTRE+DECA] feature: {feature_path}")
        feature_bgr = cv2.imread(feature_path)
        if feature_bgr is None:
            raise RuntimeError(f"Cannot read feature image: {feature_path}")
        feature_rgb = cv2.cvtColor(feature_bgr, cv2.COLOR_BGR2RGB)
        feature_rgb = cv2.resize(feature_rgb, (512, 512))

        ref_codedict = fir.image_to_3dcoeff(feature_rgb)
        shape_ref = ref_codedict["shape"]
        cam_ref   = ref_codedict["cam"]
        tform_ref = ref_codedict["tform"]
        h_ref     = ref_codedict["height"]
        w_ref     = ref_codedict["width"]

        # B) lighting
        print(f"[SPECTRE+DECA] light: {light_path}")
        light_bgr = cv2.imread(light_path)
        if light_bgr is None:
            raise RuntimeError(f"Cannot read lighting image: {light_path}")
        light_rgb = cv2.cvtColor(light_bgr, cv2.COLOR_BGR2RGB)
        light_rgb = cv2.resize(light_rgb, (512, 512))

        light_codedict = fir.image_to_3dcoeff(light_rgb)
        light_target = light_codedict["light"]

        # C1) head video
        print(f"[SPECTRE+DECA] head video: {head_video_path}")
        head_exp_seq, head_pose_seq, fps_head, _ = extract_spectre_motion(
            head_video_path, self.spectre, device, self.face_tracker
        )
        T_head = head_exp_seq.shape[0]

        # C2) speech video
        print(f"[SPECTRE+DECA] speech video: {speech_video_path}")
        speech_exp_seq, speech_pose_seq, fps_speech, _ = extract_spectre_motion(
            speech_video_path, self.spectre, device, self.face_tracker
        )
        T_speech = speech_exp_seq.shape[0]

        # align lengths
        T = min(T_head, T_speech)
        if T_head != T_speech:
            print(f"[WARN] Head and speech length differ, truncating to {T}")
        head_pose_seq   = head_pose_seq[:T]
        speech_pose_seq = speech_pose_seq[:T]
        speech_exp_seq  = speech_exp_seq[:T]

        # D) fuse pose & exp
        pose_fused = speech_pose_seq.clone()
        pose_fused[:, :3] = head_pose_seq[:, :3]
        exp_fused = speech_exp_seq.clone()
        if self.exp_scale != 1.0:
            exp_fused = exp_fused * self.exp_scale

        # E) render the shading with DECA
        shading_frames = []
        shape_ref = shape_ref.to(device)
        cam_ref = cam_ref.to(device)
        light_target = light_target.to(device)

        print("[SPECTRE+DECA] rendering shading frames ...")
        for t in tqdm(range(T), desc="Render shading (SPECTRE+DECA)"):
            exp_t  = exp_fused[t:t+1].to(device)
            pose_t = pose_fused[t:t+1].to(device)

            shape_img, _ = fir.render_shape(
                shape=shape_ref,
                exp=exp_t,
                pose=pose_t,
                cam=cam_ref,
                light=light_target,
                tform=tform_ref,
                h=h_ref,
                w=w_ref
            )
            shape_img = shape_img.astype(np.uint8)

            # convert to 3-channel pseudo-greyscale
            gray = cv2.cvtColor(shape_img, cv2.COLOR_RGB2GRAY)
            gray3 = np.stack([gray, gray, gray], axis=-1)
            shading_frames.append(gray3)

        return shading_frames


# ============================================================
#             Wav2Vec2 + linearly interpolated audio features (MEMO)
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
        # Do not modify self.config.output_attentions
        if output_attentions is None:
            output_attentions = False
        if output_hidden_states is None:
            output_hidden_states = True
        if return_dict is None:
            return_dict = True

        # feature extractor
        extract_features = self.feature_extractor(input_values)   # (B, C, T')
        extract_features = extract_features.transpose(1, 2)       # (B, T', C)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)  # (B, T, C)

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

    def feature_extract(
        self,
        input_values,
        seq_len,
    ):
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)
        return extract_features

    def encode(
        self,
        extract_features,
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
    """
    features: (B, T', C)
    Linearly interpolate to seq_len frames.
    """
    import torch.nn.functional as F

    features = features.transpose(1, 2)  # (B, C, T')
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode="linear")
    return output_features.transpose(1, 2)  # (B, T, C)


class AudioProcessor:
    """
    Simplified MEMO AudioProcessor:
    - optional vocal separation (audio_separator)
    - multi-layer Wav2Vec2 hidden_states
    - linear interpolation to video_length
    - returns audio_emb: [video_length, n_layers, dim]
    """
    def __init__(
        self,
        wav2vec_model_name="facebook/wav2vec2-base-960h",
        vocal_separator_model=None,
        sample_rate=16000,
        device="cuda",
        vocals_root=None,
    ):
        self.device = device
        self.sample_rate = sample_rate

        # Wav2Vec encoder
        self.audio_encoder = Wav2VecModel.from_pretrained(
            wav2vec_model_name,
            local_files_only=True
        ).to(device)
        # freeze the feature_extractor parameters
        self.audio_encoder.feature_extractor._freeze_parameters()

        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_model_name,
            local_files_only=True
        )

        # vocal separation model (Kim_Vocal_2.onnx)
        if vocal_separator_model is not None and os.path.exists(vocal_separator_model):
            audio_separator_model_path = os.path.dirname(vocal_separator_model)
            audio_separator_model_name = os.path.basename(vocal_separator_model)

            if vocals_root is None:
                vocals_root = Path("./tmp/vocals")
            else:
                vocals_root = Path(vocals_root)
            vocals_root.mkdir(parents=True, exist_ok=True)

            self.vocals_root = vocals_root
            self.vocal_separator = Separator(
                output_dir=self.vocals_root,
                output_single_stem="vocals",
                model_file_dir=audio_separator_model_path,
            )
            self.vocal_separator.load_model(audio_separator_model_name)
            assert self.vocal_separator.model_instance is not None, "Fail to load audio separator model."
        else:
            self.vocal_separator = None
            self.vocals_root = None


    def _extract_wave(self, video_path: str):
        """
        Get speech_array (mono) from a video.
        Preferred path: ffmpeg -> wav -> separator -> vocals;
        on failure, read the audio straight from the video file with librosa.
        """
        video_path = str(video_path)
        stem = Path(video_path).stem

        # Start from the original video audio as a fallback
        fallback_audio, _ = librosa.load(video_path, sr=self.sample_rate)

        if self.vocal_separator is None:
            return fallback_audio

        try:
            # 1) Extract the original audio track to wav with ffmpeg
            raw_audio_path = self.vocals_root / f"{stem}-raw.wav"
            ffmpeg_command = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(self.sample_rate),
                "-ac",
                "2",
                str(raw_audio_path),
            ]
            subprocess.run(ffmpeg_command, check=True)

            # 2) Vocal separation
            #   Note: the path is passed as str on purpose
            outputs = self.vocal_separator.separate(str(raw_audio_path))

            # 3) Handle the different return types
            from pathlib import Path as _Path

            if isinstance(outputs, (_Path, str)):
                # a single path
                output_paths = [outputs]
            elif isinstance(outputs, (list, tuple)):
                output_paths = outputs
            else:
                # unexpected type: treat as a failure
                raise TypeError(f"Unexpected type from Separator.separate: {type(outputs)}")

            if len(output_paths) == 0:
                raise RuntimeError("Separator returned empty outputs")

            # 4) Use the first output path
            first_out = output_paths[0]
            first_out = _Path(first_out)

            # If it is a relative path, join it with vocals_root
            if not first_out.is_absolute():
                vocal_path = self.vocals_root / first_out
            else:
                vocal_path = first_out

            # 5) Read the vocals wav and resample
            y, sr = librosa.load(vocal_path.as_posix(), sr=None)
            speech_array = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)

            return speech_array

        except Exception as e:
            print(f"[WARN] vocal separation failed for {video_path}, fallback to raw audio. Error: {e}")
            return fallback_audio


    def get_audio_embedding_from_video(self, video_path: str, video_length: int):
        """
        Main entry:
        - input: video path and target video_length (number of frames)
        - output: audio_emb [video_length, n_layers, dim]
        """
        speech_array = self._extract_wave(video_path)

        # wav2vec preprocessing
        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(speech_array, sampling_rate=self.sample_rate).input_values
        )  # (T_audio,)
        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)
        audio_feature = audio_feature.unsqueeze(0)  # (1, T_audio)

        with torch.no_grad():
            outputs = self.audio_encoder(
                audio_feature,
                seq_len=video_length,
                output_hidden_states=True
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) <= 1:
            raise RuntimeError("Fail to get hidden_states from wav2vec")

        # hidden_states[1:] = output of each Transformer layer (B, T, D)
        audio_emb = torch.stack(hidden_states[1:], dim=1).squeeze(0)  # (L, T, D)
        # reshape to (T, L, D)
        audio_emb = rearrange(audio_emb, "l t d -> t l d")  # [frame, layer, dim]

        return audio_emb.cpu().detach()


# ============================================================
#                LCVD MakeDataset + Audio
# ============================================================
class MakeDataset(InferVideo):
    def __init__(self,
                 use_spectre=False,
                 spectre_head_root=None,
                 spectre_feature_root=None,
                 spectre_light_root=None,
                 spectre_exp_scale=1.0,
                 device="cuda",
                 use_audio=True,
                 audio_processor: AudioProcessor = None,
                 audio_emb_dir: str = None):
        super().__init__()
        self.use_spectre = use_spectre
        self.spectre_head_root = spectre_head_root
        self.spectre_feature_root = spectre_feature_root
        self.spectre_light_root = spectre_light_root

        self.use_audio = use_audio
        self.audio_processor = audio_processor
        self.audio_emb_dir = Path(audio_emb_dir) if audio_emb_dir is not None else None

        if self.use_spectre:
            print("[INIT] Use SPECTRE+DECA shading")
            self.spectre_retargetor = SpectreDecaRetargetor(
                deca_renderer=self.fir,
                device=device,
                exp_scale=spectre_exp_scale
            )
        else:
            self.spectre_retargetor = None

        if self.use_audio and (self.audio_processor is None or self.audio_emb_dir is None):
            raise ValueError("use_audio=True but audio_processor or audio_emb_dir was not provided")

    def make_dataset(self, video_root, video_name, save_root, dataset="VFHQ"):
        # Create a unique tmp directory for concurrency / retries
        stem = Path(video_name).stem
        tmp_dir = Path(save_root) / "_tmp" / f"target_{stem}_{os.getpid()}"
        if tmp_dir.exists():
            os.system(f'rm -rf "{tmp_dir.as_posix()}"')
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Extract frames (the speech video defaults to the current video)
        input_video = Path(video_root) / video_name
        os.system(f'ffmpeg -i "{input_video.as_posix()}" "{(tmp_dir / "%05d.png").as_posix()}" -y')

        # Read frames (RGB 512x512)
        frame_files = sorted(tmp_dir.glob("*.png"))
        if len(frame_files) == 0:
            raise RuntimeError("ffmpeg frame extraction failed or produced no frames")
        frames = [np.array(Image.open(p).convert("RGB").resize((512, 512))) for p in frame_files]

        # Keypoints
        kpmaps = self.fkpd.motion_self(frames)
        if kpmaps is None or len(kpmaps) != len(frames):
            raise RuntimeError("face detection failed in kp extraction")

        # Matting
        alphas = []
        try:
            for frame in frames:
                alphas.append(self.fm.portrait_matting(frame))
        except Exception:
            raise RuntimeError("matting failed")

        # Shading: different paths depending on whether SPECTRE is enabled
        if self.use_spectre:
            speech_video_path = input_video.as_posix()

            # head video
            if self.spectre_head_root is not None:
                head_video_path = (Path(self.spectre_head_root) / video_name).as_posix()
            else:
                head_video_path = speech_video_path

            # feature image
            if self.spectre_feature_root is not None:
                feature_path = (Path(self.spectre_feature_root) / f"{stem}.png").as_posix()
            else:
                feature_path = (tmp_dir / "00001.png").as_posix()

            # light image
            if self.spectre_light_root is not None:
                light_path = (Path(self.spectre_light_root) / f"{stem}.png").as_posix()
            else:
                light_path = feature_path

            shadings = self.spectre_retargetor.fuse(
                feature_path=feature_path,
                head_video_path=head_video_path,
                speech_video_path=speech_video_path,
                light_path=light_path
            )

            if shadings is None or len(shadings) != len(frames):
                raise RuntimeError("SPECTRE+DECA shading failed or length mismatch")
        else:
            shadings = []
            try:
                for frame in frames:
                    shadings.append(self.fir.render_motion_single(frame))
            except Exception:
                raise RuntimeError("DECA shading failed")

        # ------------------------------------------
        # Audio: wav2vec features resampled to video_length
        # ------------------------------------------
        audio_emb_path = None
        if self.use_audio:
            try:
                video_length = len(frames)
                audio_emb = self.audio_processor.get_audio_embedding_from_video(
                    input_video.as_posix(),
                    video_length=video_length
                )
                audio_emb_path = (self.audio_emb_dir / f"{stem}.pt").as_posix()
                torch.save(audio_emb, audio_emb_path)
            except Exception as e:
                # Raise on audio failure so the caller marks the sample as Fail and no incomplete sample is kept
                raise RuntimeError(f"Audio feature extraction failed: {e}")

        # Output directories
        ds_dir = Path(save_root) / dataset
        video_dir = ds_dir / f"{dataset}-video"
        kp_dir    = ds_dir / f"{dataset}-kpmap"
        mask_dir  = ds_dir / f"{dataset}-mask"
        mesh_dir  = ds_dir / f"{dataset}-mesh"
        for d in [video_dir, kp_dir, mask_dir, mesh_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Export each stream
        # 1) Write the original frames back to a video
        for idx, frame in enumerate(frames):
            Image.fromarray(np.uint8(frame)).save(tmp_dir / f"{idx+1:05d}.png")
        os.system(
            f'ffmpeg -r 20 -i "{(tmp_dir / "%05d.png").as_posix()}" '
            f'-pix_fmt yuv420p -c:v libx264 "{(video_dir / video_name).as_posix()}" -y'
        )

        # 2) kpmap
        for idx, kpmap in enumerate(kpmaps):
            Image.fromarray(np.uint8(kpmap)).save(tmp_dir / f"{idx+1:05d}.png")
        os.system(
            f'ffmpeg -r 20 -i "{(tmp_dir / "%05d.png").as_posix()}" '
            f'-pix_fmt yuv420p -c:v libx264 "{(kp_dir / video_name).as_posix()}" -y'
        )

        # 3) mask
        for idx, alpha in enumerate(alphas):
            Image.fromarray(np.uint8(alpha)).save(tmp_dir / f"{idx+1:05d}.png")
        os.system(
            f'ffmpeg -r 20 -i "{(tmp_dir / "%05d.png").as_posix()}" '
            f'-pix_fmt yuv420p -c:v libx264 "{(mask_dir / video_name).as_posix()}" -y'
        )

        # 4) mesh/shading (a greyscale shaded image; 3-channel pseudo-greyscale also works)
        for idx, shading in enumerate(shadings):
            Image.fromarray(np.uint8(shading)).save(tmp_dir / f"{idx+1:05d}.png")
        os.system(
            f'ffmpeg -r 20 -i "{(tmp_dir / "%05d.png").as_posix()}" '
            f'-pix_fmt yuv420p -c:v libx264 "{(mesh_dir / video_name).as_posix()}" -y'
        )

        # Clean up tmp
        os.system(f'rm -rf "{tmp_dir.as_posix()}"')

        # Return the five paths (including audio_emb_path)
        payload = {
            "video_name": video_name,
            "video_path": (video_dir / video_name).as_posix(),
            "kpmap_path": (kp_dir / video_name).as_posix(),
            "mask_path":  (mask_dir / video_name).as_posix(),
            "mesh_path":  (mesh_dir / video_name).as_posix(),
        }
        if self.use_audio:
            payload["audio_emb_path"] = audio_emb_path

        return payload


# ============================================================
#                       main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="./VFHQ", help="Directory of speech videos (or the original VFHQ path)")
    parser.add_argument("--save_path",  type=str, default="./TalkingHeadVideo", help="Output root directory")
    parser.add_argument("--dataset",    type=str, default="VFHQ", help="Dataset name (used for sub-directory and file names)")
    parser.add_argument(
        "--retry_fail",
        type=int,
        default=0,
        help="Retry videos previously marked as Fail: 1=retry, 0=skip"
    )
    # -------- SPECTRE options --------
    parser.add_argument(
        "--use_spectre",
        type=int,
        default=0,
        help="Use SPECTRE+DECA to build the shading: 1=yes, 0=no (DECA only)"
    )
    parser.add_argument(
        "--spectre_head_root",
        type=str,
        default=None,
        help="Root directory of head-motion videos (same file names as the speech videos; defaults to the speech video itself)"
    )
    parser.add_argument(
        "--spectre_feature_root",
        type=str,
        default=None,
        help="Root directory of reference identity images named <stem>.png (defaults to the first frame of the speech video)"
    )
    parser.add_argument(
        "--spectre_light_root",
        type=str,
        default=None,
        help="Root directory of lighting images named <stem>.png (defaults to the same image as the feature image)"
    )
    parser.add_argument(
        "--spectre_exp_scale",
        type=float,
        default=1.0,
        help="SPECTRE expression scale; values < 1 reduce mouth/expression amplitude"
    )
    # -------- Audio options --------
    parser.add_argument(
        "--use_audio",
        type=int,
        default=1,
        help="Extract audio features and align them to video frames: 1=yes, 0=no"
    )
    parser.add_argument(
        "--wav2vec_model",
        type=str,
        default="facebook/wav2vec2-base-960h",
        help="Wav2Vec2 model name or local path (must already be in the HF cache)"
    )
    parser.add_argument(
        "--vocal_separator_model",
        type=str,
        default="checkpoints/misc/vocal_separator/Kim_Vocal_2.onnx",
        help="Full path to Kim_Vocal_2.onnx; if None, vocal separation is skipped"
    )
    parser.add_argument(
        "--audio_sample_rate",
        type=int,
        default=16000,
        help="Audio sample rate, 16k by default"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare directories
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    ds_dir = Path(args.save_path) / args.dataset
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Audio directory & processor
    if args.use_audio:
        audio_emb_dir = ds_dir / f"{args.dataset}-audioemb"
        audio_emb_dir.mkdir(parents=True, exist_ok=True)

        vocals_root = ds_dir / f"{args.dataset}-vocals"
        vocals_root.mkdir(parents=True, exist_ok=True)

        audio_processor = AudioProcessor(
            wav2vec_model_name=args.wav2vec_model,
            vocal_separator_model=args.vocal_separator_model,
            sample_rate=args.audio_sample_rate,
            device=device,
            vocals_root=vocals_root,
        )
    else:
        audio_processor = None
        audio_emb_dir = None

    md = MakeDataset(
        use_spectre=bool(args.use_spectre),
        spectre_head_root=args.spectre_head_root,
        spectre_feature_root=args.spectre_feature_root,
        spectre_light_root=args.spectre_light_root,
        spectre_exp_scale=args.spectre_exp_scale,
        device=device,
        use_audio=bool(args.use_audio),
        audio_processor=audio_processor,
        audio_emb_dir=audio_emb_dir,
    )

    status_file = (ds_dir / STATUS_FILENAME_TMPL.format(dataset=args.dataset)).as_posix()
    json_file   = (ds_dir / JSON_FILENAME_TMPL.format(dataset=args.dataset)).as_posix()

    # Load the processing status
    status_map = load_status_map(status_file)

    # Decide whether to retry Fail videos
    if args.retry_fail:
        processed_names = {name for name, (status, _) in status_map.items() if status == "Done"}
    else:
        processed_names = set(status_map.keys())

    video_files = sorted(os.listdir(args.video_path))
    for video_name in tqdm(video_files):
        if video_name in processed_names:
            continue
        try:
            payload = md.make_dataset(args.video_path, video_name, args.save_path, dataset=args.dataset)
            append_status(status_file, video_name, "Done", payload)
        except Exception as e:
            # Write a payload even on failure, to help debugging
            payload = {
                "video_name": video_name,
                "video_path": str(Path(args.save_path) / args.dataset / f"{args.dataset}-video" / video_name),
                "kpmap_path": str(Path(args.save_path) / args.dataset / f"{args.dataset}-kpmap" / video_name),
                "mask_path":  str(Path(args.save_path) / args.dataset / f"{args.dataset}-mask" / video_name),
                "mesh_path":  str(Path(args.save_path) / args.dataset / f"{args.dataset}-mesh" / video_name),
                "error":      str(e),
            }
            if args.use_audio:
                stem = Path(video_name).stem
                payload["audio_emb_path"] = str(ds_dir / f"{args.dataset}-audioemb" / f"{stem}.pt")
            append_status(status_file, video_name, "Fail", payload)

    build_json_from_status(status_file, json_file)

    print(f"[OK] Status file: {status_file}")
    print(f"[OK] Data list: {json_file}")
