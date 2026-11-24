<div align="center">

# MoCoTalk

### Multi-Conditional Diffusion with Adaptive Router for Controllable Talking Head Generation

[Xinyan Ye](https://github.com/xy2119), Jiankang Deng, Abbas Edalat

Imperial College London

[![arXiv](https://img.shields.io/badge/arXiv-2605.08050-b31b1b.svg)](https://arxiv.org/abs/2605.08050)
[![Paper](https://img.shields.io/badge/Paper-HTML-blue.svg)](https://arxiv.org/html/2605.08050v1)
[![Project Page](https://img.shields.io/badge/Project-Page-233f7a.svg)](https://xy2119.github.io/MoCoTalk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<img src="https://arxiv.org/html/2605.08050v1/img/train5.png" width="95%">

</div>

Official implementation of **MoCoTalk**, a multi-conditional video diffusion model for talking-head generation built on Stable Video Diffusion. It is conditioned on four streams at once (a reference image, facial keypoints, a mouth-augmented 3DMM shading mesh and speech audio) and fuses them with an **Adaptive Multi-Condition Router**: channel-wise, timestep-aware gating that decides how much each condition contributes at every denoising step. A **lip consistency loss** tightens audio-visual alignment of the mouth.

Videos and comparisons are on the [project page](https://xy2119.github.io/MoCoTalk/).

## Contents

- [Installation](#installation)
- [Models and external assets](#models-and-external-assets)
- [Inference](#inference)
- [Training](#training)
- [Repository structure](#repository-structure)
- [Citation](#citation)

## Installation

Tested with Python 3.8, PyTorch 2.2.2 and CUDA 12.1 on Linux.

```bash
git clone https://github.com/xy2119/MoCoTalk_official.git
cd MoCoTalk_official

# Option A: reproduce the full environment
conda env create -f environment.yaml
conda activate mocotalk

# Option B: minimal install
conda create -n mocotalk python=3.8 -y && conda activate mocotalk
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.7+pt2.2.2cu121
pip install -r requirements.txt
```

`ffmpeg` must be on your `PATH`. Vocal separation (`separate_vocals.py`, and `preprocess_dataset.py` when a separator model is given) also needs `pip install audio-separator`.

All commands below are run from the repository root.

## Models and external assets

These files are not part of the repository. Download them and place them as shown:

```
MoCoTalk_official/
├── pretrained_weights/
│   ├── stable-video-diffusion-img2vid-xt/        # stabilityai/stable-video-diffusion-img2vid-xt
│   │   └── sd-vae-ft-mse/                        # stabilityai/sd-vae-ft-mse, placed INSIDE the SVD folder
│   └── mocotalk/                                 # MoCoTalk checkpoint (see below)
│       ├── unet.pth
│       ├── head_embedder.pth
│       ├── light_embedder.pth
│       ├── ref_embedder.pth
│       └── audio_proj.pth
├── mocotalk/third_party/decalib/data/            # DECA assets: deca_model.tar, generic_model.pkl (FLAME), FLAME masks, ...
├── mocotalk/third_party/facematting/u2net_human_seg.pth   # U^2-Net human segmentation
├── spectre/                                      # git clone https://github.com/filby89/spectre (+ its pretrained models)
└── checkpoints/misc/vocal_separator/Kim_Vocal_2.onnx       # only needed for vocal separation
```

- **DECA / FLAME:** follow the [DECA](https://github.com/yfeng95/DECA) instructions. FLAME requires registering on the [FLAME website](https://flame.is.tue.mpg.de/).
- **SPECTRE:** clone [SPECTRE](https://github.com/filby89/spectre) into `./spectre`, then follow its README to fetch `pretrained/spectre_model.tar` and the lip-reading model. The lip consistency loss uses the same lip-reading model.
- **Wav2Vec2:** `facebook/wav2vec2-base-960h` is downloaded from Hugging Face on first use.
- **Initialisation for training:** the U-Net is initialised from the [LCVD](https://github.com/MingtaoGuo/Relightable-Portrait-Animation) relightable-portrait checkpoint (`--pretrain_unet`).

> **MoCoTalk checkpoint:** release pending. This README will be updated with a download link.

## Inference

### Quick start: self-reenactment

Input is a talking-head video with audio. Its first frame is used as the identity and lighting reference, and the result is written with the audio muxed back in.

```bash
# 1) separate vocals once (needs audio-separator)
python separate_vocals.py --input_dir demo_videos --output_dir demo_videos/vocals_offline

# 2) generate
python demo_self_reenactment.py \
    --input_dir demo_videos \
    --output_dir results/self_reenactment \
    --offline_vocals_root demo_videos/vocals_offline \
    --pretrained_model_name_or_path pretrained_weights/stable-video-diffusion-img2vid-xt \
    --checkpoint_path pretrained_weights/mocotalk
```

### Cross-reenactment

Videos in `--input_dir` are paired in a circle: video *i* drives the identity and lighting taken from the first frame of video *i+1*.

```bash
python demo_cross_reenactment.py \
    --input_dir demo_videos \
    --output_dir results/cross_reenactment \
    --offline_vocals_root demo_videos/vocals_offline \
    --checkpoint_path pretrained_weights/mocotalk
```

### Full control: mix attributes from different sources

**Step 1. Build the conditions.** This writes a packed condition video (each frame holds `[ref | ref mask | target | keypoints | shading]` tiles, 512×512 each) and a Wav2Vec2 embedding `.pt` file.

```bash
# --video_path:        mouth motion + expression + audio
# --spectre_head_path: head pose (defaults to --video_path)
# --source_path:       identity
# --light_path:        lighting (defaults to --source_path)
python prepare_conditions.py \
    --video_path speech.mp4 \
    --spectre_head_path head_motion.mp4 \
    --source_path identity.png \
    --light_path lighting.png \
    --save_shading_path resources/shading.mp4 \
    --offline_vocals_root vocals_offline \
    --use_spectre 1 --use_audio 1
```

**Step 2. Generate.**

```bash
python inference.py \
    --pretrained_model_name_or_path pretrained_weights/stable-video-diffusion-img2vid-xt \
    --checkpoint_path pretrained_weights/mocotalk \
    --shading_video resources/shading.mp4 \
    --audio_emb resources/shading.pt \
    --save_path results/result.mp4 \
    --guidance 2.5 --steps 25
```

Notes:
- Without `--audio_emb`, generation runs without the audio branch (video-driven only); the router masks it out.
- Long videos are generated in overlapping temporal tiles (`--tile_size 16 --tile_overlap 6`).
- `--spectre_exp_scale` (< 1) reduces expression and mouth amplitude.
- Speed: about 2 minutes for 100 frames at 512×512 on one RTX 6000 (32 GB).

## Training

### 1. Data

The paper trains on **CelebV-HQ, MultiTalk, TFHP and MEAD**. Videos with a FasterVQA score below 0.6 are removed, which leaves about 34k clips. For each dataset, run:

```bash
# extracts frames, keypoint maps, mouth-augmented shading meshes, mattes and Wav2Vec2 embeddings
python preprocess_dataset.py \
    --video_path /path/to/CelebV-HQ/videos \
    --save_path ./TalkingHeadVideo \
    --dataset CelebV-HQ \
    --use_spectre 1 --use_audio 1

# keeps only clips where every modality exists, then writes TalkingHeadVideo/CelebV-HQ/CelebV-HQ-data-consistent.json
python build_dataset_index.py --save_path ./TalkingHeadVideo --dataset CelebV-HQ
```

Each JSON entry contains `video_path`, `kps_path`, `light_path` (shading mesh), `mask_path` and `audio_emb_path`. To train on several datasets, concatenate their JSON lists into one file.

### 2. Train

```bash
accelerate launch train.py \
    --pretrained_model_name_or_path pretrained_weights/stable-video-diffusion-img2vid-xt \
    --pretrain_unet /path/to/lcvd_unet \
    --data_meta_path ./TalkingHeadVideo/all-data-consistent.json \
    --output_dir outputs/mocotalk \
    --num_frames 8 --width 512 --height 512 \
    --per_gpu_batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 --max_train_steps 30000 \
    --lip_loss_weight 1.0 \
    --mixed_precision fp16
```

Paper setting: 8-frame clips at 512×512, effective batch size 8, 8-bit Adam, lr 1e-5, 30k steps, about 7 days on a single RTX 6000 (32 GB). Gradient checkpointing, xFormers and 8-bit Adam are on by default.

## Repository structure

```
MoCoTalk_official/
├── preprocess_dataset.py        # dataset preprocessing: frames, keypoints, shading mesh, matte, audio features
├── build_dataset_index.py       # keep clips with every modality and write the training JSON
├── train.py                     # training (adaptive router + lip consistency loss)
├── separate_vocals.py           # offline vocal separation for evaluation videos
├── prepare_conditions.py        # build the condition video + audio embedding for one input
├── inference.py                 # generate from prepared conditions
├── demo_self_reenactment.py     # end-to-end self-reenactment demo
├── demo_cross_reenactment.py    # end-to-end cross-reenactment demo
└── mocotalk/
    ├── data/dataset.py                        # aligned video / keypoint / mesh / mask / audio loader
    ├── models/unet_router.py                  # SVD U-Net + Adaptive Multi-Condition Router
    ├── models/{ref,head,light}_net.py         # reference, keypoint and shading adapters
    ├── models/audio_adapter.py                # windowed audio -> spatial latent adapter
    ├── pipelines/pipeline_mocotalk.py         # inference pipeline (tiled long-video generation)
    ├── pipelines/pipeline_mocotalk_train.py   # pipeline used for validation during training
    ├── preprocessing/face_video.py            # DECA rendering, keypoints and matting for a video
    ├── preprocessing/conditions_cross.py      # condition building for cross-reenactment
    └── third_party/                           # DECA (FLAME fitting + renderer), MediaPipe landmarks, U^2-Net matting
```

## Limitations

- Identity preservation and lip-sync (Sync-C) leave room for improvement compared with methods optimised for a single objective.
- The SVD backbone produces a short clip per pass (8 frames in the paper). Longer videos are stitched from overlapping windows, which can cause visible discontinuities at window boundaries.

## Citation

```bibtex
@article{ye2026mocotalk,
  title   = {MoCoTalk: Multi-Conditional Diffusion with Adaptive Router for Controllable Talking Head Generation},
  author  = {Ye, Xinyan and Deng, Jiankang and Edalat, Abbas},
  journal = {arXiv preprint arXiv:2605.08050},
  year    = {2026}
}
```

## Acknowledgements

This codebase builds on [LCVD / Relightable-Portrait-Animation](https://github.com/MingtaoGuo/Relightable-Portrait-Animation) by Mingtao Guo et al. We also use [Stable Video Diffusion](https://github.com/Stability-AI/generative-models), [DECA](https://github.com/yfeng95/DECA), [FLAME](https://flame.is.tue.mpg.de/), [SPECTRE](https://github.com/filby89/spectre), [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h), [MEMO](https://github.com/memoavatar/memo), [MediaPipe](https://github.com/google/mediapipe), [U^2-Net](https://github.com/xuebinqin/U-2-Net) and [audio-separator](https://github.com/nomadkaraoke/python-audio-separator). We thank the authors for releasing their code.

## License

Code is released under the [MIT License](LICENSE), inherited from the upstream LCVD repository. Code under `mocotalk/third_party/` and all external models (SVD, FLAME/DECA, SPECTRE, etc.) remain subject to their own licenses.
