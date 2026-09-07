# LitePose (mobilenet-pose-pytorch)

Lightweight multi-person pose estimation with a MobileNetV3-Small backbone, trained via knowledge distillation from a YOLOPose (YOLO11-Pose) ONNX teacher model.

## Features

- **MobileNetV3-Small backbone + FPN**: compact student network aimed at edge/embedded deployment (e.g. Jetson).
- **Knowledge distillation training**: a frozen YOLOPose ONNX teacher (via ONNX Runtime) supervises the student with box, objectness/score, and keypoint distillation losses, blended with ground-truth loss and a linearly decaying distillation weight.
- **YOLO-Pose output format**: 3-scale detection heads concatenated into a single `(B, 56, N)` tensor (`56 = 5 + 17*3`, `N = 80*80 + 40*40 + 20*20` at 640 input).
- **Full training pipeline**: AdamW optimizer with reduced backbone LR, cosine/step LR scheduling, warmup, EMA, gradient clipping, and backbone freezing for early epochs.
- **YAML-based configuration** with CLI overrides.
- **ONNX export** for deployment, with an optional `onnx-simplifier` pass and PyTorch-vs-ONNX output verification.
- **Multi-mode inference**: image, video, webcam, and directory batch processing.

## Project Structure

```
mobilenet-pose-pytorch/
├── setup.py                      # Package install script
├── requirements.txt
├── configs/
│   └── distill.yaml              # Distillation training config
│
├── lite_pose/                    # Core package
│   ├── models/
│   │   ├── backbone.py           # MobileNetV3-Small backbone
│   │   ├── pose_head.py          # YOLOPose-style detection head + FPN
│   │   ├── lite_pose.py          # LitePose student model
│   │   ├── teacher.py            # ONNX Runtime teacher wrapper
│   │   ├── distill_loss.py       # Distillation losses (box/score/keypoint)
│   │   └── yolo_loss.py          # Ground-truth YOLO-Pose loss
│   │
│   ├── training/
│   │   ├── config.py             # YAML + CLI configuration
│   │   └── distill_trainer.py    # DistillTrainer (train/validate loops)
│   │
│   ├── utils/
│   │   ├── data/                 # YOLO-Pose dataset & path utilities
│   │   ├── callbacks/            # LR scheduler, EMA
│   │   ├── metrics/              # PCK / AP / OKS evaluation
│   │   └── yolo_utils.py         # Letterbox, decode, NMS, draw utilities
│   │
│   └── runtime.py                # Shared CLI logging/command conventions
│
└── scripts/
    ├── train.py                  # Training entry point (wraps train_distill.py)
    ├── train_distill.py          # Distillation training script
    ├── detect.py                 # Inference (image/video/camera/directory)
    └── export_onnx.py            # ONNX export + simplification + verification
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Data

Expects a YOLO-Pose format dataset (see `configs/distill.yaml` → `data.yaml_path`), and an ONNX teacher model (`teacher.onnx_path`, default `yolo11s-pose.onnx`) used only for distillation, loaded via ONNX Runtime — no `ultralytics` dependency required.

## Quick Start

### Train (distillation)

```bash
python scripts/train.py --config configs/distill.yaml

# CLI overrides YAML values
python scripts/train.py --config configs/distill.yaml --lr 1e-3 --epochs 100 --batch-size 16
```

### Inference

```bash
# Single image
python scripts/detect.py --weights runs/train/exp/weights/best.pt --source image.jpg --show

# Video
python scripts/detect.py --weights runs/train/exp/weights/best.pt --source video.mp4 --output result.mp4

# Directory batch
python scripts/detect.py --weights runs/train/exp/weights/best.pt --source images/ --output results/

# Webcam
python scripts/detect.py --weights runs/train/exp/weights/best.pt --camera 0 --show
```

### Export to ONNX

```bash
python scripts/export_onnx.py --weights runs/train/exp/weights/best.pt --output lite_pose.onnx --simplify
```

## Key Configuration (`configs/distill.yaml`)

| Section | Key | Description |
|---|---|---|
| `model` | `fpn_channels` | FPN output channels (default 96) |
| `teacher` | `onnx_path` / `conf_thresh` | Teacher ONNX model path and confidence threshold for soft labels |
| `distill` | `weight_max` / `weight_min` | Distillation weight, linearly decayed from `weight_max` to `weight_min` over training |
| `distill` | `box_weight` / `score_weight` / `kpt_weight` | Per-component distillation loss weights |
| `training` | `epochs`, `batch_size`, `lr`, `lr_scheduler`, `warmup_epochs`, `ema`, `freeze_backbone`, `freeze_epochs`, `grad_clip` | Standard training hyperparameters |
| `output` | `dir`, `project`, `log_interval`, `val_interval`, `save_interval` | Logging/checkpoint output control |

## Inference Options (`scripts/detect.py`)

| Argument | Default | Description |
|---|---|---|
| `--weights` | (required) | Path to checkpoint (`best.pt`/`last.pt`), supports EMA weights |
| `--source` / `--camera` | — | Image/video/directory path, or camera index |
| `--img-size` | 640 640 | Inference resolution (H W) |
| `--conf-thresh` | 0.25 | Detection confidence threshold |
| `--iou-thresh` | 0.65 | NMS IoU threshold |
| `--kpt-thresh` | 0.3 | Keypoint visibility threshold |
| `--device` | cuda | `cuda` or `cpu` |

## COCO Keypoints

```
0: nose
1: left_eye      2: right_eye
3: left_ear      4: right_ear
5: left_shoulder 6: right_shoulder
7: left_elbow    8: right_elbow
9: left_wrist    10: right_wrist
11: left_hip     12: right_hip
13: left_knee    14: right_knee
15: left_ankle   16: right_ankle
```

## References

- [YOLOv8/YOLO11-Pose](https://github.com/ultralytics/ultralytics) — teacher model architecture
- [MobileNetV3](https://arxiv.org/abs/1905.02244) — Searching for MobileNetV3
- Knowledge Distillation — Hinton et al., *Distilling the Knowledge in a Neural Network*
