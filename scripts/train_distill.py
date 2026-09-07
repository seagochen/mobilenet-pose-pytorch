#!/usr/bin/env python3
"""
LitePose 蒸馏训练入口脚本

用法:
    python scripts/train_distill.py \
      --data /home/cxt/datasets/coco_pose_yolo/data.yaml \
      --teacher yolo11s-pose.onnx \
      --epochs 300 --batch-size 32 --lr 5e-4 --ema
"""

import sys
from pathlib import Path

import torch

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from lite_pose.training.config import get_parser, build_config
from lite_pose.training.distill_trainer import DistillTrainer
from lite_pose.runtime import command


@command("train")
def main():
    parser = get_parser()
    parser.add_argument('--debug-nan', action='store_true',
                        help='Enable NaN detection (slower but helps debug)')
    args = parser.parse_args()

    if args.debug_nan:
        print("=" * 60)
        print("[DEBUG] NaN detection enabled - training will be slower")
        print("=" * 60)
        torch.autograd.set_detect_anomaly(True)

    config = build_config(args)
    trainer = DistillTrainer(config)
    trainer.train()


if __name__ == '__main__':
    raise SystemExit(main())
