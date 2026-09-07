#!/usr/bin/env python3
"""
LitePose ONNX 导出脚本

将学生模型导出为 ONNX 格式，用于 Jetson TensorRT 部署。
输出 3 个尺度的原始预测 (不解码)，解码在推理端处理。

用法:
    python scripts/export_onnx.py \
      --weights runs/train/exp/weights/best.pt \
      --output litpose.onnx \
      --simplify
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from lite_pose.models import LitePose
from lite_pose.runtime import command


def load_model(weights_path: str, num_keypoints: int = 17,
               fpn_channels: int = 96) -> torch.nn.Module:
    """加载模型权重"""
    model = LitePose(
        num_keypoints=num_keypoints,
        fpn_channels=fpn_channels,
        pretrained_backbone=False,
    )

    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)

    # 支持多种 checkpoint 格式
    if 'ema' in state_dict and state_dict['ema'] is not None:
        ema_state = state_dict['ema']
        if isinstance(ema_state, dict) and 'ema' in ema_state:
            model_dict = ema_state['ema']
        else:
            model_dict = ema_state
    elif 'model' in state_dict:
        model_dict = state_dict['model']
    else:
        model_dict = state_dict

    model.load_state_dict(model_dict)
    model.eval()
    return model


class LitePoseExport(torch.nn.Module):
    """ONNX 导出包装器

    将 3 个尺度的原始预测拼接为单个输出张量，
    格式与 YOLOPose 类似: (B, C, N) 其中 N = sum(Hi*Wi)

    输出通道 C = 56 = 4(box) + 1(obj) + 17*3(kpts)
    """

    def __init__(self, model: LitePose):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) 输入图像

        Returns:
            output: (B, 56, N) 拼接后的原始预测
                N = 80*80 + 40*40 + 20*20 = 8000 (for 640x640 input)
        """
        outputs = self.model(x)  # List of (B, H, W, C)

        flattened = []
        for out in outputs:
            B, H, W, C = out.shape
            # (B, H, W, C) -> (B, C, H*W)
            flattened.append(out.permute(0, 3, 1, 2).reshape(B, C, -1))

        # (B, C, N)
        return torch.cat(flattened, dim=2)


def export_onnx(model, output_path, img_size=(640, 640), opset=13,
                dynamic_batch=True):
    """导出 ONNX"""
    export_model = LitePoseExport(model)
    export_model.eval()

    dummy_input = torch.randn(1, 3, *img_size)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'images': {0: 'batch'},
            'output': {0: 'batch'},
        }

    print(f"Exporting ONNX (opset {opset})...")
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )
    print(f"Exported to {output_path}")

    return output_path


def simplify_onnx(onnx_path):
    """使用 onnx-simplifier 优化"""
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("Warning: onnx-simplifier not installed, skipping simplification")
        print("  pip install onnx onnxsim")
        return onnx_path

    print("Simplifying ONNX model...")
    model = onnx.load(str(onnx_path))
    model_simplified, check = simplify(model)

    if check:
        onnx.save(model_simplified, str(onnx_path))
        print(f"Simplified model saved to {onnx_path}")
    else:
        print("Warning: Simplified model validation failed, keeping original")

    return onnx_path


def verify_onnx(onnx_path, pytorch_model, img_size=(640, 640), tolerance=1e-5):
    """验证 ONNX 输出与 PyTorch 一致"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("Warning: onnxruntime not installed, skipping verification")
        return False

    print("Verifying ONNX output...")

    # PyTorch 推理
    export_model = LitePoseExport(pytorch_model)
    export_model.eval()
    dummy_input = torch.randn(1, 3, *img_size)

    with torch.no_grad():
        pytorch_output = export_model(dummy_input).numpy()

    # ONNX 推理
    session = ort.InferenceSession(str(onnx_path))
    onnx_output = session.run(None, {'images': dummy_input.numpy()})[0]

    # 比较
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))

    print(f"  Max diff: {max_diff:.8f}")
    print(f"  Mean diff: {mean_diff:.8f}")
    print(f"  Tolerance: {tolerance}")

    if max_diff < tolerance:
        print("  PASSED: ONNX output matches PyTorch")
        return True
    else:
        print(f"  WARNING: Max diff {max_diff:.8f} exceeds tolerance {tolerance}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description='Export LitePose to ONNX')

    parser.add_argument('--weights', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX path (default: <weights_stem>.onnx)')
    parser.add_argument('--fpn-channels', type=int, default=96,
                        help='FPN channels (must match training)')
    parser.add_argument('--num-keypoints', type=int, default=17,
                        help='Number of keypoints')
    parser.add_argument('--img-size', type=int, nargs=2, default=[640, 640],
                        help='Input size (H, W)')
    parser.add_argument('--opset', type=int, default=13,
                        help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true',
                        help='Apply onnx-simplifier')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip verification')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                        help='Verification tolerance')

    return parser.parse_args()


@command("export")
def main():
    args = parse_args()

    # 确定输出路径
    if args.output is None:
        weights_path = Path(args.weights)
        args.output = str(weights_path.parent / f"{weights_path.stem}.onnx")

    # 加载模型
    model = load_model(args.weights, args.num_keypoints, args.fpn_channels)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: LitePose (FPN={args.fpn_channels}), {params:.2f}M params")
    print(f"Input size: {tuple(args.img_size)}")

    # 导出
    onnx_path = export_onnx(
        model, args.output,
        img_size=tuple(args.img_size),
        opset=args.opset,
    )

    # 简化
    if args.simplify:
        onnx_path = simplify_onnx(onnx_path)

    # 验证
    if not args.no_verify:
        verify_onnx(onnx_path, model,
                     img_size=tuple(args.img_size),
                     tolerance=args.tolerance)

    # 输出信息
    import os
    file_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"\nExport complete:")
    print(f"  File: {onnx_path}")
    print(f"  Size: {file_size:.2f} MB")
    print(f"  Input: images (B, 3, {args.img_size[0]}, {args.img_size[1]})")
    print(f"  Output: output (B, 56, N) where N = sum(Hi*Wi)")


if __name__ == '__main__':
    raise SystemExit(main())
