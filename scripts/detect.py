#!/usr/bin/env python3
"""
LitePose 多人姿态估计推理脚本

支持:
- 单张图像推理
- 视频推理
- 摄像头实时推理
- 目录批量推理

用法:
    # 图像推理
    python scripts/detect.py --weights runs/train/exp/weights/best.pt --source image.jpg --show

    # 视频推理
    python scripts/detect.py --weights runs/train/exp/weights/best.pt --source video.mp4 --output result.mp4

    # 摄像头
    python scripts/detect.py --weights runs/train/exp/weights/best.pt --camera 0 --show
"""

import sys
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from lite_pose.models import LitePose
from lite_pose.utils.yolo_utils import letterbox, draw_pose, postprocess


class MultiPersonPoseEstimator:
    """多人姿态估计器

    Args:
        weights: 模型权重路径
        input_size: 输入尺寸 (H, W)
        device: 运行设备
        conf_thresh: 置信度阈值
        iou_thresh: NMS IoU 阈值
    """

    def __init__(self,
                 weights: str,
                 num_keypoints: int = 17,
                 fpn_channels: int = 96,
                 input_size: tuple = (640, 640),
                 device: str = 'cuda',
                 conf_thresh: float = 0.25,
                 iou_thresh: float = 0.65):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.num_keypoints = num_keypoints

        # 构建模型
        self.model = LitePose(
            num_keypoints=num_keypoints,
            fpn_channels=fpn_channels,
            pretrained_backbone=False,
        )

        # 加载权重
        self._load_weights(weights)

        self.model = self.model.to(self.device)
        self.model.eval()

        # ImageNet 归一化参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        print(f"Model loaded from {weights}")
        print(f"Device: {self.device}")
        print(f"Input size: {input_size}")
        print(f"Conf threshold: {conf_thresh}")
        print(f"IoU threshold: {iou_thresh}")

    def _load_weights(self, weights: str):
        """加载模型权重"""
        state_dict = torch.load(weights, map_location='cpu', weights_only=False)

        # 支持多种检查点格式
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

        self.model.load_state_dict(model_dict)

    def preprocess(self, image: np.ndarray):
        """图像预处理

        Args:
            image: 输入图像 (H, W, 3) BGR格式

        Returns:
            input_tensor: 预处理后的tensor (1, 3, H, W)
            ratio: 缩放比例
            pad: padding大小 (pad_w, pad_h)
            orig_size: 原图大小 (H, W)
        """
        # Letterbox 缩放
        img_resized, ratio, pad = letterbox(image, self.input_size)

        # BGR -> RGB, 归一化
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_norm = (img_norm - self.mean) / self.std

        # 转换为tensor
        input_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1))
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        return input_tensor, ratio, pad, image.shape[:2]

    def scale_coords_back(self, coords, ratio, pad, orig_size):
        """将坐标缩放回原图

        Args:
            coords: 坐标 (..., 2) 或 (..., 4)
            ratio: 缩放比例
            pad: padding (pad_w, pad_h)
            orig_size: 原图大小 (H, W)
        """
        coords = coords.copy()
        pad_w, pad_h = pad

        if coords.shape[-1] == 4:
            coords[..., [0, 2]] -= pad_w
            coords[..., [1, 3]] -= pad_h
            coords[..., :4] /= ratio
            coords[..., [0, 2]] = np.clip(coords[..., [0, 2]], 0, orig_size[1])
            coords[..., [1, 3]] = np.clip(coords[..., [1, 3]], 0, orig_size[0])
        else:
            coords[..., 0] -= pad_w
            coords[..., 1] -= pad_h
            coords[..., :2] /= ratio
            coords[..., 0] = np.clip(coords[..., 0], 0, orig_size[1])
            coords[..., 1] = np.clip(coords[..., 1], 0, orig_size[0])

        return coords

    @torch.no_grad()
    def predict(self, image: np.ndarray):
        """预测单张图像

        Args:
            image: 输入图像 (H, W, 3) BGR格式

        Returns:
            bboxes: (N, 4) [x1, y1, x2, y2]
            scores: (N,)
            keypoints: (N, K, 3) [x, y, conf]
        """
        input_tensor, ratio, pad, orig_size = self.preprocess(image)

        outputs = self.model(input_tensor)
        bboxes, scores, keypoints = self.model.decode(
            outputs, self.conf_thresh, self.input_size)

        bboxes, scores, keypoints = postprocess(
            bboxes, scores, keypoints,
            conf_thresh=self.conf_thresh,
            iou_thresh=self.iou_thresh)

        bboxes = bboxes.cpu().numpy()
        scores = scores.cpu().numpy()
        keypoints = keypoints.cpu().numpy()

        if len(bboxes) > 0:
            bboxes = self.scale_coords_back(bboxes, ratio, pad, orig_size)
            keypoints[..., :2] = self.scale_coords_back(
                keypoints[..., :2], ratio, pad, orig_size)

        return bboxes, scores, keypoints


def process_image(estimator, image_path, output_path=None,
                  show=False, kpt_thresh=0.3):
    """处理单张图像"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    start = time.time()
    bboxes, scores, keypoints = estimator.predict(image)
    elapsed = time.time() - start

    print(f"Image: {image_path}")
    print(f"Detected {len(bboxes)} persons")
    print(f"Inference time: {elapsed*1000:.1f}ms")

    vis = draw_pose(image, keypoints, scores, bboxes, kpt_thresh=kpt_thresh)

    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"Saved to {output_path}")

    if show:
        cv2.imshow('LitePose', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return bboxes, scores, keypoints


def process_video(estimator, video_path, output_path=None,
                  show=True, kpt_thresh=0.3):
    """处理视频"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        bboxes, scores, keypoints = estimator.predict(frame)
        elapsed = time.time() - start
        total_time += elapsed

        vis = draw_pose(frame, keypoints, scores, bboxes, kpt_thresh=kpt_thresh)

        frame_count += 1
        avg_fps = frame_count / total_time

        cv2.putText(vis, f'FPS: {avg_fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, f'Persons: {len(bboxes)}', (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if writer:
            writer.write(vis)

        if show:
            cv2.imshow('LitePose', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved to {output_path}")
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames, Average FPS: {avg_fps:.1f}")


def process_camera(estimator, camera_id=0, output_path=None,
                   show=True, kpt_thresh=0.3):
    """处理摄像头"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Failed to open camera: {camera_id}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {camera_id}, Resolution: {width}x{height}")
    print("Press 'q' to quit, 's' to save snapshot")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    frame_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        bboxes, scores, keypoints = estimator.predict(frame)
        elapsed = time.time() - start
        frame_times.append(elapsed)

        vis = draw_pose(frame, keypoints, scores, bboxes, kpt_thresh=kpt_thresh)

        if len(frame_times) > 30:
            frame_times.pop(0)
        fps = 1.0 / np.mean(frame_times)

        cv2.putText(vis, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis, f'Persons: {len(bboxes)}', (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if writer:
            writer.write(vis)

        if show:
            cv2.imshow('LitePose', vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                snapshot_path = f'snapshot_{int(time.time())}.jpg'
                cv2.imwrite(snapshot_path, vis)
                print(f"Saved {snapshot_path}")

    cap.release()
    if writer:
        writer.release()
        print(f"Saved to {output_path}")
    cv2.destroyAllWindows()


def process_directory(estimator, source_dir, output_dir=None,
                      kpt_thresh=0.3, save_txt=False):
    """批量处理目录中的图像"""
    source_path = Path(source_dir)
    img_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in source_path.iterdir()
                   if f.suffix.lower() in img_formats]

    if not image_files:
        print(f"No images found in {source_dir}")
        return

    print(f"Found {len(image_files)} images")

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    total_persons = 0

    for i, img_path in enumerate(sorted(image_files)):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load: {img_path}")
            continue

        bboxes, scores, keypoints = estimator.predict(image)
        total_persons += len(bboxes)
        print(f"[{i+1}/{len(image_files)}] {img_path.name}: {len(bboxes)} persons")

        if output_dir:
            vis = draw_pose(image, keypoints, scores, bboxes, kpt_thresh=kpt_thresh)
            cv2.imwrite(str(output_path / img_path.name), vis)

            if save_txt:
                txt_path = output_path / f"{img_path.stem}.txt"
                with open(txt_path, 'w') as f:
                    for j in range(len(bboxes)):
                        bbox = bboxes[j]
                        score = scores[j]
                        kpts = keypoints[j]
                        line = f"{score:.4f} {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f}"
                        for k in range(len(kpts)):
                            line += f" {kpts[k, 0]:.2f} {kpts[k, 1]:.2f} {kpts[k, 2]:.4f}"
                        f.write(line + "\n")

    print(f"Done! Total persons detected: {total_persons}")
    if output_dir:
        print(f"Results saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='LitePose Multi-Person Pose Estimation')

    parser.add_argument('--weights', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--fpn-channels', type=int, default=96,
                        help='FPN channels (must match training)')
    parser.add_argument('--img-size', type=int, nargs=2, default=[640, 640],
                        help='Input size (H, W)')
    parser.add_argument('--num-keypoints', type=int, default=17,
                        help='Number of keypoints')

    parser.add_argument('--source', type=str, default=None,
                        help='Input: image/video/directory path')
    parser.add_argument('--camera', type=int, default=None,
                        help='Camera ID')

    parser.add_argument('--output', type=str, default=None,
                        help='Output path')
    parser.add_argument('--show', action='store_true',
                        help='Display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save detections as text files')

    parser.add_argument('--conf-thresh', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou-thresh', type=float, default=0.65,
                        help='NMS IoU threshold')
    parser.add_argument('--kpt-thresh', type=float, default=0.3,
                        help='Keypoint visualization threshold')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    return parser.parse_args()


def main():
    args = parse_args()

    estimator = MultiPersonPoseEstimator(
        weights=args.weights,
        num_keypoints=args.num_keypoints,
        fpn_channels=args.fpn_channels,
        input_size=tuple(args.img_size),
        device=args.device,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh,
    )

    if args.source:
        source = Path(args.source)

        if source.is_file():
            img_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            vid_formats = ('.mp4', '.avi', '.mov', '.mkv', '.webm')

            if source.suffix.lower() in img_formats:
                process_image(estimator, str(source), args.output,
                              show=args.show, kpt_thresh=args.kpt_thresh)
            elif source.suffix.lower() in vid_formats:
                process_video(estimator, str(source), args.output,
                              show=args.show, kpt_thresh=args.kpt_thresh)
            else:
                print(f"Unsupported file format: {source.suffix}")

        elif source.is_dir():
            process_directory(estimator, str(source), args.output,
                              kpt_thresh=args.kpt_thresh, save_txt=args.save_txt)
        else:
            print(f"Source not found: {source}")

    elif args.camera is not None:
        process_camera(estimator, args.camera, args.output,
                       show=args.show, kpt_thresh=args.kpt_thresh)
    else:
        print("Please specify --source or --camera")
        print("\nExamples:")
        print("  python scripts/detect.py --weights best.pt --source image.jpg --show")
        print("  python scripts/detect.py --weights best.pt --source video.mp4 --output result.mp4")
        print("  python scripts/detect.py --weights best.pt --source images/ --output results/")
        print("  python scripts/detect.py --weights best.pt --camera 0 --show")


if __name__ == '__main__':
    main()
