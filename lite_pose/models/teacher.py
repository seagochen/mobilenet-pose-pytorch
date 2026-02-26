"""
TeacherYOLOPose: ONNX-based YOLOPose teacher model

通过 onnxruntime 加载 YOLOPose ONNX 文件，用于知识蒸馏。
不依赖 ultralytics，纯 ONNX 推理。

ONNX 输出格式 (yolo11s-pose):
- shape: [1, 56, 8400]
- 56 = 4 (cx, cy, w, h) + 1 (score) + 17*3 (kpt x, y, conf)
- 所有坐标为 640x640 输入空间的像素坐标
"""

import numpy as np
import onnxruntime as ort
from typing import List, Dict, Optional


def _nms_numpy(
    bboxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.65
) -> np.ndarray:
    """Numpy NMS

    Args:
        bboxes: (N, 4) [x1, y1, x2, y2]
        scores: (N,)
        iou_thresh: IoU 阈值

    Returns:
        keep: 保留的索引
    """
    if len(bboxes) == 0:
        return np.empty(0, dtype=np.int64)

    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)

        mask = iou <= iou_thresh
        order = order[1:][mask]

    return np.array(keep, dtype=np.int64)


class TeacherYOLOPose:
    """YOLOPose 教师模型 (ONNX)

    通过 onnxruntime 加载 ONNX 模型，纯推理无梯度。
    用于知识蒸馏时生成软标签。

    Args:
        onnx_path: ONNX 文件路径
        providers: onnxruntime providers, 默认自动选择
    """

    def __init__(
        self,
        onnx_path: str,
        providers: Optional[List[str]] = None,
    ):
        if providers is None:
            providers = ort.get_available_providers()

        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # 获取输入输出信息
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape  # [1, 3, 640, 640]
        self.input_size = (inp.shape[2], inp.shape[3])  # (H, W)

    def _decode_output(
        self,
        raw_output: np.ndarray,
        conf_thresh: float,
        iou_thresh: float,
        max_det: int,
    ) -> Dict[str, np.ndarray]:
        """解码单张图的 ONNX 原始输出

        Args:
            raw_output: (56, 8400) 原始输出
            conf_thresh: 置信度阈值
            iou_thresh: NMS IoU 阈值
            max_det: 最大检测数量

        Returns:
            dict with 'bboxes' (N,4) xyxy, 'scores' (N,), 'keypoints' (N,17,3)
        """
        empty = {
            'bboxes': np.empty((0, 4), dtype=np.float32),
            'scores': np.empty(0, dtype=np.float32),
            'keypoints': np.empty((0, 17, 3), dtype=np.float32),
        }

        # (56, 8400) -> (8400, 56)
        pred = raw_output.T

        # 置信度过滤
        scores = pred[:, 4]
        mask = scores > conf_thresh
        if not mask.any():
            return empty

        pred = pred[mask]
        scores = scores[mask]

        # bbox: cx, cy, w, h -> x1, y1, x2, y2
        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        bboxes = np.stack([x1, y1, x2, y2], axis=1)

        # keypoints: (N, 51) -> (N, 17, 3)
        keypoints = pred[:, 5:].reshape(-1, 17, 3)

        # NMS
        keep = _nms_numpy(bboxes, scores, iou_thresh)
        keep = keep[:max_det]

        return {
            'bboxes': bboxes[keep].astype(np.float32),
            'scores': scores[keep].astype(np.float32),
            'keypoints': keypoints[keep].astype(np.float32),
        }

    def predict_batch(
        self,
        images,
        conf_thresh: float = 0.1,
        iou_thresh: float = 0.65,
        max_det: int = 100,
    ) -> List[Dict[str, np.ndarray]]:
        """批量推理

        Args:
            images: (B, 3, H, W) 归一化张量 (torch.Tensor 或 np.ndarray)
                    值域 [0, 1], 已完成 letterbox 等预处理
            conf_thresh: 置信度阈值 (蒸馏用低阈值获取更多软标签)
            iou_thresh: NMS IoU 阈值
            max_det: 每张图最大检测数量

        Returns:
            results: 每张图的检测结果列表
                [{'bboxes': (N,4) xyxy, 'scores': (N,), 'keypoints': (N,17,3)}, ...]
        """
        # torch.Tensor -> numpy
        if hasattr(images, 'cpu'):
            images = images.detach().cpu().numpy()

        images = images.astype(np.float32)
        batch_size = images.shape[0]

        results = []
        for i in range(batch_size):
            # ONNX 模型输入 batch=1
            inp = images[i:i+1]
            raw = self.session.run(None, {self.input_name: inp})[0]  # (1, 56, 8400)
            det = self._decode_output(raw[0], conf_thresh, iou_thresh, max_det)
            results.append(det)

        return results
