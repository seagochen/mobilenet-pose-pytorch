"""
Response-based 知识蒸馏损失

将教师模型的解码检测结果分配到学生模型的 grid cells，
然后比较学生解码后的输出与教师输出。

分配逻辑:
1. 根据教师 bbox 大小选择学生尺度: max(w,h)/stride 最接近 4
2. 根据 bbox 中心确定 grid cell: (floor(cx/stride), floor(cy/stride))

损失组成:
- Box: SmoothL1 (归一化坐标)
- Score: BCE with logits (学生 obj logit vs 教师 confidence)
- Keypoint: SmoothL1 (归一化坐标)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple


class DistillationLoss(nn.Module):
    """Response-based 知识蒸馏损失

    Args:
        num_keypoints: 关键点数量
        strides: 学生模型各尺度的步长
        box_weight: Box 蒸馏损失权重
        score_weight: Score 蒸馏损失权重
        kpt_weight: Keypoint 蒸馏损失权重
    """

    def __init__(self,
                 num_keypoints: int = 17,
                 strides: List[int] = None,
                 box_weight: float = 1.0,
                 score_weight: float = 1.0,
                 kpt_weight: float = 2.0):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.strides = strides or [8, 16, 32]
        self.box_weight = box_weight
        self.score_weight = score_weight
        self.kpt_weight = kpt_weight

    def _assign_to_scale(self, w: float, h: float) -> int:
        """根据 bbox 尺寸选择最合适的尺度

        选择 max(w,h)/stride 最接近 4 的尺度。
        """
        max_dim = max(w, h)
        best_idx = 0
        best_diff = float('inf')

        for idx, stride in enumerate(self.strides):
            ratio = max_dim / stride
            diff = abs(ratio - 4.0)
            if diff < best_diff:
                best_idx = idx
                best_diff = diff

        return best_idx

    def forward(self,
                student_outputs: List[torch.Tensor],
                teacher_results: List[Dict[str, np.ndarray]],
                input_size: Tuple[int, int] = (640, 640)) -> Dict[str, torch.Tensor]:
        """计算蒸馏损失

        Args:
            student_outputs: 学生多尺度原始输出 [(B, H, W, 56), ...]
            teacher_results: 教师解码结果, 每张图一个 dict
                [{'bboxes': (N,4) xyxy, 'scores': (N,), 'keypoints': (N,17,3)}, ...]
            input_size: 输入尺寸 (H, W)

        Returns:
            loss_dict: {distill_loss, distill_box, distill_score, distill_kpt, distill_num_pos}
        """
        device = student_outputs[0].device
        batch_size = student_outputs[0].shape[0]
        input_h, input_w = input_size

        feat_sizes = [(out.shape[1], out.shape[2]) for out in student_outputs]

        # 保持梯度流到所有尺度，即使某些尺度无教师检测
        zero_loss = sum(out.sum() * 0 for out in student_outputs)
        total_box_loss = zero_loss.clone()
        total_score_loss = zero_loss.clone()
        total_kpt_loss = zero_loss.clone()
        num_pos = 0

        for b in range(batch_size):
            t_bboxes = teacher_results[b]['bboxes']     # (N, 4) xyxy
            t_scores = teacher_results[b]['scores']     # (N,)
            t_kpts = teacher_results[b]['keypoints']    # (N, 17, 3)

            if len(t_scores) == 0:
                continue

            for d in range(len(t_scores)):
                x1, y1, x2, y2 = t_bboxes[d]
                w_det = x2 - x1
                h_det = y2 - y1
                cx_det = (x1 + x2) / 2
                cy_det = (y1 + y2) / 2

                # 分配到尺度
                scale_idx = self._assign_to_scale(float(w_det), float(h_det))
                stride = self.strides[scale_idx]
                H, W = feat_sizes[scale_idx]

                # Grid cell
                gi = max(0, min(int(cx_det / stride), W - 1))
                gj = max(0, min(int(cy_det / stride), H - 1))

                # 学生在该 cell 的原始预测
                raw = student_outputs[scale_idx][b, gj, gi]  # (56,)

                # === 解码学生 box ===
                s_cx = (gi + raw[0].sigmoid()) * stride
                s_cy = (gj + raw[1].sigmoid()) * stride
                s_w = raw[2].clamp(-10, 10).exp() * stride
                s_h = raw[3].clamp(-10, 10).exp() * stride
                s_x1 = s_cx - s_w / 2
                s_y1 = s_cy - s_h / 2
                s_x2 = s_cx + s_w / 2
                s_y2 = s_cy + s_h / 2

                # 归一化到 [0, 1] 计算 loss
                s_bbox = torch.stack([s_x1 / input_w, s_y1 / input_h,
                                      s_x2 / input_w, s_y2 / input_h])
                t_bbox = torch.tensor(
                    [x1 / input_w, y1 / input_h, x2 / input_w, y2 / input_h],
                    dtype=torch.float32, device=device)

                total_box_loss = total_box_loss + F.smooth_l1_loss(s_bbox, t_bbox)

                # === Score: BCE with logits ===
                t_score_tensor = torch.tensor(
                    float(t_scores[d]), dtype=torch.float32, device=device)
                total_score_loss = total_score_loss + F.binary_cross_entropy_with_logits(
                    raw[4], t_score_tensor)

                # === 解码学生 keypoints ===
                s_kpts_raw = raw[5:].view(self.num_keypoints, 3)
                s_kpt_dx = s_kpts_raw[:, 0].sigmoid() * 2 - 1  # [-1, 1]
                s_kpt_dy = s_kpts_raw[:, 1].sigmoid() * 2 - 1
                s_kpt_x = (s_cx + s_kpt_dx * s_w) / input_w
                s_kpt_y = (s_cy + s_kpt_dy * s_h) / input_h
                s_kpt_xy = torch.stack([s_kpt_x, s_kpt_y], dim=-1)  # (17, 2)

                t_kpt_xy = torch.tensor(np.stack([
                    t_kpts[d, :, 0] / input_w,
                    t_kpts[d, :, 1] / input_h,
                ], axis=-1), dtype=torch.float32, device=device)  # (17, 2)

                total_kpt_loss = total_kpt_loss + F.smooth_l1_loss(s_kpt_xy, t_kpt_xy)

                num_pos += 1

        # 归一化
        num_pos = max(num_pos, 1)
        total_box_loss = self.box_weight * total_box_loss / num_pos
        total_score_loss = self.score_weight * total_score_loss / num_pos
        total_kpt_loss = self.kpt_weight * total_kpt_loss / num_pos

        distill_loss = total_box_loss + total_score_loss + total_kpt_loss

        return {
            'distill_loss': distill_loss,
            'distill_box': total_box_loss,
            'distill_score': total_score_loss,
            'distill_kpt': total_kpt_loss,
            'distill_num_pos': torch.tensor(num_pos, device=device),
        }
