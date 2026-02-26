"""
LitePose: MobileNetV3-Small + FPN + YOLOPoseHead

轻量级多人姿态估计模型，用于从 YOLOPose 教师模型蒸馏。
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from .backbone import MobileNetV3Backbone
from .pose_head import YOLOPoseHead


class LitePose(nn.Module):
    """轻量级多人姿态估计模型

    Args:
        num_keypoints: 关键点数量
        fpn_channels: FPN 输出通道数
        pretrained_backbone: 是否加载 backbone 预训练权重
    """

    def __init__(self,
                 num_keypoints: int = 17,
                 fpn_channels: int = 96,
                 pretrained_backbone: bool = True):
        super().__init__()

        self.num_keypoints = num_keypoints

        # Backbone
        self.backbone = MobileNetV3Backbone(pretrained=pretrained_backbone)

        # Detection Head (含 FPN)
        self.head = YOLOPoseHead(
            in_channels=self.backbone.out_channels,
            num_keypoints=num_keypoints,
            fpn_channels=fpn_channels,
            use_fpn=True,
            strides=self.backbone.strides,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """前向传播

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            outputs: 每个尺度的预测 [(B, H, W, 56), ...]
        """
        features = self.backbone(x)
        outputs = self.head(features)
        return outputs

    def decode(self,
               outputs: List[torch.Tensor],
               conf_thresh: float = 0.25,
               input_size: Tuple[int, int] = (640, 640)):
        """解码网络输出为检测结果，委托给 head"""
        return self.head.decode(outputs, conf_thresh, input_size)

    def freeze_backbone(self):
        """冻结 backbone 参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """解冻 backbone 参数"""
        for param in self.backbone.parameters():
            param.requires_grad = True
