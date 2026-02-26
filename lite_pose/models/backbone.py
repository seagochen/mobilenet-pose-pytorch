"""
MobileNetV3-Small Backbone

使用 timm 封装 MobileNetV3-Small，提取多尺度特征图用于 FPN。
输出 3 个尺度: channels=[24, 48, 576], strides=[8, 16, 32]
"""

import timm
import torch
import torch.nn as nn
from typing import List


class MobileNetV3Backbone(nn.Module):
    """MobileNetV3-Small 特征提取器

    Args:
        pretrained: 是否加载 ImageNet 预训练权重
        out_indices: 输出特征图的阶段索引
    """

    def __init__(self, pretrained: bool = True, out_indices: List[int] = [2, 3, 4]):
        super().__init__()

        self.backbone = timm.create_model(
            'mobilenetv3_small_100',
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        # 自动获取通道数和步长
        self.out_channels = self.backbone.feature_info.channels()
        self.strides = self.backbone.feature_info.reduction()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """提取多尺度特征图

        Args:
            x: 输入图像 (B, 3, H, W)

        Returns:
            features: 多尺度特征图列表
        """
        return self.backbone(x)
