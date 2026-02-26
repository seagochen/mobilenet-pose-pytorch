# 项目任务清单

> **项目名称**: mobilenet-pose-pytorch (MobileNetV3 蒸馏 YOLOPose 姿态估计)
> **创建日期**: 2026-02-26
> **最后更新**: 2026-02-26 (任务一完成)

---

## 基本信息

| 项目 | 说明 |
|------|------|
| **相关工程位置** | `/home/cxt/models/convnext-pose-pytorch` |
| **开发环境** | `conda activate aigc` |
| **YOLOPose 教师模型** | 用户提供 ONNX 文件，使用 onnxruntime 推理；**不要在 aigc 环境中安装 ultralytics** |

---

## 进行中的任务

_(暂无)_

---

## 未完成的任务

### [ ] 任务二：实现 MobileNetV3-Small Backbone

- **需求描述**: 使用 timm 库封装 MobileNetV3-Small，提取 3 个尺度的特征图用于 FPN
- **涉及文件**:
  - `lite_pose/models/backbone.py` - 新建，MobileNetV3-Small 封装
- **操作步骤**:
  1. [ ] 实现 `MobileNetV3Backbone` 类
     - 使用 `timm.create_model('mobilenetv3_small_100', pretrained=True, features_only=True, out_indices=[2,3,4])`
     - 输出 3 个尺度: channels=[24, 48, 576], strides=[8, 16, 32]
     - 通过 `backbone.feature_info.channels()` 和 `.reduction()` 自动获取通道数和步长
  2. [ ] Smoke test: 传入 `(1, 3, 640, 640)` 张量，验证 3 个输出的形状分别为 `(1, 24, 80, 80)`, `(1, 48, 40, 40)`, `(1, 576, 20, 20)`
- **上下文备注**:
  - timm 的 `features_only=True` 模式自动去除分类头，只保留特征提取部分
  - MobileNetV3-Small backbone 约 1.5M 参数（不含分类头）
  - ImageNet 预训练权重通过 `pretrained=True` 自动下载

---

### [ ] 任务三：组装学生模型 LitePose

- **需求描述**: 将 MobileNetV3-Small backbone + FPN + YOLOPoseHead 组装为完整的学生模型
- **涉及文件**:
  - `lite_pose/models/lite_pose.py` - 新建，完整学生模型
  - `lite_pose/models/__init__.py` - 更新导出
- **操作步骤**:
  1. [ ] 实现 `LitePose` 类
     - 构造: `MobileNetV3Backbone` + `YOLOPoseHead(in_channels=[24,48,576], fpn_channels=96, strides=[8,16,32])`
     - forward: backbone 提取特征 → head 检测 → 返回 3 尺度输出
     - decode: 委托给 head.decode()
     - freeze_backbone / unfreeze_backbone 方法
  2. [ ] 验证输出形状: `[(B, 80, 80, 56), (B, 40, 40, 56), (B, 20, 20, 56)]`，其中 56 = 4(bbox) + 1(obj) + 17×3(kpts)
  3. [ ] 验证总参数量约 2.3M
- **上下文备注**:
  - fpn_channels=96（从原工程的 256 缩减，匹配轻量 backbone）
  - 3 个检测尺度（原工程 4 个，去掉 stride=64）
  - 17 个 COCO 关键点

---

### [ ] 任务四：实现 YOLOPose 教师模型封装（ONNX）

- **需求描述**: 通过 onnxruntime 加载用户提供的 YOLOPose ONNX 文件作为知识蒸馏的教师（**不使用 ultralytics**）
- **涉及文件**:
  - `lite_pose/models/teacher.py` - 新建，教师模型封装
- **操作步骤**:
  1. [ ] 实现 `TeacherYOLOPose` 类
     - 通过 `onnxruntime.InferenceSession(onnx_path)` 加载用户提供的 ONNX 文件
     - 无需梯度，纯推理模式
     - `predict_batch(images, conf_thresh=0.1)` 方法:
       - 输入: `(B, 3, H, W)` 归一化张量
       - 需自行实现 ONNX 原始输出的解码和 NMS 后处理
       - 输出: 每张图的 `{'bboxes': (N,4) xyxy, 'scores': (N,), 'keypoints': (N,17,3)}` 像素坐标
     - 使用低置信度阈值 0.1 获取更多软标签
  2. [ ] Smoke test: 对 `/home/cxt/datasets/coco_pose_yolo/images/val/` 中的样本图推理，验证输出格式和合理性
- **上下文备注**:
  - 使用 onnxruntime 推理，不依赖 ultralytics
  - 需要根据 ONNX 模型的输出格式实现解码逻辑（bbox 解码 + keypoint 解码 + NMS）
  - 教师推理在 CPU/GPU 上通过 onnxruntime 完成，不占 PyTorch 显存

---

### [ ] 任务五：实现响应蒸馏损失

- **需求描述**: 实现 response-based 知识蒸馏损失，匹配教师和学生的解码输出
- **涉及文件**:
  - `lite_pose/models/distill_loss.py` - 新建，蒸馏损失模块
- **操作步骤**:
  1. [ ] 实现教师检测到学生 grid cell 的分配逻辑
     - 根据教师 bbox 中心 `(cx, cy)` 确定对应的学生尺度和 grid cell
     - 尺度选择: `max(w, h) / stride` 在 2~8 之间的尺度
     - grid cell: `(floor(cx/stride), floor(cy/stride))`
  2. [ ] 实现 `DistillationLoss` 类
     - Box 蒸馏: SmoothL1 (学生 vs 教师 bbox)，权重 1.0
     - Score 蒸馏: BCE (学生 objectness vs 教师 confidence)，权重 1.0
     - Keypoint 蒸馏: SmoothL1 (学生 vs 教师 keypoints)，权重 2.0
     - 仅对教师有检测结果的位置计算损失
  3. [ ] 单元测试: mock 数据验证损失值有限、梯度正确流向学生
- **上下文备注**:
  - 分配逻辑参考原工程 `pose_head.py` 中 `get_targets_for_scale()` (将 GT 分配到 grid cell 的反向过程)
  - 教师无检测时该图蒸馏损失为 0（仍有 GT 损失）
  - keypoint 权重设为 2.0 因为姿态估计是主要目标

---

### [ ] 任务六：实现蒸馏训练配置系统

- **需求描述**: 基于 argparse 创建蒸馏训练的配置系统
- **涉及文件**:
  - `lite_pose/training/config.py` - 新建，配置解析
- **操作步骤**:
  1. [ ] 实现 argparse 配置，包含以下参数组:
     - **数据**: `--data`(yaml路径), `--img-size`(640), `--num-keypoints`(17)
     - **学生模型**: `--fpn-channels`(96), `--backbone-pretrained`(True)
     - **教师模型**: `--teacher`(ONNX文件路径), `--teacher-conf`(0.1)
     - **蒸馏**: `--distill-weight-max`(1.0), `--distill-weight-min`(0.25), `--distill-box-weight`(1.0), `--distill-score-weight`(1.0), `--distill-kpt-weight`(2.0)
     - **训练**: `--epochs`(300), `--batch-size`(32), `--lr`(5e-4), `--weight-decay`(0.01), `--lr-scheduler`(cosine), `--warmup-epochs`(10)
     - **高级**: `--ema`(True), `--ema-decay`(0.9999), `--freeze-backbone`(True), `--freeze-epochs`(5), `--grad-clip`(10.0)
     - **输出**: `--output-dir`(runs/train), `--project`(exp)
  2. [ ] 返回 config 字典供 trainer 使用
- **上下文备注**:
  - 参考原工程 `convnext_pose/training/config.py` 的模式
  - 学习率 5e-4（比原工程 1e-3 低，适合小模型蒸馏）
  - 训练 300 epochs（蒸馏需要更长训练时间）

---

### [ ] 任务七：实现蒸馏训练器

- **需求描述**: 实现完整的蒸馏训练循环，联合 GT 损失和蒸馏损失
- **涉及文件**:
  - `lite_pose/training/distill_trainer.py` - 新建，蒸馏训练器
- **操作步骤**:
  1. [ ] 实现 `DistillTrainer.__init__()`:
     - 构建学生模型 (LitePose)
     - 加载教师模型 (TeacherYOLOPose)
     - 构建 GT 损失 (YOLOPoseLoss) 和蒸馏损失 (DistillationLoss)
     - 构建 optimizer (AdamW，backbone 0.1x lr)、scheduler、EMA
     - 构建 DataLoader (复用 YOLOPoseDataset)
  2. [ ] 实现 `train_one_epoch()`:
     - 每批: 教师推理(no_grad) → 学生 forward → GT loss + distill loss → backward → step
     - 蒸馏权重衰减: `beta = beta_max - (beta_max - beta_min) * epoch / total_epochs`
     - NaN 检测、梯度裁剪、EMA 更新
  3. [ ] 实现 `validate()`:
     - 仅用学生模型（无教师）
     - 计算 PCK@0.2、AP50、OKS 指标
     - 复用 `YOLOPoseEvaluator`
  4. [ ] 实现 `train()` 主循环:
     - warmup → freeze backbone 阶段 → unfreeze → 正常训练
     - 每 N epoch 验证、保存 checkpoint
     - best model 跟踪
  5. [ ] 实现 checkpoint 保存/恢复
  6. [ ] 日志系统 (logging + tensorboard)
- **上下文备注**:
  - 重度参考原工程 `convnext_pose/training/trainer.py` 的 `YOLOPoseTrainer` 模式
  - 核心区别: 每批额外跑教师推理 + 蒸馏损失
  - 组合损失: `total = 1.0 * gt_loss + beta * distill_loss`
  - beta 从 1.0 线性衰减到 0.25（早期多学教师，后期回归 GT）

---

### [ ] 任务八：实现训练入口脚本

- **需求描述**: 创建蒸馏训练的命令行入口
- **涉及文件**:
  - `scripts/train_distill.py` - 新建，训练入口
- **操作步骤**:
  1. [ ] 实现 main 函数: 解析配置 → 创建 trainer → 开始训练
  2. [ ] Overfit test: 用 10 张图跑 100 epochs，验证 GT loss 和 distill loss 均下降
- **上下文备注**:
  - 训练命令:
    ```bash
    python scripts/train_distill.py \
      --data /home/cxt/datasets/coco_pose_yolo/data.yaml \
      --teacher /path/to/yolopose.onnx \
      --epochs 300 --batch-size 32 --lr 5e-4 --ema
    ```

---

### [ ] 任务九：实现推理脚本

- **需求描述**: 创建单图/视频推理脚本
- **涉及文件**:
  - `scripts/detect.py` - 新建，推理脚本
- **操作步骤**:
  1. [ ] 加载训练好的学生模型 checkpoint
  2. [ ] 图像预处理 (letterbox resize, normalize)
  3. [ ] 模型推理 → 解码 → NMS → 绘制骨架
  4. [ ] 支持单图、目录、视频输入
- **上下文备注**:
  - 参考原工程 `scripts/detect.py`

---

### [ ] 任务十：实现 ONNX 导出脚本

- **需求描述**: 将学生模型导出为 ONNX 格式，用于 Jetson TensorRT 部署
- **涉及文件**:
  - `scripts/export_onnx.py` - 新建，ONNX 导出
- **操作步骤**:
  1. [ ] 加载 checkpoint → 导出 ONNX (opset 13)
  2. [ ] 使用 onnx-simplifier 优化
  3. [ ] 验证 ONNX 输出与 PyTorch 一致（tolerance 1e-5）
  4. [ ] 输出 3 个尺度的原始预测，解码在推理端处理
- **上下文备注**:
  - Jetson 上使用 `trtexec` 将 ONNX 转为 TensorRT engine
  - MobileNetV3-Small + FP16 预计 Jetson Nano 上 30-50ms 延迟
  - dynamic_axes 支持 batch 维度

---

## 已完成的任务（归档）

### [x] 任务一：工程骨架搭建与文件复用（2026-02-26）

- **完成内容**:
  1. [x] 创建 8 个 `__init__.py` 文件（lite_pose, models, training, utils, callbacks, data, metrics, visualization）
  2. [x] 创建 `requirements.txt`（含 onnxruntime，不含 ultralytics）
  3. [x] 创建 `setup.py`
  4. [x] 从 `convnext-pose-pytorch` 复制 9 个文件（pose_head, yolo_loss, dataset, path_utils, ema, lr_scheduler, metrics, yolo_utils, plots）
  5. [x] 所有复制文件均使用相对 import，无需修改包名
  6. [x] 验证通过：`import lite_pose` 及全部子模块 import 均无报错
