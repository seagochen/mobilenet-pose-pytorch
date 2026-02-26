# 项目任务清单

> **项目名称**: mobilenet-pose-pytorch (MobileNetV3 蒸馏 YOLOPose 姿态估计)
> **创建日期**: 2026-02-26
> **最后更新**: 2026-02-26 (任务十完成 - 全部完成)

---

## 基本信息

| 项目 | 说明 |
|------|------|
| **相关工程位置** | `/home/cxt/models/convnext-pose-pytorch` |
| **开发环境** | `conda activate aigc` |
| **数据集** | `/home/cxt/datasets/coco_pose_yolo` |
| **YOLOPose 教师模型** | 用户提供 ONNX 文件，使用 onnxruntime 推理；**不要在 aigc 环境中安装 ultralytics** |

---

## 进行中的任务

_(暂无)_

---

## 未完成的任务

_(全部完成)_

---

## 已完成的任务（归档）

### [x] 任务十：实现 ONNX 导出脚本（2026-02-26）

- **完成内容**:
  1. [x] 实现 `scripts/export_onnx.py`
     - `LitePoseExport` 包装器: 3 尺度 (B,H,W,C) → 拼接为 (B,56,N)
     - `export_onnx()`: opset 13, dynamic_axes 支持 batch 维度
     - `simplify_onnx()`: 可选 onnx-simplifier 优化
     - `verify_onnx()`: PyTorch vs ONNX 输出比较
  2. [x] 验证通过:
     - 导出成功: 6.72 MB ONNX 文件
     - 输出格式: (B, 56, N) where N = 80*80 + 40*40 + 20*20 = 8000
     - 一致性: max diff 3.8e-5 < tolerance 1e-4

---

### [x] 任务九：实现推理脚本（2026-02-26）

- **完成内容**:
  1. [x] 实现 `scripts/detect.py`
     - `MultiPersonPoseEstimator` 类: 加载 checkpoint (支持 EMA/model/raw)、letterbox 预处理、推理解码 NMS、坐标缩放回原图
     - `process_image()` / `process_video()` / `process_camera()` / `process_directory()` 四种输入模式
     - 命令行参数: weights, source/camera, output, conf/iou/kpt thresh, device
  2. [x] Smoke test 通过: 加载 overfit checkpoint, 单图推理 55ms (CPU), 输出保存正常

---

### [x] 任务八：实现训练入口脚本（2026-02-26）

- **完成内容**:
  1. [x] 实现 `scripts/train_distill.py`
     - `main()`: 解析配置 → 创建 DistillTrainer → 开始训练
     - 支持 `--debug-nan` 标志开启异常检测
  2. [x] Overfit test 通过 (10 张图, 100 epochs):
     - Distill loss: 2.01 → 0.47 (下降 77%)
     - Total loss: 3.58 → 2.47 (下降 31%)
     - Beta 衰减正常: 1.00 → 0.25
     - 训练完成无错误

---

### [x] 任务七：实现蒸馏训练器（2026-02-26）

- **完成内容**:
  1. [x] 实现 `DistillTrainer` 类（`lite_pose/training/distill_trainer.py`，637 行）
     - `__init__()`: 构建学生模型、教师模型、GT 损失、蒸馏损失、optimizer(AdamW, backbone 0.1x lr)、scheduler、EMA、DataLoader
     - `train_one_epoch()`: 教师推理(no_grad) → 学生 forward → GT loss + beta*distill loss → backward → step; NaN 检测、梯度裁剪、EMA 更新
     - `validate()`: 仅学生模型（或 EMA），YOLOPoseEvaluator 计算 PCK/AP/OKS
     - `train()`: warmup → freeze backbone 阶段 → unfreeze（重建 optimizer/scheduler）→ 正常训练; 每 N epoch 验证/保存
     - `save_checkpoint()` / `_resume_training()` / `_load_weights()`: checkpoint 保存/恢复
     - `_get_distill_beta()`: beta 线性衰减 (1.0→0.25)
  2. [x] tensorboard 可选导入 (try/except)，未安装时自动跳过
  3. [x] 验证通过:
     - Import 成功，15 个方法齐全
     - Beta 衰减公式: epoch0=1.0, epoch150=0.6237, epoch299=0.25

---

### [x] 任务六：实现蒸馏训练配置系统（2026-02-26）

- **完成内容**:
  1. [x] 实现 `lite_pose/training/config.py`
     - `get_parser()`: 9 个参数组 (Data, Student, Teacher, Distillation, Training, Advanced, Output, Resume, Misc)
     - `build_config()`: 返回嵌套 dict (data, model, teacher, distill, training, output, ...)
     - 所有默认值与 TASKS.md 规格一致
  2. [x] 验证通过: 默认值解析、自定义值覆盖均正确

---

### [x] 任务五：实现响应蒸馏损失（2026-02-26）

- **完成内容**:
  1. [x] 实现 `DistillationLoss` 类（`lite_pose/models/distill_loss.py`）
     - 尺度分配: `max(w,h)/stride` 最接近 4 的尺度
     - Grid cell 定位: `(floor(cx/stride), floor(cy/stride))`
     - 解码学生原始输出后与教师像素坐标比较
     - Box: SmoothL1 (归一化坐标), 权重 1.0
     - Score: BCE with logits (学生 obj logit vs 教师 confidence), 权重 1.0
     - Keypoint: SmoothL1 (归一化坐标), 权重 2.0
     - 教师无检测时损失为 0，梯度流保持 (`sum(out.sum()*0)`)
  2. [x] 更新 `models/__init__.py` 导出 `DistillationLoss`
  3. [x] 单元测试全部通过:
     - mock 数据: loss 有限 (box=0.009, score=0.597, kpt=0.174)
     - 梯度正确流向所有 3 个尺度的学生输出
     - 空教师: loss=0，梯度图保持
     - 真实 LitePose 模型: forward + backward 正常

---

### [x] 任务四：实现 YOLOPose 教师模型封装 ONNX（2026-02-26）

- **完成内容**:
  1. [x] 实现 `TeacherYOLOPose` 类（`lite_pose/models/teacher.py`）
     - 通过 `onnxruntime.InferenceSession` 加载 ONNX 文件
     - ONNX 输出格式: `[1, 56, 8400]` → 解码为 bbox(cx,cy,w,h→xyxy) + score + keypoints(17×3)
     - 自带 numpy NMS 实现 (`_nms_numpy`)
     - `predict_batch()` 支持 torch.Tensor 和 np.ndarray 输入
  2. [x] 更新 `models/__init__.py` 导出 `TeacherYOLOPose`
  3. [x] Smoke test 通过:
     - 单人图 (000000000785.jpg): 1 detection, score=0.914
     - 多人图 (000000001000.jpg): 14 detections
     - 空白图: 0 detections
     - torch.Tensor 输入与 numpy 输入结果一致
     - batch 推理正常

---

### [x] 任务三：组装学生模型 LitePose（2026-02-26）

- **完成内容**:
  1. [x] 实现 `LitePose` 类（`lite_pose/models/lite_pose.py`）
  2. [x] 更新 `models/__init__.py` 导出 `MobileNetV3Backbone` 和 `LitePose`
  3. [x] 输出形状验证通过：`(B,80,80,56)`, `(B,40,40,56)`, `(B,20,20,56)`
  4. [x] 总参数量 1.75M（backbone 0.93M + head 0.83M）
  5. [x] freeze/unfreeze_backbone 方法验证通过

---

### [x] 任务二：实现 MobileNetV3-Small Backbone（2026-02-26）

- **完成内容**:
  1. [x] 实现 `MobileNetV3Backbone` 类（`lite_pose/models/backbone.py`）
  2. [x] Smoke test 通过：输出形状 `(1,24,80,80)`, `(1,48,40,40)`, `(1,576,20,20)`
  3. [x] 实际参数量 0.93M

---

### [x] 任务一：工程骨架搭建与文件复用（2026-02-26）

- **完成内容**:
  1. [x] 创建 8 个 `__init__.py` 文件（lite_pose, models, training, utils, callbacks, data, metrics, visualization）
  2. [x] 创建 `requirements.txt`（含 onnxruntime，不含 ultralytics）
  3. [x] 创建 `setup.py`
  4. [x] 从 `convnext-pose-pytorch` 复制 9 个文件（pose_head, yolo_loss, dataset, path_utils, ema, lr_scheduler, metrics, yolo_utils, plots）
  5. [x] 所有复制文件均使用相对 import，无需修改包名
  6. [x] 验证通过：`import lite_pose` 及全部子模块 import 均无报错
