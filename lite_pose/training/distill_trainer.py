"""
蒸馏训练器

联合 GT 损失和教师蒸馏损失训练 LitePose 学生模型。
核心区别于普通训练: 每批额外跑教师 ONNX 推理 + 蒸馏损失。

组合损失: total = gt_loss + beta * distill_loss
beta 从 weight_max 线性衰减到 weight_min (早期多学教师, 后期回归 GT)
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from ..models import LitePose, TeacherYOLOPose
from ..models.yolo_loss import YOLOPoseLoss, build_yolo_loss
from ..models.distill_loss import DistillationLoss
from ..utils.data import YOLOPoseDataset, collate_fn_yolo
from ..utils.callbacks import build_scheduler, ModelEMA
from ..utils.yolo_utils import YOLOPoseEvaluator, postprocess


class DistillTrainer:
    """蒸馏训练器

    Args:
        config: 配置字典 (由 build_config 生成)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))

        # 设置输出目录
        self.output_dir = self._setup_output_dir()

        # 设置日志
        self.logger = self._setup_logging()

        # 设置 tensorboard (可选)
        if SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard'))
        else:
            self.writer = None

        # 构建组件
        self.model = self._build_model()
        self.teacher = self._build_teacher()
        self.gt_criterion = self._build_gt_criterion()
        self.distill_criterion = self._build_distill_criterion()
        self.optimizer = self._build_optimizer()
        self.train_loader, self.val_loader = self._build_dataloaders()
        self.scheduler = self._build_scheduler()

        # EMA
        self.ema = self._build_ema() if config['training']['ema'] else None

        # 训练状态
        self.start_epoch = 0
        self.best_metric = float('inf')
        self.global_step = 0

        # 输入尺寸
        self.input_size = tuple(config['data']['input_size'])

        # 恢复训练
        if config.get('resume'):
            self._resume_training(config['resume'])
        elif config.get('weights'):
            self._load_weights(config['weights'])

    # ===== Setup =====

    def _setup_output_dir(self) -> Path:
        """设置输出目录"""
        output_cfg = self.config['output']
        base_dir = Path(output_cfg['dir'])
        name = output_cfg['project']

        # 恢复训练时使用原目录
        if self.config.get('resume'):
            resume_path = Path(self.config['resume'])
            if resume_path.is_file():
                return resume_path.parent.parent

        output_dir = base_dir / name
        idx = 1
        while output_dir.exists():
            output_dir = base_dir / f"{name}{idx}"
            idx += 1

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'weights').mkdir(exist_ok=True)

        return output_dir

    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('distill_trainer')
        logger.setLevel(logging.INFO)
        logger.handlers = []

        fh = logging.FileHandler(self.output_dir / 'train.log')
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    # ===== Build components =====

    def _build_model(self) -> nn.Module:
        """构建学生模型"""
        model_cfg = self.config['model']

        model = LitePose(
            num_keypoints=model_cfg['num_keypoints'],
            fpn_channels=model_cfg['fpn_channels'],
            pretrained_backbone=model_cfg['backbone_pretrained'],
        )

        if self.config['training']['freeze_backbone']:
            model.freeze_backbone()

        model = model.to(self.device)

        params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        self.logger.info(f"Student: LitePose (FPN={model_cfg['fpn_channels']})")
        self.logger.info(f"Parameters: {params:.2f}M (trainable: {trainable:.2f}M)")

        return model

    def _build_teacher(self) -> TeacherYOLOPose:
        """构建教师模型 (ONNX)"""
        teacher_cfg = self.config['teacher']
        teacher = TeacherYOLOPose(teacher_cfg['onnx_path'])
        self.logger.info(f"Teacher: {teacher_cfg['onnx_path']}")
        self.logger.info(f"Teacher conf thresh: {teacher_cfg['conf_thresh']}")
        return teacher

    def _build_gt_criterion(self) -> nn.Module:
        """构建 GT 损失"""
        strides = self.model.head.strides
        return build_yolo_loss(
            num_keypoints=self.config['model']['num_keypoints'],
            strides=strides,
        ).to(self.device)

    def _build_distill_criterion(self) -> DistillationLoss:
        """构建蒸馏损失"""
        distill_cfg = self.config['distill']
        strides = self.model.head.strides
        return DistillationLoss(
            num_keypoints=self.config['model']['num_keypoints'],
            strides=strides,
            box_weight=distill_cfg['box_weight'],
            score_weight=distill_cfg['score_weight'],
            kpt_weight=distill_cfg['kpt_weight'],
        ).to(self.device)

    def _build_optimizer(self) -> optim.Optimizer:
        """构建优化器 (backbone 0.1x lr)"""
        train_cfg = self.config['training']

        backbone_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {'params': backbone_params, 'lr': train_cfg['lr'] * 0.1},
            {'params': head_params, 'lr': train_cfg['lr']},
        ]

        return optim.AdamW(
            param_groups,
            lr=train_cfg['lr'],
            weight_decay=train_cfg['weight_decay'],
        )

    def _build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """构建数据加载器"""
        data_cfg = self.config['data']
        train_cfg = self.config['training']

        train_dataset = YOLOPoseDataset(
            data_yaml=data_cfg['yaml_path'],
            split='train',
            input_size=data_cfg['input_size'],
            max_persons=20,
            augment=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=True,
            num_workers=data_cfg['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn_yolo,
            drop_last=True,
        )

        val_dataset = YOLOPoseDataset(
            data_yaml=data_cfg['yaml_path'],
            split='val',
            input_size=data_cfg['input_size'],
            max_persons=20,
            augment=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=False,
            num_workers=data_cfg['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn_yolo,
        )

        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Val samples: {len(val_dataset)}")
        return train_loader, val_loader

    def _build_scheduler(self):
        """构建学习率调度器"""
        train_cfg = self.config['training']
        return build_scheduler(
            optimizer=self.optimizer,
            scheduler_type=train_cfg['lr_scheduler'],
            epochs=train_cfg['epochs'],
            steps_per_epoch=len(self.train_loader),
            warmup_epochs=train_cfg['warmup_epochs'],
        )

    def _build_ema(self) -> Optional[ModelEMA]:
        """构建 EMA"""
        return ModelEMA(
            self.model,
            decay=self.config['training']['ema_decay'],
        )

    # ===== Training =====

    def _get_distill_beta(self, epoch: int) -> float:
        """计算蒸馏权重 beta (线性衰减)

        beta = weight_max - (weight_max - weight_min) * epoch / total_epochs
        """
        distill_cfg = self.config['distill']
        total_epochs = self.config['training']['epochs']
        beta_max = distill_cfg['weight_max']
        beta_min = distill_cfg['weight_min']
        return beta_max - (beta_max - beta_min) * epoch / max(total_epochs - 1, 1)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        train_cfg = self.config['training']
        teacher_cfg = self.config['teacher']
        beta = self._get_distill_beta(epoch)

        # 累计器
        sum_loss = 0.0
        sum_gt = 0.0
        sum_distill = 0.0
        sum_gt_box = 0.0
        sum_gt_obj = 0.0
        sum_gt_kpt = 0.0
        sum_d_box = 0.0
        sum_d_score = 0.0
        sum_d_kpt = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{train_cfg['epochs']}",
            unit='batch',
            ncols=160,
            leave=True,
        )

        for step, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets_device = {
                'bboxes': targets['bboxes'].to(self.device),
                'keypoints': targets['keypoints'].to(self.device),
                'num_persons': targets['num_persons'].to(self.device),
            }

            # --- 教师推理 (no grad, CPU/ONNX) ---
            teacher_results = self.teacher.predict_batch(
                images, conf_thresh=teacher_cfg['conf_thresh'])

            # --- 学生 forward ---
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # --- GT loss ---
            gt_loss_dict = self.gt_criterion(outputs, targets_device, self.input_size)
            gt_loss = gt_loss_dict['loss']

            # --- Distill loss ---
            distill_loss_dict = self.distill_criterion(
                outputs, teacher_results, self.input_size)
            distill_loss = distill_loss_dict['distill_loss']

            # --- Combined loss ---
            total_loss = gt_loss + beta * distill_loss

            # NaN 检测
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                self.logger.warning(
                    f"[Step {step}] Loss NaN/Inf: gt={gt_loss.item():.4f}, "
                    f"distill={distill_loss.item():.4f}")
                self.optimizer.zero_grad()
                continue

            total_loss.backward()

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=train_cfg['grad_clip'])
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                self.logger.warning(f"[Step {step}] Gradient NaN/Inf, skipping")
                self.optimizer.zero_grad()
                continue

            self.optimizer.step()
            self.scheduler.step()

            if self.ema:
                self.ema.update(self.model)

            # 累计
            sum_loss += total_loss.item()
            sum_gt += gt_loss.item()
            sum_distill += distill_loss.item()
            sum_gt_box += gt_loss_dict['box_loss'].item()
            sum_gt_obj += gt_loss_dict['obj_loss'].item()
            sum_gt_kpt += gt_loss_dict['kpt_loss'].item()
            sum_d_box += distill_loss_dict['distill_box'].item()
            sum_d_score += distill_loss_dict['distill_score'].item()
            sum_d_kpt += distill_loss_dict['distill_kpt'].item()

            self.global_step += 1

            # 进度条
            lr = self.optimizer.param_groups[1]['lr']
            pbar.set_postfix({
                'loss': f'{total_loss.item():.3f}',
                'gt': f'{gt_loss.item():.3f}',
                'dist': f'{distill_loss.item():.3f}',
                'β': f'{beta:.2f}',
                'lr': f'{lr:.1e}',
            })

        # 平均
        n = max(num_batches, 1)
        losses = {
            'loss': sum_loss / n,
            'gt_loss': sum_gt / n,
            'distill_loss': sum_distill / n,
            'gt_box': sum_gt_box / n,
            'gt_obj': sum_gt_obj / n,
            'gt_kpt': sum_gt_kpt / n,
            'distill_box': sum_d_box / n,
            'distill_score': sum_d_score / n,
            'distill_kpt': sum_d_kpt / n,
            'beta': beta,
        }

        self.logger.info(
            f'Epoch [{epoch+1}/{train_cfg["epochs"]}] '
            f'Loss: {losses["loss"]:.4f} '
            f'(GT: {losses["gt_loss"]:.4f}, Distill: {losses["distill_loss"]:.4f}, β={beta:.2f}) '
            f'LR: {self.optimizer.param_groups[1]["lr"]:.6f}'
        )

        # Tensorboard
        if self.writer:
            self.writer.add_scalar('train/loss', losses['loss'], epoch)
            self.writer.add_scalar('train/gt_loss', losses['gt_loss'], epoch)
            self.writer.add_scalar('train/distill_loss', losses['distill_loss'], epoch)
            self.writer.add_scalar('train/gt_box', losses['gt_box'], epoch)
            self.writer.add_scalar('train/gt_obj', losses['gt_obj'], epoch)
            self.writer.add_scalar('train/gt_kpt', losses['gt_kpt'], epoch)
            self.writer.add_scalar('train/distill_box', losses['distill_box'], epoch)
            self.writer.add_scalar('train/distill_score', losses['distill_score'], epoch)
            self.writer.add_scalar('train/distill_kpt', losses['distill_kpt'], epoch)
            self.writer.add_scalar('train/beta', beta, epoch)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[1]['lr'], epoch)

        return losses

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证 (仅学生模型, 无教师)"""
        model = self.ema.ema if self.ema else self.model
        model.eval()

        total_loss = 0
        total_box_loss = 0
        total_obj_loss = 0
        total_kpt_loss = 0
        num_batches = len(self.val_loader)

        evaluator = YOLOPoseEvaluator(
            num_keypoints=self.config['model']['num_keypoints'],
            iou_thresh=0.5,
            pck_thresh=0.2,
        )

        pbar = tqdm(
            self.val_loader, desc="Validating",
            unit='batch', ncols=120, leave=False)

        for images, targets in pbar:
            images = images.to(self.device)
            targets_device = {
                'bboxes': targets['bboxes'].to(self.device),
                'keypoints': targets['keypoints'].to(self.device),
                'num_persons': targets['num_persons'].to(self.device),
            }
            batch_size = images.shape[0]

            outputs = model(images)
            loss_dict = self.gt_criterion(outputs, targets_device, self.input_size)

            total_loss += loss_dict['loss'].item()
            total_box_loss += loss_dict['box_loss'].item()
            total_obj_loss += loss_dict['obj_loss'].item()
            total_kpt_loss += loss_dict['kpt_loss'].item()

            # 逐图像评估
            for b in range(batch_size):
                n_persons = targets['num_persons'][b].item()

                single_outputs = [out[b:b+1] for out in outputs]
                pred_bboxes, pred_scores, pred_keypoints = model.decode(
                    single_outputs, conf_thresh=0.25, input_size=self.input_size)

                if pred_bboxes.numel() > 0:
                    pred_bboxes, pred_scores, pred_keypoints = postprocess(
                        pred_bboxes, pred_scores, pred_keypoints,
                        conf_thresh=0.25, iou_thresh=0.65)

                if n_persons > 0:
                    img_h, img_w = self.input_size
                    gt_bboxes_norm = targets['bboxes'][b, :n_persons]
                    gt_kpts_encoded = targets['keypoints'][b, :n_persons]

                    # bbox: norm cxcywh -> pixel xyxy
                    cx_p = gt_bboxes_norm[:, 0] * img_w
                    cy_p = gt_bboxes_norm[:, 1] * img_h
                    w_p = gt_bboxes_norm[:, 2] * img_w
                    h_p = gt_bboxes_norm[:, 3] * img_h
                    gt_boxes_xyxy = torch.stack([
                        cx_p - w_p / 2, cy_p - h_p / 2,
                        cx_p + w_p / 2, cy_p + h_p / 2,
                    ], dim=-1).cpu().numpy()

                    # keypoints: relative -> pixel
                    gt_kpts_abs = gt_kpts_encoded.clone()
                    bcx = gt_bboxes_norm[:, 0:1]
                    bcy = gt_bboxes_norm[:, 1:2]
                    bw = gt_bboxes_norm[:, 2:3]
                    bh = gt_bboxes_norm[:, 3:4]
                    gt_kpts_abs[:, :, 0] = (bcx + gt_kpts_encoded[:, :, 0] * bw) * img_w
                    gt_kpts_abs[:, :, 1] = (bcy + gt_kpts_encoded[:, :, 1] * bh) * img_h
                    gt_kpts_abs = gt_kpts_abs.cpu().numpy()

                    evaluator.update(
                        pred_boxes=pred_bboxes.cpu().numpy(),
                        pred_keypoints=pred_keypoints.cpu().numpy(),
                        gt_boxes=gt_boxes_xyxy,
                        gt_keypoints=gt_kpts_abs,
                    )

        eval_metrics = evaluator.compute()
        n = max(num_batches, 1)
        metrics = {
            'loss': total_loss / n,
            'box_loss': total_box_loss / n,
            'obj_loss': total_obj_loss / n,
            'kpt_loss': total_kpt_loss / n,
            'PCK@0.2': eval_metrics['PCK@0.2'],
            'AP50': eval_metrics['AP50'],
            'OKS': eval_metrics['OKS'],
        }

        self.logger.info(
            f'Val - Loss: {metrics["loss"]:.4f} '
            f'(box: {metrics["box_loss"]:.4f}, obj: {metrics["obj_loss"]:.4f}, '
            f'kpt: {metrics["kpt_loss"]:.4f}) '
            f'| PCK@0.2: {metrics["PCK@0.2"]:.4f}, '
            f'AP50: {metrics["AP50"]:.4f}, OKS: {metrics["OKS"]:.4f}'
        )

        # Tensorboard
        if self.writer:
            self.writer.add_scalar('val/loss', metrics['loss'], epoch)
            self.writer.add_scalar('val/PCK@0.2', metrics['PCK@0.2'], epoch)
            self.writer.add_scalar('val/AP50', metrics['AP50'], epoch)
            self.writer.add_scalar('val/OKS', metrics['OKS'], epoch)

        return metrics

    # ===== Checkpoint =====

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'global_step': self.global_step,
            'config': self.config,
        }
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()

        torch.save(checkpoint, self.output_dir / 'weights' / 'last.pt')

        if is_best:
            torch.save(checkpoint, self.output_dir / 'weights' / 'best.pt')
            self.logger.info(f'Saved best model (loss={self.best_metric:.4f})')

    def _resume_training(self, checkpoint_path: str):
        """恢复训练"""
        self.logger.info(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_metric = ckpt.get('best_metric', float('inf'))
        self.global_step = ckpt.get('global_step', 0)

        if self.ema and 'ema' in ckpt:
            self.ema.load_state_dict(ckpt['ema'])

        self.logger.info(f"Resumed from epoch {self.start_epoch}")

    def _load_weights(self, weights_path: str):
        """加载模型权重 (不恢复优化器)"""
        self.logger.info(f"Loading weights from {weights_path}")
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        state = ckpt['model'] if 'model' in ckpt else ckpt
        self.model.load_state_dict(state)

    # ===== Main loop =====

    def _log_config(self):
        """记录训练配置"""
        self.logger.info("=" * 60)
        self.logger.info("LitePose Distillation Training")
        self.logger.info("=" * 60)

        cfg = self.config
        self.logger.info(f"Data: {cfg['data']['yaml_path']}")
        self.logger.info(f"Input size: {cfg['data']['input_size']}")
        self.logger.info(f"Student: FPN={cfg['model']['fpn_channels']}")
        self.logger.info(f"Teacher: {cfg['teacher']['onnx_path']} (conf={cfg['teacher']['conf_thresh']})")

        d = cfg['distill']
        self.logger.info(f"Distill: beta {d['weight_max']}→{d['weight_min']}, "
                         f"box={d['box_weight']}, score={d['score_weight']}, kpt={d['kpt_weight']}")

        t = cfg['training']
        self.logger.info(f"Epochs: {t['epochs']}, BS: {t['batch_size']}, LR: {t['lr']}")
        self.logger.info(f"Scheduler: {t['lr_scheduler']}, Warmup: {t['warmup_epochs']}")
        self.logger.info(f"EMA: {t['ema']}, Freeze: {t['freeze_backbone']} ({t['freeze_epochs']}ep)")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info("=" * 60)

    def train(self):
        """完整训练流程"""
        train_cfg = self.config['training']
        output_cfg = self.config['output']

        self._log_config()
        self.logger.info(f"Starting training for {train_cfg['epochs']} epochs")

        for epoch in range(self.start_epoch, train_cfg['epochs']):
            # 解冻 backbone
            if (train_cfg['freeze_backbone']
                    and epoch == train_cfg['freeze_epochs']):
                self.model.unfreeze_backbone()
                # 重建 optimizer 以包含 backbone 参数
                self.optimizer = self._build_optimizer()
                self.scheduler = self._build_scheduler()
                self.logger.info(f'Unfreezing backbone at epoch {epoch}')

            # 训练
            train_losses = self.train_one_epoch(epoch)

            # 验证
            if (epoch + 1) % output_cfg['val_interval'] == 0:
                metrics = self.validate(epoch)

                is_best = metrics['loss'] < self.best_metric
                if is_best:
                    self.best_metric = metrics['loss']
                self.save_checkpoint(epoch, is_best)

            elif (epoch + 1) % output_cfg['save_interval'] == 0:
                self.save_checkpoint(epoch)

        # 最终保存
        self.save_checkpoint(train_cfg['epochs'] - 1)
        if self.writer:
            self.writer.close()
        self.logger.info(f'Training completed. Best val loss: {self.best_metric:.4f}')
