"""
蒸馏训练配置系统

支持两种使用方式:
  1. YAML 配置文件 + CLI 覆盖 (推荐):
     python scripts/train_distill.py --config configs/distill.yaml
     python scripts/train_distill.py --config configs/distill.yaml --lr 1e-3 --epochs 100

  2. 纯 CLI (向后兼容):
     python scripts/train_distill.py --data data.yaml --teacher model.onnx --epochs 300

优先级: CLI > YAML > argparse defaults
"""

import argparse
from pathlib import Path
from typing import Dict, Any

import yaml


def _load_yaml(yaml_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _yaml_to_flat_args(yaml_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """将 YAML 嵌套结构映射到 argparse flat 命名空间

    YAML 结构 → argparse dest 的映射关系:
        data.yaml_path      → data
        data.img_size        → img_size
        data.num_keypoints   → num_keypoints
        model.*              → fpn_channels, backbone_pretrained
        teacher.onnx_path    → teacher
        teacher.conf_thresh  → teacher_conf
        distill.*            → distill_weight_max, distill_weight_min, ...
        training.*           → epochs, batch_size, lr, ...
        output.*             → output_dir, project, ...
        misc.*               → num_workers, seed, device
        resume, weights      → resume, weights
    """
    flat = {}

    # data
    data = yaml_cfg.get('data', {})
    if 'yaml_path' in data:
        flat['data'] = data['yaml_path']
    if 'img_size' in data:
        flat['img_size'] = data['img_size']
    if 'num_keypoints' in data:
        flat['num_keypoints'] = data['num_keypoints']

    # model
    model = yaml_cfg.get('model', {})
    if 'fpn_channels' in model:
        flat['fpn_channels'] = model['fpn_channels']
    if 'backbone_pretrained' in model:
        flat['backbone_pretrained'] = model['backbone_pretrained']

    # teacher
    teacher = yaml_cfg.get('teacher', {})
    if 'onnx_path' in teacher:
        flat['teacher'] = teacher['onnx_path']
    if 'conf_thresh' in teacher:
        flat['teacher_conf'] = teacher['conf_thresh']

    # distill
    distill = yaml_cfg.get('distill', {})
    for key in ('weight_max', 'weight_min', 'box_weight', 'score_weight', 'kpt_weight'):
        if key in distill:
            flat[f'distill_{key}'] = distill[key]

    # training
    training = yaml_cfg.get('training', {})
    direct_keys = ('epochs', 'batch_size', 'lr', 'weight_decay', 'lr_scheduler',
                   'warmup_epochs', 'ema', 'ema_decay', 'freeze_backbone',
                   'freeze_epochs', 'grad_clip')
    for key in direct_keys:
        if key in training:
            flat[key] = training[key]

    # output
    output = yaml_cfg.get('output', {})
    if 'dir' in output:
        flat['output_dir'] = output['dir']
    if 'project' in output:
        flat['project'] = output['project']
    for key in ('log_interval', 'val_interval', 'save_interval'):
        if key in output:
            flat[key] = output[key]

    # top-level
    if 'resume' in yaml_cfg and yaml_cfg['resume'] is not None:
        flat['resume'] = yaml_cfg['resume']
    if 'weights' in yaml_cfg and yaml_cfg['weights'] is not None:
        flat['weights'] = yaml_cfg['weights']

    # misc
    misc = yaml_cfg.get('misc', {})
    for key in ('num_workers', 'seed', 'device'):
        if key in misc:
            flat[key] = misc[key]

    return flat


def get_parser() -> argparse.ArgumentParser:
    """获取命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='LitePose Distillation Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # YAML 配置文件
    parser.add_argument('--config', type=str, default=None,
                        help='YAML 配置文件路径 (优先级: CLI > YAML > defaults)')

    # 数据相关
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data', type=str, default=None,
                           help='YOLO 数据集配置文件路径 (data.yaml)')
    data_group.add_argument('--img-size', type=int, default=640,
                           help='输入图像大小')
    data_group.add_argument('--num-keypoints', type=int, default=17,
                           help='关键点数量')

    # 学生模型
    student_group = parser.add_argument_group('Student Model')
    student_group.add_argument('--fpn-channels', type=int, default=96,
                              help='FPN 输出通道数')
    student_group.add_argument('--backbone-pretrained', action='store_true',
                              default=True,
                              help='加载 backbone ImageNet 预训练权重')
    student_group.add_argument('--no-backbone-pretrained', dest='backbone_pretrained',
                              action='store_false',
                              help='不加载 backbone 预训练权重')

    # 教师模型
    teacher_group = parser.add_argument_group('Teacher Model')
    teacher_group.add_argument('--teacher', type=str, default=None,
                              help='教师 ONNX 文件路径')
    teacher_group.add_argument('--teacher-conf', type=float, default=0.1,
                              help='教师置信度阈值 (低阈值获取更多软标签)')

    # 蒸馏参数
    distill_group = parser.add_argument_group('Distillation')
    distill_group.add_argument('--distill-weight-max', type=float, default=1.0,
                              help='蒸馏权重最大值 (训练初期)')
    distill_group.add_argument('--distill-weight-min', type=float, default=0.25,
                              help='蒸馏权重最小值 (训练后期)')
    distill_group.add_argument('--distill-box-weight', type=float, default=1.0,
                              help='蒸馏 Box 损失权重')
    distill_group.add_argument('--distill-score-weight', type=float, default=1.0,
                              help='蒸馏 Score 损失权重')
    distill_group.add_argument('--distill-kpt-weight', type=float, default=2.0,
                              help='蒸馏 Keypoint 损失权重')

    # 训练参数
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--epochs', type=int, default=300,
                            help='训练轮数')
    train_group.add_argument('--batch-size', type=int, default=32,
                            help='批大小')
    train_group.add_argument('--lr', type=float, default=5e-4,
                            help='初始学习率')
    train_group.add_argument('--weight-decay', type=float, default=0.01,
                            help='权重衰减')
    train_group.add_argument('--lr-scheduler', type=str, default='cosine',
                            choices=['cosine', 'step'],
                            help='学习率调度器')
    train_group.add_argument('--warmup-epochs', type=int, default=10,
                            help='Warmup 轮数')

    # 高级训练选项
    adv_group = parser.add_argument_group('Advanced Training')
    adv_group.add_argument('--ema', action='store_true',
                          help='使用指数移动平均')
    adv_group.add_argument('--ema-decay', type=float, default=0.9999,
                          help='EMA 衰减率')
    adv_group.add_argument('--freeze-backbone', action='store_true',
                          default=True,
                          help='训练初期冻结 backbone')
    adv_group.add_argument('--no-freeze-backbone', dest='freeze_backbone',
                          action='store_false',
                          help='不冻结 backbone')
    adv_group.add_argument('--freeze-epochs', type=int, default=5,
                          help='冻结 backbone 的轮数')
    adv_group.add_argument('--grad-clip', type=float, default=10.0,
                          help='梯度裁剪最大范数')

    # 输出和日志
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output-dir', type=str, default='./runs/train',
                             help='输出目录')
    output_group.add_argument('--project', type=str, default='exp',
                             help='实验名称')
    output_group.add_argument('--log-interval', type=int, default=50,
                             help='日志打印间隔 (步)')
    output_group.add_argument('--val-interval', type=int, default=5,
                             help='验证间隔 (轮)')
    output_group.add_argument('--save-interval', type=int, default=10,
                             help='保存间隔 (轮)')

    # 恢复和加载
    resume_group = parser.add_argument_group('Resume')
    resume_group.add_argument('--resume', type=str, default=None,
                             help='恢复训练: 检查点路径')
    resume_group.add_argument('--weights', type=str, default=None,
                             help='加载模型权重 (不恢复优化器状态)')

    # 其他
    misc_group = parser.add_argument_group('Misc')
    misc_group.add_argument('--num-workers', type=int, default=4,
                           help='数据加载线程数')
    misc_group.add_argument('--seed', type=int, default=42,
                           help='随机种子')
    misc_group.add_argument('--device', type=str, default='cuda',
                           help='设备 (cuda 或 cpu)')

    return parser


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """从命令行参数构建配置字典

    如果指定了 --config，先加载 YAML 作为基线，再用 CLI 显式传入的参数覆盖。
    优先级: CLI > YAML > argparse defaults

    Args:
        args: 解析后的命令行参数

    Returns:
        配置字典
    """
    # 1. 如果指定了 YAML，先用 YAML 值覆盖 argparse defaults
    if args.config:
        yaml_cfg = _load_yaml(args.config)
        yaml_flat = _yaml_to_flat_args(yaml_cfg)

        # 找出 CLI 显式传入的参数 (通过重新解析不带 --config 的参数)
        parser = get_parser()
        # 解析一遍得到 defaults
        defaults = vars(parser.parse_args([]))
        # 当前 args
        current = vars(args)

        for key, yaml_val in yaml_flat.items():
            if key == 'config':
                continue
            # 仅当用户没有在 CLI 显式覆盖时，才用 YAML 值
            if key in current and current[key] == defaults.get(key):
                setattr(args, key, yaml_val)

    # 2. 验证必填项
    if args.data is None:
        raise ValueError("--data is required (via CLI or YAML config)")
    if args.teacher is None:
        raise ValueError("--teacher is required (via CLI or YAML config)")

    # 3. 构建嵌套 config dict
    config = {
        'data': {
            'yaml_path': args.data,
            'input_size': (args.img_size, args.img_size),
            'num_keypoints': args.num_keypoints,
            'num_workers': args.num_workers,
        },
        'model': {
            'fpn_channels': args.fpn_channels,
            'backbone_pretrained': args.backbone_pretrained,
            'num_keypoints': args.num_keypoints,
        },
        'teacher': {
            'onnx_path': args.teacher,
            'conf_thresh': args.teacher_conf,
        },
        'distill': {
            'weight_max': args.distill_weight_max,
            'weight_min': args.distill_weight_min,
            'box_weight': args.distill_box_weight,
            'score_weight': args.distill_score_weight,
            'kpt_weight': args.distill_kpt_weight,
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'lr_scheduler': args.lr_scheduler,
            'warmup_epochs': args.warmup_epochs,
            'ema': args.ema,
            'ema_decay': args.ema_decay,
            'freeze_backbone': args.freeze_backbone,
            'freeze_epochs': args.freeze_epochs,
            'grad_clip': args.grad_clip,
        },
        'output': {
            'dir': args.output_dir,
            'project': args.project,
            'log_interval': args.log_interval,
            'val_interval': args.val_interval,
            'save_interval': args.save_interval,
        },
        'resume': args.resume,
        'weights': args.weights,
        'seed': args.seed,
        'device': args.device,
    }

    return config
