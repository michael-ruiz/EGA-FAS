import argparse


def get_input_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LMFFNet')
    parser.add_argument('--image_modality', type=str, default=None)
    parser.add_argument('--image_size', type=int, default=48)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--vis_model', type=bool, default=True)

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer_test'])
    parser.add_argument('--pretrained_model', type=str, default=None)

    parser.add_argument('--num_modalities', type=int, default=1,
                        help="Number of modalities: 1=single-modal, 3=color+depth+ir, 4=color+depth+ir+thermal")
    parser.add_argument('--guidance_modality', type=str, default='depth',
                        choices=['depth', 'color', 'ir', 'thermal'],
                        help="Modality for cross-attention guidance: 'depth' (default), 'color', 'ir', 'thermal'")
    parser.add_argument('--adaptive_guidance', action='store_true', default=False,
                        help="Enable adaptive guidance selection (learns per-sample weights)")
    parser.add_argument('--fusion_type', type=str, default=None,
                        choices=['soft', 'hard', 'gumbel'],
                        help="Override fusion type: 'soft' (differentiable weighted), 'hard' (argmax), 'gumbel' (Gumbel-Softmax)")
    parser.add_argument('--gumbel_tau', type=float, default=1.0,
                        help="Initial Gumbel-Softmax temperature (annealed linearly to 0.1 during training)")
    parser.add_argument('--aux_loss_weight', type=float, default=0.0,
                        help="Weight for per-branch auxiliary loss in Gumbel fusion (0 = disabled)")
    parser.add_argument('--balance_loss_weight', type=float, default=0.0,
                        help="MoE load balancing loss weight (0 = disabled)")
    parser.add_argument('--guidance_temperature', type=float, default=1.0,
                        help="Temperature for guidance softmax (lower = sharper selection, e.g. 0.5)")
    parser.add_argument('--ir_prior', type=float, default=0.0,
                        help="Learnable IR guide prior initialisation. e.g. 2.0 → ~76%% IR at start.")
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help="Label smoothing factor for BCE loss (e.g. 0.1). Helps generalize to unseen attacks.")
    parser.add_argument('--strong_augment', action='store_true', default=False,
                        help="Use stronger augmentations (blur, noise, color shift, compression, cutout)")

    parser.add_argument('--bce', type=float, default=None,
                        help="BCE loss weight (overrides default per-modal setting)")
    parser.add_argument('--pwl', type=float, default=None,
                        help="Pixel-wise loss weight (overrides default per-modal setting)")

    parser.add_argument('--dataset_name', type=str, default='CASIA',
                        help="data_name in ['OULU-NPU', 'CASIA', 'SIW', 'RA', 'MSU', 'CASIA-SURF']")
    parser.add_argument('--prot', type=str, default=None, help="OULU-NPU: prot in ['1', '2', '3', '4']; " 
                                                               "SIW: prot in ['1', '2', '3', '4']")
    parser.add_argument('--sub_prot', type=str, default=None,
                        help="where OULU-NPU prot = '3' or '4', sub_prot is None or ['1', '2', '3', '4', '5', '6'];"
                             "SIW: prot_sub_prot in ['1_1', '2_1', '2_2', '2_3', '2_4', '3_1', '3_2']")

    config = parser.parse_args()

    return config


class All_Config_setting(object):
    def __init__(self, config):
        self.model = config.model
        self.image_modality = config.image_modality
        self.image_size = config.image_size
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.vis_model = config.vis_model
        # 测试
        self.mode = config.mode
        self.pretrained_model = config.pretrained_model
        # 单模态、多模态
        self.num_modalities = config.num_modalities
        self.is_Multi = config.num_modalities > 1
        self.is_Wave = False
        self.guidance_modality = config.guidance_modality
        self.adaptive_guidance = config.adaptive_guidance
        self.fusion_type = config.fusion_type
        self.gumbel_tau = config.gumbel_tau
        self.aux_loss_weight = config.aux_loss_weight
        self.balance_loss_weight = config.balance_loss_weight
        self.guidance_temperature = config.guidance_temperature
        self.ir_prior = config.ir_prior
        self.label_smoothing = config.label_smoothing
        self.strong_augment = config.strong_augment
        # 数据集名称、协议+子协议
        self.dataset_name = config.dataset_name
        self.prot = config.prot
        self.sub_prot = config.sub_prot
        self.num_workers = 4
        # 损失函数、梯度下降算法、学习率调整策略相关
        self.lr = 0.01
        self.optimizer_name = 'SGD'  # Adam
        self.weight_decay = 0.0005
        self.momentum = 0.9
        self.train_fold_index = -1
        """
        self.optimizer_name = 'Adam'
        self.weight_decay = 0.05
        self.eps = 1e-8
        self.betas=(0.5, 0.999)
        """
        self.lr_type = 'cosine_repeat_lr'  # step_lr
        if self.mode == 'infer_test' or self.dataset_name in ['RA-CASIA']:
            self.cycle = 1
        else:
            self.cycle = 10


        if self.is_Multi:
            self.bce =0.4 # 0.8
            self.pwl = 0.6 # 0.2
            self.cmfl = 1.5
            self.cce = 0.0
            self.ml = 0.0
        else:
            self.bce = 0.4
            self.pwl = 0.6
            self.cmfl = 0.0
            self.cce = 0.0
            self.ml = 0.0

        # CLI overrides for loss weights
        if config.bce is not None:
            self.bce = config.bce
        if config.pwl is not None:
            self.pwl = config.pwl


config_input = get_input_config()
config = All_Config_setting(config_input)