"""ShuffleNetV2 Hybrid - Depthwise early, Ghost late

Hybrid architecture options:
- Option A ('hybrid_a'): Depthwise (Stage 2,3) + Ghost (Stage 4) - Conservative
- Option B ('hybrid_b'): Depthwise (Stage 2) + Ghost (Stage 3,4) - Aggressive

Based on shufflenetv2 in pytorch
[1] Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
    ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    https://arxiv.org/abs/1807.11164
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.Common_fun import SELayer, ECALayer, GhostModule


def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """
    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels // groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride, use_ghost=False):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = int(in_channels / 2)
        # 1*1 Conv
        self.primary_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True))

        if stride != 1 or in_channels != out_channels:
            if use_ghost:
                # Ghost version
                self.residual = nn.Sequential(
                    GhostModule(mid_channels, mid_channels, kernel_size=3, stride=stride, relu=True),
                    nn.Conv2d(mid_channels, int(out_channels / 2), 1),
                    nn.BatchNorm2d(int(out_channels / 2)),
                    nn.ReLU(inplace=True)
                )
            else:
                # Depthwise version
                self.residual = nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=mid_channels),
                    nn.BatchNorm2d(mid_channels),
                    nn.Conv2d(mid_channels, int(out_channels / 2), 1),
                    nn.BatchNorm2d(int(out_channels / 2)),
                    nn.ReLU(inplace=True)
                )

            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, stride=2),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, int(out_channels / 2), 1),
            )

        else:
            if use_ghost:
                # Ghost version
                self.residual = nn.Sequential(
                    GhostModule(mid_channels, mid_channels, kernel_size=3, stride=stride, relu=True),
                    nn.Conv2d(mid_channels, mid_channels, 1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                # Depthwise version
                self.residual = nn.Sequential(
                    nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=mid_channels),
                    nn.BatchNorm2d(mid_channels),
                    nn.Conv2d(mid_channels, mid_channels, 1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True)
                )
            self.shortcut = nn.Sequential()

    def forward(self, x):

        primary_conv = self.primary_conv(x)
        shortcut = self.shortcut(primary_conv)
        residual = self.residual(primary_conv)
        x1 = torch.cat([shortcut, residual], dim=1)

        if self.stride == 1 and self.out_channels == self.in_channels:
            x1 = x + x1  # residual

        x = channel_shuffle(x1, 2)

        return x


class ShuffleNetV2Hybrid(nn.Module):
    """
    Hybrid ShuffleNetV2 with configurable Depthwise/Ghost per stage.

    Args:
        class_num: number of classes
        input_c: input channels
        hybrid_mode:
            - 'hybrid_a': Depthwise (Stage 2,3) + Ghost (Stage 4) - Conservative, stable
            - 'hybrid_b': Depthwise (Stage 2) + Ghost (Stage 3,4) - More feature diversity
            - 'ghost': Ghost everywhere (same as original use_eca_ghost=True)
            - 'depthwise': Depthwise everywhere (same as original use_eca_ghost=False)
        use_eca: if True, use ECA attention; if False, use SE attention
    """

    def __init__(self, class_num=100, input_c=3, hybrid_mode='hybrid_a', use_eca=True):
        super().__init__()
        self.hybrid_mode = hybrid_mode
        self.use_eca = use_eca

        # Define which stages use Ghost based on hybrid_mode
        # stage_ghost[i] = True means stage i+2 uses Ghost
        if hybrid_mode == 'hybrid_a':
            # Conservative: Depthwise early, Ghost only at the end + Soft Fusion
            self.stage_ghost = [False, False, True]  # Stage 2, 3, 4
        elif hybrid_mode == 'hybrid_b':
            # Aggressive: Depthwise only at start, Ghost mid+late + Soft Fusion
            self.stage_ghost = [False, True, True]   # Stage 2, 3, 4
        elif hybrid_mode == 'hybrid_c':
            # Same backbone as hybrid_a + Hard Argmax
            self.stage_ghost = [False, False, True]  # Stage 2, 3, 4
        elif hybrid_mode == 'hybrid_d':
            # Same backbone as hybrid_b + Hard Argmax
            self.stage_ghost = [False, True, True]   # Stage 2, 3, 4
        elif hybrid_mode == 'ghost':
            # All Ghost (equivalent to use_eca_ghost=True)
            self.stage_ghost = [True, True, True]
        elif hybrid_mode == 'depthwise':
            # All Depthwise (equivalent to use_eca_ghost=False)
            self.stage_ghost = [False, False, False]
        else:
            raise ValueError(f"Unknown hybrid_mode: {hybrid_mode}. "
                           f"Use 'hybrid_a', 'hybrid_b', 'hybrid_c', 'hybrid_d', 'ghost', or 'depthwise'")

        stage_layers = [2, 6, 3]
        out_channels = [16, 32, 48, 64]
        init_c = 24

        self.pre = nn.Sequential(
            nn.Conv2d(input_c, init_c, 3, stride=2, padding=1),
            nn.BatchNorm2d(init_c)
        )

        # Stage 2: 2 units, 24 -> 16 channels
        self.stage2 = self._make_stage(init_c, out_channels[0], stage_layers[0], self.stage_ghost[0])
        self.se2 = ECALayer(out_channels[0]) if use_eca else SELayer(out_channels[0])

        # Stage 3: 6 units, 16 -> 32 channels
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], stage_layers[1], self.stage_ghost[1])
        self.se3 = ECALayer(out_channels[1]) if use_eca else SELayer(out_channels[1])

        # Stage 4: 3 units, 32 -> 48 channels
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], stage_layers[2], self.stage_ghost[2])
        self.se4 = ECALayer(out_channels[2]) if use_eca else SELayer(out_channels[2])

        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.stage2(x)
        x = self.se2(x)
        x = self.stage3(x)
        x = self.se3(x)
        x = self.stage4(x)
        x = self.se4(x)
        x = self.conv5(x)

        return x

    def _make_stage(self, in_channels, out_channels, repeat, use_ghost):
        layers = []
        layers.append(ShuffleUnit(in_channels, out_channels, 2, use_ghost=use_ghost))

        while repeat:
            layers.append(ShuffleUnit(out_channels, out_channels, 1, use_ghost=use_ghost))
            repeat -= 1

        return nn.Sequential(*layers)

    def get_config_str(self):
        """Return a string describing the current configuration."""
        stage_types = ['DW' if not g else 'Ghost' for g in self.stage_ghost]
        attn_type = 'ECA' if self.use_eca else 'SE'
        return f"Stage2:{stage_types[0]}, Stage3:{stage_types[1]}, Stage4:{stage_types[2]}, Attn:{attn_type}"


def shufflenetv2_hybrid_a(use_eca=True):
    """Conservative hybrid: Depthwise (Stage 2,3) + Ghost (Stage 4)"""
    return ShuffleNetV2Hybrid(input_c=3, hybrid_mode='hybrid_a', use_eca=use_eca)


def shufflenetv2_hybrid_b(use_eca=True):
    """Aggressive hybrid: Depthwise (Stage 2) + Ghost (Stage 3,4)"""
    return ShuffleNetV2Hybrid(input_c=3, hybrid_mode='hybrid_b', use_eca=use_eca)


if __name__ == '__main__':
    # Test and compare all configurations
    import torch
    from torchinfo import summary

    print("=" * 70)
    print("ShuffleNetV2 Hybrid Configurations Comparison")
    print("=" * 70)

    configs = [
        ('hybrid_a', 'Depthwise(2,3) + Ghost(4) - Conservative'),
        ('hybrid_b', 'Depthwise(2) + Ghost(3,4) - Aggressive'),
        ('ghost', 'Ghost everywhere'),
        ('depthwise', 'Depthwise everywhere'),
    ]

    x = torch.randn(1, 3, 112, 112)

    for mode, desc in configs:
        model = ShuffleNetV2Hybrid(input_c=3, hybrid_mode=mode, use_eca=True)
        model.eval()

        # Count parameters
        params = sum(p.numel() for p in model.parameters())

        # Forward pass
        with torch.no_grad():
            out = model(x)

        print(f"\n{mode.upper()}: {desc}")
        print(f"  Config: {model.get_config_str()}")
        print(f"  Parameters: {params:,}")
        print(f"  Output shape: {out.shape}")
