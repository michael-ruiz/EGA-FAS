"""
ShuffleNetV2 Hybrid Multi-Fusion Model

Hybrid architecture options:
- 'hybrid_a': Depthwise (Stage 2,3) + Ghost (Stage 4) - Conservative, stable
- 'hybrid_b': Depthwise (Stage 2) + Ghost (Stage 3,4) - More feature diversity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.ShffleNetv2_base_hd_v1_hybrid import ShuffleNetV2Hybrid


class GuidanceSelector(nn.Module):
    """Learns per-sample weights for guidance modality selection."""
    def __init__(self, channel=64, hidden_dim=64):
        super(GuidanceSelector, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channel * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, depth_feas, color_feas, ir_feas):
        # GAP: [B, 64, H, W] -> [B, 64]
        depth_vec = self.gap(depth_feas).view(depth_feas.size(0), -1)
        color_vec = self.gap(color_feas).view(color_feas.size(0), -1)
        ir_vec = self.gap(ir_feas).view(ir_feas.size(0), -1)

        # Concat -> MLP -> Softmax
        concat_vec = torch.cat([depth_vec, color_vec, ir_vec], dim=1)
        weights = F.softmax(self.mlp(concat_vec), dim=1)
        return weights  # [B, 3]


class CrossAtten(nn.Module):
    def __init__(self, channel=64):
        super(CrossAtten, self).__init__()

        self.to_q = nn.Linear(channel, channel, bias=False)
        self.to_k = nn.Linear(channel, channel, bias=False)
        self.to_v = nn.Linear(channel, channel, bias=False)
        self.scale = channel ** -0.5

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        # Reshape: [B, C, H, W] -> [B, H*W, C]
        x1_flat = x1.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        x2_flat = x2.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]

        q = self.to_q(x1_flat)  # [B, H*W, C]
        k = self.to_k(x2_flat)  # [B, H*W, C]
        v = self.to_v(x2_flat)  # [B, H*W, C]

        # Attention
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B, H*W, H*W]
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)  # [B, H*W, C]

        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


class Multi_FusionNet_Hybrid(nn.Module):
    """
    Multi-modal fusion network with hybrid ShuffleNetV2 backbone.

    Args:
        num_class: number of output classes
        hybrid_mode:
            - 'hybrid_a': DW(2,3) + Ghost(4) + Soft Fusion (differentiable)
            - 'hybrid_b': DW(2) + Ghost(3,4) + Soft Fusion (differentiable)
            - 'hybrid_c': DW(2,3) + Ghost(4) + Hard Argmax (same backbone as A)
            - 'hybrid_d': DW(2) + Ghost(3,4) + Hard Argmax (same backbone as B)
            - 'ghost': Ghost everywhere
            - 'depthwise': Depthwise everywhere
        use_eca: whether to use ECA (True) or SE (False) attention
        guidance_modality: fixed guidance modality ('depth', 'color', 'ir')
        adaptive_guidance: if True, use learned guidance weights
    """

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def print_info(self):
        print(f'Model: Multi_FusionNet_Hybrid')
        print(f'Hybrid mode: {self.hybrid_mode}')
        print(f'Backbone config: {self.rgb_backbone.get_config_str()}')
        print(f'Fusion type: {"Soft Weighted" if self.use_soft_fusion else "Hard Argmax"}')
        print(f'Adaptive guidance: {self.adaptive_guidance}')
        print(f'Parameters: {self.count_parameters():,}')
        print('')

    def __init__(self, num_class=10, hybrid_mode='hybrid_a', use_eca=True,
                 guidance_modality='depth', adaptive_guidance=False):
        super(Multi_FusionNet_Hybrid, self).__init__()
        self.hybrid_mode = hybrid_mode
        self.guidance_modality = guidance_modality
        self.adaptive_guidance = adaptive_guidance

        # Determine fusion type based on hybrid_mode
        # hybrid_a, hybrid_b use soft fusion; hybrid_c, hybrid_d use hard argmax
        self.use_soft_fusion = hybrid_mode in ['hybrid_a', 'hybrid_b', 'ghost', 'depthwise']

        # Three backbone branches with hybrid architecture
        self.rgb_backbone = ShuffleNetV2Hybrid(input_c=3, hybrid_mode=hybrid_mode, use_eca=use_eca)
        self.depth_backbone = ShuffleNetV2Hybrid(input_c=3, hybrid_mode=hybrid_mode, use_eca=use_eca)
        self.ir_backbone = ShuffleNetV2Hybrid(input_c=3, hybrid_mode=hybrid_mode, use_eca=use_eca)

        init_channel = 64
        self.cross_atten = CrossAtten(channel=init_channel)

        if self.adaptive_guidance:
            self.guidance_selector = GuidanceSelector(channel=init_channel, hidden_dim=64)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(init_channel * 3, init_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(init_channel),
            nn.ReLU(inplace=True)
        )

        self.dec = nn.Conv2d(init_channel, 1, kernel_size=1, stride=1, padding=0)
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_class)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        # 4-modality version:
        # Input x expected shape: (batch, 12, H, W) where channels are:
        # [0:3] = Color, [3:6] = Depth, [6:9] = IR, [9:12] = Thermal
        color, depth, ir, thermal = x[:, 0:3, :, :], x[:, 3:6, :, :], x[:, 6:9, :, :], x[:, 9:12, :, :]

        # Backbone feature extraction
        color_feas = self.rgb_backbone(color)
        depth_feas = self.depth_backbone(depth)
        ir_feas = self.ir_backbone(ir)

        # Cross-attention fusion
        guidance_weights = None

        if self.adaptive_guidance:
            # Compute per-sample weights
            guidance_weights = self.guidance_selector(depth_feas, color_feas, ir_feas)  # [B, 3]

            if self.use_soft_fusion:
                # Soft Weighted Fusion: compute ALL cross-attention outputs (vectorized)
                # Then combine them using learned weights - fully differentiable!

                # Depth as guide: other modalities attend to depth
                ca_c2d = self.cross_atten(color_feas, depth_feas)
                ca_i2d = self.cross_atten(ir_feas, depth_feas)
                fea_depth_guide = torch.cat([ca_c2d, ca_i2d, depth_feas], dim=1)

                # Color as guide: other modalities attend to color
                ca_d2c = self.cross_atten(depth_feas, color_feas)
                ca_i2c = self.cross_atten(ir_feas, color_feas)
                fea_color_guide = torch.cat([ca_d2c, ca_i2c, color_feas], dim=1)

                # IR as guide: other modalities attend to ir
                ca_d2i = self.cross_atten(depth_feas, ir_feas)
                ca_c2i = self.cross_atten(color_feas, ir_feas)
                fea_ir_guide = torch.cat([ca_d2i, ca_c2i, ir_feas], dim=1)

                # Soft weighted combination (fully differentiable)
                B, C, H, W = fea_depth_guide.shape
                w = guidance_weights.view(B, 3, 1, 1, 1)
                fea = (w[:, 0] * fea_depth_guide +
                       w[:, 1] * fea_color_guide +
                       w[:, 2] * fea_ir_guide)
            else:
                # Hard Argmax Selection: select single modality per sample (non-differentiable)
                selected_modality = torch.argmax(guidance_weights, dim=1)  # [B]

                B = depth_feas.size(0)
                fea_list = []
                for i in range(B):
                    mod = selected_modality[i].item()
                    if mod == 0:  # depth is guide
                        fea_i = torch.cat([
                            self.cross_atten(color_feas[i:i+1], depth_feas[i:i+1]),
                            self.cross_atten(ir_feas[i:i+1], depth_feas[i:i+1]),
                            depth_feas[i:i+1]
                        ], dim=1)
                    elif mod == 1:  # color is guide
                        fea_i = torch.cat([
                            self.cross_atten(depth_feas[i:i+1], color_feas[i:i+1]),
                            self.cross_atten(ir_feas[i:i+1], color_feas[i:i+1]),
                            color_feas[i:i+1]
                        ], dim=1)
                    else:  # ir is guide (mod == 2)
                        fea_i = torch.cat([
                            self.cross_atten(depth_feas[i:i+1], ir_feas[i:i+1]),
                            self.cross_atten(color_feas[i:i+1], ir_feas[i:i+1]),
                            ir_feas[i:i+1]
                        ], dim=1)
                    fea_list.append(fea_i)
                fea = torch.cat(fea_list, dim=0)
        else:
            # Fixed guidance modality
            if self.guidance_modality == 'depth':
                guide_other1 = self.cross_atten(depth_feas, color_feas)
                guide_other2 = self.cross_atten(depth_feas, ir_feas)
                guide_feas = depth_feas
            elif self.guidance_modality == 'color':
                guide_other1 = self.cross_atten(color_feas, depth_feas)
                guide_other2 = self.cross_atten(color_feas, ir_feas)
                guide_feas = color_feas
            elif self.guidance_modality == 'ir':
                guide_other1 = self.cross_atten(ir_feas, depth_feas)
                guide_other2 = self.cross_atten(ir_feas, color_feas)
                guide_feas = ir_feas
            else:
                raise ValueError(f"Unknown guidance_modality: {self.guidance_modality}")
            fea = torch.cat([guide_other1, guide_feas, guide_other2], dim=1)

        x = self.bottleneck(fea)
        x_map = torch.sigmoid(self.dec(x))

        regmap8 = self.avgpool8(x)
        x = self.fc(self.drop(regmap8.squeeze(-1).squeeze(-1)))

        # Flatten features for auxiliary outputs
        depth_feas_flat = depth_feas.view(depth_feas.size(0), -1)
        color_feas_flat = color_feas.view(color_feas.size(0), -1)
        ir_feas_flat = ir_feas.view(ir_feas.size(0), -1)

        return x, depth_feas_flat, color_feas_flat, ir_feas_flat, x_map, guidance_weights


def get_hybrid_a_model(num_class=2, adaptive_guidance=True):
    """
    Conservative hybrid: Depthwise (Stage 2,3) + Ghost (Stage 4)
    - Most stable training
    - Best for smaller datasets
    - Clean early features
    """
    return Multi_FusionNet_Hybrid(
        num_class=num_class,
        hybrid_mode='hybrid_a',
        use_eca=True,
        adaptive_guidance=adaptive_guidance
    )


def get_hybrid_b_model(num_class=2, adaptive_guidance=True):
    """
    Aggressive hybrid: Depthwise (Stage 2) + Ghost (Stage 3,4)
    - More feature diversity
    - Better for complex attack patterns
    - Higher capacity
    """
    return Multi_FusionNet_Hybrid(
        num_class=num_class,
        hybrid_mode='hybrid_b',
        use_eca=True,
        adaptive_guidance=adaptive_guidance
    )


def get_hybrid_c_model(num_class=2, adaptive_guidance=True):
    """
    Same backbone as Hybrid A + Hard Argmax selection
    - DW(Stage 2,3) + Ghost(Stage 4)
    - Hard argmax guidance (non-differentiable, like original model)
    - Compare with Hybrid A to see effect of soft vs hard fusion
    """
    return Multi_FusionNet_Hybrid(
        num_class=num_class,
        hybrid_mode='hybrid_c',
        use_eca=True,
        adaptive_guidance=adaptive_guidance
    )


def get_hybrid_d_model(num_class=2, adaptive_guidance=True):
    """
    Same backbone as Hybrid B + Hard Argmax selection
    - DW(Stage 2) + Ghost(Stage 3,4)
    - Hard argmax guidance (non-differentiable, like original model)
    - Compare with Hybrid B to see effect of soft vs hard fusion
    """
    return Multi_FusionNet_Hybrid(
        num_class=num_class,
        hybrid_mode='hybrid_d',
        use_eca=True,
        adaptive_guidance=adaptive_guidance
    )


if __name__ == '__main__':
    # Test all configurations
    print("=" * 70)
    print("Multi_FusionNet_Hybrid Model Comparison")
    print("=" * 70)

    x = torch.randn(2, 12, 112, 112)  # Batch of 2, 4 modalities * 3 channels

    configs = [
        ('hybrid_a', True, 'Hybrid A: DW(2,3)+Ghost(4) + Soft Fusion'),
        ('hybrid_b', True, 'Hybrid B: DW(2)+Ghost(3,4) + Soft Fusion'),
        ('hybrid_c', True, 'Hybrid C: DW(2,3)+Ghost(4) + Hard Argmax'),
        ('hybrid_d', True, 'Hybrid D: DW(2)+Ghost(3,4) + Hard Argmax'),
        ('ghost', True, 'Full Ghost + Soft Fusion'),
    ]

    for mode, adaptive, desc in configs:
        model = Multi_FusionNet_Hybrid(
            num_class=2,
            hybrid_mode=mode,
            use_eca=True,
            adaptive_guidance=adaptive
        )
        model.eval()

        params = model.count_parameters()

        with torch.no_grad():
            out, d, c, i, x_map, gw = model(x)

        print(f"\n{desc}")
        print(f"  Backbone: {model.rgb_backbone.get_config_str()}")
        print(f"  Parameters: {params:,}")
        print(f"  Output shape: {out.shape}")
        if gw is not None:
            print(f"  Guidance weights: {gw[0].tolist()}")
