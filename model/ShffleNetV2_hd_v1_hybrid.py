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
    def __init__(self, num_modalities=3, channel=64, hidden_dim=64, ir_prior=0.0):
        super(GuidanceSelector, self).__init__()
        self.num_modalities = num_modalities
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channel * num_modalities, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_modalities),
        )
        # Learnable global modality preference (ir is index 2)
        prior = [0.0] * num_modalities
        if num_modalities > 2:
            prior[2] = ir_prior
        self.prior = nn.Parameter(torch.tensor(prior))

    def forward(self, feas_list):
        """feas_list: list of [B, C, H, W] feature maps, one per modality."""
        vecs = [self.gap(f).view(f.size(0), -1) for f in feas_list]
        concat_vec = torch.cat(vecs, dim=1)
        logits = self.mlp(concat_vec) + self.prior
        return logits  # [B, num_modalities]


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
        num_modalities: number of modalities (3=color+depth+ir, 4=+thermal)
        hybrid_mode: backbone variant ('hybrid_a', 'hybrid_b', 'hybrid_c', 'hybrid_d', etc.)
        use_eca: whether to use ECA (True) or SE (False) attention
        guidance_modality: fixed guidance modality ('depth', 'color', 'ir', 'thermal')
        adaptive_guidance: if True, use learned guidance weights
    """

    # Modality names in canonical order (matches input channel layout)
    MODALITY_NAMES = ['color', 'depth', 'ir', 'thermal']

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def print_info(self):
        print(f'Model: Multi_FusionNet_Hybrid ({self.num_modalities} modalities)')
        print(f'Hybrid mode: {self.hybrid_mode}')
        print(f'Attention: {"ECA" if self.use_eca else "SE"}')
        print(f'Backbone config: {self.backbones[0].get_config_str()}')
        fusion_name = "Gumbel-Softmax" if self.use_gumbel else ("Soft Weighted" if self.use_soft_fusion else "Hard Argmax")
        print(f'Fusion type: {fusion_name}')
        print(f'Adaptive guidance: {self.adaptive_guidance}')
        if self.adaptive_guidance:
            print(f'Guidance temperature: {self.guidance_temperature}')
        print(f'Parameters: {self.count_parameters():,}')
        print('')

    def set_gumbel_tau(self, tau):
        """Set Gumbel-Softmax temperature for annealing."""
        self.gumbel_tau = tau

    def __init__(self, num_class=10, num_modalities=3, hybrid_mode='hybrid_a', use_eca=True,
                 guidance_modality='depth', adaptive_guidance=False, fusion_type=None,
                 guidance_temperature=1.0, ir_prior=0.0):
        super(Multi_FusionNet_Hybrid, self).__init__()
        self.num_modalities = num_modalities
        self.hybrid_mode = hybrid_mode
        self.use_eca = use_eca
        self.guidance_modality = guidance_modality
        self.adaptive_guidance = adaptive_guidance
        self.guidance_temperature = guidance_temperature

        self.use_soft_fusion = True
        self.use_gumbel = False
        if fusion_type is not None:
            self.use_soft_fusion = (fusion_type == 'soft')
            self.use_gumbel = (fusion_type == 'gumbel')

        self.gumbel_tau = 1.0

        # Create one backbone per modality
        self.backbones = nn.ModuleList([
            ShuffleNetV2Hybrid(input_c=3, hybrid_mode=hybrid_mode, use_eca=use_eca)
            for _ in range(num_modalities)
        ])

        init_channel = 64
        self.cross_atten = CrossAtten(channel=init_channel)

        if self.adaptive_guidance:
            self.guidance_selector = GuidanceSelector(
                num_modalities=num_modalities, channel=init_channel,
                hidden_dim=64, ir_prior=ir_prior)

        # Each guide option: (num_modalities-1) cross-attention outputs + guide = num_modalities * 64
        self.bottleneck = nn.Sequential(
            nn.Conv2d(init_channel * num_modalities, init_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(init_channel),
            nn.ReLU(inplace=True)
        )

        self.dec = nn.Conv2d(init_channel, 1, kernel_size=1, stride=1, padding=0)
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_class)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        # Split input into per-modality 3-channel tensors
        # x shape: (batch, num_modalities*3, H, W)
        modality_inputs = [x[:, i*3:(i+1)*3, :, :] for i in range(self.num_modalities)]

        # Backbone feature extraction
        all_feas = [backbone(inp) for backbone, inp in zip(self.backbones, modality_inputs)]

        # Cross-attention fusion
        guidance_weights = None
        aux_logits = None

        if self.adaptive_guidance:
            logits = self.guidance_selector(all_feas)  # [B, num_modalities]
            guidance_weights = F.softmax(logits / self.guidance_temperature, dim=1)

            if self.use_gumbel:
                guide_feas_list = self._compute_all_guides(all_feas)

                if self.training:
                    one_hot = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True)
                else:
                    one_hot = torch.zeros_like(logits)
                    one_hot.scatter_(1, torch.argmax(logits, dim=1, keepdim=True), 1.0)

                # Auxiliary per-branch classification
                if self.training:
                    aux_logits = tuple(
                        self.fc(self.drop(self.avgpool8(self.bottleneck(gf)).squeeze(-1).squeeze(-1)))
                        for gf in guide_feas_list
                    )
                else:
                    aux_logits = None

                B = all_feas[0].size(0)
                w = one_hot.view(B, self.num_modalities, 1, 1, 1)
                fea = sum(w[:, i] * guide_feas_list[i] for i in range(self.num_modalities))

            elif self.use_soft_fusion:
                guide_feas_list = self._compute_all_guides(all_feas)

                B = all_feas[0].size(0)
                w = guidance_weights.view(B, self.num_modalities, 1, 1, 1)
                fea = sum(w[:, i] * guide_feas_list[i] for i in range(self.num_modalities))
            else:
                # Hard Argmax Selection
                selected_modality = torch.argmax(guidance_weights, dim=1)  # [B]
                B = all_feas[0].size(0)
                fea_list = []
                for i in range(B):
                    mod = selected_modality[i].item()
                    guide = all_feas[mod]
                    others = [f for j, f in enumerate(all_feas) if j != mod]
                    parts = [self.cross_atten(o[i:i+1], guide[i:i+1]) for o in others]
                    parts.append(guide[i:i+1])
                    fea_list.append(torch.cat(parts, dim=1))
                fea = torch.cat(fea_list, dim=0)
        else:
            # Fixed guidance modality
            mod_names = self.MODALITY_NAMES[:self.num_modalities]
            if self.guidance_modality not in mod_names:
                raise ValueError(f"Unknown guidance_modality: {self.guidance_modality}")
            guide_idx = mod_names.index(self.guidance_modality)
            guide_feas = all_feas[guide_idx]
            others = [f for j, f in enumerate(all_feas) if j != guide_idx]
            parts = [self.cross_atten(o, guide_feas) for o in others]
            parts.append(guide_feas)
            fea = torch.cat(parts, dim=1)

        x = self.bottleneck(fea)
        x_map = torch.sigmoid(self.dec(x))

        regmap8 = self.avgpool8(x)
        x = self.fc(self.drop(regmap8.squeeze(-1).squeeze(-1)))

        # Flatten first 3 modality features for ICMFL loss (color, depth, ir)
        feas_flat = [f.view(f.size(0), -1) for f in all_feas[:3]]

        return x, feas_flat[1], feas_flat[0], feas_flat[2], x_map, guidance_weights, aux_logits

    def _compute_all_guides(self, all_feas):
        """Compute cross-attention fusion for all guide options."""
        guide_feas_list = []
        for guide_idx in range(self.num_modalities):
            guide = all_feas[guide_idx]
            parts = [self.cross_atten(all_feas[j], guide)
                     for j in range(self.num_modalities) if j != guide_idx]
            parts.append(guide)
            guide_feas_list.append(torch.cat(parts, dim=1))
        return guide_feas_list


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
    # Test with 3 and 4 modalities
    print("=" * 70)
    print("Multi_FusionNet_Hybrid Model Comparison")
    print("=" * 70)

    for num_mod in [3, 4]:
        x = torch.randn(2, num_mod * 3, 112, 112)
        print(f"\n--- {num_mod} modalities (input: {x.shape}) ---")

        for mode, adaptive, desc in [
            ('hybrid_d', True, f'Hybrid D + Soft Fusion ({num_mod} mod)'),
        ]:
            model = Multi_FusionNet_Hybrid(
                num_class=2,
                num_modalities=num_mod,
                hybrid_mode=mode,
                use_eca=True,
                adaptive_guidance=adaptive
            )
            model.eval()
            params = model.count_parameters()

            with torch.no_grad():
                out, d, c, i, x_map, gw, _ = model(x)

            print(f"\n{desc}")
            print(f"  Backbone: {model.backbones[0].get_config_str()}")
            print(f"  Parameters: {params:,}")
            print(f"  Output shape: {out.shape}")
            if gw is not None:
                print(f"  Guidance weights: {gw[0].tolist()}")
