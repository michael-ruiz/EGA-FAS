import torch.nn as nn
import torch
from model.backbone.ShffleNetv2_base_hd_v1 import ShuffleNetV2
from model.backbone.ShffleNetv2_base_hd_v1_hybrid import ShuffleNetV2Hybrid
from model.backbone.Common_fun import SELayer, h_swish
import torch.nn.functional as F

###########################################################################################
# Set to True to use ECA + Ghost modules, False for original SE + Depthwise
USE_ECA_GHOST = True

class Single_branchNet(nn.Module):
    def load_pretrain(self, pretrain_file):
        #raise NotImplementedError
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        print('')


    def __init__(self, num_class=2, use_eca_ghost=USE_ECA_GHOST):
        super(Single_branchNet,self).__init__()
        self.raw_img_moudle = ShuffleNetV2(use_eca_ghost=use_eca_ghost)
        last_channel = 64
        self.bottleneck = nn.Sequential(nn.Conv2d(last_channel, last_channel, kernel_size=1, padding=0),
                                        nn.BatchNorm2d(last_channel),
                                        nn.ReLU(inplace=True))
        self.dec = nn.Conv2d(last_channel, 1, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(last_channel, num_class)
   
    def forward(self, x):
        batch_size,C,H,W = x.shape
        x = self.raw_img_moudle.forward(x)
        x = self.bottleneck(x)
        x_map = torch.sigmoid(self.dec(x))

        output = self.avg_pool(x)

        x = self.fc(self.drop(output.squeeze(-1).squeeze(-1)))
        return x, x_map


class Single_branchNet_Hybrid(nn.Module):
    """Single-branch version of hybrid_d using Ghost+ECA backbone.

    Mirrors Single_branchNet but substitutes ShuffleNetV2Hybrid so that the
    improved backbone (Ghost modules in stages 3-4, ECA attention) is used
    without multi-modal cross-attention.  Compatible with the single-modal
    training loop in train_test/train_test_amplification.py.
    """

    def __init__(self, num_class=2, hybrid_mode='hybrid_d'):
        super(Single_branchNet_Hybrid, self).__init__()
        self.raw_img_moudle = ShuffleNetV2Hybrid(hybrid_mode=hybrid_mode, use_eca=True)
        last_channel = 64
        self.bottleneck = nn.Sequential(
            nn.Conv2d(last_channel, last_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(last_channel),
            nn.ReLU(inplace=True))
        self.dec = nn.Conv2d(last_channel, 1, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(last_channel, num_class)

    def forward(self, x):
        x = self.raw_img_moudle(x)
        x = self.bottleneck(x)
        x_map = torch.sigmoid(self.dec(x))
        output = self.avg_pool(x)
        x = self.fc(self.drop(output.squeeze(-1).squeeze(-1)))
        return x, x_map


"""
###################TWO MODALITY##########################################
class Multi_FusionNet(nn.Module):
    def load_pretrain(self, pretrain_file):
        #raise NotImplementedError
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        print('')


    def __init__(self, num_class=2):
        super(Multi_FusionNet,self).__init__()
        self.one_backbone = ShuffleNetV2()
        self.two_backbone = ShuffleNetV2()
        init_channel = 64
        self.cross_atten = CrossAtten(channel=init_channel)
        self.bottleneck = nn.Sequential(nn.Conv2d(init_channel * 2, init_channel, kernel_size=1, padding=0),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True))

        self.dec = nn.Conv2d(init_channel, 1, kernel_size=1, stride=1, padding=0)
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        color, depth, ir = x[:, 0:3,:,:], x[:, 3:6,:,:], x[:, 6:9,:,:]
        one = ir
        two = depth
        # two = color
        # backbone net
        one_feas = self.one_backbone(one)
        two_feas = self.two_backbone(two)

        two_one = self.cross_atten(two_feas, one_feas)

        # 融合后的后续操作
        fea = torch.cat([two_one, two_feas], dim=1)
        x = self.bottleneck(fea)
    
        x_map = torch.sigmoid(self.dec(x))

        # x = x.view(x.size(0), -1)

        regmap8 =  self.avgpool8(x)
        x = self.fc(self.drop(regmap8.squeeze(-1).squeeze(-1)))

        two_feas = two_feas.view(two_feas.size(0), -1)
        one_feas = one_feas.view(one_feas.size(0), -1)
        
        return x, two_feas, one_feas, None, x_map
"""
"""
###################SCORE FUSION##########################################
class Multi_FusionNet(nn.Module):
    def load_pretrain(self, pretrain_file):
        #raise NotImplementedError
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        print('')


    def __init__(self, num_class=10):
        super(Multi_FusionNet,self).__init__()
        self.rgb_backbone = ShuffleNetV2()
        self.depth_backbone = ShuffleNetV2()
        self.ir_backbone = ShuffleNetV2()
    
    def forward(self, x):
        color, depth, ir = x[:, 0:3,:,:], x[:, 3:6,:,:], x[:, 6:9,:,:]
        # backbone net
        color_feas = self.rgb_backbone(color)
        depth_feas = self.depth_backbone(depth)
        ir_feas = self.ir_backbone(ir)
        
        x = depth_feas

        depth_feas = depth_feas.view(depth_feas.size(0), -1)
        color_feas = color_feas.view(color_feas.size(0), -1)
        ir_feas = ir_feas.view(ir_feas.size(0), -1)

        return x, depth_feas, color_feas, ir_feas, ir_feas
"""
###########################################################################################
class Multi_FusionNet(nn.Module):
    def load_pretrain(self, pretrain_file):
        #raise NotImplementedError
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        print('')


    def __init__(self, num_class=10, use_eca_ghost=USE_ECA_GHOST,
                 guidance_modality='depth', adaptive_guidance=False):
        super(Multi_FusionNet,self).__init__()
        self.guidance_modality = guidance_modality
        self.adaptive_guidance = adaptive_guidance
        self.rgb_backbone = ShuffleNetV2(use_eca_ghost=use_eca_ghost)
        self.depth_backbone = ShuffleNetV2(use_eca_ghost=use_eca_ghost)
        self.ir_backbone = ShuffleNetV2(use_eca_ghost=use_eca_ghost)
        init_channel = 64
        self.cross_atten = CrossAtten(channel=init_channel)

        if self.adaptive_guidance:
            self.guidance_selector = GuidanceSelector(channel=init_channel, hidden_dim=64)

        self.bottleneck = nn.Sequential(nn.Conv2d(init_channel*3, init_channel, kernel_size=1, padding=0),
                                        nn.BatchNorm2d(init_channel),
                                        nn.ReLU(inplace=True))

        # self.final_DW = nn.Sequential(nn.Conv2d(init_channel, init_channel, kernel_size=3, stride=1, padding=1, groups=64, bias=False))
        self.dec = nn.Conv2d(init_channel, 1, kernel_size=1, stride=1, padding=0)
        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        # 3-modality version (commented out):
        # color, depth, ir = x[:, 0:3,:,:], x[:, 3:6,:,:], x[:, 6:9,:,:]

        # 4-modality version (FIXED - removed duplicate channel indices):
        # Input x expected shape: (batch, 12, H, W) where channels are:
        # [0:3] = Color, [3:6] = Depth, [6:9] = IR, [9:12] = Thermal
        color, depth, ir, thermal = x[:, 0:3,:,:], x[:, 3:6,:,:], x[:, 6:9,:,:], x[:, 9:12,:,:]
        # backbone net
        color_feas = self.rgb_backbone(color)
        depth_feas = self.depth_backbone(depth)
        ir_feas = self.ir_backbone(ir)

        # Cross-attention fusion
        guidance_weights = None  # Will be set if adaptive_guidance is True
        if self.adaptive_guidance:
            # Compute per-sample weights
            guidance_weights = self.guidance_selector(depth_feas, color_feas, ir_feas)  # [B, 3]

            # Select the modality with highest weight per sample
            selected_modality = torch.argmax(guidance_weights, dim=1)  # [B]

            # Build fused features based on selected modality per sample
            # The selected modality is the GUIDE - the other 2 modalities attend to it
            B = depth_feas.size(0)
            fea_list = []
            for i in range(B):
                mod = selected_modality[i].item()
                if mod == 0:  # depth is guide
                    fea_i = torch.cat([
                        self.cross_atten(color_feas[i:i+1], depth_feas[i:i+1]),  # color attends to depth
                        self.cross_atten(ir_feas[i:i+1], depth_feas[i:i+1]),     # ir attends to depth
                        depth_feas[i:i+1]                                         # guide features
                    ], dim=1)
                elif mod == 1:  # color is guide
                    fea_i = torch.cat([
                        self.cross_atten(depth_feas[i:i+1], color_feas[i:i+1]),  # depth attends to color
                        self.cross_atten(ir_feas[i:i+1], color_feas[i:i+1]),     # ir attends to color
                        color_feas[i:i+1]                                         # guide features
                    ], dim=1)
                else:  # ir is guide (mod == 2)
                    fea_i = torch.cat([
                        self.cross_atten(depth_feas[i:i+1], ir_feas[i:i+1]),     # depth attends to ir
                        self.cross_atten(color_feas[i:i+1], ir_feas[i:i+1]),     # color attends to ir
                        ir_feas[i:i+1]                                            # guide features
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
        # x = self.final_DW(x)

        x_map = torch.sigmoid(self.dec(x))

        # x = x.view(x.size(0), -1)
        regmap8 =  self.avgpool8(x)
        x = self.fc(self.drop(regmap8.squeeze(-1).squeeze(-1)))


        depth_feas = depth_feas.view(depth_feas.size(0), -1)
        color_feas = color_feas.view(color_feas.size(0), -1)
        ir_feas = ir_feas.view(ir_feas.size(0), -1)

        return x, depth_feas, color_feas, ir_feas, x_map, guidance_weights


class CrossAtten(nn.Module):
    def __init__(self, channel=64):
        super(CrossAtten, self).__init__()

        self.to_q = nn.Linear(channel, channel, bias = False)
        self.to_k = nn.Linear(channel, channel, bias = False)
        self.to_v = nn.Linear(channel, channel, bias = False)
        self.scale = channel ** -0.5


    def forward(self, x1, x2):

        classtoken1 = x1
        classtoken2 = x2
        
        B,C,H,W = classtoken1.shape
        h1_temp = classtoken1.view(B,C,-1)  # B, C, H*W
        h2_temp = classtoken2.view(B,C,-1)
        
        q = self.to_q(h1_temp.transpose(-2, -1))  # B, H*W, C
        k = self.to_k(h2_temp.transpose(-2, -1))
        v = self.to_v(h2_temp.transpose(-2, -1))
        crossh1_h2 = (q.transpose(-2, -1) @ k) * self.scale # B, C, C
        crossh1_h2 =F.softmax(crossh1_h2, dim=-1)
        crossedh1_h2 = (crossh1_h2 @ v.transpose(-2, -1)).contiguous()  # B, C, H*W
        crossedh1_h2 = crossedh1_h2.view(B,C,H,W)  # B, C, H, W
        """
        crossh1_h2 = h2_temp @ h1_temp.transpose(-2, -1)   # B, C, C
        crossh1_h2 =F.softmax(crossh1_h2, dim=-1)  
        crossedh1_h2 = (crossh1_h2 @ h2_temp).contiguous()  # B, C, H*W
        crossedh1_h2 = crossedh1_h2.view(B,C,H,W)  # B, C, H, W
        """
        logit = crossedh1_h2
        return logit


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