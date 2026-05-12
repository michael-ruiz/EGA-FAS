def get_model(config, num_class, is_pruning=False):
    myself_net = ['FeatherNetB','FeatherNetA', 'ResNet_hd', 'ShffleNetV2_hd', 'ShffleNetV2_hd_v3','ShffleNetV2_hd_v4','ShffleNetV2_hd_v5',
    'ShffleNetV2_hd_v1','ShffleNetV2_hd_v2', 'MobileNetV2_hd', 'GhostNet_hd', 'FiveNet','FiveNet_5','FiveNet_1','FiveNet_6', 'FiveNet_4',
    'FiveNet_3','FiveNet_2', 'FiveNet_2_1','FiveNet_2_2','FiveNet_2_3','FiveNet_2_4','FiveNet_2_5','SixNet',
    'ShffleNetV2_hd_v1_hybrid_a', 'ShffleNetV2_hd_v1_hybrid_b', 'ShffleNetV2_hd_v1_hybrid_c', 'ShffleNetV2_hd_v1_hybrid_d',
    'ShffleNetV2_hd_v1_ablation_eca', 'ShffleNetV2_hd_v1_ablation_ghost', 'ShffleNetV2_hd_v1_ablation_adaptive',
    'ECA_FAS_ir',
    'ViT_hd_v1']

    # Handle ablation models (hardcoded params to prevent misconfiguration)
    ablation_map = {
        'ShffleNetV2_hd_v1_ablation_eca': {
            'hybrid_mode': 'depthwise', 'use_eca': True,
            'adaptive_guidance': False, 'fusion_type': None,
        },
        'ShffleNetV2_hd_v1_ablation_ghost': {
            'hybrid_mode': 'hybrid_d', 'use_eca': False,
            'adaptive_guidance': False, 'fusion_type': None,
        },
        'ShffleNetV2_hd_v1_ablation_adaptive': {
            'hybrid_mode': 'depthwise', 'use_eca': False,
            'adaptive_guidance': True, 'fusion_type': 'hard',
        },
        'ECA_FAS_ir': {
            'hybrid_mode': 'hybrid_d', 'use_eca': True,
            'adaptive_guidance': True, 'fusion_type': None,
        },
    }
    if config.model in ablation_map:
        from model.ShffleNetV2_hd_v1_hybrid import Multi_FusionNet_Hybrid
        params = ablation_map[config.model]
        guidance_modality = getattr(config, 'guidance_modality', 'depth')
        ir_prior = getattr(config, 'ir_prior', 0.0)
        num_modalities = getattr(config, 'num_modalities', 3)
        net = Multi_FusionNet_Hybrid(
            num_class=num_class,
            num_modalities=num_modalities,
            hybrid_mode=params['hybrid_mode'],
            use_eca=params['use_eca'],
            guidance_modality=guidance_modality,
            adaptive_guidance=params['adaptive_guidance'],
            fusion_type=params['fusion_type'],
            ir_prior=ir_prior,
        )
        net.print_info()
        return net

    # Handle hybrid models separately
    if config.model in ['ShffleNetV2_hd_v1_hybrid_a', 'ShffleNetV2_hd_v1_hybrid_b',
                        'ShffleNetV2_hd_v1_hybrid_c', 'ShffleNetV2_hd_v1_hybrid_d']:
        mode_map = {
            'ShffleNetV2_hd_v1_hybrid_a': 'hybrid_a',
            'ShffleNetV2_hd_v1_hybrid_b': 'hybrid_b',
            'ShffleNetV2_hd_v1_hybrid_c': 'hybrid_c',
            'ShffleNetV2_hd_v1_hybrid_d': 'hybrid_d',
        }
        hybrid_mode = mode_map[config.model]

        if not config.is_Multi:
            # Single-modal: use hybrid backbone without cross-attention
            from model.ShffleNetV2_hd_v1 import Single_branchNet_Hybrid
            net = Single_branchNet_Hybrid(num_class=num_class, hybrid_mode=hybrid_mode)
            return net

        from model.ShffleNetV2_hd_v1_hybrid import Multi_FusionNet_Hybrid
        guidance_modality = getattr(config, 'guidance_modality', 'depth')
        adaptive_guidance = getattr(config, 'adaptive_guidance', False)
        fusion_type = getattr(config, 'fusion_type', None)
        guidance_temperature = getattr(config, 'guidance_temperature', 1.0)
        num_modalities = getattr(config, 'num_modalities', 3)
        net = Multi_FusionNet_Hybrid(
            num_class=num_class,
            num_modalities=num_modalities,
            hybrid_mode=hybrid_mode,
            use_eca=True,
            guidance_modality=guidance_modality,
            adaptive_guidance=adaptive_guidance,
            fusion_type=fusion_type,
            guidance_temperature=guidance_temperature
        )
        net.print_info()
        return net

    # Handle ViT models
    if config.model == 'ViT_hd_v1':
        from model.ViT_hd_v1 import Multi_FusionNet, Single_branchNet
        if config.is_Multi:
            guidance_modality = getattr(config, 'guidance_modality', 'depth')
            adaptive_guidance = getattr(config, 'adaptive_guidance', False)
            net = Multi_FusionNet(guidance_modality=guidance_modality,
                                  adaptive_guidance=adaptive_guidance)
        else:
            net = Single_branchNet()
        return net

    if config.model in myself_net:
        if config.model in ['FeatherNetB','FeatherNetA']:
            from model.FeatherNet import Multi_FusionNet, Two_StreamNet, Single_branchNet
        elif config.model == 'ResNet_hd':
            from model.ResNet_hd import Multi_FusionNet, Single_branchNet
        elif config.model == 'ShffleNetV2_hd':
            from model.ShffleNetV2_hd import Multi_FusionNet, Single_branchNet
        elif config.model in ['ShffleNetV2_hd_v1','ShffleNetV2_hd_v2','ShffleNetV2_hd_v3','ShffleNetV2_hd_v4','ShffleNetV2_hd_v5']:
            from model.ShffleNetV2_hd_v1 import Multi_FusionNet, Single_branchNet
        elif config.model == 'MobileNetV2_hd':
            from model.MobileNetV2_hd import Multi_FusionNet, Single_branchNet
        elif config.model == 'GhostNet_hd':
            from model.GhostNet_hd import Multi_FusionNet, Single_branchNet
        elif config.model == 'SixNet':
            from model.SixNet import Multi_FusionNet, Single_branchNet
        elif config.model in ['FiveNet_5','FiveNet_1','FiveNet_6', 'FiveNet_4', 'FiveNet_3','FiveNet_2','FiveNet']:
            from model.FiveNet import Multi_FusionNet, Two_StreamNet, Single_branchNet, Two_StreamNet_Pruning
        elif config.model in ['FiveNet_2_1','FiveNet_2_2','FiveNet_2_3','FiveNet_2_4','FiveNet_2_5'] :
            from model.FiveNet_1 import Multi_FusionNet, Two_StreamNet, Single_branchNet
        else:
            raise Exception('This model name is not implemented yet.')

        if config.is_Multi:
            guidance_modality = getattr(config, 'guidance_modality', 'depth')
            adaptive_guidance = getattr(config, 'adaptive_guidance', False)
            net = Multi_FusionNet(guidance_modality=guidance_modality,
                                  adaptive_guidance=adaptive_guidance)
        elif config.is_Wave:
            if is_pruning:
                net = Two_StreamNet_Pruning()
            else:
                net = Two_StreamNet()
        else:
            net = Single_branchNet()
    elif config.model == 'Two_stream':  # 仅限于图像进行了小波变换处理
        from model_single.Two_stream import FusionNet
        net = FusionNet()
    elif config.model == 'LMFFNet':
        from model_single.LMFFNet import LFEM_B
        net = LFEM_B()
    elif config.model == 'Two_stream1':  # 仅限于图像进行了小波变换处理
        from model_single.Two_stream1 import FusionNet
        net = FusionNet()
    elif config.model == 'FaceBagNet':
        if config.is_Multi:
            from model.FaceBagNet import FusionNet
            net = FusionNet()
        else:
            from model.FaceBagNet import Net
            net = Net()
    elif config.model == 'FourNet_5':  # 仅限于图像进行了小波变换处理
        from model.FourNet_2 import Two_StreamNet
        net = Two_StreamNet()
    elif config.model == 'inceptionv4':
        from model.ref_model.InceptionV4 import inceptionv4
        net = inceptionv4()
    elif config.model == 'LightFASNet':
        from model.LightFASNet import FeatherNet_G_B
        net = FeatherNet_G_B()
    return net