import os


def get_save_path(config, localtime):
    out_root = './results'
    if config.is_Multi:
        # 多模态 /results/CASIA-SURF/fusion/CDCN_112
        config.flod_name = config.model + '_Multi' + '_' + str(config.image_size)
        save_path = os.path.join(out_root, config.dataset_name, 'fusion', config.flod_name)
        if config.prot is not None:
            save_path = os.path.join(out_root, config.dataset_name, 'fusion', config.flod_name, config.prot)
        file_name = config.model + '_' + 'fusion' + '_' + str(config.image_size) + '_' + localtime + '_' + '.txt'
        return save_path, file_name
    else:
        if config.is_Wave:
            config.flod_name = config.model + '_Two' + '_' + str(config.image_size)
        else:
            config.flod_name = config.model + '_Single' + '_' + str(config.image_size)

        if config.image_modality is None:
            # 单模态 /results/OULU-NPU/CDCN_112
            save_path = os.path.join(out_root, config.dataset_name, config.flod_name)
            if config.prot is not None:
                save_path = os.path.join(save_path, config.prot)
            if config.sub_prot is not None:
                save_path = os.path.join(save_path, config.sub_prot)
            # 文件命名 CDCN_112_port3.1__时间
            if config.prot is not None and config.sub_prot is not None:
                file_name = config.model + '_' + str(config.image_size) + '_prot' + config.prot + '.' + config.sub_prot + '_' +localtime + '.txt'
            elif config.prot is not None:
                file_name = config.model + '_' + str(config.image_size) + '_prot' + config.prot + '_' + localtime + '.txt'
            else:
                file_name = config.model + '_' + str(config.image_size) + '_' + localtime + '.txt'
        else:
            # 多模态 /results/CASIA-SURF/color/CDCN_112
            # save_path = os.path.join(out_root, config.dataset_name, config.flod_name)
            save_path = os.path.join(out_root, config.dataset_name, config.image_modality, config.flod_name)
            # save_path = os.path.join(out_root, config.dataset_name, config.prot, config.image_modality, config.flod_name)
            file_name = config.model + '_' + config.image_modality + '_' + str(config.image_size) + '_' + localtime + '_' + '.txt'
        return save_path, file_name