from torch.utils.data import DataLoader
from data_process.load_single_data import FAS_single_Dataset
from data_process.load_multi_data import FAS_multi_Dataset
from data_process.get_path import Get_path

import math

def get_dataset(config, t_path, isVal):
    load_dataset = None
    if config.is_Multi:
        FAS_Dataset = FAS_multi_Dataset
    else:
        FAS_Dataset = FAS_single_Dataset
    num_workers = config.num_workers
    batch_size_test = math.ceil(config.batch_size)
    balance = False
    if config.dataset_name in ['OULU-NPU', 'SIW', 'CASIA_FASD-RA', 'RA-CASIA_FASD', 'CASIA-FASD', 'Replay-Attack',
                                'WMCA-HQWMCA', 'HQWMCA-WMCA']:
        balance = True

    if isVal == 'train':
        curr_data = FAS_Dataset(t_path['image_dir'], t_path['prot_list'], config=config, balance=balance, isVal=isVal)
        curr_loader = DataLoader(curr_data, shuffle=True, batch_size=config.batch_size, drop_last=True,
                                 num_workers=num_workers)
    else:
        curr_data = FAS_Dataset(t_path['image_dir'], t_path['prot_list'], config=config, balance=False, isVal=isVal)
        curr_loader = DataLoader(curr_data, shuffle=False, batch_size=batch_size_test, drop_last=False,
                                 num_workers=num_workers)

    load_dataset = curr_loader
    return load_dataset


def bulid_dataset(config):
    train_path, val_path, test_path = Get_path(data_name=config.dataset_name, prot=config.prot, sub_prot=config.sub_prot)
    train_loader = get_dataset(config, train_path, isVal='train')
    val_loader = get_dataset(config, val_path, isVal='val')
    test_loader = get_dataset(config, test_path, isVal='test')
    return train_loader, val_loader, test_loader
