import os
import cv2
import torch
import numpy as np
import random
from skimage.color import rgb2ycbcr
from torch.utils.data import Dataset, DataLoader

from data_process.augmentation import *

RESIZE_SIZE = 112

def Data_Cut(config):
    """
    根据不同数据集的图片数量进行数据缩减，加快模型训练速度
    data_name in ['OULU-NPU', 'CASIA', 'SIW', 'RA', 'MSU', 'CASIA-SURF']
    OULU-NPU: prot in ['1', '2', '3', '4'], where prot = '3' or '4', sub_prot is None or ['1', '2', '3', '4', '5', '6']
    SIW: prot_sub_prot in ['1_1', '2_1', '2_2', '2_3', '2_4', '3_1', '3_2']
    """
    data_cut = {}
    if config.dataset_name == 'CASIA-SURF':
        data_cut = {'train': 1, 'val': 2, 'test': 30}
        if config.mode == 'infer_test':
            data_cut['test'] = 1
    elif config.dataset_name == 'WMCA':
        data_cut = {'train': 1, 'val': 10, 'test': 5}
        if config.mode == 'infer_test':
            data_cut['test'] = 1
    elif config.dataset_name == 'CASIA':
        data_cut = {'train': 2, 'val': 2, 'test': 1}
    elif config.dataset_name == 'MSU':
        data_cut = {'train': 1, 'val': 10, 'test': 10}
        if config.mode == 'infer_test':
            data_cut['test'] = 1
    elif config.dataset_name == 'RA':
        data_cut = {'train': 4, 'val': 2, 'test': 1}
    elif config.dataset_name == 'RA-CASIA':
        data_cut = {'train': 9, 'val': 1, 'test': 1}
    elif config.dataset_name == 'CASIA-RA':
        data_cut = {'train': 2, 'val': 1, 'test': 1}
    elif config.dataset_name == 'OULU-NPU':
        if config.prot in ['1', '2']:
            data_cut = {'train': 1, 'val': 5, 'test': 5}
        elif config.prot == '3':
            data_cut = {'train': 1, 'val': 5, 'test': 5}
        elif config.prot == '4':
            data_cut = {'train': 1, 'val': 5, 'test': 2}
        if config.mode == 'infer_test':
            data_cut['test'] = 1
    elif config.dataset_name == 'SIW':
        if config.prot == '1':
            data_cut = {'train': 1, 'val': 300, 'test': 300}
        elif config.prot == '2' and config.sub_prot in ['2']:
            data_cut = {'train': 15, 'val': 350, 'test': 350}
        elif config.prot == '3' or (config.prot == '2' and config.sub_prot in ['1', '3', '4']):
            data_cut = {'train': 15, 'val': 150, 'test': 150}
    elif config.dataset_name == 'VFPAD':
        data_cut = {'train': 1, 'val': 1, 'test': 1}
        if config.mode == 'infer_test':
            data_cut['test'] = 1
    elif config.dataset_name in ['CASIA-FASD', 'Replay-Attack', 'CASIA_FASD-RA', 'RA-CASIA_FASD',
                                    'WMCA-HQWMCA', 'HQWMCA-WMCA']:
        data_cut = {'train': 1, 'val': 1, 'test': 1}

    return data_cut


def load_list(list_path, data_cut):
    """
    :param list_path: train_list/val_list/test_list的路径
    :return: list中的path和label
    """
    list = []
    f = open(list_path)
    lines = f.readlines()
    i = 0
    for line in lines:
        i = i + 1
        if i % data_cut == 0:
            line = line.strip().split(' ')
            list.append(line)
    return list


def transform_balance(train_list, modality):
    print('train data balance!')
    idx = 1
    if modality is not None:
        idx = 3
    pos_list, neg_list = [], []
    for tmp in train_list:
        if int(tmp[idx]) == 1:
            pos_list.append(tmp)
        else:
            neg_list.append(tmp)
    print(f"balance: pos sample: {len(pos_list)}, neg sample: {len(neg_list)}")

    return [pos_list, neg_list]


class FAS_single_Dataset(Dataset):
    # modality = config.image_modality, image_size = config.image_size,
    # fold_index = config.train_fold_index, balance = False, isVal = 'train'
    def __init__(self, data_root, list_path, config, balance=False, isVal='train'):
        super(FAS_single_Dataset, self).__init__()

        self.data_root = data_root
        self.list_path = list_path
        self.data_name = config.dataset_name
        self.raw_modality = config.image_modality
        self.modality = config.image_modality
        if self.modality == 'ycbcr':
            self.modality = 'color'
        self.fold_index = config.train_fold_index
        self.balance = balance
        self.isVal = isVal
        self.isLocal = RESIZE_SIZE>config.image_size
        if self.isLocal:
            self.image_size = config.image_size
        else:
            self.image_size = RESIZE_SIZE
        self.channels = 3
        self.label_weight = 0.99
        self.map_size = int(config.image_size/16)
        self.get_augment()
        self.set_mode(config)


    def get_augment(self):
        if self.modality is None:
            self.augment = color_augumentor
        elif self.modality == 'color':
            self.augment = color_augumentor
        elif self.modality == 'depth':
            self.augment = depth_augumentor
        elif self.modality == 'ir':
            self.augment = ir_augumentor
        else:
            self.augment = ir_augumentor
        # if self.data_name == 'OULU-NPU':
        #     self.augment = augumentor_OULU

    def set_mode(self, config):
        data_cut = Data_Cut(config)
        if self.isVal == 'train':
            self.train_list = load_list(self.list_path, data_cut[self.isVal])
            self.num_data = len(self.train_list)
            if self.balance:
                self.train_list = transform_balance(self.train_list, self.modality)
                # self.num_data = len(self.train_list)
                # print(self.train_list)
            # random.shuffle(self.train_list)
            print(f'Load training set, the number of pictures is {self.num_data}')
        elif self.isVal == 'val':
            self.val_list = load_list(self.list_path, data_cut[self.isVal])
            self.num_data = len(self.val_list)
            print(f'Load val set, the number of pictures is {self.num_data}')
        elif self.isVal == 'test':
            self.val_list = load_list(self.list_path, data_cut[self.isVal])
            self.num_data = len(self.val_list)
            print(f'Load test set, the number of pictures is {self.num_data}')

    def single_path(self, index):
        if self.isVal == 'train':
            if self.balance:
                if random.randint(0, 1) == 0:
                    tmp_list = self.train_list[0]
                else:
                    tmp_list = self.train_list[1]

                pos = random.randint(0, len(tmp_list) - 1)
                color, label = tmp_list[pos]
            else:
                color, label = self.train_list[index]
        else:
            color, label = self.val_list[index]

        return color, label

    def Muti_path(self, index):
        img_sub_path = None
        if self.data_name == 'WMCA':
            # RGB、Color、Depth、Infrared、Thermal
            if self.isVal == 'train':
                rgb, color, depth, ir, thermal, label = self.train_list[index]
            else:
                rgb, color, depth, ir, thermal, label = self.val_list[index]
        elif self.data_name == 'CASIA-SURF':
            if self.isVal == 'train':
                color, depth, ir, label = self.train_list[index]
            else:
                color, depth, ir, label = self.val_list[index]
        elif self.data_name == 'VFPAD':
            # 2-column format: path label (single NIR modality)
            if self.isVal == 'train':
                img_sub_path, label = self.train_list[index]
            else:
                img_sub_path, label = self.val_list[index]
            return img_sub_path, label

        if self.modality == 'color':
            img_sub_path = color
        elif self.modality == 'depth':
            img_sub_path = depth
        elif self.modality == 'ir':
            img_sub_path = ir
        elif self.modality == 'rgb':
            img_sub_path = rgb
        elif self.modality == 'thermal':
            img_sub_path = thermal

        return img_sub_path, label


    def __getitem__(self, index):
        if self.modality is not None:
            img_sub_path, label = self.Muti_path(index)
        else:
            img_sub_path, label = self.single_path(index)
        if int(label) == 1:
            label = 1
            mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (self.label_weight)
        else:
            label = 0
            mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32) * (1 - self.label_weight)

        img_path = os.path.join(self.data_root, img_sub_path)
        # print(img_path)
        image = cv2.imread(img_path,1)
        if self.raw_modality == 'ycbcr':
            image = rgb2ycbcr(image)
        image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

        if self.isVal == 'train':

            image = self.augment(image, label=label, target_shape=(self.image_size, self.image_size, self.channels), isLocal=self.isLocal)
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([self.channels, self.image_size, self.image_size])
            image = image / 255.0

            return torch.FloatTensor(image), torch.LongTensor(mask), torch.LongTensor(np.asarray(label).reshape([-1]))

        else:
            image = self.augment(image, label=label, target_shape=(self.image_size, self.image_size, self.channels), is_infer = True, isLocal=self.isLocal)
            n = len(image)
            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels, self.image_size, self.image_size])
            image = image / 255.0

            return torch.FloatTensor(image), torch.LongTensor(mask), torch.LongTensor(np.asarray(label).reshape([-1]))

    def __len__(self):
        return self.num_data

if __name__ == '__main__':
    # data_root = 'E:/python-project/pytorch17/FAS/data/CASIA-SURF/'
    # list_path =  'E:/python-project/pytorch17/FAS/data/CASIA-SURF/train_list.txt'
    # list_path =  'E:/python-project/pytorch17/FAS/data/CASIA-SURF/val_private_list.txt'
    from augmentation import *
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default=-1)
    parser.add_argument('--image_modality', type=str, default=None)
    parser.add_argument('--image_size', type=int, default=48)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer_test'])
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='CASIA',
                        help="data_name in ['OULU-NPU', 'CASIA', 'SIW', 'RA', 'MSU', 'CASIA-SURF']")
    parser.add_argument('--prot', type=str, default=None, help="OULU-NPU: prot in ['1', '2', '3', '4']; "
                                                               "SIW: prot in ['1', '2', '3', '4']")
    parser.add_argument('--sub_prot', type=str, default=None,
                        help="where OULU-NPU prot = '3' or '4', sub_prot is None or ['1', '2', '3', '4', '5', '6'];"
                             "SIW: prot_sub_prot in ['1_1', '2_1', '2_2', '2_3', '2_4', '3_1', '3_2']")

    config = parser.parse_args()

    data_root, list_path = None, None
    data_name = 'OULU-NPU'
    if data_name == 'OULU-NPU':
        # OULU-NPU Dataset root
        print("========loader OUNU-NPU=========")  # 1200 videos
        data_root = 'E:/python-project/pytorch17/FAS/data/OULU-NPU-1/'
        list_path = 'E:/python-project/pytorch17/FAS/data/OULU-NPU-1/Prot/Protocol_1/Train.txt'
        modality = None

    elif data_name == 'CASIA':
        # CASIA Dataset root
        print("========loader CASIA=========")  # 64077 frames
        data_root = "E:/python-project/pytorch17/FAS/data"
        list_path = "E:/python-project/pytorch17/FAS/data/cbnData/CASIA_train.txt"
        modality = None

    image_size = 64
    isVal = 'train'
    dataset = FAS_single_Dataset(data_root, list_path, config, balance=True, isVal=isVal)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=16, drop_last=True, num_workers=0)
    num = 5
    for input, truth in train_loader:
        print(input.shape, truth)
        break